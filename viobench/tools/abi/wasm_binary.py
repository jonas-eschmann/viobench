"""Minimal reader/writer for the section structure of a WebAssembly binary.

Only what the bundler and the harness need: walk the sections, read and replace the memory and data sections. Code
is never touched, so the bundled module runs exactly the runtime's code.
"""
from dataclasses import dataclass

MAGIC = b"\0asm"
VERSION = b"\x01\0\0\0"
CUSTOM_SECTION_ID = 0


def read_leb128(data, position):
    """Unsigned LEB128 at position -> (value, next position)."""
    result = 0
    shift = 0
    while True:
        if position >= len(data):
            raise ValueError("truncated LEB128 integer")
        byte = data[position]
        position += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, position
        shift += 7
        if shift > 35:
            raise ValueError("LEB128 integer too large")


def write_leb128(value):
    if value < 0:
        raise ValueError("unsigned LEB128 cannot encode a negative value")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


@dataclass(frozen=True)
class Section:
    id: int
    name: str | None  # custom sections only
    start: int  # offset of the section id byte
    payload_start: int  # offset of the payload (after the name for custom sections)
    end: int  # offset one past the section


def iter_sections(wasm):
    if wasm[:4] != MAGIC or wasm[4:8] != VERSION:
        raise ValueError("not a WebAssembly binary (bad magic/version)")
    position = 8
    while position < len(wasm):
        section_id = wasm[position]
        start = position
        size, position = read_leb128(wasm, position + 1)
        end = position + size
        if end > len(wasm):
            raise ValueError(f"section {section_id} at {start} overruns the file")
        name = None
        payload_start = position
        if section_id == CUSTOM_SECTION_ID:
            name_length, payload_start = read_leb128(wasm, position)
            name = bytes(wasm[payload_start:payload_start + name_length]).decode("utf-8")
            payload_start += name_length
        yield Section(section_id, name, start, payload_start, end)
        position = end


# --- memory / data sections (the bundler embeds the student's zip as a data segment) -----------------------------

MEMORY_SECTION_ID = 5
DATA_SECTION_ID = 11
DATA_COUNT_SECTION_ID = 12
PAGE_BYTES = 65536
I32_CONST = 0x41
GLOBAL_GET = 0x23
END = 0x0B


def read_sleb128(data, position):
    result = 0
    shift = 0
    while True:
        if position >= len(data):
            raise ValueError("truncated signed LEB128 integer")
        byte = data[position]
        position += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            if byte & 0x40:
                result -= 1 << shift
            return result, position
        if shift > 63:
            raise ValueError("signed LEB128 integer too large")


def write_sleb128(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        done = (value == 0 and not byte & 0x40) or (value == -1 and byte & 0x40)
        out.append(byte | (0 if done else 0x80))
        if done:
            return bytes(out)


@dataclass
class Memory:
    flags: int
    minimum: int  # pages
    maximum: int | None  # pages


def parse_memory_section(payload):
    count, position = read_leb128(payload, 0)
    memories = []
    for _ in range(count):
        flags = payload[position]
        position += 1
        if flags & 0x04:
            raise ValueError("memory64 is not supported")
        minimum, position = read_leb128(payload, position)
        maximum = None
        if flags & 0x01:
            maximum, position = read_leb128(payload, position)
        memories.append(Memory(flags, minimum, maximum))
    return memories


def encode_memory_section(memories):
    out = bytearray(write_leb128(len(memories)))
    for memory in memories:
        out.append(memory.flags)
        out += write_leb128(memory.minimum)
        if memory.flags & 0x01:
            out += write_leb128(memory.maximum)
    return bytes(out)


@dataclass
class DataSegment:
    flags: int  # 0 active (memory 0), 1 passive, 2 active with explicit memory index
    memory: int
    offset: int | None  # constant offset for `i32.const` expressions, None otherwise
    offset_expression: bytes  # raw expression including the end opcode (active segments)
    data: bytes

    @staticmethod
    def active(offset, data):
        return DataSegment(0, 0, offset, bytes([I32_CONST]) + write_sleb128(offset) + bytes([END]), bytes(data))


def _parse_expression(payload, position):
    start = position
    opcode = payload[position]
    if opcode == I32_CONST:
        value, position = read_sleb128(payload, position + 1)
    elif opcode == GLOBAL_GET:
        value, position = None, read_leb128(payload, position + 1)[1]
    else:
        raise ValueError(f"unsupported data segment offset expression opcode {opcode:#x}")
    if payload[position] != END:
        raise ValueError("data segment offset expression does not end with `end`")
    return value, payload[start:position + 1], position + 1


def parse_data_section(payload):
    count, position = read_leb128(payload, 0)
    segments = []
    for _ in range(count):
        flags, position = read_leb128(payload, position)
        memory = 0
        offset, expression = None, b""
        if flags == 2:
            memory, position = read_leb128(payload, position)
        if flags in (0, 2):
            offset, expression, position = _parse_expression(payload, position)
        elif flags != 1:
            raise ValueError(f"unsupported data segment flags {flags}")
        size, position = read_leb128(payload, position)
        segments.append(DataSegment(flags, memory, offset, expression, bytes(payload[position:position + size])))
        position += size
    return segments


def encode_data_section(segments):
    out = bytearray(write_leb128(len(segments)))
    for segment in segments:
        out += write_leb128(segment.flags)
        if segment.flags == 2:
            out += write_leb128(segment.memory)
        if segment.flags in (0, 2):
            out += segment.offset_expression
        out += write_leb128(len(segment.data)) + segment.data
    return bytes(out)


def replace_sections(wasm, payloads):
    """A copy of the binary where the (non-custom) sections listed in payloads {id: payload} are replaced; a None
    payload removes the section. Sections not listed, custom sections included, are copied byte for byte."""
    out = bytearray(wasm[:8])
    for section in iter_sections(wasm):
        if section.id != CUSTOM_SECTION_ID and section.id in payloads:
            payload = payloads[section.id]
            if payload is None:
                continue
            out += bytes([section.id]) + write_leb128(len(payload)) + payload
        else:
            out += wasm[section.start:section.end]
    return bytes(out)


def section_payload(wasm, section_id):
    for section in iter_sections(wasm):
        if section.id == section_id:
            return bytes(wasm[section.payload_start:section.end])
    return None
