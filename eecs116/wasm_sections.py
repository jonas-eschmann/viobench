"""Minimal reader/writer for the section structure of a WebAssembly binary.

Only what the bundler needs: walk the sections, find/strip/append a custom section (id 0). Code is never touched,
so the compiled module is byte-identical before and after bundling except for the appended section.
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


def find_custom(wasm, name):
    """Payload bytes of the (last) custom section called name, or None."""
    payload = None
    for section in iter_sections(wasm):
        if section.id == CUSTOM_SECTION_ID and section.name == name:
            payload = bytes(wasm[section.payload_start:section.end])
    return payload


def strip_custom(wasm, name):
    """The binary without any custom section called name."""
    out = bytearray(wasm[:8])
    for section in iter_sections(wasm):
        if section.id == CUSTOM_SECTION_ID and section.name == name:
            continue
        out += wasm[section.start:section.end]
    return bytes(out)


def append_custom(wasm, name, payload):
    """The binary with a custom section called name appended (any existing one of that name is removed first)."""
    encoded_name = name.encode("utf-8")
    body = write_leb128(len(encoded_name)) + encoded_name + bytes(payload)
    return strip_custom(wasm, name) + bytes([CUSTOM_SECTION_ID]) + write_leb128(len(body)) + body


def split_bundle(wasm, name):
    """(binary without the section, payload or None) in one pass."""
    return strip_custom(wasm, name), find_custom(wasm, name)
