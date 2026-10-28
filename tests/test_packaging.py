import pytest

from relpose.tools.abi.wasm_binary import iter_sections, read_leb128, write_leb128

# the smallest valid module: magic + version, no sections
EMPTY_MODULE = b"\0asm\x01\0\0\0"
# a type section with one empty function type and a custom section "name" with an empty payload
TINY_MODULE = EMPTY_MODULE + bytes([1, 4, 1, 0x60, 0, 0]) + bytes([0, 5, 4]) + b"name"


@pytest.mark.parametrize("value", [0, 1, 127, 128, 300, 2**21, 2**32 - 1])
def test_leb128_roundtrip(value):
    encoded = write_leb128(value)
    assert read_leb128(encoded, 0) == (value, len(encoded))


def test_iter_sections_reads_ids_and_names():
    sections = list(iter_sections(TINY_MODULE))
    assert [section.id for section in sections] == [1, 0]
    assert sections[1].name == "name"
    assert sections[1].end == len(TINY_MODULE)


def test_rejects_non_wasm():
    with pytest.raises(ValueError):
        list(iter_sections(b"not a module"))


def test_memory_and_data_section_roundtrip():
    from relpose.tools.abi.wasm_binary import (DATA_SECTION_ID, MEMORY_SECTION_ID, DataSegment, Memory, encode_data_section, encode_memory_section,
                                       parse_data_section, parse_memory_section, read_sleb128, replace_sections, section_payload, write_sleb128)

    for value in (0, 1, 63, 64, -1, -64, -65, 2**31 - 1, -2**31):
        encoded = write_sleb128(value)
        assert read_sleb128(encoded, 0) == (value, len(encoded))
    memory = encode_memory_section([Memory(0x01, 3, 10)])
    data = encode_data_section([DataSegment.active(1024, b"hello"), DataSegment(1, 0, None, b"", b"passive")])
    module = EMPTY_MODULE + bytes([MEMORY_SECTION_ID, len(memory)]) + memory + bytes([DATA_SECTION_ID, len(data)]) + data
    assert parse_memory_section(section_payload(module, MEMORY_SECTION_ID)) == [Memory(0x01, 3, 10)]
    segments = parse_data_section(section_payload(module, DATA_SECTION_ID))
    assert [(s.flags, s.offset, s.data) for s in segments] == [(0, 1024, b"hello"), (1, None, b"passive")]
    grown = replace_sections(module, {MEMORY_SECTION_ID: encode_memory_section([Memory(0x01, 5, 10)])})
    assert parse_memory_section(section_payload(grown, MEMORY_SECTION_ID))[0].minimum == 5
    assert section_payload(grown, DATA_SECTION_ID) == section_payload(module, DATA_SECTION_ID)
