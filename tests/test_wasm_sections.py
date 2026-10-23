import pytest

from eecs116.wasm_sections import append_custom, find_custom, iter_sections, read_leb128, split_bundle, strip_custom, write_leb128

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


def test_append_find_strip_custom():
    payload = bytes(range(256)) * 3
    bundled = append_custom(TINY_MODULE, "eecs116.bundle", payload)
    assert find_custom(bundled, "eecs116.bundle") == payload
    assert find_custom(bundled, "name") == b""
    assert strip_custom(bundled, "eecs116.bundle") == TINY_MODULE
    assert split_bundle(bundled, "eecs116.bundle") == (TINY_MODULE, payload)
    # appending again replaces instead of duplicating
    rebundled = append_custom(bundled, "eecs116.bundle", b"x")
    assert find_custom(rebundled, "eecs116.bundle") == b"x"
    assert strip_custom(rebundled, "eecs116.bundle") == TINY_MODULE


def test_rejects_non_wasm():
    with pytest.raises(ValueError):
        list(iter_sections(b"not a module"))
