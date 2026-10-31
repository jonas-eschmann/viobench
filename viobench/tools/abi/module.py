"""Read-only inspection of a solution module: does it carry the Python runtime's bundle descriptor, and is it filled?

The descriptor (`magic[16] | address u32 | length u32 | version u32 | reserved u32`) lives in the runtime's data
(viobench/solution/runtime/python/reactor.c); the bundler patches it when it embeds a zip. The runner only needs to
*read* it, to tell a bare runtime from a finished solution; the writing side lives with the Python runtime.
"""
import struct
from dataclasses import dataclass

from . import BUNDLE_MAGIC, AbiError
from .wasm_binary import DATA_SECTION_ID, parse_data_section, section_payload


@dataclass
class Descriptor:
    segment: int  # index of the data segment holding the descriptor
    position: int  # byte offset of the descriptor inside that segment's data
    memory_address: int  # linear-memory address of the descriptor
    address: int  # linear-memory address of the embedded zip (0 = none)
    length: int  # bytes of the embedded zip (0 = none)


def runtime_descriptor(wasm):
    """The Python runtime's bundle descriptor, or None for a module without one (a native solution)."""
    payload = section_payload(wasm, DATA_SECTION_ID)
    if payload is None:
        return None
    found = None
    for index, segment in enumerate(parse_data_section(payload)):
        position = segment.data.find(BUNDLE_MAGIC)
        while position != -1:
            if found is not None:
                raise AbiError("the module contains more than one bundle descriptor")
            if segment.offset is None:
                raise AbiError("the bundle descriptor lives in a segment with a non-constant offset")
            address, length = struct.unpack_from("<II", segment.data, position + len(BUNDLE_MAGIC))
            found = Descriptor(index, position, segment.offset + position, address, length)
            position = segment.data.find(BUNDLE_MAGIC, position + 1)
    return found


def embedded_bundle(wasm):
    """The zip embedded by the Python runtime's bundler, or None (read-only; the bundler lives with the runtime)."""
    descriptor = runtime_descriptor(wasm)
    if descriptor is None or descriptor.length == 0:
        return None
    for segment in parse_data_section(section_payload(wasm, DATA_SECTION_ID)):
        if segment.offset == descriptor.address and len(segment.data) == descriptor.length:
            return segment.data
    raise AbiError("the embedded bundle segment does not match its descriptor")
