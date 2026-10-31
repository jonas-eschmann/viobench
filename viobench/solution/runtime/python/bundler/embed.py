"""Embedding a zip into the Python runtime as a data segment, and finding it again.

The runtime (viobench/solution/runtime/python/reactor.c) keeps a 32-byte descriptor in its data: `magic[16] | address | length |
version | reserved`. The bundler appends an active data segment holding the zip just above the runtime's initial
memory, grows the memory minimum accordingly and patches the descriptor's address/length in place. wasi-libc's
allocator seeds its heap only up to the initial memory end and grows beyond it afterwards, so the zip is never
touched by the heap. The module is self-contained: in frame_width() the guest shim reads the descriptor and imports
from the zip in its own memory; the host never handles the zip.
"""
import struct

from viobench.tools import abi
from viobench.tools.abi.module import runtime_descriptor
from viobench.tools.abi.wasm_binary import (DATA_COUNT_SECTION_ID, DATA_SECTION_ID, MEMORY_SECTION_ID, PAGE_BYTES, DataSegment, encode_data_section,
                                           encode_memory_section, parse_data_section, parse_memory_section, replace_sections, section_payload,
                                           write_leb128)

from . import BundleError as PackagingError


def _memory(wasm):
    payload = section_payload(wasm, MEMORY_SECTION_ID)
    if payload is None:
        raise PackagingError("the module defines no memory (imported memories are not supported)")
    memories = parse_memory_section(payload)
    if len(memories) != 1:
        raise PackagingError(f"expected one memory, found {len(memories)}")
    return memories


def _patched_descriptor(segments, descriptor, address, length):
    segment = segments[descriptor.segment]
    data = bytearray(segment.data)
    struct.pack_into("<II", data, descriptor.position + len(abi.BUNDLE_MAGIC), address, length)
    segments[descriptor.segment] = DataSegment(segment.flags, segment.memory, segment.offset, segment.offset_expression, bytes(data))


def _rebuild(wasm, memories, segments):
    payloads = {MEMORY_SECTION_ID: encode_memory_section(memories), DATA_SECTION_ID: encode_data_section(segments)}
    if section_payload(wasm, DATA_COUNT_SECTION_ID) is not None:
        payloads[DATA_COUNT_SECTION_ID] = write_leb128(len(segments))
    return replace_sections(wasm, payloads)


def strip_bundle(wasm):
    """The runtime without an embedded zip (the given binary if it has none)."""
    descriptor = runtime_descriptor(wasm)
    if descriptor is None or descriptor.length == 0:
        return wasm
    memories = _memory(wasm)
    segments = parse_data_section(section_payload(wasm, DATA_SECTION_ID))
    kept = [s for s in segments if not (s.offset == descriptor.address and len(s.data) == descriptor.length)]
    if len(kept) != len(segments) - 1:
        raise PackagingError("the embedded bundle segment does not match its descriptor")
    memories[0].minimum = descriptor.address // PAGE_BYTES
    _patched_descriptor(kept, descriptor, 0, 0)
    return _rebuild(wasm, memories, kept)


def embed_bundle(wasm, payload):
    """The runtime with payload embedded above its initial memory and the descriptor pointing at it."""
    wasm = strip_bundle(wasm)
    descriptor = runtime_descriptor(wasm)
    if descriptor is None:
        raise PackagingError("not a Python runtime: the module has no bundle descriptor (bundle on runtime.wasm)")
    memories = _memory(wasm)
    address = memories[0].minimum * PAGE_BYTES  # just above the heap's initial arena; growth continues beyond the zip
    pages = (len(payload) + PAGE_BYTES - 1) // PAGE_BYTES
    memories[0].minimum += pages
    if memories[0].maximum is not None and memories[0].maximum < memories[0].minimum:
        memories[0].maximum = memories[0].minimum
    segments = parse_data_section(section_payload(wasm, DATA_SECTION_ID))
    segments.append(DataSegment.active(address, payload))
    _patched_descriptor(segments, descriptor, address, len(payload))
    return _rebuild(wasm, memories, segments)
