"""Convert the pinned ORB vocabulary to compact records and emit an incbin assembly source."""
import argparse
import struct
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('archive', type=Path)
parser.add_argument('binary', type=Path)
parser.add_argument('assembly', type=Path)
parser.add_argument('platform', choices=['WASI', 'Darwin', 'other'])
args = parser.parse_args()
record = struct.Struct('<IB3x32sd')
with tarfile.open(args.archive, 'r:gz') as archive, args.binary.open('wb') as output:
    source = archive.extractfile('ORBvoc.txt')
    header = list(map(int, source.readline().split()))
    if header != [10, 6, 0, 0]:
        raise ValueError(f'Unexpected ORB vocabulary header: {header}')
    output.write(struct.pack('<5I4x', *header, 0))
    count = 0
    for row in source:
        fields = row.split()
        if not fields:
            continue
        if len(fields) != 35:
            raise ValueError('Invalid ORB vocabulary record')
        parent, leaf = map(int, fields[:2])
        if not 0 <= parent <= count or leaf not in (0, 1):
            raise ValueError('Invalid ORB vocabulary tree')
        output.write(record.pack(parent, leaf, bytes(map(int, fields[2:34])), float(fields[34])))
        count += 1
    output.seek(16)
    output.write(struct.pack('<I', count))
filename = args.binary.resolve().as_posix().replace('"', '\\"')
if args.platform == 'Darwin':
    assembly = '.section __TEXT,__const\n.globl _orb_vocabulary\n.p2align 4\n_orb_vocabulary:\n'
else:
    section = '.section .rodata.orb_vocabulary,"",@\n' if args.platform == 'WASI' else '.section .rodata\n'
    assembly = section + '.globl orb_vocabulary\n.type orb_vocabulary,@object\n.p2align 4\norb_vocabulary:\n'
assembly += f'.incbin "{filename}"\n'
if args.platform != 'Darwin':
    assembly += '.size orb_vocabulary, .-orb_vocabulary\n'
args.assembly.write_text(assembly)
print(f'Embedded {count:,} vocabulary nodes ({args.binary.stat().st_size:,} bytes)')
