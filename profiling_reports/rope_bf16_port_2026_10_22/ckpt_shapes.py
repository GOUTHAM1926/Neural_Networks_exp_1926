# Print the weights stored in a DCP checkpoint's metadata.bin (2026-10-22). Format (Metadata.h, TensorMetadata):
# u64 name_len, name, u64 n_dims, int64 dims[n_dims], int dtype, ... ; only names "param_NNNNNN" are read.
# usage: python3 ckpt_shapes.py <path to metadata.bin>
import re, struct, sys
b = open(sys.argv[1], "rb").read()
shapes = {}
for m in re.finditer(rb"param_\d{6}", b):
    e = m.end()
    if struct.unpack_from("<Q", b, m.start() - 8)[0] != 12:   # must be a length-prefixed name of 12 chars
        continue
    n = struct.unpack_from("<Q", b, e)[0]
    if 1 <= n <= 4:
        dims = struct.unpack_from("<%dq" % n, b, e + 8)
        dt = struct.unpack_from("<i", b, e + 8 + 8 * n)[0]
        shapes.setdefault(m.group(0).decode(), (dims, dt))
names = sorted(shapes)
print(f"{len(names)} weights (param_000000 .. {names[-1]})")
for k in names[:11] + ["..."] + names[-12:]:
    print(k if k == "..." else f"{k}  shape {list(shapes[k][0])}  dtype-code {shapes[k][1]}")
