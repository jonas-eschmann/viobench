import marshal, pathlib, sys

# usage: freeze.py OUTPUT_DIR HOST_DIR:GUEST_DIR:MODULE_PREFIX ...; must run on the same CPython minor version as the target (bytecode format)
output = pathlib.Path(sys.argv[1]).resolve()
roots = [argument.split(":") for argument in sys.argv[2:]]
excluded_directories = {"__pycache__", "test", "tests", "idlelib", "tkinter", "turtledemo", "ensurepip", "venv", "__phello__", "pydoc_data", "site-packages", "ctypes", "sqlite3", "curses", "_examples", "f2py", "_pyinstaller"}
excluded_files = {"turtle.py", "__hello__.py"}

output.mkdir(parents=True, exist_ok=True)
blob = bytearray()
entries = []
for host, guest, module_prefix in roots:
    host = pathlib.Path(host)
    for path in sorted(host.rglob("*.py")):
        relative = path.relative_to(host)
        if excluded_directories & set(relative.parts[:-1]) or relative.name in excluded_files:
            continue
        parts = list(relative.with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts = parts[:-1]
        name = ".".join(([module_prefix] if module_prefix else []) + parts)
        code = compile(path.read_bytes(), f"{guest}/{relative.as_posix()}", "exec", dont_inherit=True)
        data = marshal.dumps(code)
        entries.append((name, len(blob), len(data), int(is_package)))
        blob += data
(output / "frozen.bin").write_bytes(blob)
table = "\n".join(f'    {{"{name}", wasm_python_frozen_blob + {offset}, {size}, {is_package}}},' for name, offset, size, is_package in entries)
(output / "frozen.c").write_text(f"""#include <Python.h>
static const unsigned char wasm_python_frozen_blob[] = {{
#embed "frozen.bin"
}};
const struct _frozen wasm_python_frozen_modules[] = {{
{table}
    {{0, 0, 0, 0}},
}};
""")
print(len(entries), "modules,", len(blob) // (1 << 20), "MiB bytecode")
