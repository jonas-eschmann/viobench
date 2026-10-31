import argparse, json, pathlib, shlex, subprocess

parser = argparse.ArgumentParser(description="relink CPython's python.wasm with statically linked extension modules, as a command (python.wasm) or as a reactor (runtime.wasm)")
parser.add_argument("build", type=pathlib.Path, help="build directory of build.sh")
parser.add_argument("sdk", type=pathlib.Path, help="wasi-sdk directory")
parser.add_argument("output", type=pathlib.Path)
parser.add_argument("manifests", nargs="+", type=pathlib.Path)
parser.add_argument("--frozen", type=pathlib.Path, help="directory with frozen.c from freeze.py; embeds the Python modules so no stdlib directory is needed at runtime")
parser.add_argument("--reactor", type=pathlib.Path, help="C file with the exported entry points (viobench/solution/runtime/python/reactor.c): links a wasm reactor without main() that calls wasm_python_register_modules() before Py_InitializeFromConfig")
arguments = parser.parse_args()
build = arguments.build.resolve()
sdk = arguments.sdk.resolve()
output = arguments.output.resolve()
cross = build / "src/cpython/cross-build/wasm32-wasip1"
manifests = [json.loads(path.read_text()) for path in arguments.manifests]
modules = [module for manifest in manifests for module in manifest["modules"]]
archives = [module["archive"] for module in modules] + [library for manifest in manifests for library in manifest["static_libraries"]]

work = output.parent / f"link-{output.stem}"
work.mkdir(parents=True, exist_ok=True)
compile_command = [str(sdk / "bin/clang"), "--target=wasm32-wasip1", "-O2", f"-I{build}/prefix/usr/local/include/python3.14", "-c"]
declarations = "\n".join(f"PyMODINIT_FUNC {module['init']}(void);" for module in modules)
registrations = "\n".join(f'    PyImport_AppendInittab("{module["name"]}", {module["init"]});' for module in modules)
objects = [work / "main.o"]
if arguments.frozen:
    declarations += "\nextern const struct _frozen wasm_python_frozen_modules[];"
    registrations += "\n    PyImport_FrozenModules = wasm_python_frozen_modules;"
    subprocess.check_call(compile_command + ["-std=c23", str(arguments.frozen.resolve() / "frozen.c"), "-o", str(work / "frozen.o")])
    objects.append(work / "frozen.o")
if arguments.reactor:
    # the reactor registers the modules itself right before initializing Python
    (work / "main.c").write_text(f"""#include <Python.h>
{declarations}
void wasm_python_register_modules(void){{
{registrations}
}}
""")
    subprocess.check_call(compile_command + [str(arguments.reactor.resolve()), "-o", str(work / "reactor.o")])
    objects.append(work / "reactor.o")
else:
    (work / "main.c").write_text(f"""#include <Python.h>
{declarations}
int main(int argc, char** argv){{
{registrations}
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.pathconfig_warnings = {0 if arguments.frozen else 1};
    PyStatus status = PyConfig_SetBytesArgv(&config, argc, argv);
    if (!PyStatus_Exception(status)) {{
        status = Py_InitializeFromConfig(&config);
    }}
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {{
        Py_ExitStatusException(status);
    }}
    return Py_RunMain();
}}
""")
subprocess.check_call(compile_command + [str(work / "main.c"), "-o", str(work / "main.o")])

link = shlex.split((build / "python-link-command.txt").read_text())
link[0] = str(sdk / "bin/clang++")
link = [token for token in link if token != "Programs/python.o"]
link[link.index("-o") + 1] = str(output)
link += [str(path) for path in objects] + ["-fwasm-exceptions", "-Wl,--strip-debug"] + archives + ["-lc++", "-lc++abi", "-lunwind", "-lc-printscan-long-double"]
if arguments.reactor:
    link.append("-mexec-model=reactor")
subprocess.check_call(link, cwd=cross)
print(output, output.stat().st_size // (1 << 20), "MiB")
