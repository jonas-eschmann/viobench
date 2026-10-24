import json, pathlib, shlex, subprocess, sys

build = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2]).resolve()
ar = sys.argv[3]
meson = sys.argv[4]
skipped = {"_multiarray_tests", "_umath_tests", "_rational_tests", "_struct_ufunc_tests", "_operand_flag_tests", "_simd"}
output.mkdir(parents=True, exist_ok=True)
targets = json.loads(subprocess.check_output([sys.executable, meson, "introspect", str(build), "--targets"]))
modules = []
static_libraries = []
for target in targets:
    if target["type"] != "shared module":
        continue
    so = pathlib.Path(target["filename"][0]).relative_to(build)
    name = so.name.split(".")[0]
    if name in skipped:
        continue
    commands = subprocess.check_output(["ninja", "-C", str(build), "-t", "commands", str(so)], text=True).strip().splitlines()
    link = shlex.split(commands[-1])
    objects = [token for token in link if token.endswith(".o")]
    for token in link:
        if token.endswith(".a") and token not in static_libraries:
            static_libraries.append(token)
    archive = output / f"libnumpy_{'_'.join(so.parent.parts[1:])}_{name}.a"
    archive.unlink(missing_ok=True)
    subprocess.check_call([ar, "rcs", str(archive)] + [str(build / o) for o in objects])
    dotted = ".".join(so.parent.parts) + "." + name
    modules.append({"name": dotted, "init": f"PyInit_{name}", "archive": str(archive)})
(output / "modules.json").write_text(json.dumps({"modules": modules, "static_libraries": [str(build / s) for s in static_libraries]}, indent=1))
for module in modules:
    print(module["name"], module["init"])
print(len(static_libraries), "static libraries")
