"""Guest-side glue between relpose/solution/runtime/python/reactor.c and the student's solution; frozen into runtime.wasm.

init() receives the config JSON and (when the wasm carries one) the zip of the student's directory, makes the
zip importable, imports `solution:Solution` and constructs it. step() wraps the raw
buffers in numpy arrays, calls Solution.step and writes the pose back. Status codes mirror relpose/abi.py.
"""
import importlib
import importlib.abc
import importlib.util
import io
import json
import os
import sys
import traceback
import types
import zipfile

import numpy as np

BUNDLE_ROOT = "/bundle"  # virtual location used for __file__ of bundled modules
BUNDLE_MODULE = "solution_bundle"  # `import solution_bundle` gives students their data files
SOLUTION_MODULE = "solution"  # the entry point is always class Solution in solution.py (or solution/__init__.py)
SOLUTION_CLASS = "Solution"
SOLUTION_DIR_VARIABLE = "RELPOSE_SOLUTION_DIR"  # dev mode: the host preopened the directory instead of bundling it
IMU_ROW_VALUES = 7
POSE_VALUES = 7

STATUS_OK = 0
STATUS_NO_ESTIMATE = 1
STATUS_EXCEPTION = -1
STATUS_PROTOCOL = -2
STATUS_RUNTIME = -3
STATUS_NO_SOLUTION = -4
STATUS_NOT_FINITE = -5


class Bundle:
    """Read-only view of the files the student shipped next to their code (`bundle.read_bytes("net.onnx")`)."""

    def __init__(self, data):
        self._zip = zipfile.ZipFile(io.BytesIO(data))
        self.names = frozenset(info.filename for info in self._zip.infolist() if not info.is_dir())

    def exists(self, name):
        return name in self.names

    def read_bytes(self, name):
        try:
            return self._zip.read(name)
        except KeyError:
            raise FileNotFoundError(f"{name!r} is not in the bundle (have {sorted(self.names)[:20]}...)") from None

    def read_text(self, name, encoding="utf-8"):
        return self.read_bytes(name).decode(encoding)

    def open(self, name):
        """File-like object for APIs that want one, e.g. np.load(bundle.open("weights.npz"))."""
        return io.BytesIO(self.read_bytes(name))

    def listdir(self, prefix=""):
        prefix = prefix.strip("/")
        prefix = prefix + "/" if prefix else ""
        return sorted(name[len(prefix):] for name in self.names if name.startswith(prefix))


class DirectoryBundle(Bundle):
    """Same interface over a preopened directory (dev mode)."""

    def __init__(self, directory):
        self.directory = directory
        names = []
        for root, _, files in os.walk(directory):
            for file in files:
                names.append(os.path.relpath(os.path.join(root, file), directory).replace(os.sep, "/"))
        self.names = frozenset(names)

    def read_bytes(self, name):
        with open(os.path.join(self.directory, name), "rb") as file:
            return file.read()


class BundleImporter(importlib.abc.MetaPathFinder, importlib.abc.SourceLoader):
    """Imports modules and packages straight from the in-memory zip; source is kept so tracebacks show lines."""

    def __init__(self, bundle):
        self.bundle = bundle

    def _locate(self, fullname):
        relative = fullname.replace(".", "/")
        for name, is_package in ((f"{relative}/__init__.py", True), (f"{relative}.py", False)):
            if name in self.bundle.names:
                return name, is_package
        return None, False

    def find_spec(self, fullname, path=None, target=None):
        name, is_package = self._locate(fullname)
        if name is None:
            return None
        search = [f"{BUNDLE_ROOT}/{name.rpartition('/')[0]}"] if is_package else None
        return importlib.util.spec_from_file_location(fullname, f"{BUNDLE_ROOT}/{name}", loader=self, submodule_search_locations=search)

    def get_filename(self, fullname):
        name, _ = self._locate(fullname)
        if name is None:
            raise ImportError(f"{fullname} is not in the bundle", name=fullname)
        return f"{BUNDLE_ROOT}/{name}"

    def get_data(self, path):
        prefix = BUNDLE_ROOT + "/"
        if not path.startswith(prefix):
            raise OSError(f"{path} is outside the bundle")
        try:
            return self.bundle.read_bytes(path[len(prefix):])
        except FileNotFoundError as error:
            raise OSError(str(error)) from None

    def path_stats(self, path):
        raise OSError("bundled sources are not cached as bytecode")


_solution = None
_shape = None


def _install_bundle_module(bundle):
    module = types.ModuleType(BUNDLE_MODULE, Bundle.__doc__)
    for attribute in ("exists", "read_bytes", "read_text", "open", "listdir"):
        setattr(module, attribute, getattr(bundle, attribute))
    module.names = bundle.names
    module.bundle = bundle
    sys.modules[BUNDLE_MODULE] = module


def init(config_view, bundle_view):
    """Returns (status, frame_bytes)."""
    global _solution, _shape
    config = json.loads(bytes(config_view).decode("utf-8"))
    height, width = int(config["height"]), int(config["width"])
    _shape = (height, width, 3)  # always RGB (docs/ABI.md)
    frame_bytes = height * width * 3

    if bundle_view is not None:
        bundle = Bundle(bytes(bundle_view))
        sys.meta_path.insert(0, BundleImporter(bundle))
    else:
        directory = os.environ.get(SOLUTION_DIR_VARIABLE)
        if not directory:
            print("runtime: no solution code (the wasm carries no bundle and no directory was mounted)", file=sys.stderr)
            return STATUS_NO_SOLUTION, 0
        bundle = DirectoryBundle(directory)
        sys.path.insert(0, directory)
    _install_bundle_module(bundle)
    try:
        module = importlib.import_module(SOLUTION_MODULE)
        factory = getattr(module, SOLUTION_CLASS)
        _solution = factory(config)
    except BaseException:
        traceback.print_exc()
        return STATUS_EXCEPTION, 0
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    return STATUS_OK, frame_bytes


def step(t, frame_view, imu_view, out_view):
    try:
        frame = np.frombuffer(frame_view, dtype=np.uint8).reshape(_shape)  # zero-copy, read-only, valid until the next step
        imu = np.frombuffer(imu_view, dtype=np.float64).reshape(-1, IMU_ROW_VALUES).copy()
        pose = _solution.step(float(t), frame, imu)
        if pose is None:
            return STATUS_NO_ESTIMATE
        values = np.asarray(pose, dtype=np.float64).reshape(-1)
        if values.shape != (POSE_VALUES,):
            raise ValueError(f"step() must return 7 values [x, y, z, qx, qy, qz, qw] or None, got shape {values.shape}")
        np.frombuffer(out_view, dtype=np.float64)[:] = values
        return STATUS_OK if np.all(np.isfinite(values)) else STATUS_NOT_FINITE
    except BaseException:
        traceback.print_exc()
        return STATUS_EXCEPTION
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
