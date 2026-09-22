"""Minimal PyInstaller hook for eager CNN inference with bundled CUDA DLLs."""

from PyInstaller import compat
from PyInstaller.utils.hooks import (
    PY_DYLIB_PATTERNS,
    collect_data_files,
    collect_dynamic_libs,
)


module_collection_mode = "pyz+py"
warn_on_missing_hiddenimports = False

datas = collect_data_files(
    "torch",
    excludes=[
        "**/*.h",
        "**/*.hpp",
        "**/*.cuh",
        "**/*.lib",
        "**/*.cpp",
        "**/*.pyi",
        "**/*.cmake",
    ],
)
binaries = collect_dynamic_libs(
    "torch",
    search_patterns=PY_DYLIB_PATTERNS + ["*.so.*"],
)
hiddenimports = [
    "torch._C",
    "torch.amp",
    "torch.backends.cudnn",
    "torch.cuda",
    "torch.nn",
    "torch.nn.functional",
    "torch.serialization",
    "torch.utils._pytree",
]

if compat.is_win:
    hiddenimports.append("torch._C._dynamo")
