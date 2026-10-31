"""Make pip-installed NVIDIA cuBLAS visible to CTranslate2 without LD_LIBRARY_PATH."""

import ctypes
import sys
from collections.abc import Iterable
from pathlib import Path

CUBLAS_RELATIVE = Path("nvidia/cublas/lib/libcublas.so.12")


def find_cublas(search_paths: Iterable[str | Path] | None = None) -> Path | None:
    for entry in sys.path if search_paths is None else search_paths:
        candidate = Path(entry) / CUBLAS_RELATIVE
        if candidate.is_file():
            return candidate
    return None


def preload_cublas(search_paths: Iterable[str | Path] | None = None) -> bool:
    path = find_cublas(search_paths)
    if path is None:
        return False
    try:
        # Once loaded under its soname, CTranslate2's later dlopen("libcublas.so.12") reuses it.
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return False
    return True


def cuda_ready() -> bool:
    if not preload_cublas():
        return False
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except (ImportError, RuntimeError):
        return False
