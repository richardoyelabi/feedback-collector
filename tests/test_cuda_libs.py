import ctypes

from feedback_collector import cuda_libs
from feedback_collector.cuda_libs import find_cublas, preload_cublas


def make_lib(root, relative="nvidia/cublas/lib/libcublas.so.12"):
    lib = root / relative
    lib.parent.mkdir(parents=True)
    lib.touch()
    return lib


def test_find_cublas_in_pip_layout(tmp_path):
    lib = make_lib(tmp_path)
    assert find_cublas([tmp_path]) == lib


def test_find_cublas_ignores_cuda_13(tmp_path):
    make_lib(tmp_path, "nvidia/cu13/lib/libcublas.so.13")
    assert find_cublas([tmp_path]) is None


def test_preload_cublas_loads_globally(tmp_path, monkeypatch):
    lib, calls = make_lib(tmp_path), []
    monkeypatch.setattr(cuda_libs.ctypes, "CDLL", lambda path, mode: calls.append((path, mode)))
    assert preload_cublas([tmp_path]) is True
    assert calls == [(str(lib), ctypes.RTLD_GLOBAL)]


def test_preload_cublas_reports_unloadable_file(tmp_path):
    make_lib(tmp_path)  # an empty file is not a shared object
    assert preload_cublas([tmp_path]) is False
