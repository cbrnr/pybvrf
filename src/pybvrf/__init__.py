# © Clemens Brunner
#
# License: BSD (3-clause)

from importlib.metadata import PackageNotFoundError, version

from pybvrf.pybvrf import read_bvrf, read_bvrf_header
from pybvrf.utils import split_participants


def __getattr__(name):
    if name == "read_raw_bvrf":
        try:
            from pybvrf.mne_io import read_raw_bvrf
        except ImportError:

            def read_raw_bvrf(*args, **kwargs):
                raise ImportError(
                    "MNE-Python is required to use read_raw_bvrf(). Install pybvrf with"
                    " the 'mne' extra (pybvrf[mne])."
                )

        return read_raw_bvrf
    raise AttributeError(f"module 'pybvrf' has no attribute {name!r}")


try:
    __version__ = version("pybvrf")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["read_bvrf", "read_bvrf_header", "read_raw_bvrf", "split_participants"]
