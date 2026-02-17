# © Clemens Brunner
#
# License: BSD (3-clause)

from importlib.metadata import PackageNotFoundError, version

from pybvrf.export import read_raw_bvrf
from pybvrf.pybvrf import read_bvrf, read_bvrf_header
from pybvrf.utils import split_participants

try:
    __version__ = version("pybvrf")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["read_bvrf", "read_bvrf_header", "read_raw_bvrf", "split_participants"]
