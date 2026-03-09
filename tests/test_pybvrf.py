# © Clemens Brunner
#
# License: BSD (3-clause)

import sys

import pytest

from pybvrf.pybvrf import _validate_fname


def test_validate_fname():
    # a single valid extension returns the filename with that extension
    assert _validate_fname("file.bvrf", ".bvrf").name == "file.bvrf"
    assert _validate_fname("file", ".bvrf").name == "file.bvrf"

    # multiple valid extensions returns the filename without extension
    assert _validate_fname("file.bvrf", [".bvrf", ".bvrh"]).name == "file"
    assert _validate_fname("file", [".bvrf", ".bvrh"]).name == "file"

    with pytest.raises(ValueError):
        _validate_fname("file.BVRF", ".bvrf")  # case-sensitive

    with pytest.raises(ValueError):
        _validate_fname("file.xyz", ".bvrf")  # invalid extension

    with pytest.raises(ValueError):
        _validate_fname("file.bvrf", [".bvrh", ".bvri"])  # invalid extension


def test_import_with_mne():
    pytest.importorskip("mne")
    import pybvrf

    assert pybvrf.read_raw_bvrf.__module__ == "pybvrf.mne_io"


def test_import_without_mne(monkeypatch):
    # remove and block mne and all mne.* submodules
    for key in [k for k in sys.modules if k == "mne" or k.startswith("mne.")]:
        monkeypatch.delitem(sys.modules, key)
    monkeypatch.setitem(sys.modules, "mne", None)

    # remove all pybvrf modules to force a fresh import
    for key in [k for k in sys.modules if k == "pybvrf" or k.startswith("pybvrf.")]:
        monkeypatch.delitem(sys.modules, key)

    import pybvrf

    with pytest.raises(ImportError, match="pybvrf\\[mne\\]"):
        pybvrf.read_raw_bvrf("dummy")
