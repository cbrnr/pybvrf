import pytest

from pybvrf.pybvrf import _validate_fname


def test_validate_fname():
    # a single valid extension returns the filename with that extension
    assert _validate_fname("file.bvrf", ".bvrf").name == "file.bvrf"
    assert _validate_fname("file", ".bvrf").name == "file.bvrf"

    # multiple valid extensions returns the filename without extension
    assert _validate_fname("file.bvrf", [".bvrf", ".bvrh"]).name == "file"

    with pytest.raises(ValueError):
        # invalid extension raises ValueError
        _validate_fname("file.BVRF", ".bvrf")
        _validate_fname("file.xyz", ".bvrf")
        _validate_fname("file.bvrf", [".bvrh", ".bvri"])
        _validate_fname("file", [".bvrf", ".bvrh"])
