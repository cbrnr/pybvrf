# © Clemens Brunner
#
# License: BSD (3-clause)

import json
import sys

import numpy as np
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


def _write_recording(path, electrodes):
    """Write a minimal two-channel-per-participant BVRF recording."""
    channels, coords = [], []
    for pid in ("P1", "P2"):
        for name in electrodes:
            channels.append(
                {
                    "Name": name,
                    "Type": "EEG" if name != "EOG" else "EOG",
                    "Unit": "µV",
                    "ResolutionPerBit": 1,
                    "ParticipantId": pid,
                }
            )
            coords.append(
                {
                    "Name": name,
                    "ParticipantId": pid,
                    "Coordinates": electrodes[name],
                }
            )
    header = {
        "BVRF": {"Version": "1.0.0"},
        "BVRFHeaderFile": {"Version": "1.0.0"},
        "Participants": [{"Id": "P1"}, {"Id": "P2"}],
        "EEGModality": {
            "BVRFFiles": {
                "DataFile": {"Version": "1.0.0", "NumericDataType": "Single"},
                "MarkerFile": {"Version": "1.0.0"},
            },
            "DataSpecific": {"SamplingFrequencyInHertz": 100},
            "CoordinateSystems": {"EEGCoordinateUnit": "mm"},
            "Channels": channels,
            "Electrodes": coords,
        },
    }
    path.with_suffix(".bvrh").write_text(json.dumps(header), encoding="utf-8")
    n = len(channels)
    np.zeros((n, 10), dtype=np.float32).T.tofile(path.with_suffix(".bvrd"))
    path.with_suffix(".bvrm").write_text(
        "Type\tCode\tValue\tSample\nStimulus\tS\t1\t5\nStimulus\tS\t2\t7\n",
        encoding="utf-8",
    )


@pytest.mark.filterwarnings("ignore:Fiducial point:RuntimeWarning")
def test_read_raw_missing_coordinates(tmp_path):
    pytest.importorskip("mne")
    from pybvrf import read_raw_bvrf

    fname = tmp_path / "rec"
    names = ["Cz", "Pz", "EOG"]
    _write_recording(fname, {"Cz": [0, 0, 100], "Pz": [0, -50, 90], "EOG": None})

    # combined and split recordings used to raise a TypeError
    raw = read_raw_bvrf(fname)
    raws = read_raw_bvrf(fname, split=True)

    for r in [raw, *raws.values()]:
        locs = {ch["ch_name"]: ch["loc"][:3] for ch in r.info["chs"]}
        for name, loc in locs.items():
            has_pos = np.any(loc != 0) and np.all(np.isfinite(loc))
            assert has_pos == (not name.startswith("EOG"))
    assert raw.ch_names == [f"{ch} ({p})" for p in ("P1", "P2") for ch in names]
    assert raws["P1"].ch_names == names
