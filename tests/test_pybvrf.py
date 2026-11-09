# © Clemens Brunner
#
# License: BSD (3-clause)

import json
import sys

import numpy as np
import pytest

from pybvrf.pybvrf import _read_bvrm, _validate_fname


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


@pytest.mark.parametrize("n_markers", [1, 2])
def test_read_bvrm_n_markers(tmp_path, n_markers):
    lines = ["Type\tCode\tValue\tSample"]
    lines += [f"Stimulus\tS\t{i}\t{10 * i}" for i in range(1, n_markers + 1)]
    fname = tmp_path / "rec.bvrm"
    fname.write_text("\n".join(lines) + "\n", encoding="utf-8")

    markers = _read_bvrm(fname)

    assert markers.shape == (n_markers,)  # also for a single marker
    assert list(markers["Sample"]) == [10 * i for i in range(1, n_markers + 1)]


def test_import_with_mne():
    pytest.importorskip("mne")
    import pybvrf

    assert pybvrf.read_raw_bvrf.__module__ == "pybvrf.mne_io"


def _make_raw(tmp_path, system):
    from pybvrf.mne_io import RawBVRF

    (tmp_path / "rec.bvrh").touch()  # MNE checks that the file exists
    header = {
        "fname": tmp_path / "rec",
        "fs": 100.0,
        "ch_names": ["Cz", "Pz"],
        "ch_types": ["eeg", "eeg"],
        "ch_positions": {"Cz": [0, 0, 0.1], "Pz": [0, -0.05, 0.09]},
        "yaml_header": {
            "EEGModality": {"CoordinateSystems": {"EEGCoordinateSystem": system}}
        },
    }
    markers = np.array(
        [("Stimulus", "S", 1, 5), ("Stimulus", "S", 2, 7)],
        dtype=[("Type", "U8"), ("Code", "U1"), ("Value", "i8"), ("Sample", "i8")],
    )
    return RawBVRF.from_data(header, np.zeros((2, 100)), markers)


def test_montage_captrak_no_fiducial_warning(tmp_path, recwarn):
    pytest.importorskip("mne")

    raw = _make_raw(tmp_path, "CapTrak")

    assert not [w for w in recwarn if "Fiducial" in str(w.message)]
    assert raw.get_montage().get_positions()["ch_pos"]["Pz"].tolist() == [
        0,
        -0.05,
        0.09,
    ]


@pytest.mark.parametrize("system", ["Easycap", None])
def test_montage_other_system_fiducial_warning(tmp_path, system):
    pytest.importorskip("mne")

    with pytest.warns(RuntimeWarning, match="Fiducial point nasion not found"):
        _make_raw(tmp_path, system)


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
