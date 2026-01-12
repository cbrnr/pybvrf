import json
import re
from pathlib import Path

import numpy as np


def read_bvrf(fname, participants=None):
    """Read BrainVision Recording Format (BVRF) recording.

    Parameters
    ----------
    fname : str | Path
        Path to the BVRF file (either without extension or one of `.bvrh`, `.bvrd`,
        `.bvrm`, or `.bvri`).
    participants : str | list of str | None
        Participant identifier(s) to read. If None (default), return data for all
        participants.

    Returns
    -------
    TODO

    Notes
    -----
    A BVRF recording consists of multiple files, which are expected to be available in
    the same directory. The required files are:
    - `<fname>.bvrh` (header file)
    - `<fname>.bvrd` (data file)
    - `<fname>.bvrm` (marker file)

    Optionally, the following file may also be present:
    - `<fname>.bvri` (impedance file)

    See https://www.brainproducts.com/download/bvrf-reference-specification/ for the
    official BVRF specification.
    """
    # sanitize fname
    fname = Path(fname).expanduser().resolve()
    if fname.suffix in (".bvrh", ".bvrd", ".bvrm", ".bvri", ""):
        fname = fname.with_suffix("")
    else:
        raise ValueError(f"Invalid file extension {fname.suffix}")

    header = _read_bvrh(f"{fname}.bvrh")
    data = _read_bvrd(
        f"{fname}.bvrd", header["dtype"], header["n_channels"], header["ch_units"]
    )
    markers = _read_bvrm(f"{fname}.bvrm")
    impedances = (
        _read_bvri(f"{fname}.bvri") if (fname.with_suffix(".bvri")).is_file() else None
    )

    return header, data, markers, impedances


def _read_bvrh(fname):
    with open(fname, "r", encoding="utf-8-sig") as f:
        header = json.load(f)

    DTYPES = {
        "Int16": np.int16,
        "Int32": np.int32,
        "Single": np.float32,
        "Double": np.float64,
    }

    if "Participants" in header:
        participant_ids = [p["Id"] for p in header["Participants"]]
    else:
        participant_ids = None

    return {
        "fname": fname,
        "dtype": DTYPES[
            header["EEGModality"]["BVRFFiles"]["DataFile"]["NumericDataType"]
        ],
        "fs": float(header["EEGModality"]["DataSpecific"]["SamplingFrequencyInHertz"]),
        "n_channels": len(header["EEGModality"]["Channels"]),
        "ch_names": [ch["Name"] for ch in header["EEGModality"]["Channels"]],
        "ch_types": [ch["Type"].lower() for ch in header["EEGModality"]["Channels"]],
        "ch_units": [ch["Unit"] for ch in header["EEGModality"]["Channels"]],
        "n_participants": len(header.get("Participants", [1])),
        "participant_ids": participant_ids,
        "yaml_header": header,
    }


def _read_bvrd(fname, dtype, n_channels, ch_units):
    data = np.fromfile(fname, dtype=dtype)
    data = data.reshape((n_channels, -1), order="F")
    UNITS = {"V": 1e0, "mV": 1e-3, "µV": 1e-6, "nV": 1e-9}
    return data * np.array([UNITS[unit] for unit in ch_units])[:, None]  # rescale to V


def _read_bvrm(fname):
    return np.genfromtxt(
        fname,
        delimiter="\t",
        names=True,
        dtype=None,
        encoding="utf-8-sig",
        autostrip=True,
    )


def _read_bvri(fname):
    lines = Path(fname).read_text(encoding="utf-8-sig").splitlines()

    electrode_lines = [s for s in lines if s.startswith("Electrode")]
    datetime_lines = [s for s in lines if re.match(r"^\d{4}-\d{2}-\d{2}", s)]

    if electrode_lines and datetime_lines:
        electrodes = electrode_lines[0].split("\t")[1:]
        values = datetime_lines[0].split("\t")[1:]
        return {e: float(v) for e, v in zip(electrodes, values)}
    return None
