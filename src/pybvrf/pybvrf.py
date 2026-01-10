import json
from pathlib import Path

import numpy as np
import re

DTYPES = {
    "Int16": np.int16,
    "Int32": np.int32,
    "Single": np.float32,
    "Double": np.float64,
}
UNITS = {
    "V": 1e0,
    "mV": 1e-3,
    "µV": 1e-6,
    "nV": 1e-9,
}


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
    fname = Path(fname)
    if fname.suffix in (".bvrh", ".bvrd", ".bvrm", ".bvri", ""):
        fname = fname.with_suffix("")
    else:
        raise ValueError(f"Invalid file extension {fname.suffix}")

    # read header (.bvrh)
    with open(f"{fname}.bvrh", "r", encoding="utf-8-sig") as f:
        header = json.load(f)

    dtype = DTYPES[header["EEGModality"]["BVRFFiles"]["DataFile"]["NumericDataType"]]

    fs = float(header["EEGModality"]["DataSpecific"]["SamplingFrequencyInHertz"])

    n_channels = len(header["EEGModality"]["Channels"])
    ch_names = [ch["Name"] for ch in header["EEGModality"]["Channels"]]
    ch_types = [ch["Type"].lower() for ch in header["EEGModality"]["Channels"]]
    ch_units = [ch["Unit"] for ch in header["EEGModality"]["Channels"]]

    # read binary data (.bvrd)
    data = np.fromfile(f"{fname}.bvrd", dtype=dtype)  # multiplexed format
    data = data.reshape((n_channels, -1), order="F")
    data = data * np.array([UNITS[unit] for unit in ch_units])[:, None]  # rescale to V

    # read marker file (.bvrm)
    markers = np.genfromtxt(
        f"{fname}.bvrm",
        delimiter="\t",
        names=True,
        dtype=None,
        encoding="utf-8-sig",
        autostrip=True,
    )

    # read impedance file (.bvri) if it exists
    impedances = None
    if (f := Path(f"{fname}.bvri")).is_file():
        lines = f.read_text(encoding="utf-8-sig").splitlines()

        electrode_lines = [s for s in lines if s.startswith("Electrode")]
        datetime_lines = [s for s in lines if re.match(r"^\d{4}-\d{2}-\d{2}", s)]

        if electrode_lines and datetime_lines:
            electrodes = electrode_lines[0].split("\t")[1:]
            values = datetime_lines[0].split("\t")[1:]
            impedances = {e: float(v) for e, v in zip(electrodes, values)}

    info = {
        "fname": fname,
        "ch_names": ch_names,
        "ch_types": ch_types,
        "ch_units": ch_units,
        "fs": fs,
    }

    return data, info, markers, impedances
