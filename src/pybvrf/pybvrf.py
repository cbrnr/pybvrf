import json
import re
from pathlib import Path

import numpy as np


def _read_bvrh(fname):
    """Read header from a BrainVision Recording Format (BVRF) recording.

    Parameters
    ----------
    fname : str | Path
        Path to the BVRF header file (with or without `.bvrh` extension).

    Returns
    -------
    header : dict
        Dictionary containing header information with the following keys:
        - "fname": path to the header file
        - "dtype": data type of the binary data
        - "fs": sampling frequency in Hz
        - "n_channels": total number of channels
        - "n_participants": number of participants
        - "ch_names": list of channel names
        - "ch_types": list of channel types
        - "ch_units": list of channel units
        - "ch_resolutions": list of channel resolutions
        - "yaml_header": original JSON header content

    Examples
    --------
    >>> header = _read_bvrh("recording.bvrh")
    >>> print(header["n_participants"])
    >>> print(header["ch_names"])
    """
    fname = Path(fname).expanduser().resolve()
    if fname.suffix not in (".bvrh", ""):
        raise ValueError(f"Invalid file extension {fname.suffix} (expected .bvrh)")
    fname = fname.with_suffix(".bvrh")

    with open(fname, "r", encoding="utf-8-sig") as f:
        header = json.load(f)

    DTYPES = {
        "Int16": np.int16,
        "Int32": np.int32,
        "Single": np.float32,
        "Double": np.float64,
    }

    n_participants = len(header["Participants"]) if "Participants" in header else 1

    ch_names = []
    ch_types = []
    ch_units = []
    ch_resolutions = []

    for ch in header["EEGModality"]["Channels"]:
        ch_names.append(
            f"{ch['Name']} ({ch['ParticipantId']})"
            if "ParticipantId" in ch
            else ch["Name"]
        )
        ch_types.append(ch["Type"].lower())
        ch_units.append(ch["Unit"])
        ch_resolutions.append(ch.get("ResolutionPerBit", 1))

    return {
        "fname": fname.with_suffix(""),
        "dtype": DTYPES[
            header["EEGModality"]["BVRFFiles"]["DataFile"]["NumericDataType"]
        ],
        "fs": float(header["EEGModality"]["DataSpecific"]["SamplingFrequencyInHertz"]),
        "n_participants": n_participants,
        "n_channels": len(header["EEGModality"]["Channels"]),
        "ch_names": ch_names,
        "ch_types": ch_types,
        "ch_units": ch_units,
        "ch_resolutions": ch_resolutions,
        "yaml_header": header,
    }


def read_bvrf(fname):
    """Read BrainVision Recording Format (BVRF) recording.

    Parameters
    ----------
    fname : str | Path
        Path to the BVRF file (either without extension or one of `.bvrh`, `.bvrd`,
        `.bvrm`, or `.bvri`).

    Returns
    -------
    header : dict
        Header information. Channel names are modified to include participant ID
        suffix `" ({Id})"` for participant-specific channels. Channels shared across
        all participants have no suffix.
    data : ndarray, shape (n_channels, n_samples)
        EEG data for all channels (in V). Rows correspond to channels in the order
        specified in the header.
    markers : ndarray
        All markers from all participants combined.
    impedances : dict or None
        Impedances for all electrodes (in kOhm), with participant-specific electrodes
        having a suffix `" ({Id})"`. None if impedance file is not available.

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

    For single-participant datasets without a participant ID in the header, the
    participant ID is set to "1", but no suffix is added to channel names.

    For multi-participant datasets, channel names and impedance electrode names get
    a suffix `" ({Id})"` where `{Id}` is the participant ID.
    """
    fname = Path(fname).expanduser().resolve()
    if fname.suffix in (".bvrh", ".bvrd", ".bvrm", ".bvri", ""):
        fname = fname.with_suffix("")
    else:
        raise ValueError(f"Invalid file extension {fname.suffix}")

    header = _read_bvrh(f"{fname}.bvrh")

    # read all data
    data = _read_bvrd(
        f"{fname}.bvrd",
        header["dtype"],
        header["n_channels"],
        header["ch_units"],
        header["ch_resolutions"],
    )

    # read all markers
    markers = _read_bvrm(f"{fname}.bvrm")

    # read all impedances
    impedances = (
        _read_bvri(f"{fname}.bvri", header["ch_names"])
        if (fname.with_suffix(".bvri")).is_file()
        else None
    )

    return header, data, markers, impedances


def _read_bvrd(fname, dtype, n_channels, ch_units, ch_resolutions):
    """Read BVRF binary data file.

    Parameters
    ----------
    fname : str
        Path to the .bvrd file.
    dtype : NumPy data type
        Data type for reading binary data.
    n_channels : int
        Number of channels in the file.
    ch_units : list of str
        Units for each channel.
    ch_resolutions : list of float
        Resolution per bit for each channel.

    Returns
    -------
    ndarray, shape (n_channels, n_samples)
        Data (in V).
    """
    UNITS = {"V": 1e0, "mV": 1e-3, "µV": 1e-6, "nV": 1e-9}

    data = np.fromfile(fname, dtype=dtype)
    data = data.reshape((n_channels, -1), order="F")
    scales = np.array([UNITS[u] * r for u, r in zip(ch_units, ch_resolutions)])
    return data * scales[:, None]


def _read_bvrm(fname):
    """Read BVRFmarker file.

    Parameters
    ----------
    fname : str
        Path to the .bvrm file.

    Returns
    -------
    ndarray
        Markers (in a structured NumPy array).
    """
    return np.genfromtxt(
        fname,
        delimiter="\t",
        names=True,
        dtype=None,
        encoding="utf-8-sig",
        autostrip=True,
    )


def _read_bvri(fname, ch_names):
    """Read BVRF impedance file.

    Parameters
    ----------
    fname : str
        Path to the .bvri file.
    ch_names : list of str
        Channel names.

    Returns
    -------
    dict or None
        Impedances for all electrodes, or None if impedances are not available.
    """
    lines = Path(fname).read_text(encoding="utf-8-sig").splitlines()

    # find participant ID, electrode, and impedance measurement lines
    pid_lines = [s for s in lines if s.startswith("ParticipantId")]
    electrode_lines = [s for s in lines if s.startswith("Electrode")]
    datetime_lines = [s for s in lines if re.match(r"^\d{4}-\d{2}-\d{2}", s)]

    if not electrode_lines or not datetime_lines:
        return None

    electrodes = electrode_lines[0].split("\t")[1:]
    values = datetime_lines[0].split("\t")[1:]
    pids = pid_lines[0].split("\t")[1:] if pid_lines else [None] * len(electrodes)

    result = {}
    for electrode, pid, value in zip(electrodes, pids, values):
        ch_name = f"{electrode} ({pid})" if pid else electrode
        if ch_name in set(ch_names):
            result[ch_name] = float(value)

    return result if result else None
