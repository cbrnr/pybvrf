import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_bvrf_header(fname):
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
        - "participants": participant IDs
        - "channels": dict with participant IDs as keys, each containing:
            - "names": list of channel names
            - "types": list of channel types
            - "units": list of channel units
            - "resolutions": list of channel resolutions
        - "yaml_header": original JSON header content

    Examples
    --------
    >>> header = read_bvrf_header("recording.bvrh")
    >>> print(header["participants"])
    >>> print(header["channels"]["1"]["names"])
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

    if "Participants" in header:
        participant_ids = [p["Id"] for p in header["Participants"]]
    else:
        participant_ids = ["1"]

    # organize channels by participant ID
    ch_info = defaultdict(
        lambda: {
            "names": [],
            "types": [],
            "units": [],
            "resolutions": [],
            "indices": [],
        }
    )

    for ch_idx, ch in enumerate(header["EEGModality"]["Channels"]):
        pids = participant_ids if "ParticipantId" not in ch else [ch["ParticipantId"]]

        for pid in pids:
            ch_info[pid]["names"].append(ch["Name"])
            ch_info[pid]["types"].append(ch["Type"].lower())
            ch_info[pid]["units"].append(ch["Unit"])
            ch_info[pid]["resolutions"].append(ch.get("ResolutionPerBit", 1))
            ch_info[pid]["indices"].append(ch_idx)

    return {
        "fname": fname.with_suffix(""),
        "dtype": DTYPES[
            header["EEGModality"]["BVRFFiles"]["DataFile"]["NumericDataType"]
        ],
        "fs": float(header["EEGModality"]["DataSpecific"]["SamplingFrequencyInHertz"]),
        "n_channels": len(header["EEGModality"]["Channels"]),
        "participants": participant_ids,
        "channels": dict(ch_info),
        "yaml_header": header,
    }


def read_bvrf(fname, participants=None):
    """Read BrainVision Recording Format (BVRF) recording.

    Parameters
    ----------
    fname : str | Path
        Path to the BVRF file (either without extension or one of `.bvrh`, `.bvrd`,
        `.bvrm`, or `.bvri`).
    participants : str | list of str | None
        Participant ID(s) to read. If None (default), return data for all participants
        in the recording.

    Returns
    -------
    dict
        The keys are participant IDs, and each value is a dict containing:
        - "header": dict with header information (see `read_bvrf_header()`)
        - "data": ndarray, shape (n_channels, n_samples) with EEG data (in V)
        - "markers": ndarray, shape (n_markers,) with marker information
        - "impedances": dict with impedances per electrode (in kOhm) or None

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
    participant ID is set to "1".
    """
    fname = Path(fname).expanduser().resolve()
    if fname.suffix in (".bvrh", ".bvrd", ".bvrm", ".bvri", ""):
        fname = fname.with_suffix("")
    else:
        raise ValueError(f"Invalid file extension {fname.suffix}")

    header = read_bvrf_header(f"{fname}.bvrh")

    # determine which participant(s) to read
    if participants is None:  # all
        participants = header["participants"]
    elif isinstance(participants, str):
        participants = [participants]
    else:
        participants = list(participants)

    # read data for each participant
    result = {}
    for pid in participants:
        data = _read_bvrd(
            f"{fname}.bvrd",
            header["dtype"],
            header["n_channels"],
            header["channels"][pid]["indices"],
            header["channels"][pid]["units"],
            header["channels"][pid]["resolutions"],
        )
        markers = _read_bvrm(f"{fname}.bvrm", pid)
        impedances = (
            _read_bvri(f"{fname}.bvri", header["channels"][pid]["names"])
            if (fname.with_suffix(".bvri")).is_file()
            else None
        )

        result[pid] = {
            "header": header,
            "data": data,
            "markers": markers,
            "impedances": impedances,
        }

    return result


def _read_bvrd(fname, dtype, n_channels, ch_indices, ch_units, ch_resolutions):
    """Read binary data file and filter for specific participant.

    Parameters
    ----------
    fname : str
        Path to the .bvrd file
    dtype : numpy dtype
        Data type for reading binary data
    n_channels : int
        Total number of channels in the file
    ch_indices : list of int
        Indices of channels belonging to this participant
    ch_units : list of str
        Units for each channel
    ch_resolutions : list of float
        Resolution per bit for each channel

    Returns
    -------
    data : ndarray, shape (n_channels, n_samples)
        Filtered data for the specified participant in Volts
    """
    # read data and reshape to (n_channels, n_samples)
    data = np.fromfile(fname, dtype=dtype)
    n_samples = len(data) // n_channels
    data = data.reshape((n_channels, n_samples), order="F")

    # filter channels for a specific participant (as given by ch_indices)
    data = data[ch_indices, :]

    UNITS = {"V": 1e0, "mV": 1e-3, "µV": 1e-6, "nV": 1e-9}
    return (
        data
        * np.array([UNITS[unit] for unit in ch_units])[:, None]
        * np.array(ch_resolutions)[:, None]
    )


def _read_bvrm(fname, pid):
    """Read marker file and filter for specific participant.

    Parameters
    ----------
    fname : str
        Path to the .bvrm file
    pid : str
        Participant ID

    Returns
    -------
    markers : ndarray
        Markers for the specified participant
    """
    markers = np.genfromtxt(
        fname,
        delimiter="\t",
        names=True,
        dtype=None,
        encoding="utf-8-sig",
        autostrip=True,
    )

    # filter by ParticipantId if the field exists
    if markers.size > 0 and "ParticipantId" in markers.dtype.names:
        mask = markers["ParticipantId"] == pid
        return markers[mask]

    return markers


def _read_bvri(fname, ch_names):
    """Read impedance file and filter for specific participant.

    Parameters
    ----------
    fname : str
        Path to the .bvri file
    ch_names : list of str
        Channel names for this participant

    Returns
    -------
    impedances : dict or None
        Impedances for channels of the specified participant
    """
    lines = Path(fname).read_text(encoding="utf-8-sig").splitlines()

    electrode_lines = [s for s in lines if s.startswith("Electrode")]
    datetime_lines = [s for s in lines if re.match(r"^\d{4}-\d{2}-\d{2}", s)]

    if electrode_lines and datetime_lines:
        electrodes = electrode_lines[0].split("\t")[1:]
        values = datetime_lines[0].split("\t")[1:]

        # # filter channels for a specific participant (as given by ch_names)
        result = {}
        for e, v in zip(electrodes, values):
            if e in ch_names:
                result[e] = float(v)

        return result if result else None

    return None
