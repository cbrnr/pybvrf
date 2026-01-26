# © Clemens Brunner
#
# License: BSD (3-clause)

"""Module for reading BrainVision Recording Format (BVRF) files."""

import json
import re
from pathlib import Path

import jsonschema
import numpy as np


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
        Header information with the following items:
        - fname : Path
            File path (without extension).
        - dtype : np.float32 | np.float64 | np.int16 | np.int32
            Data type of the binary data.
        - fs : float
            Sampling frequency (in Hz).
        - n_participants : int
            Number of participants.
        - n_channels : int
            Number of channels.
        - ch_names : list of str
            Channel names.
        - ch_types : list of str
            Channel types.
        - ch_units : list of str
            Channel units.
        - ch_resolutions : list of float
            Channel resolutions.
        - yaml_header : dict
            Original JSON header.
    data : ndarray, shape (n_channels, n_samples)
        EEG data.
    markers : ndarray
        Markers.
    impedances : dict | None
        Electrode impedances (in kΩ) or None if not available.

    Notes
    -----
    A BVRF recording consists of multiple files which are expected to be available in
    the same directory. The required files are:

    - `<fname>.bvrh` (header file)
    - `<fname>.bvrd` (data file)
    - `<fname>.bvrm` (marker file)

    Optionally, `<fname>.bvri` (impedance file) may also be present.

    See https://www.brainproducts.com/download/bvrf-reference-specification/ for the
    official BVRF specification.
    """
    fname = _validate_fname(fname, [".bvrh", ".bvrd", ".bvrm", ".bvri"])

    header = read_bvrf_header(fname.with_suffix(".bvrh"))

    data = _read_bvrd(
        fname.with_suffix(".bvrd"),
        header["dtype"],
        header["n_channels"],
        header["ch_units"],
        header["ch_resolutions"],
    )

    markers = _read_bvrm(fname.with_suffix(".bvrm"))

    impedances = (
        _read_bvri(fname.with_suffix(".bvri"), header["ch_names"])
        if (fname.with_suffix(".bvri")).is_file()
        else None
    )

    return header, data, markers, impedances


def read_bvrf_header(fname):
    """Read BVRF header file.

    Parameters
    ----------
    fname : str | Path
        Path to the BVRF header file (with or without `.bvrh` extension).

    Returns
    -------
    dict
        Header information, see `read_bvrf()` for details.

    Raises
    ------
    ValueError
        If header validation fails.

    Examples
    --------
    >>> header = read_bvrf_header("recording.bvrh")
    >>> print(header["n_participants"])
    >>> print(header["ch_names"])
    """
    fname = _validate_fname(fname, ".bvrh")

    with open(fname, encoding="utf-8-sig") as f:
        header = json.load(f)

    with open(Path(__file__).parent / "BVRFHeader-1.0.0.json") as f:
        schema = json.load(f)
    try:
        jsonschema.validate(instance=header, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"Could not parse {fname}: {e.message}") from e

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


def _read_bvrd(fname, dtype, n_channels, ch_units, ch_resolutions):
    """Read BVRF binary data file.

    Parameters
    ----------
    fname : str | Path
        Path to the BVRF data file (with or without `.bvrd` extension).
    dtype : np.float32 | np.float64 | np.int16 | np.int32
        Data type of the binary data.
    n_channels : int
        Number of channels.
    ch_units : list of str
        Channel units.
    ch_resolutions : list of float
        Channel resolutions.

    Returns
    -------
    ndarray, shape (n_channels, n_samples)
        Data (in V).
    """
    fname = _validate_fname(fname, ".bvrd")

    UNITS = {"V": 1e0, "mV": 1e-3, "µV": 1e-6, "nV": 1e-9}

    data = np.fromfile(fname, dtype=dtype)
    data = data.reshape((n_channels, -1), order="F")
    scales = np.array([UNITS[u] * r for u, r in zip(ch_units, ch_resolutions)])
    return data * scales[:, None]


def _read_bvrm(fname):
    """Read BVRF marker file.

    Parameters
    ----------
    fname : str | Path
        Path to the BVRF marker file (with or without `.bvrm` extension).

    Returns
    -------
    ndarray
        Markers (in a structured NumPy array).
    """
    fname = _validate_fname(fname, ".bvrm")

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
    fname : str | Path
        Path to the BVRF impedance file (with or without `.bvri` extension).
    ch_names : list of str
        Channel names.

    Returns
    -------
    dict | None
        Impedances for all electrodes or None if impedances are not available.

    Raises
    ------
    ValueError
        If the impedance file cannot be parsed.
    """
    fname = _validate_fname(fname, ".bvri")

    lines = Path(fname).read_text(encoding="utf-8-sig").splitlines()

    # find participant ID, electrode, and impedance measurement lines
    pid_lines = [s for s in lines if s.startswith("ParticipantId")]
    electrode_lines = [s for s in lines if s.startswith("Electrode")]
    datetime_lines = [s for s in lines if re.match(r"^\d{4}-\d{2}-\d{2}", s)]

    if not electrode_lines or not datetime_lines:
        raise ValueError(f"Could not parse {fname} (missing required lines).")

    electrodes = electrode_lines[0].split("\t")[1:]
    values = datetime_lines[0].split("\t")[1:]
    pids = pid_lines[0].split("\t")[1:] if pid_lines else [None] * len(electrodes)

    result = {}
    for electrode, pid, value in zip(electrodes, pids, values):
        ch_name = f"{electrode} ({pid})" if pid else electrode
        if ch_name in set(ch_names):
            result[ch_name] = float(value)

    return result


def _validate_fname(fname, extensions):
    """Validate and normalize BVRF file path.

    Parameters
    ----------
    fname : str | Path
        Path to the BVRF file.
    extensions : str | list of str
        Expected file extension (e.g., ".bvrh") or list of allowed extensions.

    Returns
    -------
    Path
        Normalized path with the expected extension (if single extension provided) or
        without extension (if list of allowed extensions provided).

    Raises
    ------
    ValueError
        If fname does not have an allowed extension.
    """
    fname = Path(fname).expanduser().resolve()

    if isinstance(extensions, str):
        if fname.suffix not in (extensions, ""):
            raise ValueError(
                f"Invalid file extension {fname.suffix} (expected {extensions})."
            )
        return fname.with_suffix(extensions)
    else:
        if fname.suffix not in (*extensions, ""):
            raise ValueError(f"Invalid file extension {fname.suffix}.")
        return fname.with_suffix("")
