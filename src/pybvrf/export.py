# © Clemens Brunner
#
# License: BSD (3-clause)

"""Module for reading BrainVision Recording Format (BVRF) files into MNE-Python."""

from mne import create_info
from mne.io import BaseRaw, get_channel_type_constants

from pybvrf.pybvrf import read_bvrf
from pybvrf.utils import _is_participant_channel, split_participants


class RawBVRF(BaseRaw):
    """BrainVision Recording Format (BVRF) recording."""

    def __init__(self, fname, *args, **kwargs):
        """Read BrainVision Recording Format (BVRF) recording.

        Parameters
        ----------
        fname : str | Path
            Path to the BVRF file (either without extension or one of `.bvrh`, `.bvrd`,
            `.bvrm`, or `.bvri`).
        """
        header, data, markers, _ = read_bvrf(fname)
        self._init_from_data(header, data, markers, *args, **kwargs)

    @classmethod
    def from_data(cls, header, data, markers, *args, **kwargs):
        """Create RawBVRF instance from already read data.

        Parameters
        ----------
        header : dict
            Header information.
        data : ndarray
            EEG data.
        markers : ndarray
            Markers.

        Returns
        -------
        RawBVRF
            The raw data object.
        """
        instance = cls.__new__(cls)
        instance._init_from_data(header, data, markers, *args, **kwargs)
        return instance

    def _init_from_data(self, header, data, markers, *args, **kwargs):
        """Initialize from header, data, and markers."""
        fs = header["fs"]

        mne_ch_types = list(get_channel_type_constants().keys())
        ch_types = [
            ch_type if ch_type in mne_ch_types else "misc"
            for ch_type in header["ch_types"]
        ]
        info = create_info(ch_names=header["ch_names"], sfreq=fs, ch_types=ch_types)

        super().__init__(
            preload=data,
            info=info,
            filenames=[str(header["fname"].with_suffix(".bvrh"))],
            *args,
            **kwargs,
        )

        self.annotations.append(
            onset=markers["Sample"] / fs,
            duration=0,
            description=[
                f"{t}/{c}{v if v > 0 else ''}"
                for t, c, v in zip(markers["Type"], markers["Code"], markers["Value"])
            ],
        )


def read_raw_bvrf(fname, participants=None, split=False, *args, **kwargs):
    """Read BrainVision Recording Format (BVRF) recording.

    Parameters
    ----------
    fname : str | Path
        Path to the BVRF file (either without extension or one of `.bvrh`, `.bvrd`,
        `.bvrm`, or `.bvri`).
    participants : None | str | list of str, optional
        Which participants to load. If None, all participants are loaded. If a
        participant ID (e.g., "P1") or a list of participant IDs, only the specified
        participants are loaded.
    split : bool, optional
        If False, combine the data into a single RawBVRF object. If True, return
        separate RawBVRF objects for each participant in a dict.

    Returns
    -------
    RawBVRF | dict of RawBVRF
        The raw data object (if `split=False`) or a dict of raw data objects with PID
        as keys (if `split=True`).
    """
    header, data, markers, _ = read_bvrf(fname)

    # get available participant IDs
    all_pids = [p["Id"] for p in header["yaml_header"]["Participants"]]

    # normalize and validate participants parameter
    if participants is not None:
        if isinstance(participants, str):
            participants = [participants]

        if not participants or any(not pid for pid in participants):
            raise ValueError(
                "Participant list cannot be empty and must contain non-empty IDs"
            )

        if invalid_pids := [pid for pid in participants if pid not in all_pids]:
            raise ValueError(
                f"Invalid participant ID(s): {invalid_pids}. Available participants: "
                f"{all_pids}"
            )
    else:
        participants = all_pids

    if split:
        participant_data = split_participants(header, data, markers, None)
        return {
            pid: RawBVRF.from_data(*participant_data[pid][:3], *args, **kwargs)
            for pid in participants
        }
    else:
        if header["n_participants"] > 1 and set(participants) != set(all_pids):
            ch_indices = [
                i
                for i, ch_name in enumerate(header["ch_names"])
                if any(_is_participant_channel(ch_name, pid) for pid in participants)
            ]

            filtered_header = header.copy()
            filtered_header["n_channels"] = len(ch_indices)
            filtered_header["ch_names"] = [header["ch_names"][i] for i in ch_indices]
            filtered_header["ch_types"] = [header["ch_types"][i] for i in ch_indices]
            filtered_header["ch_units"] = [header["ch_units"][i] for i in ch_indices]
            filtered_header["ch_resolutions"] = [
                header["ch_resolutions"][i] for i in ch_indices
            ]

            filtered_data = data[ch_indices, :]

            return RawBVRF.from_data(
                filtered_header, filtered_data, markers, *args, **kwargs
            )

        return RawBVRF.from_data(header, data, markers, *args, **kwargs)
