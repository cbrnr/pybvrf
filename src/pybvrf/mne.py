from mne import create_info
from mne.io import BaseRaw, get_channel_type_constants

from pybvrf.pybvrf import read_bvrf


class RawBVRF(BaseRaw):
    """Raw data from BrainVision Recording Format (BVRF) recording."""

    def __init__(self, fname, participants=None, *args, **kwargs):
        """Read BrainVision Recording Format (BVRF) recording.

        Parameters
        ----------
        fname : str | Path
            Path to the BVRF file (either without extension or one of `.bvrh`, `.bvrd`,
            `.bvrm`, or `.bvri`).
        participants : str | list of str | None
            Participant identifier(s) to read. If None (default), return data for all
            participants.
        """
        data, info, markers, impedances = read_bvrf(fname, participants)

        fs = info["fs"]

        mne_ch_types = list(get_channel_type_constants().keys())
        ch_types = [
            ch_type if ch_type in mne_ch_types else "misc"
            for ch_type in info["ch_types"]
        ]
        info = create_info(ch_names=info["ch_names"], sfreq=fs, ch_types=ch_types)

        super().__init__(
            preload=data,
            info=info,
            filenames=[info["fname"].with_suffix(".bvrh")],
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


def read_raw_bvrf(fname, participants=None, *args, **kwargs):
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
    RawBVRF
        The raw data.
    """
    return RawBVRF(fname, participants, *args, **kwargs)
