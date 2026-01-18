from mne import create_info
from mne.io import BaseRaw, get_channel_type_constants

from pybvrf.pybvrf import read_bvrf


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
        header, data, markers, _ = read_bvrf(fname)  # MNE does not support impedances

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


def read_raw_bvrf(fname, *args, **kwargs):
    """Read BrainVision Recording Format (BVRF) recording.

    Parameters
    ----------
    fname : str | Path
        Path to the BVRF file (either without extension or one of `.bvrh`, `.bvrd`,
        `.bvrm`, or `.bvri`).

    Returns
    -------
    RawBVRF
        The raw data.
    """
    return RawBVRF(fname, *args, **kwargs)
