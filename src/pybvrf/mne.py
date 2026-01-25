from mne import create_info
from mne.io import BaseRaw, get_channel_type_constants

from pybvrf.pybvrf import read_bvrf, split_participants


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

        super(RawBVRF, self).__init__(
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


def read_raw_bvrf(fname, split=False, *args, **kwargs):
    """Read BrainVision Recording Format (BVRF) recording.

    Parameters
    ----------
    fname : str | Path
        Path to the BVRF file (either without extension or one of `.bvrh`, `.bvrd`,
        `.bvrm`, or `.bvri`).
    split : bool, optional
        If True, split multi-participant recordings into separate RawBVRF objects (one
        per participant). Default is False.

    Returns
    -------
    RawBVRF | list of RawBVRF
        The raw data object (if `split` is False) or a list of raw data objects (if
        `split` is True).
    """
    header, data, markers, _ = read_bvrf(fname)
    if split and header["n_participants"] > 1:
        participants = split_participants(header, data, markers, None)
        return [
            RawBVRF.from_data(p_header, p_data, p_markers, *args, **kwargs)
            for p_header, p_data, p_markers, _ in participants
        ]
    return RawBVRF.from_data(header, data, markers, *args, **kwargs)
