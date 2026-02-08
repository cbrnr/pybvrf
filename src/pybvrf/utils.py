# © Clemens Brunner
#
# License: BSD (3-clause)

"""Utility functions for BVRF processing."""

import re


def split_participants(header, data, markers, impedances):
    """Split multi-participant recording into individual participant recordings.

    Parameters
    ----------
    header : dict
        Header information.
    data : ndarray, shape (n_channels, n_samples)
        EEG data.
    markers : ndarray
        Markers.
    impedances : dict | None
        Impedances.

    Returns
    -------
    dict
        Dict with PID as keys and (header, data, markers, impedances) tuples as values,
        one per participant.

    Raises
    ------
    ValueError
        If the recording has only one participant.
    """
    if header["n_participants"] == 1:
        raise ValueError("Recording has only one participant.")

    participant_ids = [p["Id"] for p in header["yaml_header"]["Participants"]]

    results = {}
    for pid in participant_ids:
        # find channels for this participant
        ch_indices = []
        for i, ch_name in enumerate(header["ch_names"]):
            if _is_participant_channel(ch_name, pid):
                ch_indices.append(i)

        # create new header for this participant
        participant_header = header.copy()
        participant_header["n_participants"] = 1
        participant_header["n_channels"] = len(ch_indices)
        participant_header["ch_names"] = [
            _remove_participant_suffix(header["ch_names"][i]) for i in ch_indices
        ]
        participant_header["ch_types"] = [header["ch_types"][i] for i in ch_indices]
        participant_header["ch_units"] = [header["ch_units"][i] for i in ch_indices]
        participant_header["ch_resolutions"] = [
            header["ch_resolutions"][i] for i in ch_indices
        ]
        participant_header["ch_positions"] = _filter_dict_by_participant(
            header["ch_positions"], pid
        )

        # filter impedances for this participant
        participant_impedances = _filter_dict_by_participant(impedances, pid)

        results[pid] = (
            participant_header,
            data[ch_indices, :],
            markers,
            participant_impedances,
        )

    return results


def _remove_participant_suffix(ch_name):
    """Remove participant ID suffix from a channel name.

    Parameters
    ----------
    ch_name : str
        Channel name, possibly with participant ID suffix (e.g., "Fz (P1)").

    Returns
    -------
    str
        Channel name without the participant ID suffix.
    """
    return re.sub(r"\s*\([^)]*\)$", "", ch_name)


def _is_participant_channel(ch_name, participant_id):
    """Check if a channel belongs to a specific participant.

    Parameters
    ----------
    ch_name : str
        Channel name, possibly with participant ID suffix (e.g., "Fz (P1)").
    participant_id : str
        Participant ID to check for.

    Returns
    -------
    bool
        True if the channel belongs to the participant (either specific to that
        participant or common to all participants).
    """
    is_participant_channel = re.search(rf"\({re.escape(participant_id)}\)$", ch_name)
    is_common_channel = not re.search(r"\(.+\)$", ch_name)
    return bool(is_participant_channel or is_common_channel)


def _filter_dict_by_participant(data_dict, participant_id):
    """Filter a dictionary by participant ID.

    Parameters
    ----------
    data_dict : dict | None
        Dictionary with channel names as keys.
    participant_id : str
        Participant ID to filter for.

    Returns
    -------
    dict | None
        Filtered dictionary with participant suffix removed from keys, or None.
    """
    if data_dict is None:
        return None
    result = {}
    for name, value in data_dict.items():
        if _is_participant_channel(name, participant_id):
            result[_remove_participant_suffix(name)] = value
    return result if result else None
