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
        one per participant. If the recording has only one participant.
    """
    n_participants = header["n_participants"]

    if n_participants == 1:
        pid = header["yaml_header"]["Participants"][0]["Id"]
        return {pid: (header, data, markers, impedances)}

    participant_ids = [p["Id"] for p in header["yaml_header"]["Participants"]]

    results = {}
    for pid in participant_ids:
        # find channels for this participant (specific to PID or common to all)
        ch_indices = []
        for i, ch_name in enumerate(header["ch_names"]):
            if _is_participant_channel(ch_name, pid):
                ch_indices.append(i)

        # create new header for this participant
        participant_header = header.copy()
        participant_header["n_participants"] = 1
        participant_header["n_channels"] = len(ch_indices)
        participant_header["ch_names"] = [
            re.sub(r"\s*\(.+\)$", "", header["ch_names"][i]) for i in ch_indices
        ]
        participant_header["ch_types"] = [header["ch_types"][i] for i in ch_indices]
        participant_header["ch_units"] = [header["ch_units"][i] for i in ch_indices]
        participant_header["ch_resolutions"] = [
            header["ch_resolutions"][i] for i in ch_indices
        ]

        # filter impedances for this participant
        participant_impedances = None
        if impedances is not None:
            participant_impedances = {}
            for ch_name, value in impedances.items():
                if _is_participant_channel(ch_name, pid):
                    # remove participant ID suffix from channel name
                    participant_impedances[re.sub(r"\s*\(.+\)$", "", ch_name)] = value
            if not participant_impedances:
                participant_impedances = None

        results[pid] = (
            participant_header,
            data[ch_indices, :],
            markers,
            participant_impedances,
        )

    return results


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
