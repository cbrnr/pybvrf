# See https://www.brainproducts.com/download/bvrf-reference-specification/

import json

import numpy as np

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


def read_bvrf(fname):
    # read header (.bvrh)
    with open(f"{fname}.bvrh", "r") as f:
        header = json.load(f)

    dtype = DTYPES[header["EEGModality"]["BVRFFiles"]["DataFile"]["NumericDataType"]]

    fs = float(header["EEGModality"]["DataSpecific"]["SamplingFrequencyInHertz"])

    n_channels = len(header["EEGModality"]["Channels"])
    ch_names = [channel["Name"] for channel in header["EEGModality"]["Channels"]]
    ch_types = [channel["Type"] for channel in header["EEGModality"]["Channels"]]
    ch_units = [channel["Unit"] for channel in header["EEGModality"]["Channels"]]

    # read binary data (.bvrd)
    data = np.fromfile(f"{fname}.bvrd", dtype=dtype)
    data = data.reshape((n_channels, -1), order="F")
    scalings = np.array([UNITS[unit] for unit in ch_units])[:, None]
    data *= scalings

    return data, ch_names, ch_types, ch_units, fs
