# PyBVRF

PyBVRF is a Python package for working with [BVRF (BrainVision Reader Format)](https://www.brainproducts.com/download/bvrf-reference-specification/) files.

The package includes the following features:

- Support for multi-participant recordings
- Seamless integration with MNE-Python
- Convenient access to metadata (including the original YAML header)
- Support for markers and impedance data

A BVRF recording consists of multiple files which are expected to be available in the same directory. The required files are:

- `<fname>.bvrh` (header file)
- `<fname>.bvrd` (data file)
- `<fname>.bvrm` (marker file)

Optionally, `<fname>.bvri` (impedance file) may also be present.


## Basic usage

Use `read_bvrf()` to load a recording. The file extension is optional (the function accepts any of the four supported extensions or just the base filename).

```python
from pybvrf import read_bvrf

header, data, markers, impedances = read_bvrf("recording")
```

Here, `header` is a dict containing metadata about the recording (such as sampling frequency, channel names, and participant information):

```python
print(f"Sampling frequency: {header['fs']} Hz")
print(f"Number of channels: {header['n_channels']}")
print(f"Channel names: {header['ch_names']}")
print(f"Number of participants: {header['n_participants']}")
```

The entire original header information is available as `header["yaml_header"]` (a dict parsed from the YAML header file).

Next, `data` is a 2D NumPy array (channels x samples) containing the EEG signals:

```python
print(f"Data shape: {data.shape}")
```

Finally, `markers` contains information about events in a NumPy structured array:

```python
print(f"Number of markers: {len(markers)}")
print(f"Marker fields: {markers.dtype.names}")
```

If impedances are available, `impedances` is a dict mapping channel names to impedance values:

```python
if impedances:
    print(f"Impedances: {impedances}")
```


## Advanced usage

Multi-participant recordings (`header["n_participants"] > 1`) are available in a single data structure by default, with channels from all participants concatenated together (but suffixed with the participant ID). For example, if there are two participants P1 and P2, the channel names might be "C3 (P1)", "Cz (P1)", "C4 (P1)", "C3 (P2)", "Cz (P2)", and "C4 (P2)". You can use `split_participants()` to split the data into separate data structures per participant:

```python
from pybvrf import split_participants

participant_data = split_participants(header, data, markers, impedances)
```

This returns a dict mapping participant IDs to their respective data (a tuple of header, data, markers, and impedances).


## MNE-Python integration

PyBVRF integrates seamlessly with [MNE-Python](https://mne.tools/) for advanced EEG analysis:

```python
from pybvrf import read_raw_bvrf

raw = read_raw_bvrf("recording.bvrh")
```

Like `read_bvrf()`, `read_raw_bvrf()` accepts any of the four supported file extensions or just the base filename. It returns a [`Raw`](https://mne.tools/stable/generated/mne.io.Raw.html) object containing the EEG data, along with the appropriate metadata and annotations.

If a recording includes multiple participants, the function returns a single `Raw` object by default, with all participants' data concatenated (channel names are suffixed with the participant ID as described above).

You can also load selected participants and/or split multi-participant recordings into separate `Raw` objects:

```python
from pybvrf import read_raw_bvrf

# load specific participants
raw = read_raw_bvrf("multi_recording.bvrh", participants="P1")
raw = read_raw_bvrf("multi_recording.bvrh", participants=["P1", "P2"])

# split into separate Raw objects per participant
raw_dict = read_raw_bvrf("multi_recording.bvrh", split=True)
for pid, raw in raw_dict.items():
    print(f"Participant {pid}: {raw.info['nchan']} channels")
    
# load specific participants and split
raw_dict = read_raw_bvrf("multi_recording.bvrh", participants=["P1", "P3"], split=True)
```


## Acknowledgements

The initial release of PyBVRF was sponsored by [Brain Products](https://www.brainproducts.com/).
