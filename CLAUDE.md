# CLAUDE.md

This file provides guidance to AI coding agents when working with code in this repository.

## Commands

- This project uses [uv](https://docs.astral.sh/uv/) for package and environment management.
- Install dependencies with `uv sync --all-groups --all-extras`.
- Run tests with `uv run pytest -W error tests` (CI promotes warnings to errors; a plain `uv run pytest` will miss these).
- Run a single test: `uv run pytest tests/test_pybvrf.py::test_validate_fname`.
- Formatting is enforced by [Ruff](https://docs.astral.sh/ruff/). Run both of the following before committing:
  ```
  uv run ruff check --fix
  uv run ruff format
  ```

CI (`.github/workflows/test.yml`) also runs two custom checks that must pass:
- `python .github/check_license_headers.py` — every `.py` file in `src/` and `tests/` must start with the exact 4-line BSD header (see any source file).
- `python .github/check_uv_build.py` — the `uv_build` requirement in `pyproject.toml` must be compatible with the latest released uv.

## Contribution requirements

- Every PR must add an entry to the `[UNRELEASED]` section of `CHANGELOG.md` under the appropriate subsection (`### ✨ Added`, `### 🔧 Fixed`, `### 🌀 Changed`, `### 🗑️ Removed`), formatted as a single sentence with a PR link and author, e.g. `- Add support for XYZ ([#123](...) by [Name](...))`.
- Line length is 88 characters (the default). This limit applies to all code, including docstrings.
- Docstrings use NumPy style but with standard Markdown (single backticks for inline code, not reStructuredText double backticks).
- Inline comments start with a lower-case letter and are a single sentence where possible.
- Commit subjects: imperative mood, capitalized, ≤72 characters.

## Architecture

PyBVRF reads [BVRF (BrainVision Recording Format)](https://www.brainproducts.com/download/bvrf-reference-specification/) EEG recordings. A recording is a set of sibling files sharing a base name in one directory: `.bvrh` (JSON header, despite the "yaml_header" naming), `.bvrd` (binary data), `.bvrm` (tab-separated markers), and optional `.bvri` (impedances).

The package has a deliberately small surface. Four public names are exported from `src/pybvrf/__init__.py`: `read_bvrf`, `read_bvrf_header`, `split_participants`, and `read_raw_bvrf`.

### Core reader (`src/pybvrf/pybvrf.py`)

`read_bvrf()` returns a 4-tuple `(header, data, markers, impedances)` and orchestrates the private per-file readers:
- `read_bvrf_header()` (public) parses the `.bvrh` JSON, validates it against the bundled schema `BVRFHeader-1.0.0.json` (loaded via `importlib.resources`), and flattens the nested spec into a flat `header` dict. The full parsed JSON is preserved under `header["yaml_header"]`.
- `_read_bvrd()` reads the raw binary with `np.fromfile`, reshapes column-major (`order="F"`) into `(n_channels, n_samples)`, and scales each channel to volts using its unit and `ResolutionPerBit`.
- `_read_bvrm()` / `_read_bvri()` parse the marker and impedance text files. All text files are read with `utf-8-sig` to tolerate a BOM.
- `_validate_fname()` is the shared entry gate: it accepts a base name or any supported extension, resolves the path, and normalizes to the expected suffix. All readers call it, so any of the four extensions (or none) works as input to the public functions.

### Multi-participant handling (`src/pybvrf/utils.py`)

Multi-participant recordings are loaded combined by default, with channels from all participants suffixed by participant ID, e.g. `"Cz (P1)"`. Channels without a `(...)` suffix are treated as common to all participants. `split_participants()` uses this naming convention (via `_is_participant_channel` / `_remove_participant_suffix`) to slice the combined arrays back into per-participant recordings. This suffix convention is the contract shared across the reader, utils, and MNE integration — changing it affects all three.

### MNE integration (`src/pybvrf/mne_io.py`)

Imported lazily. `__init__.py` defines `__getattr__` so that `import pybvrf` works without MNE installed; `read_raw_bvrf` only fails (with a clear "install pybvrf[mne]" message) when actually called. `mne` is a `dev` dependency and an optional `mne` extra, so tests exercise both the present and absent cases (`test_import_with_mne` / `test_import_without_mne`).

`RawBVRF` subclasses `mne.io.BaseRaw`. `read_raw_bvrf()` reuses `read_bvrf` + `split_participants` and adds `participants=` selection and `split=` (one combined `Raw` vs. a dict of per-participant `Raw` objects). Unknown channel types are mapped to MNE's `"misc"`; markers become annotations.
