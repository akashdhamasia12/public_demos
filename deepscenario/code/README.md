# README

Minimal code to run the data preprocessing on one of our recordings.

### Setup

1. Make sure you have ExifTool installed (e.g. version 12.58 works). See: https://exiftool.org/install.html. Check its correctly installed by running "`exiftool -ver`".
2. Create a fresh environment:

```bash
conda create --name dsc_intel python=3.8
```

1. In the development dir run:

```bash
conda activate dsc_intel
cd data_preprocessing
pip install -e .
cd -
pip install -r data_preprocessing/requirements.txt
```

### Execution

See file: *data_preprocessing/data_preprocessing/preprocess_recording.py*.

Arguments e.g.:

```bash
python data_preprocessing/preprocess_recording.py --recording_dir <PATH_TO_RECORDING_DIR>/2022-10-12T18-23-26
```