## Flat Field Correction Repo

### Description

This repository contains the code for the flat field correction of images. The code is written in Python and uses the OpenCV library. The code is written in a modular way, so that it can be easily integrated into other projects. The code is also written in a way that it can be easily understood by someone who is new to the field of image processing.

It contains the following modules:
- **pre_processing**: include intensity compressor and grid_noise filter.
- **correct**: include three option for flat field correction:
    1. NaiveEstimate: Our methods
    2. BaSicEstimate. BaSiC
    3. None: No correction
- **post_processing**: include post processing methods (median filter).

### Usage

For simplicity, we provide `run.py` for our two-photon images. It requires the following arguments:
- `--input_folder`: the path to the input image's folder.
- `--method`: the method to use for flat field correction. It can be either `NaiveEstimate`(default) or `BaSicEstimate`.

The input folder structure should be as follows:
```bash
input_folder
--- CellVideo
--- CellVideo_CHB_Info.tdms
--- CellVideo_CHB_Info.tdms_index
--- Information.txt
```
where `CellVideo` is the folder containing the images, `CellVideo_CHB_Info.tdms` is the file containing the metadata of the images, and `Information.txt` is the file containing the information of the images.

Just run the following command in the src folder:
```bash
python run.py --input_folder path_to_input_folder --method method_to_use
```
