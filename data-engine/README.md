# Data Engine

Goal: To build an engine that can take in any EEG dataset, and with minimal scripting edits, and output a machine learning dataset (raw & clean signals).
Why: To support the training of new EEG cleaning machine learning models.

## Module Overview
This module is focued on generating a machine learning dataset. The dataset will comprise of two copies of signal data, one raw and one cleaned from artifacts. Each data peice will comprise of a single channel 2-second epoch (256 Hz). This will be used to train an artifact removal model.

This project drew from the following papers:

"...it is presently recommended that an EEG recording have at least 30 * (the number of channels)2 data samples to undergo ICA decomposition"

Information-based modeling of event-related brain dynamics by [(Onton & Makeig, 2006)](https://doi.org/10.1016/S0079-6123(06)59007-7)

"W-ICA followed by ICA is supported by prior work showing that using wavelet-thresholding approaches before ICA improves the resulting ICA decomposition of the EEG data"

Blind source separation of multichannel electroencephalogram based on wavelet transform and ICA by [(Rong-Yi & Zhong, 2005)](https://iopscience.iop.org/article/10.1088/1009-1963/14/11/006/meta)

This module implements the complete automated RELAX pipeline for cleaning 

Introducing RELAX: An automated pre-processing pipeline for cleaning EEG data - Part 1: Algorithm and application to oscillations by [(N.W. Bailey et al., 2023)](https://doi.org/10.1016/j.clinph.2023.01.017)

### To-do
- Check if interpolation is happening to bad channels
- Print out bad channels and delete them in both datasets
- Silence MATLAB warnings
- Make channel prefixes and suffixes a configuration
- Make sure notch and band pass filters are equivalent in raw and clean pipelines for consistency

### Dependencies
- Matlab r2024b
- [EEGLab](https://sccn.ucsd.edu/eeglab/download.php)

#### MATLAB Toolboxes
- Statistics and Machine Learning
- Signal Processing
- readyaml
- Wavelet Toolbox

#### EEGLab plugins
- RELAX2.0.0
- ICLabel
- [mwf-artifact-removal](https://github.com/exporl/mwf-artifact-removal)
- PrepPipeline0.57.0
- clean_rawdata
- dipfit
- EEG-BIDS10.2
- firfilt
- zapline-plus
- amica
- [fieldtrip](https://www.fieldtriptoolbox.org/download/)

## Source code alterations
- Change paths in `pop_RELAX.m`
- Change `\` to `/` in `RELAX_Wrapper.m:113` (if on mac)
- Comment out `Save Statistics...` section in `RELAX_Wrapper.m`
