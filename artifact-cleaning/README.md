# EEG Denoising Models 

This repo contains denoising models that take an input of raw, 512 length eeg signals (vector or 1d matrix) and outputs a clean signal. 
It contains 3 different models that are based on a convolutional net work (1) CLEnet (2) DuoCL (3) U-Net. Below are the research papers from the respective model
- [CLEnet](https://www.nature.com/articles/s41598-025-98653-1)
- [Duo CL](https://ieeexplore.ieee.org/document/9973303/)
- [IC-U-Net](https://www.sciencedirect.com/science/article/pii/S1053811922007017)

## How to navigate directory

### src
dataloader.py contains the class to load and match eeg data and returns raw and clean eeg data from the same patient
metrics.py contains a functions to measure how successful the model performed against a test dataset

### notebook
Contains 2 versions of jupyter notebooks. Kaggle version is meant to be imported directly into Kaggle and includes the model class, metric functions, and dataloader function (original files can be found in src). Regular version (without any additional names) is also meant to be ran without any additional code as well, except it won't run on Kaggle (or Google colab). Both versions are meant to be train the model.

### model
Contains .py files of each model
