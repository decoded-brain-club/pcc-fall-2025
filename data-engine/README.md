# Data Engine

## Module Overview
This module is focued on generating a machine learning dataset. The dataset will comprise of two copies of signal data, one raw and one cleaned from artifacts. Each data peice will comprise of a single channel 2-second epoch. This will be used to train an artifact removal model.

### Tasks
- Apply minimal preprocessing (filters and resampling) to raw data
- Outline how the data will be stored so it can be accessed easily and scaled to other datasets
- Generate first half of data (single channel 2-second segments)
- Perform ICA on preprocessed data
- Calculate features of independent components
- Use MNE-ICLabel to flag bad ICs
- Perform Wavlet Transform on bad ICs
- Reconstruct data
- Generate second half of data from post-processed data