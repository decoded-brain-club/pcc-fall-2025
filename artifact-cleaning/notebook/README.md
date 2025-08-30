# How to run notebook

The purpose of these notebooks is to provider others with the finished models, training loop and testing metrics so that the code can run smoothly once the repo is downloaded. The only thing users would need to do
for the code to run is to download the [dataset](https://www.kaggle.com/datasets/jasonhhi/eeg-clean-raw), and edit the file path(s) to lead to the data. All files in src must be downloaded.

### Dataloader.py

This file contains the class to create the custom dataset. It takes 2 arguments: 
1. data_dir (str): File path to the directory
2. split (str): Dataset split between training, testing and validation (eg. training_epochs, testing_epochs, valid_epochs)

Here's an example of what a split could look like

```
# ===== Training =====
train_dataset = EEG_Dataset(
    "/kaggle/input/eeg-clean-raw/dat-dataset-2-pro-max-supreme",
    "training_epochs"
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


# ===== Valid =====
valid_dataset = EEG_Dataset(
    "/kaggle/input/eeg-clean-raw/dat-dataset-2-pro-max-supreme",
    "validation_epochs"
)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)


# ===== Test =====
test_dataset = EEG_Dataset(
    "/kaggle/input/eeg-clean-raw/dat-dataset-2-pro-max-supreme",
    "testing_epochs"
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
```

### Metrics.py

Contains custom function to calculate EEG regression metrics: Signal to noise ratio, RRMSE (time), RRMSE (frequency) and Correlation Coefficient. The function returns a dict of a name of the metrics as keys and their respective values.

Here's an example

```
# Printing metrics
metrics = evaluate_model_metrics(model, test_loader)

print("Test metrics:")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
```
