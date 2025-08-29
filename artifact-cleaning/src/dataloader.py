import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from torch.utils.data import Subset 

import numpy as np
import joblib
# from IPython.display import FileLink

from pathlib import Path

class EEG_Dataset(Dataset):
    """
    A PyTorch Dataset for loading raw and clean EEG epoch pairs.
    It expects the following structure:
    - data/
        - raw/
            - raw_training_epochs/
                - subject1/
                    - c3_epoch0_raw.pt
        - clean/
            - clean_training_epochs/
                - subject1/
                    - c3_epoch0_clean.pt
    """
    def __init__(self, data_dir, split, transform=None):
        """
        Args:
            data_dir (str): The path to the root 'data' directory.
            split (str): The dataset split (e.g., 'training_epochs', 'validation_epochs', or 'test_epochs').
        """
        self.transform = transform
        
        # Construct the paths to the raw and clean data folders for the specified split
        self.raw_dir = Path(data_dir) / 'raw' / f'raw_{split}'
        self.clean_dir = Path(data_dir) / 'clean' / f'clean_{split}'

        if not self.raw_dir.is_dir() or not self.clean_dir.is_dir():
            raise FileNotFoundError(f"One of the specified directories does not exist: {self.raw_dir} or {self.clean_dir}")

        self.file_pairs = []

        # Traverse the directory to find all raw files and their corresponding clean files
        for subject_dir in self.raw_dir.iterdir():
            if subject_dir.is_dir():
                for raw_file_path in subject_dir.glob('*.pt'):
                    # The clean file path is found by replacing the directory and file suffix
                    relative_path = raw_file_path.relative_to(self.raw_dir)
                    clean_file_path = self.clean_dir / relative_path.with_name(
                        raw_file_path.stem.replace('_raw', '_clean') + '.pt'
                    )

                    if clean_file_path.is_file():
                        self.file_pairs.append((raw_file_path, clean_file_path))
                    # else:
                    #     print(f"Warning: Corresponding clean file not found for {raw_file_path}")

    def __len__(self):
        """Returns the total number of data samples."""
        return len(self.file_pairs)

    def __getitem__(self, idx):
      """Loads and returns a raw and clean pair, normalized to [-1, 1]."""
      raw_path, clean_path = self.file_pairs[idx]

      # Load the tensors from their file paths
      raw_tensor = torch.load(raw_path)
      clean_tensor = torch.load(clean_path)

      # Add channel dim: [512] -> [1, 512]
      raw_tensor = raw_tensor.unsqueeze(0)
      clean_tensor = clean_tensor.unsqueeze(0)

      # Standardize each sample to [-1, 1]
      def normalize(tensor): # Comment out normalize function if already normalizing in data transformation
          min_val = tensor.min()
          max_val = tensor.max()
          if max_val > min_val:  # Avoid division by zero
              tensor = 2 * (tensor - min_val) / (max_val - min_val) - 1
          else:
              tensor = torch.zeros_like(tensor)
          return tensor

      raw_tensor = normalize(raw_tensor)
      
      if self.transform:
          raw_tensor = self.transform(raw_tensor)
          
      clean_tensor = normalize(clean_tensor)

      return raw_tensor, clean_tensor



# Dataset & Loaders examples

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