from abc import ABC, abstractmethod
import os
import shutil
import pickle
import io
from datetime import datetime
from typing import List, Tuple
from eeg_data import EEGData
import processor as processor
import torch
from pathlib import Path
import matlab.engine
import shutil

# Auto-detects environment
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

class DatasetLoader(ABC):
    def __init__(self, config, raw: bool, cap: int):
        self.config = config # Configurations
        self.raw = raw # Mode
        self.cap = cap # Maximum number of data files to load (for testing)

        self.dir_path = Path(__file__).parent.parent / config.raw_data_path
        self.file_paths = self.__load_file_paths(cap)

        # Data containers
        self.training_data: List[EEGData] = []
        self.validation_data: List[EEGData] = []
        self.testing_data: List[EEGData] = []

    def raw_mode(self):
        """Switch loader to raw mode when working with unprocessed data."""
        self.raw = True

    def clean_mode(self):
        """Switch loader to clean mode when working with preprocessed (filtered) data."""
        self.raw = False

    def __load_file_paths(self, cap) -> List[str]:
        """Load .edf and .set file paths from the dataset directory and return them as a list of strings.

        Returns:
            List[str]: List of EEG file paths as strings.
        """
        if not self.dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.dir_path}")

        file_paths = []
        for root, _, files in os.walk(str(self.dir_path)):
            for f in files:
                if f.lower().endswith((".edf", ".set")):
                    file_paths.append(str(Path(root) / f))

        file_paths.sort()  # Sort for deterministic ordering

        if cap is not None:  # Limit the number of files (for testing)
            file_paths = file_paths[:cap]

        print(f"Found {len(file_paths)} EEG files.")
        return file_paths

    @abstractmethod
    def _split(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Split the dataset file paths into training, validation, and testing sets.

        Implementers should define the logic for splitting self.file_paths into three non-overlapping lists,
        typically based on random shuffling, subject IDs, or chronological order, depending on the dataset and use case.

        Returns:
            Tuple[List[str], List[str], List[str]]: Three lists containing file paths for training, validation, and testing, respectively.
        """
        pass

    def load_data(self):
        """Load EEG data into training, validation, and testing sets."""
        train_files, val_files, test_files = self._split()

        # Clear previous data (if any)
        self.training_data.clear()
        self.validation_data.clear()
        self.testing_data.clear()

        # Load training data with progress bar
        for file in tqdm(train_files, desc="Loading training data...", unit="file"):
            self.training_data.append(EEGData(self.config, file))
        
        # Load validation data with progress bar
        for file in tqdm(val_files, desc="Loading validation data...", unit="file"):
            self.validation_data.append(EEGData(self.config, file))
        
        # Load testing data with progress bar
        for file in tqdm(test_files, desc="Loading testing data...", unit="file"):
            self.testing_data.append(EEGData(self.config, file))

        print(f"Loaded {len(self.training_data)} training files, "
              f"{len(self.validation_data)} validation files, "
              f"{len(self.testing_data)} testing files.")
        
    def generate_single_channel_epochs(self, process: str):
        """
        Generate single-channel EEG epochs, label them, and store them as PyTorch tensors in designated folders.

        Parameters:
            process (str): Specifies the processing step to perform. 
                - "filter": Applies filtering to raw data (only allowed in raw mode).
                - "artifact_correction": Applies artifact correction using MATLAB EEGLAB (only allowed in clean mode).

        Expected Behavior:
            - Iterates over all EEGData objects in the loader (training, validation, testing).
            - For each EEG file, creates a directory to store its epochs.
            - For each channel in the EEG data, splits the signal into overlapping epochs of fixed duration and step size.
            - Converts each epoch into a PyTorch tensor and saves it to disk with a descriptive filename.

        Side Effects:
            - Writes epoch tensor files to disk in the appropriate directory structure.
            - May start a MATLAB engine session and interact with EEGLAB for artifact correction.
            - Prints progress and warnings to the console.
            - Raises exceptions if processing is attempted in an invalid mode or if data is insufficient.

        Raises:
            ValueError: If an invalid process is specified or if processing is attempted in the wrong mode.
            FileNotFoundError: If required directories do not exist.
        """
        tag, engine = self._prepare_mode() # Prepare mode and engine

        for eeg_data, save_path in tqdm(self, desc="Generating single-channel epochs"):

            # Check if recording is long enough for ICA to run effectively
            minimum_segment_length = int(self.config.minimum_time_per_weight * (eeg_data.num_channels ** 2))
            if eeg_data.raw.n_times < minimum_segment_length:
                print(f"Skipping {eeg_data.file_name} due to insufficient data length: {eeg_data.raw.n_times} < {minimum_segment_length}")
                continue

            # Make a directory for each EEG file
            file_base = os.path.splitext(eeg_data.file_name)[0]
            dir_path = save_path / file_base
            os.makedirs(dir_path, exist_ok=True)

            # Process
            if process == "filter":
                if not self.raw: raise ValueError("You cannot filter data in clean mode.")
                processor.filter(self.config, eeg_data)
            elif process == "relax":
                if self.raw: raise ValueError("You cannot correct artifacts in raw mode.")
                # Use a simpler progress indicator for RELAX
                with tqdm(total=None, desc=f"RELAX: {eeg_data.file_name}", 
                         bar_format='{desc} {elapsed}', leave=False) as pbar:
                    processor.relax_pipeline(self.config, engine, eeg_data)
            else:
                raise ValueError(f"You must specify a valid process: 'filter' or 'artifact_correction'. Got: {process}")

            channels, data = eeg_data.get_channels_and_data() # np.2d array (n_channels, n_times)

            for i, channel in enumerate(tqdm(channels, desc=f"Processing channels in {eeg_data.file_name}", unit="channel", leave=False)):
                
                channel_data = data[i, :]

                sfreq = eeg_data.raw.info["sfreq"]
                if sfreq != self.config.target_sfreq: # make sure resampling occured on all data
                    raise ValueError(f"Unsupported sampling frequency: {sfreq}. Expected {self.config.target_sfreq} Hz. For file: {eeg_data.file_name}")
                
                epoch_length = int(sfreq * self.config.epoch_duration)  # window
                step_size = int(sfreq * self.config.step_duration)      # step

                for start in range(0, len(channel_data) - epoch_length + 1, step_size):
                    end = start + epoch_length
                    epoch = channel_data[start:end]
                    # Convert epoch to tensor
                    epoch_tensor = torch.tensor(epoch, dtype=torch.float32)
                    # Filename
                    epoch_number = start // step_size
                    filename = f"{channel}_epoch{epoch_number}_{tag}.pt"
                    full_path = dir_path / filename
                    # Save
                    torch.save(epoch_tensor, full_path)

        # Shut down MATLAB
        if engine:
            # Create output buffers for cleanup operations
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            try:
                engine.eval("clear EEG", nargout=0, stdout=stdout_buffer, stderr=stderr_buffer)
            except matlab.engine.MatlabExecutionError as e:
                print(f"[clear EEG] MATLAB error: {e}")
                raise
            
            engine.quit()

    def _prepare_mode(self):
        """
        Helper method to prepare paths and engine based on raw/clean mode.
        Returns:
            tag (str): 'raw' or 'clean'
            engine (matlab.engine.MatlabEngine or None): MATLAB engine if clean mode, else None
        """
        engine = None
        if not self.raw:
            tag = "clean"
            train = Path(__file__).parent.parent / self.config.clean_training_epochs_path
            val = Path(__file__).parent.parent / self.config.clean_validation_epochs_path
            test = Path(__file__).parent.parent / self.config.clean_testing_epochs_path
            # Clear and recreate directories to avoid mixing with previous runs
            print("Clearing directories for clean mode...")
            for path in [train, val, test]:
                if path.exists():
                    shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)
            print("Cleaning done.")
            # Start MATLAB with output suppression
            # Capture stdout/stderr to suppress output
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            engine = matlab.engine.start_matlab()
            # Suppress MATLAB warnings and outputs
            engine.eval("warning('off', 'all');", nargout=0, stdout=stdout_buffer, stderr=stderr_buffer)
            engine.cd(self.config.eeglab_absolute_path, nargout=0)
            engine.eval("eeglab; close;", nargout=0, stdout=stdout_buffer, stderr=stderr_buffer)
        else:
            tag = "raw"
            train = Path(__file__).parent.parent / self.config.raw_training_epochs_path
            val = Path(__file__).parent.parent / self.config.raw_validation_epochs_path
            test = Path(__file__).parent.parent / self.config.raw_testing_epochs_path
            # Clear and recreate directories to avoid mixing with previous runs
            print("Clearing directories for raw mode...")
            for path in [train, val, test]:
                if path.exists():
                    shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)
            print("Cleaning done.")
        return tag, engine

    def __iter__(self):
        root = Path(__file__).parent.parent
        if self.raw:
            self._iterator_data = (
                [(eeg_data, root / self.config.raw_training_epochs_path) for eeg_data in self.training_data] +
                [(eeg_data, root / self.config.raw_validation_epochs_path) for eeg_data in self.validation_data] +
                [(eeg_data, root / self.config.raw_testing_epochs_path) for eeg_data in self.testing_data]
            )
        else:
            self._iterator_data = (
                [(eeg_data, root / self.config.clean_training_epochs_path) for eeg_data in self.training_data] +
                [(eeg_data, root / self.config.clean_validation_epochs_path) for eeg_data in self.validation_data] +
                [(eeg_data, root / self.config.clean_testing_epochs_path) for eeg_data in self.testing_data]
            )
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self._iterator_data):
            raise StopIteration
        result = self._iterator_data[self._index]
        self._index += 1
        return result
    
    def get_eeg_data_by_index(self, idx):
        """
        Retrieve an EEGData object by its index from training, validation, or testing data.
        """
        train = len(self.training_data)
        val = len(self.validation_data)
        test = len(self.testing_data)

        if idx < train:
            return self.training_data[idx]
        elif idx < train + val:
            return self.validation_data[idx - train]
        elif idx < train + val + test:
            return self.testing_data[idx - train - val]
        else:
            raise IndexError(f"Index {idx} out of range in DataLoader (valid range: 0 to {train + val + test - 1})")
        
    def get_eeg_data_by_file_name(self, name: str):
        """
        Retrieve an EEGData object by its file name from training, validation, or testing data.
        """
        for eeg in self.training_data:
            if eeg.file_name == name: return eeg
        for eeg in self.validation_data:
            if eeg.file_name == name: return eeg
        for eeg in self.testing_data:
            if eeg.file_name == name: return eeg
        raise FileNotFoundError(f"No EEGData object found with file name: {name}")
    
    def export_state(self) -> str:
        """
        Export the current DatasetLoader state to a pickle file in the log directory.
        
        Args:
            filename (str, optional): Custom filename. If None, generates timestamp-based name.
            
        Returns:
            str: Path to the exported file
        """
        # Create log directory
        log_dir = Path(__file__).parent.parent / self.config.log_path
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "raw" if self.raw else "clean"
        filename = f"dataset_loader_{mode}_{timestamp}.pkl"
        
        # Ensure .pkl extension
        if not filename.endswith('.pkl'):
            filename += '.pkl'
            
        save_path = log_dir / filename
        
        # Create state dictionary with all important attributes
        state = {
            'config': self.config,
            'raw': self.raw,
            'cap': self.cap,
            'dir_path': self.dir_path,
            'file_paths': self.file_paths,
            'training_data': self.training_data,
            'validation_data': self.validation_data,
            'testing_data': self.testing_data,
            'export_timestamp': datetime.now().isoformat(),
            'class_name': self.__class__.__name__
        }
        
        # Save to pickle file
        with open(save_path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"DatasetLoader state exported to: {save_path}")
        return str(save_path)
    
    @classmethod
    def load_state(cls, filepath: str, config=None):
        """
        Load a DatasetLoader state from a pickle file.
        
        Args:
            filepath (str): Path to the pickle file
            config: Optional new config to override the saved one
            
        Returns:
            DatasetLoader: New instance with loaded state
        """
        # Load state from pickle file
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Use provided config or fallback to saved config
        use_config = config if config is not None else state['config']
        
        # Create new instance
        instance = cls.__new__(cls)
        
        # Set all attributes from saved state
        instance.config = use_config
        instance.raw = state['raw']
        instance.cap = state['cap']
        instance.dir_path = state['dir_path']
        instance.file_paths = state['file_paths']
        instance.training_data = state['training_data']
        instance.validation_data = state['validation_data']
        instance.testing_data = state['testing_data']
        
        print(f"DatasetLoader state loaded from: {filepath}")
        print(f"Loaded {len(instance.training_data)} training, "
              f"{len(instance.validation_data)} validation, "
              f"{len(instance.testing_data)} testing files")
        print(f"Export timestamp: {state.get('export_timestamp', 'Unknown')}")
        
        return instance
    
    

