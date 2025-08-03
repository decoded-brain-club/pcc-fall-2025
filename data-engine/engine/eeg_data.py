import mne
import os

class EEGData:
    def __init__(self, file_path):
        """Initializes the EEGData object by loading an EEG file.
        Args:
            file_path (str): File path to the EEG data file. Supports .edf and .set files.
        Raises:
            FileNotFoundError: If the file path does not point to an existing file.
            ValueError: If the file extension is not .edf or .set.
        """
        # Check if path exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        # Initialize variables
        self.file_path = file_path 
        self.raw = None
        ext = os.path.splitext(file_path)[1].lower() # .set or .edf
        self.samp_freq=256 # sample frequency
        # Load file type
        if ext == '.edf':
            self.raw = mne.io.read_raw_edf(file_path, preload=True)
        elif ext == '.set':
            self.raw = mne.io.read_raw_eeglab(file_path, preload=True)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        print(f"Loaded {ext} file: {file_path}")
        
    @property
    def info(self):
        """Returns metadata of the raw MNE data."""
        return self.raw.info
    
    def get_channels_and_data(self):
        """Retrives names and data of EEG channels from raw MNE data
        Returns:
            Dict: _description_
        """
        # Get list of indices of only EEG data
        picks = mne.pick_types(self.raw.info, eeg=True)
        # Get channel names from list of EEG indices
        eeg_channel_names = [self.raw.info['ch_names'][i] for i in picks]
        # Get data for channels
        eeg_data = self.raw.get_data(picks=picks) # np.2d array (n_channels, n_times)
        
        return eeg_channel_names, eeg_data
    
    def save_as_edf(self, file_name):
        """Saves the EEG data (channel names and signals) to a new edf file.
        Args:
            file_name (str): Name of the output EDF file. Must end with '.edf'.
        """
        # Retrieve channel names and their data
        ch_names, eeg_data = self.ch_names_data()  
        # Create metadata
        edf_info = mne.create_info(ch_names=ch_names, sfreq=self.samp_freq, ch_types=['eeg']*len(ch_names))  
        # Create a raw object
        edf_raw = mne.io.RawArray(eeg_data, edf_info)  
        # Save
        if file_name.endswith('.edf'):
            mne.export.export_raw(file_name, edf_raw, fmt='edf') 
            print(f"Saved edf file to: {file_name}")
        else:
            print(f"{file_name} needs to end with '.edf'")