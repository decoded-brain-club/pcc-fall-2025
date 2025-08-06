import mne
import os
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt

class EEGData:
    def __init__(self, config, file_path):
        """Initializes the EEGData object by loading an EEG file.
        Args:
            file_path (str): File path to the EEG data file. Supports .edf and .set files.
        Raises:
            FileNotFoundError: If the file path does not point to an existing file.
            ValueError: If the file extension is not .edf or .set.
        """
        # Configurations
        self.config = config
        # Check if path exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Initialize variables
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)

        # raw MNE data
        self.raw = None

        # Load file type (.set or .edf)
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.edf':
            self.raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        elif ext == '.set':
            self.raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # Keep EEG channels (only work if data is not screwed)
        picks = mne.pick_types(self.raw.info, eeg=True)
        self.raw.pick(picks)

        self._remove_channels(self.config.exclude_channels)  # Remove user specified channels if any

        channels = self.raw.info['ch_names']
        self.num_channels = len(channels) # Number of EEG channels

        if self.num_channels is not self.config.expected_channels:
            print(f"WARNING: inspect file: {self.file_name}, channels: {self.num_channels}")

    def _remove_channels(self, channels_to_remove: List[str]):
        """Remove specified channels from the raw data."""
        if not channels_to_remove:
            return
        original_channels = self.raw.info['ch_names']
        # Find channels to remove that actually exist
        channels_to_drop = [ch for ch in channels_to_remove if ch in original_channels]
        if channels_to_drop:
            self.raw.drop_channels(channels_to_drop)


    def get_channels_and_data(self):
        """Retrieves names and data of EEG channels from raw MNE data."""
        channels = self.raw.info['ch_names']
        data = self.raw.get_data()
        return channels, data
    
    def log_plot(self, label: str):
        """Save a plot screenshot to a directory with appropriate labeling."""
        # Determine label and output directory based on raw flag
        label = label.lower()
        log_dir = Path(__file__).parent.parent / self.config.log_path / "plots"
        os.makedirs(log_dir, exist_ok=True)

        # Create filename with label
        base_name = os.path.splitext(self.file_name)[0]
        plot_filename = f"{base_name}_{label}.png"
        save_path = log_dir / plot_filename
        
        # Create plot
        fig = self.raw.plot(
            n_channels=len(self.raw.ch_names),
            scalings='auto',
            title=f"{self.file_name} ({label})",
            duration=20.0,
            show=False  # Don't display the plot
        )
        
        # Position text in the top-right corner with background box for better visibility
        fig.text(0.60, 0.95, f"FILE: {self.file_name}\nLABEL: {label.upper()}", 
                ha='left', va='top', fontsize=10, fontweight='medium', 
                color='red', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                         edgecolor="red", linewidth=2, alpha=0.9))

        # Save the figure
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close to free memory
        
        print(f"Plot saved to: {save_path}")

    def plot(self):
        """Plot the EEG recording using MNE viewer and print channel types."""
        print("Number of channels:", len(self.raw.ch_names))
        self.raw.plot(
            n_channels=len(self.raw.ch_names), 
            scalings='auto', 
            title=self.file_name,
            duration=20.0  # Show 20 seconds per window
        )

    def save_to_edf(self, output_dir):
        """Saves the EEG data (channel names and signals) to a specified directory."""
        output_file = os.path.join(output_dir, os.path.splitext(self.file_name)[0] + ".edf")

        ch_names, eeg_data = self.get_channels_and_data()
        edf_info = mne.create_info(ch_names=ch_names, sfreq=self.raw.info['sfreq'], ch_types=['eeg'] * len(ch_names))
        edf_raw = mne.io.RawArray(eeg_data, edf_info)

        mne.export.export_raw(output_file, edf_raw, fmt='edf')
        print(f"Saved edf file to: {output_file}")

    def save_to_set(self, save_dir: Path) -> tuple[Path, str]:
        """
        Save the data to a .set file. Returns a tuple containing the full path to the saved .set file and the set file name.
        """
        base_name = os.path.splitext(self.file_name)[0]
        set_name = f"{base_name}.set"
        set_path = save_dir / set_name
        self.raw.export(str(set_path), fmt='eeglab', overwrite=True)
        return set_path, set_name

    def load_set(self, set_path: Path):
        """
        Load .set file. Updates self.raw and self.file_name accordingly.
        """
        raw = mne.io.read_raw_eeglab(str(set_path), preload=True)
        self.raw = raw
        self.file_name = set_path.name