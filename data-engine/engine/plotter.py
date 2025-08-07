from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src")) # Add root directory to path

from config import get_config # type: ignore
from eeg_data import EEGData  # type: ignore

def main():

    file_name = "FILE NAME" # Replace with your file name

    config = get_config()
    path = Path(__file__).parent / config.raw_data_path / file_name
    eeg = EEGData(config, file_path=path)
    total_samples = eeg.raw.n_times
    sampling_rate = eeg.raw.info['sfreq']
    duration_seconds = total_samples / sampling_rate # Use to see end of recording (e.g. duration_seconds - 60.0 for last minute)

    # Adjust parameters as needed (label, start_time, duration, scale)
    eeg.log_plot("test", start_time=10.0, duration=60.0, scale=100e-6)

if __name__ == "__main__":
    import sys
    sys.exit(main())
