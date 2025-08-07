import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve().parent / "src")) # Add src/loaders directory to path

from loaders.tuh_loader import TUHLoader # type: ignore
from config import get_config # type: ignore

def main():

    # Get configurations
    config = get_config()

    # Initialize the TUHLoader and load TUH dataset
    loader = TUHLoader(config, True)
    loader.load_data()

    ## Unprocessed

    loader.unprocessed_mode() # Ensure loader is in default mode
    # No filters
    # Save as tensors in respective directories
    loader.generate_single_channel_epochs("unprocessed")
    # Save the plots of the first 5 recordings before minimal processing
    loader.get_eeg_data_by_index(1).log_plot("unprocessed", start_time=0.0, duration=60.0, scale=20e-6)
    loader.get_eeg_data_by_index(3).log_plot("unprocessed", start_time=0.0, duration=60.0, scale=20e-6)
    loader.get_eeg_data_by_index(5).log_plot("unprocessed", start_time=0.0, duration=60.0, scale=20e-6)
    loader.get_eeg_data_by_index(7).log_plot("unprocessed", start_time=0.0, duration=60.0, scale=20e-6)
    loader.get_eeg_data_by_index(9).log_plot("unprocessed", start_time=0.0, duration=60.0, scale=20e-6)

    loader.export_state() # Export the current state of the loader for testing
    # unprocessed_loader = TUHLoader.load_state("PATH TO .pkl FILE, CHECK LOGS FOLDER", config) # For testing

    ## Raw

    loader.raw_mode() # Activate engine's raw mode
    # Apply bandpass, notch filters and resample
    # Save as tensors in respective directories
    loader.generate_single_channel_epochs("filter")
    # Save the plots of the first 5 recordings after minimal processing
    loader.get_eeg_data_by_index(1).log_plot("raw", start_time=0.0, duration=60.0, scale=20e-6)
    loader.get_eeg_data_by_index(3).log_plot("raw", start_time=0.0, duration=60.0, scale=20e-6)
    loader.get_eeg_data_by_index(5).log_plot("raw", start_time=0.0, duration=60.0, scale=20e-6)
    loader.get_eeg_data_by_index(7).log_plot("raw", start_time=0.0, duration=60.0, scale=20e-6)
    loader.get_eeg_data_by_index(9).log_plot("raw", start_time=0.0, duration=60.0, scale=20e-6)

    loader.export_state() # Export the current state of the loader for testing
    # raw_loader = TUHLoader.load_state("PATH TO .pkl FILE, CHECK LOGS FOLDER", config) # For testing

    ## Clean

    loader.clean_mode() # Activate engine's clean mode
    # Runs data through RELAX pipeline
    # Save as tensors in respective directories
    loader.generate_single_channel_epochs("relax")
    # Save the plots of the first 5 recordings after RELAX processing
    loader.get_eeg_data_by_index(1).log_plot("clean", start_time=0.0, duration=60.0, scale=20e-6)
    loader.get_eeg_data_by_index(3).log_plot("clean", start_time=0.0, duration=60.0, scale=20e-6)
    loader.get_eeg_data_by_index(5).log_plot("clean", start_time=0.0, duration=60.0, scale=20e-6)
    loader.get_eeg_data_by_index(7).log_plot("clean", start_time=0.0, duration=60.0, scale=20e-6)
    loader.get_eeg_data_by_index(9).log_plot("clean", start_time=0.0, duration=60.0, scale=20e-6)

    loader.export_state() # Export the current state of the loader for testing
    # clean_loader = TUHLoader.load_state("PATH TO .pkl FILE, CHECK LOGS FOLDER", config) # For testing

if __name__ == "__main__":
    import sys
    sys.exit(main())