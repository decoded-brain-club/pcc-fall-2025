import numpy as np
import matlab.engine
from pathlib import Path
import os
import io

def filter(config, eeg_data):
    """
    Applies notch and bandpass filter to EEGData's raw MNE data.
    """
    # Notch filter
    notch_range = config.filter_config['notch_filter_range']
    eeg_data.raw.notch_filter(
        freqs=np.arange(notch_range[0], notch_range[1] + 1),
        picks='eeg',
        method='spectrum_fit',
        filter_length='auto',
        phase='zero',
        verbose=False
    )
    # Bandpass filter
    eeg_data.raw.filter(
        l_freq=config.filter_config['low_cutoff'],
        h_freq=config.filter_config['high_cutoff'],
        picks='eeg',
        method='fir',
        fir_design='firwin',
        phase='zero',
        verbose=False
    )

def relax_pipeline(config, engine, eeg_data):
    """
    Custom RELAX-inspired pipeline for EEGData.
    """

    matlab_workspace = Path(__file__).parent.parent / config.matlab_workspace_path  # MATLAB workspace directory
    os.makedirs(matlab_workspace, exist_ok=True)

    # Create output buffers to suppress MATLAB output
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    # Save raw data to a temp .set file
    set_path, set_name = eeg_data.save_to_set(matlab_workspace)

    # Change working directory to MATLAB workspace
    engine.cd(str(matlab_workspace), nargout=0)

    try:
        # Run RELAX
        matlab_cmd = f"""
        RELAX_config = load_RELAX_config('RELAX_config.yaml');
        RELAX_config.myPath = '{matlab_workspace}/';
        RELAX_config.filename = '{set_path}';
        """
        engine.eval(matlab_cmd, nargout=0)
        engine.eval("pop_RELAX(RELAX_config);", nargout=0)

        # Save the cleaned data to a new .set file
        cleaned_set_name = f"{set_name.replace('.set', '')}_RELAX.set"  # Construct the new filename
        cleaned_set_path = matlab_workspace / "RELAXProcessed/Cleaned_Data" / cleaned_set_name

    except matlab.engine.MatlabExecutionError as e:
        print(f"[RELAX] MATLAB error: {e}")
        raise

    # Update eeg_data with clean data
    eeg_data.load_set(cleaned_set_path)

    # Delete temp .set files
    set_path.unlink()
    cleaned_set_path.unlink()
