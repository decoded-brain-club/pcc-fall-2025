import numpy as np
import matlab.engine
from pathlib import Path
import os
import io
from contextlib import redirect_stdout, redirect_stderr
from filelock import FileLock

def filter(config, eeg_data):
    """
    Applies notch, bandpass filter, and resampling to EEGData's raw MNE data.
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
    # Resample
    eeg_data.raw.resample(config.target_sfreq, npad='auto', window='boxcar', verbose=False)

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

    # Change working directory to temp
    engine.cd(str(matlab_workspace), nargout=0) 

    try:
        # Suppress MATLAB warnings for this session
        engine.eval("warning('off', 'all');", nargout=0, stdout=stdout_buffer, stderr=stderr_buffer)
        
        # Load the .set file into MATLAB's EEG structure
        engine.eval(f"EEG = pop_loadset('filename', '{set_name}');", nargout=0, stdout=stdout_buffer, stderr=stderr_buffer)
        
        ###################### RELAX

        # Extract bad channel labels from findNoisyChannels results
        engine.eval("bad_channel_indices = EEG.etc.badChannels;", nargout=0, stdout=stdout_buffer, stderr=stderr_buffer)
        engine.eval("if ~isempty(bad_channel_indices), bad_channel_labels = {EEG.chanlocs(bad_channel_indices).labels}; else, bad_channel_labels = {}; end", nargout=0, stdout=stdout_buffer, stderr=stderr_buffer)
        bad_channels = engine.workspace['bad_channel_labels']
        # Convert MATLAB cell array to Python list of strings
        if hasattr(bad_channels, 'tolist'):
            bad_channels = [str(ch) for ch in bad_channels.tolist()]
        else:
            bad_channels = [str(ch) for ch in bad_channels]
        log_path = Path(__file__).parent.parent / config.log_path
        report_path = log_path / "bad_channel_report.txt"
        os.makedirs(log_path, exist_ok=True)
        lock_path = str(report_path) + ".lock"
        with FileLock(lock_path):
            with open(report_path, 'a') as f: # add directory path and channels
                f.write(f"{eeg_data.file_path},{','.join(bad_channels)}\n")

        # Save the cleaned data to a new .set file
        cleaned_set_name = f"{set_name.replace('.set', '')}_relax.set"  # Construct the new filename
        cleaned_set_path = matlab_workspace / cleaned_set_name
        engine.eval(f"EEG = pop_saveset(EEG, 'filename', '{cleaned_set_name}');", nargout=0, stdout=stdout_buffer, stderr=stderr_buffer)  # Save the cleaned data

    except matlab.engine.MatlabExecutionError as e:
        print(f"[RELAX] MATLAB error: {e}")
        raise

    # Update eeg_data with clean data
    eeg_data.load_set(matlab_workspace / cleaned_set_name)

    # Delete temp .set files
    set_path.unlink()
    cleaned_set_path.unlink()
