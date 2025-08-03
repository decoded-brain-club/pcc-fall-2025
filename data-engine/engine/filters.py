from scipy.stats import zscore

def filter_data(eeg_data, low_pass=0.5, high_pass=40, notch_freq=60):
    """Applies notch, bandpass filter, and resampling to EEGData's raw MNE data.
    Args:
        eeg_data (EEGData): Instance of EEGData class.
        low_pass (float, optional): Lower bound of bandpass filter. Defaults to 0.5.
        high_pass (int, optional): Upper bound of bandpass filter. Defaults to 40.
        notch_freq (int, optional): Frequency for notch filtering. Defaults to 60.
    """
    eeg_data.raw.notch_filter(freqs=notch_freq, inplace=True) # Notch filter
    eeg_data.raw.filter(l_freq=low_pass, h_freq=high_pass, inplace=True) # Low and high pass filter
    eeg_data.raw.resample(eeg_data.samp_freq) # Resample frequency
