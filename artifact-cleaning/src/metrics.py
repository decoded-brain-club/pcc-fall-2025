import numpy as np
import torch

# Metrics to measure success for regression task

def compute_snr(raw, clean):
    noise = raw - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)

    return snr

def compute_rrmse_time(raw, clean):
    numerator = np.sqrt(np.mean((raw - clean) ** 2))
    denominator = np.sqrt(np.mean(clean ** 2))
    return numerator / denominator

def compute_rrmse_freq(raw, clean):
    raw_fft = np.abs(np.fft.rfft(raw))
    clean_fft = np.abs(np.fft.rfft(clean))
    numerator = np.sqrt(np.mean((raw_fft - clean_fft) ** 2))
    denominator = np.sqrt(np.mean(clean_fft ** 2))
    return numerator / denominator

def compute_average_cc(raw, clean):
    correlations = []
    for i in range(raw.shape[0]):
        r = np.corrcoef(raw[i], clean[i])[0, 1]
        correlations.append(r)
    return np.mean(correlations)


def evaluate_model_metrics(model, dataloader):
    model.eval()
    snr_list = []
    rrmse_time_list = []
    rrmse_freq_list = []
    cc_list = []

    # Testing
    with torch.no_grad():
        for raw, clean in dataloader:
            # Run model forward
            model_output_batch = model(raw)

            # Move tensors to CPU and convert to numpy
            raw_np = raw.cpu().numpy()
            clean_np = clean.cpu().numpy()
            output_np = model_output_batch.cpu().numpy()

            # Compute metrics per sample in batch
            for i in range(raw_np.shape[0]):
                # raw_sample = raw_np[i]
                clean_sample = clean_np[i]
                model_output_sample = output_np[i]

                # Use model output and clean target (or raw if comparing raw to clean)
                snr_val = compute_snr(model_output_sample, clean_sample)
                rrmse_time_val = compute_rrmse_time(model_output_sample, clean_sample)
                rrmse_freq_val = compute_rrmse_freq(model_output_sample, clean_sample)
                cc_val = compute_average_cc(model_output_sample, clean_sample)

                snr_list.append(snr_val)
                rrmse_time_list.append(rrmse_time_val)
                rrmse_freq_list.append(rrmse_freq_val)
                cc_list.append(cc_val)

    # Aggregate metric results across entire dataset
    metrics = {
        "SNR": np.mean(snr_list),
        "RRMSE_time": np.mean(rrmse_time_list),
        "RRMSE_freq": np.mean(rrmse_freq_list),
        "Average_CC": np.mean(cc_list)
    }
    return metrics