#include "Func_EEG_Wear.h"
#include <iomanip>
#include <iostream>

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // ---------------- CHANNEL CHECKING (REAL-TIME OR BATCH) ---------------- //
    const std::string channel_filename = "channels.txt";
    const std::vector<std::vector<double>> all_channels = readAllChannels(channel_filename);

    if (all_channels.empty()) {
        std::cerr << "No channels loaded from file." << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> good_channels;

    for (const auto& target_channel : all_channels) {
        const double mean_val = calculateMean(target_channel);
        const double variance_val = calculateVariance(target_channel, mean_val);
        const double variance_z = (variance_val - 0.0) / 1.0; // placeholder z-score

        const double mean_corr = calc_mean_corr(target_channel, all_channels);
        const double corr_z = (mean_corr - 0.0) / 1.0; // placeholder z-score

        const double hurst = estimateHurstExponent(target_channel);
        const double hurst_z = (hurst - 0.0) / 1.0; // placeholder z-score

        bool is_bad = (variance_z > 3 || corr_z > 3 || hurst_z > 3);

        if (is_bad) {
            std::cout << "Channel flagged as BAD and removed.\n";
        } else {
            good_channels.push_back(target_channel);
        }
    }

    if (good_channels.empty()) {
        std::cerr << "No good channels available for ICA." << std::endl;
        return 1;
    }

    // -------------------- ICA COMPONENT ANALYSIS -------------------- //
    // ICA placeholder: here you’d normally run ICA to get independent components
    std::vector<std::vector<double>> ica_components = good_channels; // placeholder

    for (const auto& component : ica_components) {
        const double kurtosis = comp_Kurtosis(component);
        const double kurtosis_z = (kurtosis - 0.0) / 1.0; // placeholder z-score

        const double hurst = estimateHurstExponent(component);
        const double hurst_z = (hurst - 0.0) / 1.0; // placeholder z-score

        const double med_grad = computeMedGrad(component);
        const double grad_z = (med_grad - 0.0) / 1.0; // placeholder z-score

        bool is_artifact = (kurtosis_z > 3 || hurst_z > 3 || grad_z > 3);

        if (is_artifact) {
            std::cout << "ICA component flagged and removed (artifact).\n";
            // Here you'd subtract/remove the ICA component from signal
        }
    }

    return 0;
}

// TIP See CLion help at <a href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>. Also, you can try interactive lessons for CLion by selecting 'Help | Learn IDE Features' from the main menu.
