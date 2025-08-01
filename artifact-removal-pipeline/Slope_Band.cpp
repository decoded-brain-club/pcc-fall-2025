#include "Func_EEG_Wear.h"
#include <cmath>
#include <vector>
#include <utility> // for std::pair
#include <numeric> // for std::inner_product


double linearRegressionSlope(const std::vector<double>& x, const std::vector<double>& y) {
    const size_t n = x.size();
    const double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / n;
    const double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / n;

    double numerator = 0.0, denominator = 0.0;
    for (size_t i = 0; i < n; ++i) {
        numerator += (x[i] - mean_x) * (y[i] - mean_y);
        denominator += (x[i] - mean_x) * (x[i] - mean_x);
    }
    return (denominator == 0.0) ? 0.0 : numerator / denominator;
}

double computeSlopeInBand(const std::vector<double>& signal, double sampling_rate, const std::pair<double, double>& freq_band) {
    // TODO: Replace with real PSD computation using FFT library
    std::vector<double> frequencies;       // Dummy
    std::vector<double> power_spectrum;    // Dummy

    // Filter freq within band
    std::vector<double> selected_freq, selected_power;
    for (size_t i = 0; i < frequencies.size(); ++i) {
        if (frequencies[i] >= freq_band.first && frequencies[i] <= freq_band.second) {
            selected_freq.push_back(std::log(frequencies[i]));
            selected_power.push_back(std::log(power_spectrum[i]));
        }
    }

    if (selected_freq.size() < 2) return 0.0;  // Not enough points

    return linearRegressionSlope(selected_freq, selected_power);
}

// Created by Jay on 7/18/2025.
//
