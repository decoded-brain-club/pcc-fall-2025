#include <cmath>
#include <algorithm>
#include <vector>

double computeMedGrad(const std::vector<double>& signal) {
    std::vector<double> gradients;

    // Step 1: Compute differences between consecutive samples
    for (size_t i = 0; i + 1 < signal.size(); ++i) {
        gradients.push_back(std::abs(signal[i + 1] - signal[i]));
    }

    // Step 2: Compute median of absolute gradients
    std::ranges::sort(gradients);
    const size_t n = gradients.size();

    if (n == 0) return 0.0;  // Return 0 if signal is too short

    if (n % 2 == 1) {
        return gradients[n / 2];
    } else {
        return (gradients[n / 2 - 1] + gradients[n / 2]) / 2.0;
    }
}

// Created by Jay on 7/18/2025.
//
