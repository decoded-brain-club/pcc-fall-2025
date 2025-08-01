#include "Func_EEG_Wear.h"
#include <cmath>
#include <algorithm>

double estimateHurstExponent(const std::vector<double>& signal) {
    const double mean_signal = calculateMean(signal);

    std::vector<double> cumulative_sum;
    double running_total = 0.0;
    for (const double val : signal) {
        running_total += (val - mean_signal);
        cumulative_sum.push_back(running_total);
    }

    const double R = *std::ranges::max_element(cumulative_sum) -
                     *std::ranges::min_element(cumulative_sum);

    const double S = calculateSDev(signal, mean_signal);

    if (S == 0.0) return 0.0;

    return std::log(R / S) / std::log(static_cast<double>(signal.size()));
}

// Created by Jay on 7/17/2025.
//
