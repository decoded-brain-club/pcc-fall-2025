#include "Func_EEG_Wear.h"
#include <cmath>

double comp_Kurtosis(const std::vector<double>& spatial_map) {
    const int n = spatial_map.size();
    if (n == 0) return 0.0;

    const double mean_value = calculateMean(spatial_map);

    double sum_fourth_moment = 0.0;
    double sum_squared = 0.0;

    for (const double value : spatial_map) {
        const double deviation = value - mean_value;
        sum_fourth_moment += std::pow(deviation, 4);
        sum_squared += std::pow(deviation, 2);
    }

    if (sum_squared == 0.0) return 0.0;

    const double kurtosis = (n * sum_fourth_moment) / std::pow(sum_squared, 2);
    return kurtosis;
}

// Created by Jay on 7/18/2025.
//
