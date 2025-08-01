#include "Func_EEG_Wear.h"
#include <cmath>
#include <fstream>
#include <sstream>

std::vector<std::vector<double>> readAllChannels(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> channels;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> channel;
        double value;

        while (ss >> value) {
            channel.push_back(value);
        }

        if (!channel.empty()) {
            channels.push_back(channel);
        }
    }
    return channels;
}

// standard pearson correlation function
double pearson_correlation(const std::vector<double>& x, const std::vector<double>& y) {
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    int n = x.size();

    for (int i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }
    const double numerator = (n * sum_xy) - (sum_x * sum_y);
    const double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    return (denominator == 0) ? 0 : numerator / denominator;
}

double calc_mean_corr(const std::vector<double>& target_channel, const std::vector<std::vector<double>>& all_channels) {
    double correlation_sum = 0.0;
    int num_other_channels = 0;

    for (const auto& channel : all_channels) {
        if (&channel != &target_channel) { // comparing address to avoid self-comparison
            const double corr = pearson_correlation(target_channel, channel);
            correlation_sum += corr;
            ++num_other_channels;
        }
    }

    if (num_other_channels == 0) return 0.0;
    return correlation_sum / num_other_channels;
}
// Created by Jay on 7/17/2025.
//
