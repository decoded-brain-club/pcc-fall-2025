#include "Func_EEG_Wear.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

std::vector<double> readVectorFile(const std::string& fileName) {
    std::vector<double> data_vector;

    if (std::ifstream file(fileName); file.is_open()) {
        double value;
        while (file >> value) {
            data_vector.push_back(value);
        }
        file.close();
    } else {
        std::cerr << "Error opening file " << fileName << std::endl;
    }
    return data_vector;
}

std::vector<double> calculateZscore(const std::vector<double>& data_vector, double mean_value, double std_dev) {
    std::vector<double> z_score;

    for (const double value : data_vector) {
        double z = (value - mean_value) / std_dev;
        z_score.push_back(z);
    }
    return z_score;
}

double calculateMean(const std::vector<double>& data_vector) {
    double sum = 0;
    for (const double val : data_vector) {
        sum += val;
    }
    return sum / data_vector.size();
}

double calculateSDev(const std::vector<double>& data_vector, double mean_value) {
    double sum_squared_diffs = 0.0;

    for (const double value : data_vector) {
        sum_squared_diffs += (value - mean_value) * (value - mean_value);
    }
    const double variance = sum_squared_diffs / data_vector.size();
    return std::sqrt(variance);
}
// Created by Jay on 7/17/2025.
//
