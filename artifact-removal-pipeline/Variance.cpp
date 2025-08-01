#include "Func_EEG_Wear.h"
#include <vector>


// Step 1 and 2 will be reused. 1. Get vector data 2. Calculate mean
double calculateVariance(const std::vector<double>& data_vector, const double mean_value) {
    double sum_squared_diffs = 0.0;
    for (const double value : data_vector) {
        sum_squared_diffs += (value - mean_value) * (value - mean_value);
    }
    return sum_squared_diffs / data_vector.size();  // Or `.size() - 1` for sample variance
}
// Created by Jay on 7/16/2025.
