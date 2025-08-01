#include <gtest/gtest.h>
#include "Func_EEG_Wear.h"
#include "cmake-build-debug/_deps/googletest-src/googletest/include/gtest/gtest.h"

TEST(MeanTest, SimpleAverage) {
    std::vector<double> values = {1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(calculateMean(values), 2.0);
}

TEST(SDevTest, BasicDeviation) {
    std::vector<double> values = {1.0, 2.0, 3.0};
    double mean = calculateMean(values);
    EXPECT_NEAR(calculateSDev(values, mean), 0.8165, 0.0001);
}

TEST(HurstTest, FlatSignal) {
    std::vector<double> flat(50, 1.0);
    EXPECT_DOUBLE_EQ(estimateHurstExponent(flat), 0.0);
}

// Created by Jay on 7/18/2025.
//
