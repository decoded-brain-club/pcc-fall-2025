cmake_minimum_required(VERSION 3.15)
project(Calculating_Z_Score)

set(CMAKE_CXX_STANDARD 20)

# ======================
# Google Test Setup
# ======================
include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(ITPP REQUIRED)

enable_testing()

# ======================
# Main Executable
# ======================
add_executable(eeg_main
        main.cpp
        Func_EEG_Wear.h
        Z_Score.cpp
        Mean_Corr.cpp
        Hurst.cpp
        kurtosis.cpp
        Slope_Band.cpp
        Med_Gradient.cpp
        Variance.cpp
        EEGData.cpp
        EEGFiltering.cpp
        RawEEGtoEEGData.cpp
)

# ======================
# Unit Tests Executable
# ======================
add_executable(eeg_tests
        test_main.cpp
        test_func.cpp
        EEGData.cpp
        EEGFiltering.cpp
        RawEEGtoEEGData.cpp
)

target_link_libraries(eeg_tests
        gtest_main
        MyProject PRIVATE itpp
)

include(GoogleTest)
gtest_discover_tests(eeg_tests)
