#include "Func_EEG_Wear.h"
#include <fstream>
#include <sstream>
#include <kfr/all.hpp>
#include <unordered_map>

std::unique_ptr<cortex::data::EEGData> load_csv_to_EEGData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open EEG CSV file.");
    }

    std::string line;
    std::getline(file, line); // First line with headers
    std::istringstream header_ss(line);
    std::vector<std::string> channel_names;
    std::string header;
    while (std::getline(header_ss, header, ',')) {
        channel_names.push_back(header);
    }

    std::unordered_map<std::string, std::vector<double>> temp_data;
    for (const auto& name : channel_names) {
        temp_data[name] = {};
    }

    // Read data lines
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string value;
        size_t i = 0;
        while (std::getline(ss, value, ',')) {
            if (i < channel_names.size()) {
                temp_data[channel_names[i++]].push_back(std::stod(value));
            }
        }
    }

    auto eeg = std::make_unique<cortex::data::EEGData>();

    for (const auto& [channel, values] : temp_data) {
        kfr::univector<double> vec(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            vec[i] = values[i];
        }
        eeg->set_channel(channel, std::move(vec));
    }

    return eeg;
}

// Created by Jay on 7/25/2025.
//
