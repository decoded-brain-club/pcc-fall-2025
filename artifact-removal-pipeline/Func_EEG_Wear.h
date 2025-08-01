//
// Created by Jay on 7/17/2025.
//

#ifndef FUNC_EEG_WEAR_H
#define FUNC_EEG_WEAR_H
#pragma once
#include <vector>
#include <string>
#include <istream>
#include <memory>
#include <stdexcept>
#include <fmt/format.h>
#include <tsl/robin_map.h>
#include <kfr/all.hpp>
#include "EEGData.hpp"

//Z-Score
std::vector<double> readVectorFile(const std::string& fileName);
std::vector<double> calculateZscore(const std::vector<double>& data_vector, double mean_value, double std_dev);
double calculateMean(const std::vector<double>& data_vector);
double calculateSDev(const std::vector<double>& data_vector, double mean_value);

//Variance
double calculateVariance(const std::vector<double>& data_vector, double mean_value);

//Mean Correlation
double calc_mean_corr(const std::vector<double>& target_channel, const std::vector<std::vector<double>>& all_channel);
std::vector<std::vector<double>> readAllChannels(const std::string& filename);

//Hurst_Exponent
double estimateHurstExponent(const std::vector<double>& signal);

//Kurtosis
double comp_Kurtosis(const std::vector<double>& spatial_map);

//Slope_Band
double computeSlopeInBand(const std::vector<double>& signal, double sampling_rate, const std::pair<double, double>& freq_band);

//Mean_Gradient
double computeMedGrad(const std::vector<double>& signal);

namespace cortex::data
{
    enum class FrequencyBand
    {
        Delta,
        Theta,
        Alpha,
        Beta,
        Gamma
    };

    namespace FrequencyRange
    {
        static constexpr double DELTA_MIN = 0.5;
        static constexpr double DELTA_MAX = 4.0;
        static constexpr double THETA_MIN = 4.0;
        static constexpr double THETA_MAX = 8.0;
        static constexpr double ALPHA_MIN = 8.0;
        static constexpr double ALPHA_MAX = 13.0;
        static constexpr double BETA_MIN = 13.0;
        static constexpr double BETA_MAX = 30.0;
        static constexpr double GAMMA_MIN = 30.0;
        static constexpr double GAMMA_MAX = 100.0;
    }

    class EEGData final
    {
    public:
        using ChannelData = tsl::robin_map<std::string, kfr::univector<double>>;

        double m_samplingRate = 128.0;

        const ChannelData& get_channels() const;
        const kfr::univector<double>& get_channel(const std::string_view channelName) const;
        void set_channel(const std::string_view channelName, kfr::univector<double> data);
        std::vector<std::string> get_channel_names() const;
        size_t get_sample_count() const;

    private:
        ChannelData channels;
    };

    class EEGDataSource
    {
    public:
        EEGDataSource() = default;
        virtual ~EEGDataSource() = default;
        EEGDataSource(const EEGDataSource&) = delete;
        EEGDataSource& operator=(const EEGDataSource&) = delete;
        EEGDataSource(EEGDataSource&&) = default;
        EEGDataSource& operator=(EEGDataSource&&) = default;

        virtual bool is_data_available() const = 0;
        virtual std::unique_ptr<EEGData> load_data() = 0;
        virtual bool open() = 0;
        virtual void close() = 0;
        virtual bool is_open() const = 0;
        virtual std::string get_source_name() const = 0;
    };
}


namespace cortex::data::filtering
{
    EEGData set_initial_reference(const EEGData& data, const std::vector<std::string>& referenceChannels);

    EEGData apply_high_pass_filter(const EEGData& data, double cutoffFreq, int filterOrder);
    EEGData apply_lowpass_filter(const EEGData& data, double cutoffFreq, int filterOrder);
    EEGData apply_notch_filter(const EEGData& data, double notchFreq, double bandwidth, int filterOrder);

    EEGData apply_common_average_reference(const EEGData& data);

    EEGData apply_min_max_normalization(const EEGData& data, bool normalizeToRange01);

    EEGData apply_baselining(const EEGData& data, size_t baselineStartSample, size_t baselineEndSample);
    EEGData apply_baselining(const EEGData& data, double baselineStartTime, double baselineEndTime);
}

std::unique_ptr<cortex::data::EEGData> load_csv_to_EEGData(const std::string& filename);

#endif //FUNC_EEG_WEAR_H
