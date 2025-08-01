#include "Func_EEG_Wear.h"
#include <stdexcept>
#include <ranges>

namespace cortex::data::filtering {

EEGData set_initial_reference(const EEGData& data, const std::vector<std::string>& referenceChannels) {
    const size_t sampleCount = data.get_sample_count();
    if (sampleCount == 0) {
        return data;
    }

    for (const auto& refChannel : referenceChannels) {
        try {
            std::ignore = data.get_channel(refChannel);
        } catch (const std::out_of_range&) {
            throw std::invalid_argument(fmt::format("Reference channel not found: {}", refChannel));
        }
    }

    kfr::univector<double> referenceSignal(sampleCount, 0.0);
    for (size_t t = 0; t < sampleCount; ++t) {
        double sum = 0.0;
        for (const auto& refChannel : referenceChannels) {
            sum += data.get_channel(refChannel)[t];
        }
        referenceSignal[t] = sum / referenceChannels.size();
    }

    EEGData referencedData = data;
    for (const auto& channel : data.get_channel_names()) {
        kfr::univector<double> channelData = data.get_channel(channel);
        channelData = channelData - referenceSignal;
        referencedData.set_channel(channel, std::move(channelData));
    }

    return referencedData;
}

EEGData apply_high_pass_filter(const EEGData& data, const double cutoffFreq, int filterOrder) {
    EEGData filteredData = data;

    if (filterOrder % 2 == 0) {
        filterOrder += 1;
    }

    const double samplingRate = data.m_samplingRate;
    const double normalizedCutoff = cutoffFreq / (samplingRate / 2.0);

    kfr::univector<double> taps(filterOrder);
    const auto kaiser = kfr::to_handle(kfr::window_kaiser<double>(taps.size(), 3.0));
    kfr::fir_highpass(taps, normalizedCutoff, kaiser, true);

    kfr::filter_fir<double> filter(taps);

    for (const auto& channel : data.get_channel_names()) {
        kfr::univector<double> channelData = data.get_channel(channel);
        filter.apply(channelData);
        filteredData.set_channel(channel, std::move(channelData));
    }

    return filteredData;
}

EEGData apply_lowpass_filter(const EEGData& data, const double cutoffFreq, const int filterOrder) {
    EEGData filteredData = data;

    const double samplingRate = data.m_samplingRate;
    const double normalizedCutoff = cutoffFreq / (samplingRate / 2.0);

    kfr::univector<double> taps(filterOrder);
    const auto kaiser = kfr::to_handle(kfr::window_kaiser<double>(taps.size(), 3.0));
    kfr::fir_lowpass(taps, normalizedCutoff, kaiser, true);

    kfr::filter_fir<double> filter(taps);

    for (const auto& channel : data.get_channel_names()) {
        kfr::univector<double> channelData = data.get_channel(channel);
        filter.apply(channelData);
        filteredData.set_channel(channel, std::move(channelData));
    }

    return filteredData;
}

EEGData apply_notch_filter(const EEGData& data, const double notchFreq, const double bandwidth, int filterOrder) {
    EEGData filteredData = data;

    if (filterOrder % 2 == 0) {
        filterOrder += 1;
    }

    const double samplingRate = data.m_samplingRate;

    const double normalizedCenter = notchFreq / (samplingRate / 2.0);
    const double normalizedWidth = bandwidth / (samplingRate / 2.0);
    const double f1 = normalizedCenter - normalizedWidth / 2.0;
    const double f2 = normalizedCenter + normalizedWidth / 2.0;

    kfr::univector<double> taps(filterOrder);
    const auto kaiser = kfr::to_handle(kfr::window_kaiser<double>(taps.size(), 4.0));
    kfr::fir_bandstop(taps, f1, f2, kaiser, true);

    kfr::filter_fir<double> filter(taps);

    for (const auto& channel : data.get_channel_names()) {
        kfr::univector<double> channelData = data.get_channel(channel);
        filter.apply(channelData);
        filteredData.set_channel(channel, std::move(channelData));
    }

    return filteredData;
}

EEGData apply_common_average_reference(const EEGData& data) {
    const size_t sampleCount = data.get_sample_count();
    if (sampleCount == 0) return data;

    const auto channelNames = data.get_channel_names();
    if (channelNames.empty()) return data;

    kfr::univector<double> averageSignal(sampleCount, 0.0);
    for (size_t t = 0; t < sampleCount; ++t) {
        double sum = 0.0;
        for (const auto& channel : channelNames) {
            sum += data.get_channel(channel)[t];
        }
        averageSignal[t] = sum / channelNames.size();
    }

    EEGData referencedData = data;
    for (const auto& channel : channelNames) {
        kfr::univector<double> channelData = data.get_channel(channel);
        channelData = channelData - averageSignal;
        referencedData.set_channel(channel, std::move(channelData));
    }

    return referencedData;
}

EEGData apply_min_max_normalization(const EEGData& data, bool normalizeToRange01) {
    const size_t sampleCount = data.get_sample_count();
    if (sampleCount == 0) return data;

    EEGData normalizedData = data;

    for (const auto& channel : data.get_channel_names()) {
        const auto& channelData = data.get_channel(channel);
        auto [minIt, maxIt] = std::ranges::minmax_element(channelData);
        double minVal = *minIt;
        double maxVal = *maxIt;
        double range = maxVal - minVal;

        kfr::univector<double> normalizedChannel(channelData.size());

        if (std::abs(range) < 1e-10) {
            normalizedChannel = kfr::univector<double>(channelData.size(), 0.0);
        } else {
            normalizedChannel = (channelData - minVal) / range;
            if (!normalizeToRange01) {
                normalizedChannel = normalizedChannel * 2.0 - 1.0;
            }
        }

        normalizedData.set_channel(channel, std::move(normalizedChannel));
    }

    return normalizedData;
}

EEGData apply_baselining(const EEGData& data, size_t baselineStartSample, size_t baselineEndSample) {
    const size_t sampleCount = data.get_sample_count();
    if (sampleCount == 0) return data;

    if (baselineStartSample >= sampleCount || baselineEndSample >= sampleCount ||
        baselineStartSample > baselineEndSample) {
        throw std::invalid_argument(fmt::format(
            "Invalid baseline range: [{}, {}] for data with {} samples",
            baselineStartSample, baselineEndSample, sampleCount));
    }

    EEGData baselinedData = data;

    for (const auto& channel : data.get_channel_names()) {
        const auto& channelData = data.get_channel(channel);

        double baselineSum = 0.0;
        size_t baselineCount = baselineEndSample - baselineStartSample + 1;

        for (size_t t = baselineStartSample; t <= baselineEndSample; ++t) {
            baselineSum += channelData[t];
        }

        double baselineMean = baselineSum / baselineCount;
        kfr::univector<double> baselinedChannel = channelData - baselineMean;

        baselinedData.set_channel(channel, std::move(baselinedChannel));
    }

    return baselinedData;
}

EEGData apply_baselining(const EEGData& data, double baselineStartTime, double baselineEndTime) {
    const size_t start = static_cast<size_t>(baselineStartTime * data.m_samplingRate);
    const size_t end = static_cast<size_t>(baselineEndTime * data.m_samplingRate);
    return apply_baselining(data, start, end);
}

} // namespace cortex::data::filtering

// Created by Jay on 7/24/2025.
//
