#include "Func_EEG_Wear.h"


namespace cortex::data
{
    const EEGData::ChannelData& EEGData::get_channels() const
    {
        return channels;
    }

    const kfr::univector<double>& EEGData::get_channel(const std::string_view channelName) const
    {
        const auto it = channels.find(channelName.data());
        if (it == channels.end())
        {
            throw std::out_of_range(fmt::format("Channel not found: {}", channelName));
        }
        return it->second;
    }

    void EEGData::set_channel(const std::string_view channelName, kfr::univector<double> data)
    {
        channels[channelName.data()] = std::move(data);
    }

    std::vector<std::string> EEGData::get_channel_names() const
    {
        std::vector<std::string> names;
        names.reserve(channels.size());
        for (const auto& pair : channels)
        {
            names.push_back(pair.first);
        }
        return names;
    }

    size_t EEGData::get_sample_count() const
    {
        if (channels.empty())
            return 0;
        return channels.begin()->second.size();
    }
}

// Created by Jay on 7/24/2025.
//
