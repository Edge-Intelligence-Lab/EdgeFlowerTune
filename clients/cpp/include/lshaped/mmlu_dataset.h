#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "lshaped/client_config.h"

namespace lshaped {

struct MMLUSample {
    std::string question;
    std::string option_a;
    std::string option_b;
    std::string option_c;
    std::string option_d;
    char answer = 'A';

    std::string Prompt() const;
    std::string AnswerLabel() const;
};

class ClientShardDataset {
public:
    ClientShardDataset() = default;
    explicit ClientShardDataset(std::vector<MMLUSample> samples);

    static ClientShardDataset FromOptions(const ClientOptions& options);

    bool empty() const { return samples_.empty(); }
    std::size_t size() const { return samples_.size(); }
    const std::vector<MMLUSample>& samples() const { return samples_; }

    std::vector<MMLUSample> NextBatch(int batch_size);

private:
    std::vector<MMLUSample> samples_;
    std::size_t cursor_ = 0;
};

}  // namespace lshaped
