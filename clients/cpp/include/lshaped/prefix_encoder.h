#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "lshaped/client_config.h"
#include "lshaped/mmlu_dataset.h"

namespace lshaped {

struct EncodedBatch {
    std::int64_t batch_size = 0;      // B
    std::int64_t seq_len = 0;         // S
    std::int64_t hidden_size = 0;     // H
    std::string task_type = "multiple_choice";
    std::vector<float> activation;    // [B, S, H], row-major, float32
    std::vector<float> target_embedding;  // [B, H], row-major, float32
    std::vector<std::int32_t> attention_mask;  // [B, S], row-major, int32
    std::vector<std::int32_t> target_token_ids;  // [B] for MMLU, [B,S] for WikiText
    std::vector<std::int32_t> valid_lengths;  // [B], int32
    std::vector<std::string> answer_labels;  // [B]

    std::size_t transmitted_bytes() const;
};

class PrefixEncoder {
public:
    virtual ~PrefixEncoder() = default;
    virtual EncodedBatch Encode(
        const std::vector<MMLUSample>& samples,
        const ClientOptions& options) = 0;
    virtual EncodedBatch EncodeNextBatch(const ClientOptions& /*options*/) {
        throw std::runtime_error("PrefixEncoder does not support internal dataset batching");
    }
};

std::unique_ptr<PrefixEncoder> CreatePrefixEncoder(const ClientOptions& options);

}  // namespace lshaped
