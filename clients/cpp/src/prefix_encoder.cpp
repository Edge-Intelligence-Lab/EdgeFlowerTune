#include "lshaped/prefix_encoder.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "lshaped/client_config.h"

#ifdef LSHAPED_ENABLE_MFT
#include "finetune_ops/core/tokenizer_hf.h"
#include "finetune_ops/core/tensor.h"
#include "finetune_ops/graph/gemma_model.h"
#include "finetune_ops/graph/qwen_model.h"
#include "finetune_ops/graph/safetensors_loader.h"
#endif

namespace lshaped {

namespace {

enum class PrefixModelFamily {
    Gemma,
    Qwen,
};

PrefixModelFamily DetectPrefixModelFamily(const std::string& model_dir) {
    std::string config_path = model_dir;
    if (!config_path.empty() && config_path.back() != '/') {
        config_path += "/";
    }
    config_path += "config.json";
    std::ifstream input(config_path);
    if (!input) {
        return PrefixModelFamily::Gemma;
    }
    std::stringstream buffer;
    buffer << input.rdbuf();
    std::string config = buffer.str();
    std::transform(config.begin(), config.end(), config.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (config.find("qwen2forcausallm") != std::string::npos ||
        config.find("\"model_type\":\"qwen2\"") != std::string::npos ||
        config.find("\"model_type\": \"qwen2\"") != std::string::npos ||
        config.find("\"architectures\":[\"qwen2") != std::string::npos ||
        config.find("\"architectures\": [\"qwen2") != std::string::npos ||
        config.find("qwen2.5") != std::string::npos ||
        config.find("\"qwen") != std::string::npos) {
        return PrefixModelFamily::Qwen;
    }
    return PrefixModelFamily::Gemma;
}

class SimpleCharTokenizer {
public:
    SimpleCharTokenizer() {
        pad_token_id_ = 0;
        eos_token_id_ = 1;
        bos_token_id_ = 2;
        unk_token_id_ = 3;
        const std::string symbols =
            "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\v\f";
        int next_id = 4;
        for (char ch : symbols) {
            if (char_to_id_.find(ch) == char_to_id_.end()) {
                char_to_id_[ch] = next_id++;
            }
        }
    }

    std::vector<std::int32_t> Encode(const std::string& text, bool add_bos) const {
        std::vector<std::int32_t> ids;
        if (add_bos) {
            ids.push_back(bos_token_id_);
        }
        ids.reserve(ids.size() + text.size());
        for (char ch : text) {
            const auto it = char_to_id_.find(ch);
            ids.push_back(it == char_to_id_.end() ? unk_token_id_ : it->second);
        }
        return ids;
    }

    std::int32_t TokenToId(const std::string& token) const {
        if (token.empty()) {
            return unk_token_id_;
        }
        const auto it = char_to_id_.find(token[0]);
        return it == char_to_id_.end() ? unk_token_id_ : it->second;
    }

    int vocab_size() const { return static_cast<int>(char_to_id_.size()) + 4; }
    std::int32_t pad_token_id() const { return pad_token_id_; }

private:
    std::unordered_map<char, std::int32_t> char_to_id_;
    std::int32_t pad_token_id_ = 0;
    std::int32_t eos_token_id_ = 1;
    std::int32_t bos_token_id_ = 2;
    std::int32_t unk_token_id_ = 3;
};

std::vector<float> BuildMockEmbeddingTable(int vocab_size, int hidden_size, int seed) {
    assert(vocab_size > 0);
    assert(hidden_size > 0);
    std::mt19937 generator(static_cast<std::uint32_t>(seed));
    std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
    std::vector<float> table(static_cast<std::size_t>(vocab_size * hidden_size));
    for (float& value : table) {
        value = dist(generator);
    }
    return table;
}

struct PackedTokenSequence {
    std::vector<std::int32_t> input_ids;
};

std::string ResolveWikiTextTrainPath(const ClientOptions& options) {
    if (!options.dataset_train_path.empty()) {
        return options.dataset_train_path;
    }
    if (!options.dataset_csv.empty()) {
        return options.dataset_csv;
    }
    throw std::runtime_error("wikitext_raw split client requires --dataset_train_path or --dataset_csv");
}

std::vector<PackedTokenSequence> LoadPackedWikiTextSequences(
    const ClientOptions& options,
    const std::function<std::vector<std::int32_t>(const std::string&)>& encode_fn) {
    const std::string train_path = ResolveWikiTextTrainPath(options);
    std::ifstream input(train_path);
    if (!input) {
        throw std::runtime_error("Failed to open WikiText train file: " + train_path);
    }

    std::vector<std::int32_t> packed_ids;
    const std::vector<std::int32_t> separator = encode_fn("\n");
    std::string line;
    std::size_t global_index = 0;
    while (std::getline(input, line)) {
        const bool is_assigned =
            static_cast<int>(global_index % static_cast<std::size_t>(options.num_clients)) == options.client_index;
        ++global_index;
        if (!is_assigned) {
            continue;
        }
        const std::vector<std::int32_t> ids = encode_fn(line);
        packed_ids.insert(packed_ids.end(), ids.begin(), ids.end());
        packed_ids.insert(packed_ids.end(), separator.begin(), separator.end());
    }

    if (packed_ids.size() < 2) {
        throw std::runtime_error("WikiText shard did not produce enough tokens for split training");
    }

    std::vector<PackedTokenSequence> sequences;
    const std::size_t seq_len = static_cast<std::size_t>(options.max_seq_len);
    for (std::size_t start = 0; start < packed_ids.size(); start += seq_len) {
        const std::size_t remaining = packed_ids.size() - start;
        const std::size_t length = std::min(seq_len, remaining);
        if (length < 2) {
            break;
        }
        PackedTokenSequence sample;
        sample.input_ids.assign(
            packed_ids.begin() + static_cast<std::ptrdiff_t>(start),
            packed_ids.begin() + static_cast<std::ptrdiff_t>(start + length));
        sequences.push_back(std::move(sample));
    }

    if (sequences.empty()) {
        throw std::runtime_error("WikiText shard did not yield any packed sequences");
    }
    return sequences;
}

class PackedWikiTextCursor {
public:
    PackedWikiTextCursor(
        const ClientOptions& options,
        const std::function<std::vector<std::int32_t>(const std::string&)>& encode_fn)
        : sequences_(LoadPackedWikiTextSequences(options, encode_fn)),
          rng_(static_cast<std::uint32_t>(options.seed)) {
        order_.resize(sequences_.size());
        for (std::size_t i = 0; i < order_.size(); ++i) {
            order_[i] = i;
        }
        std::shuffle(order_.begin(), order_.end(), rng_);
    }

    std::vector<PackedTokenSequence> NextBatch(int batch_size) {
        assert(batch_size > 0);
        std::vector<PackedTokenSequence> batch;
        batch.reserve(static_cast<std::size_t>(batch_size));
        for (int i = 0; i < batch_size; ++i) {
            if (cursor_ >= order_.size()) {
                cursor_ = 0;
                std::shuffle(order_.begin(), order_.end(), rng_);
            }
            batch.push_back(sequences_[order_[cursor_++]]);
        }
        return batch;
    }

private:
    std::vector<PackedTokenSequence> sequences_;
    std::vector<std::size_t> order_;
    std::size_t cursor_ = 0;
    std::mt19937 rng_;
};

class MockPrefixEncoder final : public PrefixEncoder {
public:
    explicit MockPrefixEncoder(const ClientOptions& options)
        : hidden_size_(options.mock_hidden_size),
          scale_(std::sqrt(static_cast<float>(options.mock_hidden_size))),
          embedding_table_(BuildMockEmbeddingTable(tokenizer_.vocab_size(), options.mock_hidden_size, options.seed)) {
        if (hidden_size_ <= 0) {
            throw std::runtime_error("mock_hidden_size must be > 0");
        }
        if (options.dataset_format == "wikitext_raw") {
            wiki_cursor_ = std::make_unique<PackedWikiTextCursor>(
                options,
                [this](const std::string& text) { return tokenizer_.Encode(text, false); });
        }
    }

    EncodedBatch Encode(const std::vector<MMLUSample>& samples, const ClientOptions& options) override {
        assert(!samples.empty());
        EncodedBatch batch;
        batch.batch_size = static_cast<std::int64_t>(samples.size());
        batch.hidden_size = hidden_size_;
        batch.task_type = "multiple_choice";
        batch.answer_labels.reserve(samples.size());

        std::vector<std::vector<std::int32_t>> encoded;
        encoded.reserve(samples.size());
        std::vector<std::int32_t> answer_ids;
        answer_ids.reserve(samples.size());
        int max_len = 0;
        for (const MMLUSample& sample : samples) {
            std::vector<std::int32_t> ids = tokenizer_.Encode(sample.Prompt(), true);
            if (static_cast<int>(ids.size()) > options.max_seq_len) {
                ids.resize(static_cast<std::size_t>(options.max_seq_len));
            }
            if (ids.empty()) {
                throw std::runtime_error("Encoded prompt must not be empty");
            }
            max_len = std::max(max_len, static_cast<int>(ids.size()));
            encoded.push_back(std::move(ids));
            answer_ids.push_back(tokenizer_.TokenToId(sample.AnswerLabel()));
            batch.answer_labels.push_back(sample.AnswerLabel());
        }

        batch.seq_len = options.max_seq_len;
        batch.activation.resize(static_cast<std::size_t>(batch.batch_size * batch.seq_len * batch.hidden_size), 0.0f);
        batch.target_embedding.resize(static_cast<std::size_t>(batch.batch_size * batch.hidden_size), 0.0f);
        batch.attention_mask.resize(static_cast<std::size_t>(batch.batch_size * batch.seq_len), 0);
        batch.target_token_ids = answer_ids;
        batch.valid_lengths.resize(static_cast<std::size_t>(batch.batch_size), 0);

        for (std::size_t row = 0; row < encoded.size(); ++row) {
            const std::vector<std::int32_t>& ids = encoded[row];
            batch.valid_lengths[row] = static_cast<std::int32_t>(ids.size());
            for (std::size_t col = 0; col < ids.size(); ++col) {
                batch.attention_mask[row * static_cast<std::size_t>(batch.seq_len) + col] = 1;
                const std::int32_t token_id = ids[col];
                if (token_id < 0 || token_id >= tokenizer_.vocab_size()) {
                    throw std::runtime_error("Mock token id out of range");
                }
                const float* src = &embedding_table_[static_cast<std::size_t>(token_id) * hidden_size_];
                float* dst = &batch.activation[(row * static_cast<std::size_t>(batch.seq_len) + col) * hidden_size_];
                for (int h = 0; h < hidden_size_; ++h) {
                    dst[h] = src[h] * scale_;
                }
            }

            const std::int32_t answer_id = answer_ids[row];
            const float* answer_src = &embedding_table_[static_cast<std::size_t>(answer_id) * hidden_size_];
            float* answer_dst = &batch.target_embedding[row * hidden_size_];
            for (int h = 0; h < hidden_size_; ++h) {
                answer_dst[h] = answer_src[h] * scale_;
            }
        }

        return batch;
    }

    EncodedBatch EncodeNextBatch(const ClientOptions& options) override {
        if (!wiki_cursor_) {
            throw std::runtime_error("MockPrefixEncoder was not initialized for wikitext_raw");
        }
        const std::vector<PackedTokenSequence> samples = wiki_cursor_->NextBatch(options.batch_size);
        EncodedBatch batch;
        batch.batch_size = static_cast<std::int64_t>(samples.size());
        batch.seq_len = options.max_seq_len;
        batch.hidden_size = hidden_size_;
        batch.task_type = "next_token_lm";

        const std::size_t batch_size = static_cast<std::size_t>(batch.batch_size);
        const std::size_t seq_len = static_cast<std::size_t>(batch.seq_len);
        batch.activation.resize(batch_size * seq_len * static_cast<std::size_t>(hidden_size_), 0.0f);
        batch.target_embedding.resize(batch_size * static_cast<std::size_t>(hidden_size_), 0.0f);
        batch.attention_mask.resize(batch_size * seq_len, 0);
        batch.target_token_ids.resize(batch_size * seq_len, tokenizer_.pad_token_id());
        batch.valid_lengths.resize(batch_size, 0);

        for (std::size_t row = 0; row < samples.size(); ++row) {
            const auto& ids = samples[row].input_ids;
            batch.valid_lengths[row] = static_cast<std::int32_t>(ids.size());
            for (std::size_t col = 0; col < ids.size(); ++col) {
                batch.attention_mask[row * seq_len + col] = 1;
                batch.target_token_ids[row * seq_len + col] = ids[col];
                const std::int32_t token_id = ids[col];
                if (token_id < 0 || token_id >= tokenizer_.vocab_size()) {
                    throw std::runtime_error("Mock token id out of range");
                }
                const float* src = &embedding_table_[static_cast<std::size_t>(token_id) * hidden_size_];
                float* dst = &batch.activation[(row * seq_len + col) * static_cast<std::size_t>(hidden_size_)];
                for (int h = 0; h < hidden_size_; ++h) {
                    dst[h] = src[h] * scale_;
                }
            }
        }

        return batch;
    }

private:
    SimpleCharTokenizer tokenizer_;
    int hidden_size_;
    float scale_;
    std::vector<float> embedding_table_;
    std::unique_ptr<PackedWikiTextCursor> wiki_cursor_;
};

#ifdef LSHAPED_ENABLE_MFT
class MobileFineTunerPrefixEncoder final : public PrefixEncoder {
public:
    explicit MobileFineTunerPrefixEncoder(const ClientOptions& options)
        : tokenizer_(options.model_dir, true) {
        if (options.model_dir.empty()) {
            throw std::runtime_error("model_dir must be provided for backend=mft");
        }
        if (!tokenizer_.load()) {
            throw std::runtime_error("Failed to load HF tokenizer from model_dir: " + options.model_dir);
        }
        family_ = DetectPrefixModelFamily(options.model_dir);
        if (family_ == PrefixModelFamily::Qwen) {
            auto cfg = ops::QwenConfig::from_pretrained(options.model_dir + "/config.json");
            vocab_size_ = cfg.vocab_size;
            hidden_size_ = cfg.hidden_size;
            input_embedding_scale_ = 1.0f;
            target_embedding_scale_ = 1.0f;
        } else {
            auto cfg = ops::GemmaTextConfig::from_pretrained(options.model_dir);
            vocab_size_ = cfg.vocab_size;
            hidden_size_ = cfg.hidden_size;
            input_embedding_scale_ = 1.0f;
            target_embedding_scale_ = 1.0f;
        }
        embedding_weight_ = LoadEmbeddingWeight(options.model_dir);
        if (options.dataset_format == "wikitext_raw") {
            wiki_cursor_ = std::make_unique<PackedWikiTextCursor>(
                options,
                [this](const std::string& text) { return tokenizer_.encode(text); });
        }
    }

    EncodedBatch Encode(const std::vector<MMLUSample>& samples, const ClientOptions& options) override {
        assert(!samples.empty());
        EncodedBatch batch;
        batch.batch_size = static_cast<std::int64_t>(samples.size());
        batch.hidden_size = hidden_size_;
        batch.task_type = "multiple_choice";
        batch.answer_labels.reserve(samples.size());

        std::vector<std::vector<std::int32_t>> encoded;
        encoded.reserve(samples.size());
        std::vector<std::int32_t> answer_ids;
        answer_ids.reserve(samples.size());

        int max_len = 0;
        for (const MMLUSample& sample : samples) {
            std::vector<std::int32_t> ids = tokenizer_.encode(sample.Prompt());
            if (static_cast<int>(ids.size()) > options.max_seq_len) {
                ids.resize(static_cast<std::size_t>(options.max_seq_len));
            }
            if (ids.empty()) {
                throw std::runtime_error("Encoded prompt must not be empty");
            }
            max_len = std::max(max_len, static_cast<int>(ids.size()));
            encoded.push_back(std::move(ids));

            const std::string answer_with_space = options.answer_prefix + sample.AnswerLabel();
            std::int32_t answer_id = tokenizer_.token_to_id(answer_with_space);
            if (answer_id < 0) {
                answer_id = tokenizer_.token_to_id(sample.AnswerLabel());
            }
            if (answer_id < 0) {
                throw std::runtime_error("Failed to map answer token for label: " + sample.AnswerLabel());
            }
            answer_ids.push_back(answer_id);
            batch.answer_labels.push_back(sample.AnswerLabel());
        }

        assert(max_len > 0);
        batch.seq_len = options.max_seq_len;
        batch.attention_mask.resize(static_cast<std::size_t>(batch.batch_size * batch.seq_len), 0);
        batch.valid_lengths.resize(static_cast<std::size_t>(batch.batch_size), 0);
        batch.target_token_ids = answer_ids;

        std::vector<std::int32_t> padded_ids(static_cast<std::size_t>(batch.batch_size * batch.seq_len), 0);
        for (std::size_t row = 0; row < encoded.size(); ++row) {
            const std::vector<std::int32_t>& ids = encoded[row];
            batch.valid_lengths[row] = static_cast<std::int32_t>(ids.size());
            for (std::size_t col = 0; col < ids.size(); ++col) {
                padded_ids[row * static_cast<std::size_t>(batch.seq_len) + col] = ids[col];
                batch.attention_mask[row * static_cast<std::size_t>(batch.seq_len) + col] = 1;
            }
        }

        auto input_ids = std::make_shared<ops::Tensor>(
            std::vector<std::int64_t>{batch.batch_size, batch.seq_len},
            padded_ids.data(),
            ops::kInt32,
            ops::kCPU);
        auto token_ids = std::make_shared<ops::Tensor>(
            std::vector<std::int64_t>{batch.batch_size},
            answer_ids.data(),
            ops::kInt32,
            ops::kCPU);

        if (options.split_layer != 0) {
            throw std::runtime_error("MobileFineTunerPrefixEncoder currently supports split_layer=0 only");
        }
        ops::TensorPtr activation = EmbedInputIds(input_ids);
        ops::TensorPtr target_embedding = EmbedTokenIds(token_ids);

        const std::size_t activation_count =
            static_cast<std::size_t>(batch.batch_size * batch.seq_len * batch.hidden_size);
        const std::size_t target_count =
            static_cast<std::size_t>(batch.batch_size * batch.hidden_size);
        batch.activation.assign(
            activation->data<float>(),
            activation->data<float>() + activation_count);
        batch.target_embedding.assign(
            target_embedding->data<float>(),
            target_embedding->data<float>() + target_count);
        return batch;
    }

    EncodedBatch EncodeNextBatch(const ClientOptions& options) override {
        if (!wiki_cursor_) {
            throw std::runtime_error("MobileFineTunerPrefixEncoder was not initialized for wikitext_raw");
        }
        const std::vector<PackedTokenSequence> samples = wiki_cursor_->NextBatch(options.batch_size);
        EncodedBatch batch;
        batch.batch_size = static_cast<std::int64_t>(samples.size());
        batch.seq_len = options.max_seq_len;
        batch.hidden_size = hidden_size_;
        batch.task_type = "next_token_lm";

        const std::size_t batch_size = static_cast<std::size_t>(batch.batch_size);
        const std::size_t seq_len = static_cast<std::size_t>(batch.seq_len);
        batch.attention_mask.resize(batch_size * seq_len, 0);
        batch.valid_lengths.resize(batch_size, 0);
        batch.target_token_ids.resize(batch_size * seq_len, 0);
        batch.target_embedding.resize(batch_size * static_cast<std::size_t>(hidden_size_), 0.0f);

        std::vector<std::int32_t> padded_ids(batch_size * seq_len, 0);
        for (std::size_t row = 0; row < samples.size(); ++row) {
            const auto& ids = samples[row].input_ids;
            batch.valid_lengths[row] = static_cast<std::int32_t>(ids.size());
            for (std::size_t col = 0; col < ids.size(); ++col) {
                padded_ids[row * seq_len + col] = ids[col];
                batch.target_token_ids[row * seq_len + col] = ids[col];
                batch.attention_mask[row * seq_len + col] = 1;
            }
        }

        auto input_ids = std::make_shared<ops::Tensor>(
            std::vector<std::int64_t>{batch.batch_size, batch.seq_len},
            padded_ids.data(),
            ops::kInt32,
            ops::kCPU);
        if (options.split_layer != 0) {
            throw std::runtime_error("MobileFineTunerPrefixEncoder currently supports split_layer=0 only");
        }
        ops::TensorPtr activation = EmbedInputIds(input_ids);

        const std::size_t activation_count =
            static_cast<std::size_t>(batch.batch_size * batch.seq_len * batch.hidden_size);
        batch.activation.assign(
            activation->data<float>(),
            activation->data<float>() + activation_count);
        for (std::size_t row = 0; row < batch_size; ++row) {
            for (std::size_t col = static_cast<std::size_t>(batch.valid_lengths[row]); col < seq_len; ++col) {
                float* dst = &batch.activation[(row * seq_len + col) * static_cast<std::size_t>(batch.hidden_size)];
                std::fill(dst, dst + batch.hidden_size, 0.0f);
            }
        }
        return batch;
    }

private:
    ops::TensorPtr LoadEmbeddingWeight(const std::string& model_dir) const {
        ops::SafeTensorsReader reader(model_dir + "/model.safetensors");
        reader.parse_header();

        const std::vector<std::string> candidates = {
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        };
        std::string last_error;
        for (const std::string& name : candidates) {
            try {
                ops::TensorPtr tensor = reader.load_tensor(name, false);
                const auto& shape = tensor->shape();
                if (shape.size() != 2) {
                    throw std::runtime_error("embedding tensor must be rank-2");
                }
                if (shape[0] != vocab_size_ || shape[1] != hidden_size_) {
                    std::ostringstream oss;
                    oss << "embedding tensor shape mismatch: got ["
                        << shape[0] << "," << shape[1] << "], expected ["
                        << vocab_size_ << "," << hidden_size_ << "]";
                    throw std::runtime_error(oss.str());
                }
                if (tensor->dtype() != ops::kFloat32) {
                    throw std::runtime_error("embedding tensor must be loaded as float32");
                }
                return tensor;
            } catch (const std::exception& ex) {
                last_error = ex.what();
            }
        }
        throw std::runtime_error("Failed to load embedding weight from model.safetensors: " + last_error);
    }

    void CopyEmbeddingRow(std::int32_t token_id, float scale, float* dst) const {
        if (token_id < 0 || token_id >= vocab_size_) {
            throw std::runtime_error("Token id out of range in split embedding lookup");
        }
        const float* src = embedding_weight_->data<float>() +
            static_cast<std::int64_t>(token_id) * hidden_size_;
        for (std::int64_t i = 0; i < hidden_size_; ++i) {
            dst[i] = src[i] * scale;
        }
    }

    ops::TensorPtr EmbedInputIds(const ops::TensorPtr& input_ids) const {
        const auto& shape = input_ids->shape();
        if (shape.size() != 2) {
            throw std::runtime_error("EmbedInputIds expects [batch, seq_len]");
        }
        if (input_ids->dtype() != ops::kInt32) {
            throw std::runtime_error("EmbedInputIds expects int32 token ids");
        }
        const std::int64_t batch = shape[0];
        const std::int64_t seq_len = shape[1];
        auto result = std::make_shared<ops::Tensor>(
            std::vector<std::int64_t>{batch, seq_len, hidden_size_},
            ops::kFloat32,
            ops::kCPU);
        const std::int32_t* ids = input_ids->data<std::int32_t>();
        float* out = result->data<float>();
        for (std::int64_t b = 0; b < batch; ++b) {
            for (std::int64_t s = 0; s < seq_len; ++s) {
                const std::int32_t token = ids[b * seq_len + s];
                float* dst = out + (b * seq_len + s) * hidden_size_;
                CopyEmbeddingRow(token, input_embedding_scale_, dst);
            }
        }
        return result;
    }

    ops::TensorPtr EmbedTokenIds(const ops::TensorPtr& token_ids) const {
        const auto& shape = token_ids->shape();
        if (shape.empty() || shape.size() > 2) {
            throw std::runtime_error("EmbedTokenIds expects [batch] or [batch, 1]");
        }
        if (token_ids->dtype() != ops::kInt32) {
            throw std::runtime_error("EmbedTokenIds expects int32 token ids");
        }
        const std::int64_t batch = shape[0];
        const std::int64_t cols = shape.size() == 1 ? 1 : shape[1];
        if (cols != 1) {
            throw std::runtime_error("EmbedTokenIds expects [batch] or [batch, 1]");
        }
        auto result = std::make_shared<ops::Tensor>(
            std::vector<std::int64_t>{batch, hidden_size_},
            ops::kFloat32,
            ops::kCPU);
        const std::int32_t* ids = token_ids->data<std::int32_t>();
        float* out = result->data<float>();
        for (std::int64_t b = 0; b < batch; ++b) {
            CopyEmbeddingRow(ids[b], target_embedding_scale_, out + b * hidden_size_);
        }
        return result;
    }

    ops::HFTokenizer tokenizer_;
    PrefixModelFamily family_ = PrefixModelFamily::Gemma;
    ops::TensorPtr embedding_weight_;
    std::int64_t vocab_size_ = 0;
    std::int64_t hidden_size_ = 0;
    float input_embedding_scale_ = 1.0f;
    float target_embedding_scale_ = 1.0f;
    std::unique_ptr<PackedWikiTextCursor> wiki_cursor_;
};
#endif

}  // namespace

std::size_t EncodedBatch::transmitted_bytes() const {
    return activation.size() * sizeof(float) +
           target_embedding.size() * sizeof(float) +
           attention_mask.size() * sizeof(std::int32_t) +
           target_token_ids.size() * sizeof(std::int32_t) +
           valid_lengths.size() * sizeof(std::int32_t);
}

std::unique_ptr<PrefixEncoder> CreatePrefixEncoder(const ClientOptions& options) {
    if (options.backend == "mock") {
        return std::make_unique<MockPrefixEncoder>(options);
    }
#ifdef LSHAPED_ENABLE_MFT
    if (options.backend == "mft") {
        return std::make_unique<MobileFineTunerPrefixEncoder>(options);
    }
#else
    if (options.backend == "mft") {
        throw std::runtime_error("This binary was built without LSHAPED_ENABLE_MFT=ON");
    }
#endif
    throw std::runtime_error("Unsupported backend: " + options.backend);
}

}  // namespace lshaped
