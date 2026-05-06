#include "lshaped/classic_client_backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#include <unistd.h>
#endif

#include "lshaped/npy_serializer.h"

#include "finetune_ops/core/lm_loss.h"
#include "finetune_ops/core/memory_manager.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/tokenizer_hf.h"
#include "finetune_ops/graph/gemma_lora_injector.h"
#include "finetune_ops/graph/gemma_model.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/optim/adam.h"

namespace lshaped {

namespace {

constexpr char kTensorType[] = "numpy.ndarray";

double CurrentRssMb() {
#if defined(__linux__) || defined(__ANDROID__)
    std::ifstream input("/proc/self/statm");
    long resident_pages = 0;
    input.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
    input >> resident_pages;
    if (!input.good() && !input.eof()) {
        return -1.0;
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0 || resident_pages < 0) {
        return -1.0;
    }
    return static_cast<double>(resident_pages) * static_cast<double>(page_size) / (1024.0 * 1024.0);
#else
    return -1.0;
#endif
}

bool HasSInt64(const flwr::proto::Scalar& scalar) {
    return scalar.scalar_case() == flwr::proto::Scalar::kSint64;
}

bool HasUInt64(const flwr::proto::Scalar& scalar) {
    return scalar.scalar_case() == flwr::proto::Scalar::kUint64;
}

bool HasDouble(const flwr::proto::Scalar& scalar) {
    return scalar.scalar_case() == flwr::proto::Scalar::kDouble;
}

bool HasBool(const flwr::proto::Scalar& scalar) {
    return scalar.scalar_case() == flwr::proto::Scalar::kBool;
}

bool HasString(const flwr::proto::Scalar& scalar) {
    return scalar.scalar_case() == flwr::proto::Scalar::kString;
}

std::int64_t ReadIntConfig(
    const google::protobuf::Map<std::string, flwr::proto::Scalar>& config,
    const std::string& key,
    std::int64_t fallback) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return fallback;
    }
    if (HasSInt64(it->second)) {
        return it->second.sint64();
    }
    if (HasUInt64(it->second)) {
        return static_cast<std::int64_t>(it->second.uint64());
    }
    throw std::runtime_error("Expected integer config for key: " + key);
}

double ReadDoubleConfig(
    const google::protobuf::Map<std::string, flwr::proto::Scalar>& config,
    const std::string& key,
    double fallback) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return fallback;
    }
    if (HasDouble(it->second)) {
        return it->second.double_();
    }
    if (HasSInt64(it->second)) {
        return static_cast<double>(it->second.sint64());
    }
    if (HasUInt64(it->second)) {
        return static_cast<double>(it->second.uint64());
    }
    throw std::runtime_error("Expected float config for key: " + key);
}

bool ReadBoolConfig(
    const google::protobuf::Map<std::string, flwr::proto::Scalar>& config,
    const std::string& key,
    bool fallback) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return fallback;
    }
    if (HasBool(it->second)) {
        return it->second.bool_();
    }
    if (HasSInt64(it->second)) {
        return it->second.sint64() != 0;
    }
    if (HasUInt64(it->second)) {
        return it->second.uint64() != 0;
    }
    throw std::runtime_error("Expected bool config for key: " + key);
}

std::string ReadStringConfig(
    const google::protobuf::Map<std::string, flwr::proto::Scalar>& config,
    const std::string& key,
    const std::string& fallback) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return fallback;
    }
    if (!HasString(it->second)) {
        throw std::runtime_error("Expected string config for key: " + key);
    }
    return it->second.string();
}

std::vector<std::string> SplitCommaSeparated(const std::string& value) {
    std::vector<std::string> tokens;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

std::vector<std::string> TargetModulesForMode(const std::string& target_mode) {
    if (target_mode == "attn") {
        return {"q_proj", "k_proj", "v_proj", "o_proj"};
    }
    if (target_mode == "qv") {
        return {"q_proj", "v_proj"};
    }
    if (target_mode == "full") {
        return {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"};
    }
    const std::vector<std::string> custom = SplitCommaSeparated(target_mode);
    if (!custom.empty()) {
        return custom;
    }
    throw std::runtime_error("Unsupported target_mode: " + target_mode);
}

std::vector<std::uint8_t> ReadFileBytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (size > 0) {
        input.read(reinterpret_cast<char*>(bytes.data()), size);
    }
    return bytes;
}

void WriteFileBytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Failed to open file for writing: " + path.string());
    }
    if (!bytes.empty()) {
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
}

int32_t ResolveTokenId(const ops::HFTokenizer& tokenizer, const std::vector<std::string>& candidates) {
    for (const std::string& token : candidates) {
        const int32_t id = tokenizer.token_to_id(token);
        if (id >= 0) {
            return id;
        }
    }
    return -1;
}

struct RoundOptions {
    std::string mode;
    std::int64_t server_round = -1;
    int batch_size = 1;
    int max_seq_len = 128;
    int local_steps = 1;
    int local_epochs = 1;
    int logging_steps = 1;
    float learning_rate = 2e-4f;
    float weight_decay = 0.0f;
    int lora_r = 8;
    float lora_alpha = 16.0f;
    float lora_dropout = 0.0f;
    std::string target_mode = "attn";
    bool shuffle = false;
};

struct TrainBatch {
    ops::TensorPtr input_ids;
    ops::TensorPtr attention_mask;
    ops::TensorPtr labels;
    int batch_size = 0;
    int seq_len = 0;
};

class GemmaClassicClientBackend final : public ClassicClientBackend {
public:
    GemmaClassicClientBackend(ClientOptions options, ClientShardDataset dataset)
        : options_(std::move(options)),
          dataset_(std::move(dataset)),
          cache_dir_(ResolveCacheDir()) {
        if (options_.backend != "mft") {
            throw std::runtime_error("GemmaClassicClientBackend only supports backend=mft");
        }
        if (options_.model_dir.empty()) {
            throw std::runtime_error("model_dir must not be empty for backend=mft");
        }
        std::filesystem::create_directories(cache_dir_);
        LoadModelAndTokenizer();
        ConfigureLora(options_.target_mode, options_.lora_r, options_.lora_alpha, options_.lora_dropout);
    }

    flwr::proto::Parameters GetParameters() override {
        const std::vector<std::uint8_t> adapter_bytes = SaveCurrentAdapterBytes();
        flwr::proto::Parameters parameters;
        parameters.set_tensor_type(kTensorType);
        const std::vector<std::uint8_t> blob = SerializeNpy(
            adapter_bytes.data(),
            adapter_bytes.size(),
            {static_cast<std::int64_t>(adapter_bytes.size())},
            NpyDType::kUInt8);
        parameters.add_tensors(blob.data(), static_cast<int>(blob.size()));
        return parameters;
    }

    LocalTrainResult Fit(
        const flwr::proto::Parameters& parameters,
        const google::protobuf::Map<std::string, flwr::proto::Scalar>& config) override {
        RoundOptions round = BuildRoundOptions(config);
        const auto round_start = std::chrono::steady_clock::now();
        const std::size_t incoming_bytes = parameters.tensors_size() > 0
            ? static_cast<std::size_t>(parameters.tensors(0).size())
            : 0U;

        LoadIncomingAdapter(parameters);

        if (round.mode != "train") {
            LocalTrainResult result;
            const auto serialize_start = std::chrono::steady_clock::now();
            result.parameters = GetParameters();
            const auto round_end = std::chrono::steady_clock::now();
            result.serialize_time_sec = std::chrono::duration<double>(round_end - serialize_start).count();
            result.round_time_sec = std::chrono::duration<double>(round_end - round_start).count();
            result.rss_mb = CurrentRssMb();
            result.transmitted_bytes = incoming_bytes +
                (result.parameters.tensors_size() > 0 ? static_cast<std::size_t>(result.parameters.tensors(0).size()) : 0U);
            result.tensor_count = static_cast<std::size_t>(result.parameters.tensors_size());
            return result;
        }

        ops::Adam optimizer(MakeAdamConfig(round));
        double loss_sum = 0.0;
        int updates = 0;
        int examples = 0;
        const auto train_start = std::chrono::steady_clock::now();
        for (int epoch = 0; epoch < round.local_epochs; ++epoch) {
            for (int step = 0; step < round.local_steps; ++step) {
                TrainBatch batch = NextTrainBatch(round.batch_size, round.max_seq_len);
                const float loss = TrainStep(batch, optimizer);
                loss_sum += static_cast<double>(loss);
                updates += 1;
                examples += batch.batch_size;
                if (options_.verbose && updates % std::max(1, round.logging_steps) == 0) {
                    std::cout << "[client] round=" << round.server_round
                              << " local_step=" << updates
                              << " loss=" << loss
                              << " examples=" << examples
                              << "\n";
                }
            }
        }
        const auto train_end = std::chrono::steady_clock::now();

        const auto serialize_start = std::chrono::steady_clock::now();
        flwr::proto::Parameters updated_parameters = GetParameters();
        const auto serialize_end = std::chrono::steady_clock::now();
        const auto round_end = std::chrono::steady_clock::now();

        LocalTrainResult result;
        result.parameters = std::move(updated_parameters);
        result.num_examples = examples;
        result.loss = updates > 0 ? loss_sum / static_cast<double>(updates) : 0.0;
        result.train_time_sec = std::chrono::duration<double>(train_end - train_start).count();
        result.serialize_time_sec = std::chrono::duration<double>(serialize_end - serialize_start).count();
        result.round_time_sec = std::chrono::duration<double>(round_end - round_start).count();
        result.rss_mb = CurrentRssMb();
        result.transmitted_bytes = incoming_bytes +
            (result.parameters.tensors_size() > 0 ? static_cast<std::size_t>(result.parameters.tensors(0).size()) : 0U);
        result.tensor_count = static_cast<std::size_t>(result.parameters.tensors_size());
        return result;
    }

private:
    ClientOptions options_;
    ClientShardDataset dataset_;
    std::filesystem::path cache_dir_;
    std::unique_ptr<ops::HFTokenizer> tokenizer_;
    std::unique_ptr<ops::GemmaModel> model_;
    std::unique_ptr<ops::GemmaLoraInjector> injector_;
    int32_t eos_id_ = 1;
    int32_t pad_id_ = 0;

    std::filesystem::path ResolveCacheDir() const {
        if (!options_.metrics_path.empty()) {
            const std::filesystem::path metrics_path(options_.metrics_path);
            return metrics_path.parent_path() / "adapter_cache";
        }
        return std::filesystem::temp_directory_path() / ("lshaped_" + options_.client_id);
    }

    void LoadModelAndTokenizer() {
        std::cout << "[client] loading Gemma model from " << options_.model_dir << "\n";
        auto cfg = ops::GemmaTextConfig::from_pretrained(options_.model_dir);
        model_ = std::make_unique<ops::GemmaModel>(cfg);

        ops::SafeTensorsReader reader(options_.model_dir + "/model.safetensors");
        reader.parse_header();
        const auto mapping = ops::GemmaKeyMapper::generate_gemma_mapping(cfg.num_hidden_layers);
        ops::SafeTensorsLoadOptions load_opts;
        load_opts.verbose = false;
        auto tensors = reader.load_tensors_mapped(mapping, load_opts);
        for (auto& kv : tensors) {
            model_->assign_weight(kv.first, kv.second);
        }

        tokenizer_ = std::make_unique<ops::HFTokenizer>(options_.model_dir, true);
        if (!tokenizer_->load()) {
            throw std::runtime_error("Failed to load tokenizer from model_dir: " + options_.model_dir);
        }
        eos_id_ = ResolveTokenId(*tokenizer_, {"<eos>", "</s>"});
        if (eos_id_ < 0) {
            eos_id_ = 1;
        }
        pad_id_ = ResolveTokenId(*tokenizer_, {"<pad>"});
        if (pad_id_ < 0) {
            pad_id_ = eos_id_;
        }
    }

    void ConfigureLora(const std::string& target_mode, int rank, float alpha, float dropout) {
        injector_ = std::make_unique<ops::GemmaLoraInjector>();
        ops::GemmaLoraSpec spec;
        spec.rank = rank;
        spec.alpha = alpha;
        spec.dropout = dropout;
        spec.target_modules = TargetModulesForMode(target_mode);
        injector_->inject(*model_, spec);
        injector_->print_info();
    }

    RoundOptions BuildRoundOptions(
        const google::protobuf::Map<std::string, flwr::proto::Scalar>& config) const {
        RoundOptions round;
        round.mode = ReadStringConfig(config, "mode", options_.run_mode);
        round.server_round = ReadIntConfig(config, "server_round", -1);
        round.batch_size = static_cast<int>(ReadIntConfig(config, "batch_size", options_.batch_size));
        round.max_seq_len = static_cast<int>(ReadIntConfig(config, "seq_len", options_.max_seq_len));
        round.local_steps = static_cast<int>(ReadIntConfig(config, "local_steps", options_.local_steps));
        round.local_epochs = static_cast<int>(ReadIntConfig(config, "local_epochs", options_.local_epochs));
        round.logging_steps = static_cast<int>(ReadIntConfig(config, "logging_steps", options_.logging_steps));
        round.learning_rate = static_cast<float>(ReadDoubleConfig(config, "learning_rate", options_.learning_rate));
        round.weight_decay = static_cast<float>(ReadDoubleConfig(config, "weight_decay", options_.weight_decay));
        round.lora_r = static_cast<int>(ReadIntConfig(config, "lora_r", options_.lora_r));
        round.lora_alpha = static_cast<float>(ReadDoubleConfig(config, "lora_alpha", options_.lora_alpha));
        round.lora_dropout = static_cast<float>(ReadDoubleConfig(config, "lora_dropout", options_.lora_dropout));
        round.target_mode = ReadStringConfig(config, "target_mode", options_.target_mode);
        round.shuffle = ReadBoolConfig(config, "shuffle", false);
        if (round.lora_r != options_.lora_r || round.target_mode != options_.target_mode) {
            throw std::runtime_error("Round LoRA config must match client launch config");
        }
        return round;
    }

    ops::AdamConfig MakeAdamConfig(const RoundOptions& round) const {
        ops::AdamConfig adam_cfg;
        adam_cfg.learning_rate = round.learning_rate;
        adam_cfg.weight_decay = round.weight_decay;
        adam_cfg.beta1 = 0.9f;
        adam_cfg.beta2 = 0.999f;
        adam_cfg.epsilon = 1e-8f;
        adam_cfg.decoupled_weight_decay = true;
        return adam_cfg;
    }

    std::vector<std::uint8_t> SaveCurrentAdapterBytes() {
        const std::filesystem::path adapter_path = cache_dir_ / "current.safetensors";
        injector_->save_lora_safetensors(adapter_path.string());
        return ReadFileBytes(adapter_path);
    }

    void LoadIncomingAdapter(const flwr::proto::Parameters& parameters) {
        if (parameters.tensors_size() != 1) {
            throw std::runtime_error("Classic client expects exactly one adapter tensor blob");
        }
        if (parameters.tensor_type() != kTensorType) {
            throw std::runtime_error("Classic client expects tensor_type=numpy.ndarray");
        }
        const ParsedNpy parsed = DeserializeNpy(parameters.tensors(0));
        if (parsed.dtype != NpyDType::kUInt8) {
            throw std::runtime_error("Adapter tensor must be uint8");
        }
        const std::filesystem::path adapter_path = cache_dir_ / "incoming.safetensors";
        WriteFileBytes(adapter_path, parsed.raw);
        injector_->load_lora_safetensors(adapter_path.string());
        for (const auto& param : model_->get_lora_parameters()) {
            if (param->grad()) {
                param->zero_grad();
            }
        }
    }

    bool EncodeExample(const MMLUSample& sample, int max_seq_len, std::vector<int32_t>& ids, std::vector<int32_t>& labels) {
        const std::vector<int32_t> prompt_ids = tokenizer_->encode(sample.Prompt());
        const std::vector<int32_t> answer_ids = tokenizer_->encode(options_.answer_prefix + sample.AnswerLabel());
        ids.clear();
        labels.clear();
        ids.reserve(prompt_ids.size() + answer_ids.size() + 1);
        labels.reserve(prompt_ids.size() + answer_ids.size() + 1);

        ids.insert(ids.end(), prompt_ids.begin(), prompt_ids.end());
        labels.insert(labels.end(), prompt_ids.size(), -100);
        ids.insert(ids.end(), answer_ids.begin(), answer_ids.end());
        labels.insert(labels.end(), answer_ids.begin(), answer_ids.end());
        ids.push_back(eos_id_);
        labels.push_back(-100);

        if (static_cast<int>(ids.size()) > max_seq_len) {
            return false;
        }

        ids.resize(static_cast<std::size_t>(max_seq_len), pad_id_);
        labels.resize(static_cast<std::size_t>(max_seq_len), -100);
        return true;
    }

    TrainBatch NextTrainBatch(int requested_batch_size, int max_seq_len) {
        std::vector<std::vector<int32_t>> batch_ids;
        std::vector<std::vector<int32_t>> batch_labels;
        batch_ids.reserve(static_cast<std::size_t>(requested_batch_size));
        batch_labels.reserve(static_cast<std::size_t>(requested_batch_size));

        const int max_attempts = std::max(requested_batch_size * 4, static_cast<int>(dataset_.size()) * 2);
        int attempts = 0;
        while (static_cast<int>(batch_ids.size()) < requested_batch_size && attempts < max_attempts) {
            std::vector<MMLUSample> samples = dataset_.NextBatch(1);
            std::vector<int32_t> ids;
            std::vector<int32_t> labels;
            if (EncodeExample(samples.front(), max_seq_len, ids, labels)) {
                batch_ids.push_back(std::move(ids));
                batch_labels.push_back(std::move(labels));
            }
            attempts += 1;
        }
        if (batch_ids.empty()) {
            throw std::runtime_error("No local samples fit within max_seq_len");
        }

        const int batch_size = static_cast<int>(batch_ids.size());
        std::vector<int32_t> input_ids(static_cast<std::size_t>(batch_size * max_seq_len), pad_id_);
        std::vector<int32_t> labels(static_cast<std::size_t>(batch_size * max_seq_len), -100);
        std::vector<float> attention_mask(static_cast<std::size_t>(batch_size * max_seq_len), 0.0f);

        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < max_seq_len; ++s) {
                const std::size_t idx = static_cast<std::size_t>(b * max_seq_len + s);
                input_ids[idx] = batch_ids[static_cast<std::size_t>(b)][static_cast<std::size_t>(s)];
                labels[idx] = batch_labels[static_cast<std::size_t>(b)][static_cast<std::size_t>(s)];
                attention_mask[idx] = input_ids[idx] == pad_id_ ? 0.0f : 1.0f;
            }
        }

        TrainBatch batch;
        batch.batch_size = batch_size;
        batch.seq_len = max_seq_len;
        batch.input_ids = std::make_shared<ops::Tensor>(
            std::vector<int64_t>{batch_size, max_seq_len},
            input_ids.data(),
            ops::kInt32,
            ops::kCPU);
        batch.labels = std::make_shared<ops::Tensor>(
            std::vector<int64_t>{batch_size, max_seq_len},
            labels.data(),
            ops::kInt32,
            ops::kCPU);
        batch.attention_mask = std::make_shared<ops::Tensor>(
            std::vector<int64_t>{batch_size, max_seq_len},
            attention_mask.data(),
            ops::kFloat32,
            ops::kCPU);
        return batch;
    }

    float TrainStep(const TrainBatch& batch, ops::Adam& optimizer) {
        auto logits = model_->forward(batch.input_ids, batch.attention_mask);
        auto loss = ops::lm_cross_entropy(logits, batch.labels, -100, "mean");
        const float loss_val = loss->data<float>()[0];
        loss->backward();

        const std::vector<ops::TensorPtr> params = model_->get_lora_parameters();
        std::vector<ops::TensorPtr> grads;
        grads.reserve(params.size());
        for (const auto& param : params) {
            grads.push_back(param->grad());
        }
        optimizer.step(params, grads);
        for (const auto& param : params) {
            if (param->grad()) {
                param->zero_grad();
            }
        }
        ops::MemoryManager::instance().force_cleanup();
        return loss_val;
    }
};

}  // namespace

std::unique_ptr<ClassicClientBackend> CreateClassicClientBackend(
    const ClientOptions& options,
    ClientShardDataset dataset) {
    return std::make_unique<GemmaClassicClientBackend>(options, std::move(dataset));
}

}  // namespace lshaped
