#include "lshaped/federated_trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <climits>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "finetune_ops/core/autograd_engine.h"
#include "finetune_ops/core/lm_loss.h"
#include "finetune_ops/core/memory_manager.h"
#include "finetune_ops/core/ops.h"
#include "lshaped/npy_serializer.h"

#include "finetune_ops/core/tokenizer_hf.h"
#include "finetune_ops/data/wikitext2_dataset.h"
#include "finetune_ops/graph/gemma_lora_injector.h"
#include "finetune_ops/graph/gemma_model.h"
#include "finetune_ops/graph/qwen_model.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/optim/adam.h"
#include "finetune_ops/optim/gemma_trainer.h"

namespace lshaped {

namespace {

struct ProcMemoryMetrics {
    double rss_mb = -1.0;
    double hwm_mb = -1.0;
    double swap_mb = -1.0;
};

ProcMemoryMetrics CurrentProcMemoryMetrics() {
#if defined(__linux__) || defined(__ANDROID__)
    ProcMemoryMetrics metrics;
    {
        std::ifstream input("/proc/self/statm");
        long resident_pages = 0;
        input.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
        input >> resident_pages;
        if ((input.good() || input.eof()) && resident_pages >= 0) {
            const long page_size = sysconf(_SC_PAGESIZE);
            if (page_size > 0) {
                metrics.rss_mb = static_cast<double>(resident_pages) * static_cast<double>(page_size) / (1024.0 * 1024.0);
            }
        }
    }

    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        auto parse_field_mb = [&](const char* prefix, double& out) {
            const std::size_t prefix_len = std::strlen(prefix);
            if (line.rfind(prefix, 0) != 0) {
                return false;
            }
            std::stringstream ss(line.substr(prefix_len));
            double kb = -1.0;
            ss >> kb;
            if (ss) {
                out = kb / 1024.0;
            }
            return true;
        };
        if (parse_field_mb("VmHWM:", metrics.hwm_mb) ||
            parse_field_mb("VmSwap:", metrics.swap_mb)) {
            continue;
        }
    }
    return metrics;
#else
    return {};
#endif
}

std::uint64_t ParameterBytes(const flwr::proto::Parameters& parameters) {
    std::uint64_t total = 0;
    for (const auto& tensor : parameters.tensors()) {
        total += static_cast<std::uint64_t>(tensor.size());
    }
    return total;
}

std::vector<std::string> SplitCsv(const std::string& value) {
    std::vector<std::string> out;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            out.push_back(token);
        }
    }
    return out;
}

std::string JsonEncodeDoubles(const std::vector<double>& values) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            oss << ",";
        }
        oss << values[index];
    }
    oss << "]";
    return oss.str();
}

ops::sharding::DiskQuantizationMode ParseShardQuantMode(const ClientOptions& options) {
    if (options.shard_quant_mode.empty()) {
        return options.shard_quantize_fp16_on_disk
            ? ops::sharding::DiskQuantizationMode::FP16
            : ops::sharding::DiskQuantizationMode::None;
    }
    std::string mode = options.shard_quant_mode;
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (mode == "none" || mode == "fp32") {
        return ops::sharding::DiskQuantizationMode::None;
    }
    if (mode == "fp16") {
        return ops::sharding::DiskQuantizationMode::FP16;
    }
    if (mode == "int8") {
        return ops::sharding::DiskQuantizationMode::INT8;
    }
    if (mode == "int4") {
        return ops::sharding::DiskQuantizationMode::INT4;
    }
    throw std::runtime_error("Unsupported shard_quant_mode: " + options.shard_quant_mode);
}

bool IsLowBitShardMode(ops::sharding::DiskQuantizationMode mode) {
    return mode == ops::sharding::DiskQuantizationMode::INT8 ||
           mode == ops::sharding::DiskQuantizationMode::INT4;
}

bool ShouldUseStreamedQuantizedGemmaLoad(const ClientOptions& options) {
    if (options.shard_max_resident_mb <= 0) {
        return false;
    }
    return IsLowBitShardMode(ParseShardQuantMode(options));
}

bool ShouldTransposeLinearWeight(const std::string& hf_key, const ops::SafeTensorInfo& info) {
    const bool is_embedding = hf_key.find("embed_tokens") != std::string::npos;
    return hf_key.find("weight") != std::string::npos &&
           hf_key.find("ln") == std::string::npos &&
           !is_embedding &&
           info.shape.size() == 2;
}

bool IsWikiTextRawDataset(const ClientOptions& options) {
    return options.dataset_format == "wikitext_raw";
}

enum class ModelFamily {
    Gemma,
    Qwen,
};

std::string ReadTextFile(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        return {};
    }
    std::stringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::string ToLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

ModelFamily DetectModelFamily(const std::string& model_dir) {
    std::string config_path = model_dir;
    if (!config_path.empty() && config_path.back() != '/') {
        config_path += "/";
    }
    config_path += "config.json";
    const std::string config = ToLowerCopy(ReadTextFile(config_path));
    if (config.find("\"model_type\"") != std::string::npos &&
        config.find("qwen") != std::string::npos) {
        return ModelFamily::Qwen;
    }
    if (config.find("qwen2forcausallm") != std::string::npos ||
        config.find("qwen2") != std::string::npos) {
        return ModelFamily::Qwen;
    }
    return ModelFamily::Gemma;
}

bool IsPathSeparator(char ch) {
    return ch == '/' || ch == '\\';
}

std::string ResolveCurrentDir() {
#if defined(__linux__) || defined(__ANDROID__)
    char buffer[PATH_MAX];
    if (::getcwd(buffer, sizeof(buffer)) != nullptr) {
        return std::string(buffer);
    }
#endif
    return ".";
}

std::string ParentDir(const std::string& path) {
    if (path.empty()) {
        return "";
    }
    std::size_t end = path.size();
    while (end > 1 && IsPathSeparator(path[end - 1])) {
        --end;
    }
    const std::size_t pos = path.find_last_of("/\\", end - 1);
    if (pos == std::string::npos) {
        return "";
    }
    if (pos == 0) {
        return path.substr(0, 1);
    }
    return path.substr(0, pos);
}

std::string JoinPath(const std::string& dir, const std::string& child) {
    if (dir.empty() || dir == ".") {
        return child;
    }
    if (IsPathSeparator(dir.back())) {
        return dir + child;
    }
    return dir + "/" + child;
}

void EnsureDirRecursive(const std::string& dir) {
#if defined(__linux__) || defined(__ANDROID__)
    if (dir.empty() || dir == ".") {
        return;
    }
    std::string normalized = dir;
    while (normalized.size() > 1 && IsPathSeparator(normalized.back())) {
        normalized.pop_back();
    }
    std::string current;
    std::size_t index = 0;
    if (!normalized.empty() && IsPathSeparator(normalized.front())) {
        current = "/";
        index = 1;
    }
    while (index <= normalized.size()) {
        const std::size_t next = normalized.find_first_of("/\\", index);
        const std::string part = normalized.substr(
            index,
            next == std::string::npos ? std::string::npos : next - index);
        if (!part.empty()) {
            if (current.empty() || current == "/") {
                current += part;
            } else {
                current += "/" + part;
            }
            if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                throw std::runtime_error("Failed to create directory: " + current);
            }
        }
        if (next == std::string::npos) {
            break;
        }
        index = next + 1;
    }
#else
    (void)dir;
#endif
}

std::string ResolveWorkDir(const ClientOptions& options) {
    if (!options.metrics_path.empty()) {
        const std::string parent = ParentDir(options.metrics_path);
        if (!parent.empty()) {
            return parent;
        }
    }
    return ResolveCurrentDir();
}

void EnsureParentDir(const std::string& path) {
    const std::string parent = ParentDir(path);
    if (!parent.empty()) {
        EnsureDirRecursive(parent);
    }
}

int32_t ResolveGemmaTokenId(const ops::HFTokenizer& tokenizer, const std::vector<std::string>& candidates) {
    for (const auto& token : candidates) {
        const int32_t id = tokenizer.token_to_id(token);
        if (id >= 0) {
            return id;
        }
    }
    throw std::runtime_error("Failed to resolve Gemma special token id");
}

std::vector<std::string> ResolveTargetModules(const ClientOptions& options) {
    if (!options.lora_targets.empty()) {
        return SplitCsv(options.lora_targets);
    }
    if (options.target_mode == "light") {
        return {"q_proj", "v_proj"};
    }
    if (options.target_mode == "attn") {
        return {"q_proj", "k_proj", "v_proj", "o_proj"};
    }
    return {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"};
}

bool QwenQvOnly(const ClientOptions& options) {
    const auto targets = ResolveTargetModules(options);
    if (targets.empty()) {
        return false;
    }
    return targets.size() == 2 &&
           std::find(targets.begin(), targets.end(), "q_proj") != targets.end() &&
           std::find(targets.begin(), targets.end(), "v_proj") != targets.end();
}

std::vector<std::string> BuildParameterNames(const ops::GemmaModel& model) {
    std::vector<std::string> names;
    auto append = [&](int layer, const std::string& module, const std::unique_ptr<ops::LoRALinear>& linear) {
        if (!linear) {
            return;
        }
        const auto& slices = linear->slices();
        for (std::size_t slice_index = 0; slice_index < slices.size(); ++slice_index) {
            std::string base = "layers." + std::to_string(layer) + "." + module;
            if (slices.size() > 1) {
                base += "." + std::to_string(slice_index);
            }
            names.push_back(base + ".lora_A");
            names.push_back(base + ".lora_B");
        }
    };

    for (int layer = 0; layer < model.config().num_hidden_layers; ++layer) {
        const auto& block = model.get_block(layer);
        append(layer, "q_proj", block.q_proj_lora);
        append(layer, "k_proj", block.k_proj_lora);
        append(layer, "v_proj", block.v_proj_lora);
        append(layer, "o_proj", block.o_proj_lora);
        append(layer, "gate_proj", block.gate_proj_lora);
        append(layer, "up_proj", block.up_proj_lora);
        append(layer, "down_proj", block.down_proj_lora);
    }
    return names;
}

std::vector<std::string> BuildQwenParameterNames(const ops::QwenModel& model, bool qv_only) {
    std::vector<std::string> names;
    auto append = [&](int layer, const std::string& module, const std::unique_ptr<ops::LoRALinear>& linear) {
        if (!linear || linear->slices().empty()) {
            return;
        }
        const auto& slice = linear->slices().front();
        if (!slice.A || !slice.B) {
            return;
        }
        const std::string base = "layers." + std::to_string(layer) + "." + module;
        names.push_back(base + ".lora_A");
        names.push_back(base + ".lora_B");
    };

    for (int layer = 0; layer < model.config().num_hidden_layers; ++layer) {
        const auto& block = model.get_block(layer);
        append(layer, "q_proj", block.q_lin);
        if (!qv_only) {
            append(layer, "k_proj", block.k_lin);
        }
        append(layer, "v_proj", block.v_lin);
        if (!qv_only) {
            append(layer, "o_proj", block.o_lin);
        }
    }
    return names;
}

std::vector<ops::TensorPtr> GetOrderedQwenTrainableParams(ops::QwenModel& model, bool qv_only) {
    std::vector<ops::TensorPtr> params;
    auto append = [&](ops::QwenBlock& block, const std::unique_ptr<ops::LoRALinear>& linear) {
        if (!linear || linear->slices().empty()) {
            return;
        }
        const auto& slice = linear->slices().front();
        if (!slice.A || !slice.B) {
            return;
        }
        params.push_back(slice.A);
        params.push_back(slice.B);
    };

    for (int layer = 0; layer < model.config().num_hidden_layers; ++layer) {
        auto& block = model.get_block(layer);
        append(block, block.q_lin);
        if (!qv_only) {
            append(block, block.k_lin);
        }
        append(block, block.v_lin);
        if (!qv_only) {
            append(block, block.o_lin);
        }
    }
    return params;
}

std::vector<float> Transpose2DFloatTensor(const ops::TensorPtr& tensor) {
    if (!tensor || tensor->ndim() != 2 || tensor->dtype() != ops::DType::kFloat32) {
        throw std::runtime_error("Expected 2D float32 tensor for Qwen LoRA transpose");
    }
    const auto rows = static_cast<std::size_t>(tensor->shape()[0]);
    const auto cols = static_cast<std::size_t>(tensor->shape()[1]);
    const float* src = tensor->data<float>();
    std::vector<float> out(rows * cols);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            out[col * rows + row] = src[row * cols + col];
        }
    }
    return out;
}

void CopyMaybeTransposed2DFloatTensor(
    const ParsedNpyArray& parsed,
    const ops::TensorPtr& tensor,
    bool transpose_external_to_internal) {
    if (!tensor || tensor->ndim() != 2 || tensor->dtype() != ops::DType::kFloat32) {
        throw std::runtime_error("Expected 2D float32 tensor for Qwen LoRA apply");
    }
    const auto target_rows = static_cast<std::size_t>(tensor->shape()[0]);
    const auto target_cols = static_cast<std::size_t>(tensor->shape()[1]);
    const float* src = reinterpret_cast<const float*>(parsed.raw_bytes.data());
    float* dst = tensor->data<float>();
    const std::vector<std::int64_t> target_shape = {
        static_cast<std::int64_t>(target_rows),
        static_cast<std::int64_t>(target_cols),
    };
    if (parsed.shape == target_shape) {
        std::memcpy(dst, src, parsed.raw_bytes.size());
        return;
    }
    if (transpose_external_to_internal &&
        parsed.shape.size() == 2 &&
        static_cast<std::size_t>(parsed.shape[0]) == target_cols &&
        static_cast<std::size_t>(parsed.shape[1]) == target_rows) {
        for (std::size_t row = 0; row < target_rows; ++row) {
            for (std::size_t col = 0; col < target_cols; ++col) {
                dst[row * target_cols + col] = src[col * target_rows + row];
            }
        }
        return;
    }
    throw std::runtime_error("Incoming Qwen LoRA parameter shape mismatch");
}

void RegisterGemmaBaseWeights(ops::GemmaModel& model, ops::sharding::ParameterSharder& sharder) {
    auto register_tensor = [&](const std::string& name, ops::TensorPtr& tensor_ref) {
        if (!tensor_ref) {
            throw std::runtime_error("Attempted to shard null Gemma tensor: " + name);
        }
        sharder.register_parameter(name, tensor_ref, false, &tensor_ref);
    };

    register_tensor("embed_tokens.weight", model.embed_weight_ref());
    register_tensor("norm.weight", model.norm_weight_ref());
    register_tensor("lm_head.weight", model.lm_head_weight_ref());

    for (int layer = 0; layer < model.config().num_hidden_layers; ++layer) {
        auto& block = model.get_block(layer);
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        register_tensor(prefix + "input_layernorm.weight", block.input_layernorm_weight);
        register_tensor(prefix + "post_attention_layernorm.weight", block.post_attention_layernorm_weight);
        register_tensor(prefix + "pre_feedforward_layernorm.weight", block.pre_feedforward_layernorm_weight);
        register_tensor(prefix + "post_feedforward_layernorm.weight", block.post_feedforward_layernorm_weight);
        register_tensor(prefix + "self_attn.q_proj.weight", block.q_proj_weight);
        register_tensor(prefix + "self_attn.k_proj.weight", block.k_proj_weight);
        register_tensor(prefix + "self_attn.v_proj.weight", block.v_proj_weight);
        register_tensor(prefix + "self_attn.o_proj.weight", block.o_proj_weight);
        register_tensor(prefix + "self_attn.q_norm.weight", block.q_norm_weight);
        register_tensor(prefix + "self_attn.k_norm.weight", block.k_norm_weight);
        register_tensor(prefix + "mlp.gate_proj.weight", block.gate_proj_weight);
        register_tensor(prefix + "mlp.up_proj.weight", block.up_proj_weight);
        register_tensor(prefix + "mlp.down_proj.weight", block.down_proj_weight);
    }
}

class MockFederatedTrainer final : public FederatedTrainer {
public:
    explicit MockFederatedTrainer(const ClientOptions& options)
        : options_(options),
          weights_(static_cast<std::size_t>(std::max(1, options.mock_hidden_size)), 0.0f),
          parameter_names_({"mock.adapter"}) {
        for (std::size_t i = 0; i < weights_.size(); ++i) {
            weights_[i] = static_cast<float>(options.client_index + 1) * 1e-3f * static_cast<float>(i + 1);
        }
    }

    flwr::proto::Parameters GetParameters() const override {
        flwr::proto::Parameters parameters;
        parameters.set_tensor_type("numpy.ndarray");
        const auto blob = SerializeNpy(
            weights_.data(),
            weights_.size(),
            {static_cast<std::int64_t>(weights_.size())},
            NpyDType::kFloat32);
        parameters.add_tensors(blob.data(), static_cast<int>(blob.size()));
        return parameters;
    }

    const std::vector<std::string>& ParameterNames() const override { return parameter_names_; }
    std::string BackendName() const override { return "mock"; }

    LocalFitSummary Fit(
        const flwr::proto::Parameters& global_parameters,
        int batch_size,
        int max_seq_len,
        int local_steps,
        int local_epochs,
        float learning_rate,
        float /*weight_decay*/,
        int server_round) override {
        (void)batch_size;
        (void)max_seq_len;
        (void)local_epochs;
        if (global_parameters.tensors_size() == 1) {
            const ParsedNpyArray parsed = ParseNpy(global_parameters.tensors(0));
            if (parsed.dtype != NpyDType::kFloat32) {
                throw std::runtime_error("Mock trainer expects float32 parameters");
            }
            if (parsed.raw_bytes.size() != weights_.size() * sizeof(float)) {
                throw std::runtime_error("Mock trainer received mismatched parameter size");
            }
            std::memcpy(weights_.data(), parsed.raw_bytes.data(), parsed.raw_bytes.size());
        }

        const auto start = std::chrono::steady_clock::now();
        const float delta = learning_rate * static_cast<float>(options_.client_index + 1);
        double step_time_sum = 0.0;
        double max_step_time_sec = 0.0;
        std::vector<double> step_times_sec;
        std::vector<double> rss_samples_mb;
        ProcMemoryMetrics peak_memory = CurrentProcMemoryMetrics();
        for (int step = 0; step < std::max(1, local_steps); ++step) {
            const auto step_start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < weights_.size(); ++i) {
                weights_[i] += delta + static_cast<float>(server_round + step) * 1e-4f;
            }
            const double step_time_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
            step_time_sum += step_time_sec;
            max_step_time_sec = std::max(max_step_time_sec, step_time_sec);
            step_times_sec.push_back(step_time_sec);
            const ProcMemoryMetrics current_memory = CurrentProcMemoryMetrics();
            if (current_memory.rss_mb >= 0.0) {
                rss_samples_mb.push_back(current_memory.rss_mb);
            }
            peak_memory.rss_mb = std::max(peak_memory.rss_mb, current_memory.rss_mb);
            peak_memory.hwm_mb = std::max(peak_memory.hwm_mb, current_memory.hwm_mb);
            peak_memory.swap_mb = std::max(peak_memory.swap_mb, current_memory.swap_mb);
        }
        const auto end = std::chrono::steady_clock::now();

        LocalFitSummary summary;
        summary.num_examples = std::max(1, options_.synthetic_samples);
        summary.steps_completed = std::max(1, local_steps);
        summary.epochs_completed = 1;
        summary.mean_loss = 1.0 / static_cast<double>(server_round + options_.client_index + 1);
        summary.mean_objective_loss = summary.mean_loss;
        summary.mean_prox_term = 0.0;
        summary.train_time_sec = std::chrono::duration<double>(end - start).count();
        summary.avg_rss_mb = rss_samples_mb.empty()
            ? peak_memory.rss_mb
            : (std::accumulate(rss_samples_mb.begin(), rss_samples_mb.end(), 0.0) /
               static_cast<double>(rss_samples_mb.size()));
        summary.client_rss_mb = peak_memory.rss_mb;
        summary.client_hwm_mb = peak_memory.hwm_mb;
        summary.client_swap_mb = peak_memory.swap_mb;
        summary.mean_step_time_sec = step_time_sum / static_cast<double>(std::max(1, local_steps));
        summary.max_step_time_sec = max_step_time_sec;
        summary.transmitted_bytes = ParameterBytes(GetParameters());
        summary.step_times_sec_json = JsonEncodeDoubles(step_times_sec);
        return summary;
    }

private:
    ClientOptions options_;
    std::vector<float> weights_;
    std::vector<std::string> parameter_names_;
};

class GemmaFederatedTrainer final : public FederatedTrainer {
public:
    GemmaFederatedTrainer(const ClientOptions& options, ClientShardDataset dataset)
        : options_(options),
          dataset_(std::move(dataset)),
          work_dir_(ResolveWorkDir(options)) {
        if (options_.model_dir.empty()) {
            throw std::runtime_error("backend=mft requires --model_dir");
        }
        if (!IsWikiTextRawDataset(options_) && dataset_.empty()) {
            throw std::runtime_error("MFT client dataset must not be empty");
        }

        tokenizer_ = std::make_unique<ops::HFTokenizer>(options_.model_dir, true);
        if (!tokenizer_->load()) {
            throw std::runtime_error("Failed to load HF tokenizer from model_dir: " + options_.model_dir);
        }
        eos_id_ = ResolveGemmaTokenId(*tokenizer_, {"<eos>", "</s>"});

        auto model_cfg = ops::GemmaTextConfig::from_pretrained(options_.model_dir);
        model_cfg.use_bf16_activations = options_.use_bf16_activations;
        model_cfg.checkpoint_every = std::max(0, options_.checkpoint_every);
        model_cfg.checkpoint_mlp = options_.checkpoint_mlp;
        model_cfg.mlp_chunk_size = std::max(0, options_.mlp_chunk_size);
        shard_quant_mode_ = ParseShardQuantMode(options_);
        use_streamed_quant_load_ = ShouldUseStreamedQuantizedGemmaLoad(options_);
        model_ = std::make_unique<ops::GemmaModel>(model_cfg, !use_streamed_quant_load_);
        if (use_streamed_quant_load_) {
            model_->set_auto_lm_head_tying(false);
        }
        lora_spec_ = BuildLoraSpec();
        injector_.inject(*model_, lora_spec_);
        if (use_streamed_quant_load_) {
            InitializeParameterSharder(false);
            LoadModelWeightsStreamed();
        } else {
            LoadModelWeights();
            InitializeParameterSharder(true);
        }
        parameter_names_ = BuildParameterNames(*model_);
        if (parameter_names_.empty()) {
            throw std::runtime_error("LoRA parameter set is empty after injection");
        }

        RebuildDatasets(options_.max_seq_len);
    }

    flwr::proto::Parameters GetParameters() const override {
        flwr::proto::Parameters parameters;
        parameters.set_tensor_type("numpy.ndarray");
        const auto trainable = injector_.get_trainable_params();
        for (const auto& tensor : trainable) {
            if (!tensor) {
                throw std::runtime_error("Encountered null LoRA tensor while serializing parameters");
            }
            if (tensor->dtype() != ops::DType::kFloat32) {
                throw std::runtime_error("Only float32 LoRA tensors are supported");
            }
            std::vector<std::int64_t> shape;
            for (const auto dim : tensor->shape()) {
                shape.push_back(dim);
            }
            const auto blob = SerializeNpy(
                tensor->data<float>(),
                static_cast<std::size_t>(tensor->numel()),
                shape,
                NpyDType::kFloat32);
            parameters.add_tensors(blob.data(), static_cast<int>(blob.size()));
        }
        return parameters;
    }

    const std::vector<std::string>& ParameterNames() const override { return parameter_names_; }
    std::string BackendName() const override { return "mft"; }

    LocalFitSummary Fit(
        const flwr::proto::Parameters& global_parameters,
        int batch_size,
        int max_seq_len,
        int local_steps,
        int local_epochs,
        float learning_rate,
        float weight_decay,
        int /*server_round*/) override {
        if (max_seq_len != current_seq_len_) {
            RebuildDatasets(max_seq_len);
        }
        ApplyParameters(global_parameters);

        train_dataset_->reset_cursor();
        eval_dataset_->reset_cursor();

        ops::GemmaTrainerConfig trainer_cfg;
        trainer_cfg.learning_rate = learning_rate;
        trainer_cfg.weight_decay = weight_decay;
        trainer_cfg.max_grad_norm = options_.grad_clip_norm;
        trainer_cfg.fedprox_mu = options_.fedprox_mu;
        trainer_cfg.logging_steps = std::max(1, options_.logging_steps);
        trainer_cfg.eval_steps = 0;
        trainer_cfg.save_every = 0;
        const int grad_accum_steps = std::max(1, options_.grad_accum_steps);
        const int effective_batch_size = std::max(1, batch_size);
        const int micro_batch_size =
            std::max(1, (effective_batch_size + grad_accum_steps - 1) / grad_accum_steps);
        trainer_cfg.micro_batch_size = micro_batch_size;
        trainer_cfg.grad_accum_steps = grad_accum_steps;
        trainer_cfg.output_dir = work_dir_;

        ops::GemmaLoRATrainer trainer(*model_, injector_, *train_dataset_, *eval_dataset_, trainer_cfg);
        const auto start = std::chrono::steady_clock::now();

        const int effective_steps = std::max(1, local_steps);
        const int effective_epochs = std::max(0, local_epochs);
        int target_updates = effective_steps;
        if (effective_epochs > 0) {
            const int steps_per_epoch = std::max(
                1,
                static_cast<int>((train_dataset_->num_sequences() + static_cast<std::size_t>(batch_size) - 1) /
                                 static_cast<std::size_t>(batch_size)));
            target_updates = std::max(target_updates, steps_per_epoch * effective_epochs);
        }

        double loss_sum = 0.0;
        double objective_loss_sum = 0.0;
        double prox_term_sum = 0.0;
        std::int64_t examples_seen = 0;
        int updates = 0;
        double step_time_sum = 0.0;
        double max_step_time_sec = 0.0;
        std::vector<double> step_times_sec;
        std::vector<double> rss_samples_mb;
        ProcMemoryMetrics peak_memory = CurrentProcMemoryMetrics();
        while (updates < target_updates) {
            const auto step_start = std::chrono::steady_clock::now();
            float objective_loss = -1.0f;
            std::int64_t update_examples = 0;
            int micro_steps = 0;
            for (int micro = 0; micro < grad_accum_steps; ++micro) {
                auto batch = train_dataset_->next_batch(static_cast<std::size_t>(micro_batch_size), true);
                if (!batch.input_ids) {
                    break;
                }
                update_examples += batch.input_ids->shape()[0];
                ++micro_steps;
                objective_loss = trainer.train_step(batch);
                if (objective_loss >= 0.0f) {
                    break;
                }
            }
            if (micro_steps <= 0) {
                break;
            }
            if (objective_loss < 0.0f) {
                continue;
            }
            const float prox_term = trainer_cfg.fedprox_mu > 0.0f ? trainer.last_prox_term() : 0.0f;
            const float base_loss = objective_loss - prox_term;
            loss_sum += base_loss;
            objective_loss_sum += objective_loss;
            prox_term_sum += prox_term;
            examples_seen += update_examples;
            ++updates;
            const double step_time_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
            step_time_sum += step_time_sec;
            max_step_time_sec = std::max(max_step_time_sec, step_time_sec);
            step_times_sec.push_back(step_time_sec);
            const ProcMemoryMetrics current_memory = CurrentProcMemoryMetrics();
            if (current_memory.rss_mb >= 0.0) {
                rss_samples_mb.push_back(current_memory.rss_mb);
            }
            peak_memory.rss_mb = std::max(peak_memory.rss_mb, current_memory.rss_mb);
            peak_memory.hwm_mb = std::max(peak_memory.hwm_mb, current_memory.hwm_mb);
            peak_memory.swap_mb = std::max(peak_memory.swap_mb, current_memory.swap_mb);
        }
        const auto end = std::chrono::steady_clock::now();

        LocalFitSummary summary;
        summary.num_examples = std::max<std::int64_t>(1, examples_seen);
        summary.steps_completed = updates;
        summary.epochs_completed = effective_epochs;
        summary.mean_loss = updates > 0 ? (loss_sum / static_cast<double>(updates)) : 0.0;
        summary.mean_objective_loss = updates > 0 ? (objective_loss_sum / static_cast<double>(updates)) : 0.0;
        summary.mean_prox_term = updates > 0 ? (prox_term_sum / static_cast<double>(updates)) : 0.0;
        summary.train_time_sec = std::chrono::duration<double>(end - start).count();
        if (sharder_) {
            sharder_->offload_all();
        }
        summary.avg_rss_mb = rss_samples_mb.empty()
            ? peak_memory.rss_mb
            : (std::accumulate(rss_samples_mb.begin(), rss_samples_mb.end(), 0.0) /
               static_cast<double>(rss_samples_mb.size()));
        summary.client_rss_mb = peak_memory.rss_mb;
        summary.client_hwm_mb = peak_memory.hwm_mb;
        summary.client_swap_mb = peak_memory.swap_mb;
        summary.mean_step_time_sec = updates > 0 ? (step_time_sum / static_cast<double>(updates)) : 0.0;
        summary.max_step_time_sec = max_step_time_sec;
        summary.transmitted_bytes = ParameterBytes(GetParameters());
        summary.step_times_sec_json = JsonEncodeDoubles(step_times_sec);
        return summary;
    }

private:
    ops::GemmaLoraSpec BuildLoraSpec() const {
        ops::GemmaLoraSpec spec = ops::GemmaLoraSpec::full_attn_mlp();
        if (options_.target_mode == "light") {
            spec = ops::GemmaLoraSpec::attention_light();
        } else if (options_.target_mode == "attn") {
            spec = ops::GemmaLoraSpec::attention_only();
        }
        spec.rank = std::max(1, options_.lora_r);
        spec.alpha = options_.lora_alpha;
        spec.dropout = options_.lora_dropout;
        spec.target_modules = ResolveTargetModules(options_);
        return spec;
    }

    void LoadModelWeights() {
        ops::SafeTensorsReader reader(options_.model_dir + "/model.safetensors");
        reader.parse_header();
        const auto mapping = ops::GemmaKeyMapper::generate_gemma_mapping(model_->config().num_hidden_layers);
        ops::SafeTensorsLoadOptions load_opts;
        load_opts.verbose = false;
        const auto tensors = reader.load_tensors_mapped(mapping, load_opts);
        for (const auto& kv : tensors) {
            model_->assign_weight(kv.first, kv.second);
        }
    }

    void LoadModelWeightsStreamed() {
        if (!sharder_) {
            throw std::runtime_error("LoadModelWeightsStreamed requires initialized sharder");
        }
        ops::SafeTensorsReader reader(options_.model_dir + "/model.safetensors");
        reader.parse_header();
        const auto mapping = ops::GemmaKeyMapper::generate_gemma_mapping(model_->config().num_hidden_layers);
        const auto embed_it = mapping.find("embed_tokens.weight");
        if (embed_it == mapping.end()) {
            throw std::runtime_error("Gemma mapping missing embed_tokens.weight");
        }
        for (const auto& kv : mapping) {
            std::string source_name = kv.second;
            bool transpose = false;
            try {
                const auto info = reader.get_tensor_info(source_name);
                transpose = ShouldTransposeLinearWeight(source_name, info);
            } catch (const std::exception&) {
                if (kv.first != "lm_head.weight") {
                    throw;
                }
                source_name = embed_it->second;
                transpose = true;
            }
            std::vector<int64_t> shape;
            ops::DType dtype = ops::kFloat32;
            auto encoded = reader.load_tensor_quantized(
                source_name, shard_quant_mode_, transpose, &shape, &dtype);
            sharder_->register_encoded(kv.first, shape, dtype, std::move(encoded), model_->weight_ref(kv.first));
        }
    }

    void InitializeParameterSharder(bool register_existing_weights) {
        if (options_.shard_max_resident_mb <= 0) {
            return;
        }
        EnsureDirRecursive(work_dir_);
        ops::sharding::ShardConfig cfg;
        cfg.offload_dir = options_.shard_offload_dir.empty()
            ? JoinPath(work_dir_, "parameter_shards")
            : options_.shard_offload_dir;
        cfg.max_resident_bytes =
            static_cast<std::size_t>(options_.shard_max_resident_mb) * 1024ULL * 1024ULL;
        cfg.quantize_fp16_on_disk = options_.shard_quantize_fp16_on_disk;
        cfg.disk_quantization = shard_quant_mode_;

        sharder_ = std::make_unique<ops::sharding::ParameterSharder>(cfg);
        model_->set_parameter_sharder(sharder_.get());
        if (register_existing_weights) {
            RegisterGemmaBaseWeights(*model_, *sharder_);
            sharder_->offload_all();
        }
    }

    void RebuildDatasets(int seq_len) {
        current_seq_len_ = seq_len;
        EnsureDirRecursive(work_dir_);
        ops::WT2Config dataset_cfg;
        dataset_cfg.seq_len = seq_len;
        dataset_cfg.streaming_mode = false;
        dataset_cfg.shuffle_train = true;
        dataset_cfg.eos_id = eos_id_;
        dataset_cfg.pad_id = eos_id_;

        if (IsWikiTextRawDataset(options_)) {
            dataset_cfg.train_path = !options_.dataset_train_path.empty()
                ? options_.dataset_train_path
                : options_.dataset_csv;
            dataset_cfg.valid_path = !options_.dataset_valid_path.empty()
                ? options_.dataset_valid_path
                : dataset_cfg.train_path;
            dataset_cfg.test_path = !options_.dataset_test_path.empty()
                ? options_.dataset_test_path
                : dataset_cfg.valid_path;
            if (dataset_cfg.train_path.empty()) {
                throw std::runtime_error("wikitext_raw requires --dataset_train_path or --dataset_csv");
            }
        } else {
            train_jsonl_path_ = JoinPath(work_dir_, "fedavg_train.jsonl");
            valid_jsonl_path_ = JoinPath(work_dir_, "fedavg_valid.jsonl");
            WriteMaskedJsonl(dataset_.samples(), train_jsonl_path_, valid_jsonl_path_, seq_len);
            dataset_cfg.jsonl_train = train_jsonl_path_;
            dataset_cfg.jsonl_valid = valid_jsonl_path_;
        }

        std::function<std::vector<int32_t>(const std::string&)> selected_encode;
        if (IsWikiTextRawDataset(options_)) {
            selected_encode = [this](const std::string& text) { return tokenizer_->encode(text); };
        } else {
            selected_encode = [](const std::string&) { return std::vector<int32_t>{}; };
        }
        train_dataset_ = std::make_unique<ops::WikiText2Dataset>(dataset_cfg, selected_encode);
        eval_dataset_ = std::make_unique<ops::WikiText2Dataset>(dataset_cfg, selected_encode);
        train_dataset_->load(ops::Split::Train);
        eval_dataset_->load(ops::Split::Valid);
    }

    void WriteMaskedJsonl(
        const std::vector<MMLUSample>& samples,
        const std::string& train_path,
        const std::string& valid_path,
        int seq_len) const {
        EnsureParentDir(train_path);
        EnsureParentDir(valid_path);
        std::ofstream train_out(train_path, std::ios::trunc);
        std::ofstream valid_out(valid_path, std::ios::trunc);
        if (!train_out || !valid_out) {
            throw std::runtime_error("Failed to open local JSONL cache files");
        }

        for (const auto& sample : samples) {
            const std::string prompt =
                sample.Prompt();
            const auto prompt_ids = tokenizer_->encode(prompt);
            const auto answer_ids = tokenizer_->encode(options_.answer_prefix + sample.AnswerLabel());

            std::vector<int32_t> ids;
            ids.reserve(static_cast<std::size_t>(seq_len));
            ids.insert(ids.end(), prompt_ids.begin(), prompt_ids.end());
            ids.insert(ids.end(), answer_ids.begin(), answer_ids.end());
            ids.push_back(eos_id_);
            if (static_cast<int>(ids.size()) > seq_len) {
                continue;
            }

            std::vector<int32_t> mask;
            mask.reserve(ids.size());
            mask.insert(mask.end(), prompt_ids.size(), 0);
            mask.insert(mask.end(), answer_ids.size(), 1);
            mask.push_back(0);

            while (static_cast<int>(ids.size()) < seq_len) {
                ids.push_back(eos_id_);
                mask.push_back(0);
            }

            const std::string json_line = BuildJsonLine(ids, mask);
            train_out << json_line << "\n";
            valid_out << json_line << "\n";
        }
    }

    static std::string BuildJsonLine(const std::vector<int32_t>& ids, const std::vector<int32_t>& mask) {
        std::ostringstream out;
        out << "{\"ids\":[";
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i > 0) {
                out << ",";
            }
            out << ids[i];
        }
        out << "],\"mask\":[";
        for (std::size_t i = 0; i < mask.size(); ++i) {
            if (i > 0) {
                out << ",";
            }
            out << mask[i];
        }
        out << "]}";
        return out.str();
    }

    void ApplyParameters(const flwr::proto::Parameters& parameters) {
        if (parameters.tensors_size() == 0) {
            return;
        }
        auto trainable = injector_.get_trainable_params();
        if (parameters.tensors_size() != static_cast<int>(trainable.size())) {
            throw std::runtime_error(
                "Incoming adapter tensor count does not match local LoRA tensor count");
        }
        for (int index = 0; index < parameters.tensors_size(); ++index) {
            auto& tensor = trainable[static_cast<std::size_t>(index)];
            if (!tensor) {
                throw std::runtime_error("Encountered null LoRA tensor while applying parameters");
            }
            const ParsedNpyArray parsed = ParseNpy(parameters.tensors(index));
            if (parsed.dtype != NpyDType::kFloat32) {
                throw std::runtime_error("Incoming LoRA parameters must be float32");
            }

            std::vector<std::int64_t> local_shape;
            for (const auto dim : tensor->shape()) {
                local_shape.push_back(dim);
            }
            if (parsed.shape != local_shape) {
                throw std::runtime_error("Incoming LoRA parameter shape mismatch");
            }
            if (parsed.raw_bytes.size() != static_cast<std::size_t>(tensor->numel()) * sizeof(float)) {
                throw std::runtime_error("Incoming LoRA parameter byte size mismatch");
            }
            std::memcpy(tensor->data<float>(), parsed.raw_bytes.data(), parsed.raw_bytes.size());
            if (tensor->grad()) {
                tensor->zero_grad();
            }
        }
    }

    ClientOptions options_;
    ClientShardDataset dataset_;
    std::string work_dir_;
    std::unique_ptr<ops::HFTokenizer> tokenizer_;
    std::unique_ptr<ops::GemmaModel> model_;
    std::unique_ptr<ops::sharding::ParameterSharder> sharder_;
    ops::GemmaLoraInjector injector_;
    ops::GemmaLoraSpec lora_spec_;
    std::vector<std::string> parameter_names_;
    bool use_streamed_quant_load_ = false;
    ops::sharding::DiskQuantizationMode shard_quant_mode_ = ops::sharding::DiskQuantizationMode::None;
    std::unique_ptr<ops::WikiText2Dataset> train_dataset_;
    std::unique_ptr<ops::WikiText2Dataset> eval_dataset_;
    std::string train_jsonl_path_;
    std::string valid_jsonl_path_;
    int current_seq_len_ = -1;
    int32_t eos_id_ = -1;
};

class QwenFederatedTrainer final : public FederatedTrainer {
public:
    QwenFederatedTrainer(const ClientOptions& options, ClientShardDataset dataset)
        : options_(options),
          dataset_(std::move(dataset)),
          work_dir_(ResolveWorkDir(options)),
          qv_only_(QwenQvOnly(options)) {
        if (options_.model_dir.empty()) {
            throw std::runtime_error("backend=mft requires --model_dir");
        }
        if (!IsWikiTextRawDataset(options_) && dataset_.empty()) {
            throw std::runtime_error("MFT client dataset must not be empty");
        }

        tokenizer_ = std::make_unique<ops::HFTokenizer>(options_.model_dir, true);
        if (!tokenizer_->load()) {
            throw std::runtime_error("Failed to load HF tokenizer from model_dir: " + options_.model_dir);
        }

        const std::string config_path = JoinPath(options_.model_dir, "config.json");
        auto qcfg = ops::QwenConfig::from_pretrained(config_path);
        eos_id_ = qcfg.eos_token_id;
        if (eos_id_ < 0) {
            eos_id_ = ResolveGemmaTokenId(*tokenizer_, {"<|endoftext|>", "<|eos|>", "</s>"});
        }
        if (qcfg.pad_token_id < 0) {
            qcfg.pad_token_id = eos_id_;
        }

        model_ = std::make_unique<ops::QwenModel>(qcfg);
        LoadModelWeights();
        model_->init_lora(
            std::max(1, options_.lora_r),
            options_.lora_alpha,
            options_.lora_dropout,
            qv_only_);
        model_->freeze_base();
        parameter_names_ = BuildQwenParameterNames(*model_, qv_only_);
        if (parameter_names_.empty()) {
            throw std::runtime_error("Qwen LoRA parameter set is empty after injection");
        }
        RebuildDatasets(options_.max_seq_len);
    }

    flwr::proto::Parameters GetParameters() const override {
        flwr::proto::Parameters parameters;
        parameters.set_tensor_type("numpy.ndarray");
        auto trainable = GetOrderedQwenTrainableParams(*model_, qv_only_);
        for (std::size_t index = 0; index < trainable.size(); ++index) {
            const auto& tensor = trainable[index];
            if (!tensor) {
                throw std::runtime_error("Encountered null Qwen LoRA tensor while serializing parameters");
            }
            const bool is_a = (index % 2 == 0);
            const auto transposed = Transpose2DFloatTensor(tensor);
            const std::vector<std::int64_t> shape = {
                static_cast<std::int64_t>(tensor->shape()[1]),
                static_cast<std::int64_t>(tensor->shape()[0]),
            };
            (void)is_a;
            const auto blob = SerializeNpy(
                transposed.data(),
                transposed.size(),
                shape,
                NpyDType::kFloat32);
            parameters.add_tensors(blob.data(), static_cast<int>(blob.size()));
        }
        return parameters;
    }

    const std::vector<std::string>& ParameterNames() const override { return parameter_names_; }
    std::string BackendName() const override { return "mft"; }

    LocalFitSummary Fit(
        const flwr::proto::Parameters& global_parameters,
        int batch_size,
        int max_seq_len,
        int local_steps,
        int local_epochs,
        float learning_rate,
        float weight_decay,
        int /*server_round*/) override {
        if (max_seq_len != current_seq_len_) {
            RebuildDatasets(max_seq_len);
        }
        ApplyParameters(global_parameters);

        train_dataset_->reset_cursor();
        eval_dataset_->reset_cursor();

        ops::AdamWConfig opt_cfg;
        opt_cfg.learning_rate = learning_rate;
        opt_cfg.beta1 = 0.9f;
        opt_cfg.beta2 = 0.999f;
        opt_cfg.epsilon = 1e-8f;
        opt_cfg.weight_decay = weight_decay;
        opt_cfg.clip_grad_norm = options_.grad_clip_norm;
        ops::AdamW optimizer(opt_cfg);

        auto trainable = GetOrderedQwenTrainableParams(*model_, qv_only_);
        std::vector<ops::TensorPtr> reference_tensors;
        reference_tensors.reserve(trainable.size());
        for (const auto& tensor : trainable) {
            reference_tensors.push_back(tensor ? tensor->clone() : nullptr);
        }

        const auto start = std::chrono::steady_clock::now();
        const int effective_steps = std::max(1, local_steps);
        const int effective_epochs = std::max(0, local_epochs);
        int target_updates = effective_steps;
        if (effective_epochs > 0) {
            const int steps_per_epoch = std::max(
                1,
                static_cast<int>((train_dataset_->num_sequences() + static_cast<std::size_t>(batch_size) - 1) /
                                 static_cast<std::size_t>(batch_size)));
            target_updates = std::max(target_updates, steps_per_epoch * effective_epochs);
        }

        double loss_sum = 0.0;
        double objective_loss_sum = 0.0;
        double prox_term_sum = 0.0;
        std::int64_t examples_seen = 0;
        int updates = 0;
        double step_time_sum = 0.0;
        double max_step_time_sec = 0.0;
        std::vector<double> step_times_sec;
        std::vector<double> rss_samples_mb;
        ProcMemoryMetrics peak_memory = CurrentProcMemoryMetrics();
        const int grad_accum_steps = std::max(1, options_.grad_accum_steps);
        const int effective_batch_size = std::max(1, batch_size);
        const int micro_batch_size =
            std::max(1, (effective_batch_size + grad_accum_steps - 1) / grad_accum_steps);

        while (updates < target_updates) {
            const auto step_start = std::chrono::steady_clock::now();
            optimizer.zero_grad(trainable);
            double accum_total_loss = 0.0;
            double accum_base_loss = 0.0;
            double accum_prox_term = 0.0;
            int micro_steps = 0;

            for (int micro = 0; micro < grad_accum_steps; ++micro) {
                auto batch = train_dataset_->next_batch(static_cast<std::size_t>(micro_batch_size), true);
                if (!batch.input_ids) {
                    break;
                }
                auto logits = model_->forward(batch.input_ids, batch.attention_mask);
                auto base_loss = ops::lm_cross_entropy(logits, batch.labels, -100, "mean");
                auto total_loss = base_loss;
                float prox_term_value = 0.0f;
                if (options_.fedprox_mu > 0.0f) {
                    ops::TensorPtr prox_term = nullptr;
                    for (std::size_t index = 0; index < trainable.size(); ++index) {
                        const auto& param = trainable[index];
                        const auto& reference = reference_tensors[index];
                        if (!param || !reference) {
                            continue;
                        }
                        auto delta = ops::sub(param, reference);
                        auto sq = ops::mul(delta, delta);
                        auto term = ops::sum(sq);
                        prox_term = prox_term ? ops::add(prox_term, term) : term;
                    }
                    if (prox_term) {
                        prox_term_value = prox_term->data<float>()[0];
                        total_loss = ops::add(total_loss, ops::mul(prox_term, 0.5f * options_.fedprox_mu));
                    }
                }
                auto scaled = ops::mul(total_loss, 1.0f / static_cast<float>(grad_accum_steps));
                scaled->backward();
                accum_total_loss += static_cast<double>(total_loss->data<float>()[0]);
                accum_base_loss += static_cast<double>(base_loss->data<float>()[0]);
                accum_prox_term += static_cast<double>(prox_term_value * 0.5f * options_.fedprox_mu);
                examples_seen += batch.input_ids->shape()[0];
                ++micro_steps;
            }

            if (micro_steps <= 0) {
                break;
            }

            std::vector<ops::TensorPtr> gradients;
            gradients.reserve(trainable.size());
            for (const auto& tensor : trainable) {
                gradients.push_back(tensor ? tensor->grad() : nullptr);
            }
            optimizer.step(trainable, gradients);
            for (auto& tensor : trainable) {
                if (tensor) {
                    tensor->zero_grad();
                }
            }
            ops::MemoryManager::instance().force_cleanup();

            const double mean_total_loss = accum_total_loss / static_cast<double>(micro_steps);
            const double mean_base_loss = accum_base_loss / static_cast<double>(micro_steps);
            const double mean_prox_term = accum_prox_term / static_cast<double>(micro_steps);
            loss_sum += mean_base_loss;
            objective_loss_sum += mean_total_loss;
            prox_term_sum += mean_prox_term;
            ++updates;

            const double step_time_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
            step_time_sum += step_time_sec;
            max_step_time_sec = std::max(max_step_time_sec, step_time_sec);
            step_times_sec.push_back(step_time_sec);
            const ProcMemoryMetrics current_memory = CurrentProcMemoryMetrics();
            if (current_memory.rss_mb >= 0.0) {
                rss_samples_mb.push_back(current_memory.rss_mb);
            }
            peak_memory.rss_mb = std::max(peak_memory.rss_mb, current_memory.rss_mb);
            peak_memory.hwm_mb = std::max(peak_memory.hwm_mb, current_memory.hwm_mb);
            peak_memory.swap_mb = std::max(peak_memory.swap_mb, current_memory.swap_mb);
        }
        const auto end = std::chrono::steady_clock::now();

        LocalFitSummary summary;
        summary.num_examples = std::max<std::int64_t>(1, examples_seen);
        summary.steps_completed = updates;
        summary.epochs_completed = effective_epochs;
        summary.mean_loss = updates > 0 ? (loss_sum / static_cast<double>(updates)) : 0.0;
        summary.mean_objective_loss = updates > 0 ? (objective_loss_sum / static_cast<double>(updates)) : 0.0;
        summary.mean_prox_term = updates > 0 ? (prox_term_sum / static_cast<double>(updates)) : 0.0;
        summary.train_time_sec = std::chrono::duration<double>(end - start).count();
        summary.avg_rss_mb = rss_samples_mb.empty()
            ? peak_memory.rss_mb
            : (std::accumulate(rss_samples_mb.begin(), rss_samples_mb.end(), 0.0) /
               static_cast<double>(rss_samples_mb.size()));
        summary.client_rss_mb = peak_memory.rss_mb;
        summary.client_hwm_mb = peak_memory.hwm_mb;
        summary.client_swap_mb = peak_memory.swap_mb;
        summary.mean_step_time_sec = updates > 0 ? (step_time_sum / static_cast<double>(updates)) : 0.0;
        summary.max_step_time_sec = max_step_time_sec;
        summary.transmitted_bytes = ParameterBytes(GetParameters());
        summary.step_times_sec_json = JsonEncodeDoubles(step_times_sec);
        return summary;
    }

private:
    void LoadModelWeights() {
        ops::SafeTensorsReader reader(JoinPath(options_.model_dir, "model.safetensors"));
        reader.parse_header();
        const auto mapping = ops::QwenKeyMapper::generate_qwen_mapping(model_->config().num_hidden_layers);
        ops::SafeTensorsLoadOptions load_opts;
        load_opts.verbose = false;
        load_opts.transpose_linear = true;
        const auto tensors = reader.load_tensors_mapped(mapping, load_opts);
        for (const auto& kv : tensors) {
            model_->assign_weight(kv.first, kv.second);
        }
    }

    void RebuildDatasets(int seq_len) {
        current_seq_len_ = seq_len;
        EnsureDirRecursive(work_dir_);
        ops::WT2Config dataset_cfg;
        dataset_cfg.seq_len = seq_len;
        dataset_cfg.streaming_mode = false;
        dataset_cfg.shuffle_train = true;
        dataset_cfg.eos_id = eos_id_;
        dataset_cfg.pad_id = eos_id_;

        if (IsWikiTextRawDataset(options_)) {
            dataset_cfg.train_path = !options_.dataset_train_path.empty()
                ? options_.dataset_train_path
                : options_.dataset_csv;
            dataset_cfg.valid_path = !options_.dataset_valid_path.empty()
                ? options_.dataset_valid_path
                : dataset_cfg.train_path;
            dataset_cfg.test_path = !options_.dataset_test_path.empty()
                ? options_.dataset_test_path
                : dataset_cfg.valid_path;
            if (dataset_cfg.train_path.empty()) {
                throw std::runtime_error("wikitext_raw requires --dataset_train_path or --dataset_csv");
            }
        } else {
            train_jsonl_path_ = JoinPath(work_dir_, "fedavg_train.jsonl");
            valid_jsonl_path_ = JoinPath(work_dir_, "fedavg_valid.jsonl");
            WriteMaskedJsonl(dataset_.samples(), train_jsonl_path_, valid_jsonl_path_, seq_len);
            dataset_cfg.jsonl_train = train_jsonl_path_;
            dataset_cfg.jsonl_valid = valid_jsonl_path_;
        }

        std::function<std::vector<int32_t>(const std::string&)> selected_encode;
        if (IsWikiTextRawDataset(options_)) {
            selected_encode = [this](const std::string& text) { return tokenizer_->encode(text); };
        } else {
            selected_encode = [](const std::string&) { return std::vector<int32_t>{}; };
        }
        train_dataset_ = std::make_unique<ops::WikiText2Dataset>(dataset_cfg, selected_encode);
        eval_dataset_ = std::make_unique<ops::WikiText2Dataset>(dataset_cfg, selected_encode);
        train_dataset_->load(ops::Split::Train);
        eval_dataset_->load(ops::Split::Valid);
    }

    void WriteMaskedJsonl(
        const std::vector<MMLUSample>& samples,
        const std::string& train_path,
        const std::string& valid_path,
        int seq_len) const {
        EnsureParentDir(train_path);
        EnsureParentDir(valid_path);
        std::ofstream train_out(train_path, std::ios::trunc);
        std::ofstream valid_out(valid_path, std::ios::trunc);
        if (!train_out || !valid_out) {
            throw std::runtime_error("Failed to open local JSONL cache files");
        }

        for (const auto& sample : samples) {
            const std::string prompt = sample.Prompt();
            const auto prompt_ids = tokenizer_->encode(prompt);
            const auto answer_ids = tokenizer_->encode(options_.answer_prefix + sample.AnswerLabel());

            std::vector<int32_t> ids;
            ids.reserve(static_cast<std::size_t>(seq_len));
            ids.insert(ids.end(), prompt_ids.begin(), prompt_ids.end());
            ids.insert(ids.end(), answer_ids.begin(), answer_ids.end());
            ids.push_back(eos_id_);
            if (static_cast<int>(ids.size()) > seq_len) {
                continue;
            }

            std::vector<int32_t> mask;
            mask.reserve(ids.size());
            mask.insert(mask.end(), prompt_ids.size(), 0);
            mask.insert(mask.end(), answer_ids.size(), 1);
            mask.push_back(0);

            while (static_cast<int>(ids.size()) < seq_len) {
                ids.push_back(eos_id_);
                mask.push_back(0);
            }

            const std::string json_line = BuildJsonLine(ids, mask);
            train_out << json_line << "\n";
            valid_out << json_line << "\n";
        }
    }

    static std::string BuildJsonLine(const std::vector<int32_t>& ids, const std::vector<int32_t>& mask) {
        std::ostringstream out;
        out << "{\"ids\":[";
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i > 0) {
                out << ",";
            }
            out << ids[i];
        }
        out << "],\"mask\":[";
        for (std::size_t i = 0; i < mask.size(); ++i) {
            if (i > 0) {
                out << ",";
            }
            out << mask[i];
        }
        out << "]}";
        return out.str();
    }

    void ApplyParameters(const flwr::proto::Parameters& parameters) {
        if (parameters.tensors_size() == 0) {
            return;
        }
        auto trainable = GetOrderedQwenTrainableParams(*model_, qv_only_);
        if (parameters.tensors_size() != static_cast<int>(trainable.size())) {
            throw std::runtime_error("Incoming adapter tensor count does not match local Qwen LoRA tensor count");
        }
        for (int index = 0; index < parameters.tensors_size(); ++index) {
            auto& tensor = trainable[static_cast<std::size_t>(index)];
            if (!tensor) {
                throw std::runtime_error("Encountered null Qwen LoRA tensor while applying parameters");
            }
            const ParsedNpyArray parsed = ParseNpy(parameters.tensors(index));
            if (parsed.dtype != NpyDType::kFloat32) {
                throw std::runtime_error("Incoming Qwen LoRA parameters must be float32");
            }
            CopyMaybeTransposed2DFloatTensor(parsed, tensor, true);
            if (tensor->grad()) {
                tensor->zero_grad();
            }
        }
    }

    ClientOptions options_;
    ClientShardDataset dataset_;
    std::string work_dir_;
    bool qv_only_ = false;
    std::unique_ptr<ops::HFTokenizer> tokenizer_;
    std::unique_ptr<ops::QwenModel> model_;
    std::vector<std::string> parameter_names_;
    std::unique_ptr<ops::WikiText2Dataset> train_dataset_;
    std::unique_ptr<ops::WikiText2Dataset> eval_dataset_;
    std::string train_jsonl_path_;
    std::string valid_jsonl_path_;
    int current_seq_len_ = -1;
    int32_t eos_id_ = -1;
};

}  // namespace

std::unique_ptr<FederatedTrainer> CreateFederatedTrainer(
    const ClientOptions& options,
    ClientShardDataset dataset) {
    if (options.backend == "mft") {
        if (DetectModelFamily(options.model_dir) == ModelFamily::Qwen) {
            return std::make_unique<QwenFederatedTrainer>(options, std::move(dataset));
        }
        return std::make_unique<GemmaFederatedTrainer>(options, std::move(dataset));
    }
    return std::make_unique<MockFederatedTrainer>(options);
}

}  // namespace lshaped
