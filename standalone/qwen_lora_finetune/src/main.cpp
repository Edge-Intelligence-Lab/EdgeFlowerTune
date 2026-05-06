#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "finetune_ops/core/autograd_engine.h"
#include "finetune_ops/core/lm_loss.h"
#include "finetune_ops/core/memory_manager.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/tokenizer_hf.h"
#include "finetune_ops/data/wikitext2_dataset.h"
#include "finetune_ops/graph/qwen_lora_saver.h"
#include "finetune_ops/graph/qwen_model.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/optim/adam.h"
#include "finetune_ops/optim/adamw.h"

using namespace ops;

namespace {

struct CliOptions {
    std::string model_dir = "Qwen2.5-0.5B";
    std::string data_dir = "data/wikitext2/wikitext-2-raw";
    std::string jsonl_train;
    std::string jsonl_valid;
    std::string pretokenized_path;
    std::string pretokenized_meta;
    std::string output_dir = "./qwen_lora";
    int epochs = 0;
    int steps = 0;
    int seq_len = 128;
    int batch = 8;
    int grad_accum = 1;
    float learning_rate = 2e-4f;
    int warmup_steps = 0;
    std::string lr_scheduler = "cosine";
    float max_grad_norm = 1.0f;
    float data_fraction = 1.0f;
    float weight_decay = 0.0f;
    std::string optimizer = "adamw";
    std::string target_mode = "qv";
    int lora_r = 8;
    float lora_alpha = 16.0f;
    float lora_dropout = 0.0f;
    int logging_steps = 1;
    int eval_steps = 0;
    int eval_batches = 50;
    int save_every = 0;
    uint64_t seed = 42;
    std::string resume_from;
    bool shuffle_train = false;
    bool dump_first_batch = false;
    int dump_first_tokens = 16;
    std::string dump_first_batch_path;
    bool exit_after_dump = false;
    std::string fixed_batch_bin;
    int fixed_batch_steps = 0;
    bool print_step0 = false;
    std::string tokenizer_json;
    std::string eval_out;
};

struct FixedBatch {
    int32_t B = 0;
    int32_t S = 0;
    std::vector<int32_t> ids;
    std::vector<float> attn;
};

void append_jsonl(const std::string& path, const std::string& json_line) {
    if (path.empty()) return;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream out(path, std::ios::app);
    if (out) out << json_line << "\n";
}

std::string make_checkpoint_path(const std::string& output_dir, int step) {
    std::ostringstream ss;
    ss << output_dir << "/lora_step" << step << ".safetensors";
    return ss.str();
}

float make_scheduler(int step,
                     int total_steps,
                     int warmup_steps,
                     float base_lr,
                     const std::string& mode) {
    constexpr float kPi = 3.14159265358979323846f;
    const int step_1indexed = step + 1;
    if (warmup_steps > 0 && step_1indexed <= warmup_steps) {
        return base_lr * static_cast<float>(step_1indexed) /
               static_cast<float>(std::max(1, warmup_steps));
    }
    const int remain = std::max(1, total_steps - warmup_steps);
    float progress = static_cast<float>(step_1indexed - warmup_steps) /
                     static_cast<float>(remain);
    progress = std::min(std::max(progress, 0.0f), 1.0f);
    if (mode == "linear") {
        return base_lr * (1.0f - progress);
    }
    return base_lr * 0.5f * (1.0f + std::cos(kPi * progress));
}

bool parse_bool_value(const std::string& v) {
    return !(v == "0" || v == "false" || v == "False");
}

FixedBatch load_fixed_batch_bin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open fixed_batch_bin: " + path);
    }
    FixedBatch fb;
    in.read(reinterpret_cast<char*>(&fb.B), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&fb.S), sizeof(int32_t));
    if (fb.B <= 0 || fb.S <= 0) {
        throw std::runtime_error("Invalid fixed batch shape in: " + path);
    }
    const size_t n = static_cast<size_t>(fb.B) * static_cast<size_t>(fb.S);
    fb.ids.resize(n);
    fb.attn.resize(n);
    in.read(reinterpret_cast<char*>(fb.ids.data()),
            static_cast<std::streamsize>(n * sizeof(int32_t)));
    in.read(reinterpret_cast<char*>(fb.attn.data()),
            static_cast<std::streamsize>(n * sizeof(float)));
    if (!in) {
        throw std::runtime_error("Failed to read fixed batch payload from: " + path);
    }
    return fb;
}

uint64_t fnv1a_hash_int32(const int32_t* data, size_t n) {
    const uint64_t kOffset = 1469598103934665603ull;
    const uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffset;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t v = static_cast<uint32_t>(data[i]);
        for (int b = 0; b < 4; ++b) {
            const uint8_t byte = (v >> (8 * b)) & 0xFF;
            hash ^= byte;
            hash *= kPrime;
        }
    }
    return hash;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char ch : s) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += ch; break;
        }
    }
    return out;
}

int64_t count_tokens(const TensorPtr& mask) {
    if (!mask) return 0;
    const float* ptr = mask->data<float>();
    int64_t total = 0;
    for (int64_t i = 0; i < mask->numel(); ++i) {
        if (ptr[i] > 0.5f) total++;
    }
    return total;
}

struct NoGradGuard {
    NoGradGuard() { ops::autograd::Engine::instance().set_enabled(false); }
    ~NoGradGuard() {
        ops::autograd::Engine::instance().clear_graph();
        ops::autograd::Engine::instance().set_enabled(true);
    }
};

CliOptions parse_cli(int argc, char** argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto get_val = [&](const std::string& key) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + key);
            }
            return std::string(argv[++i]);
        };
        if (arg == "--model_dir") opts.model_dir = get_val(arg);
        else if (arg == "--data_dir") opts.data_dir = get_val(arg);
        else if (arg == "--jsonl_train") opts.jsonl_train = get_val(arg);
        else if (arg == "--jsonl_valid") opts.jsonl_valid = get_val(arg);
        else if (arg == "--pretokenized_path") opts.pretokenized_path = get_val(arg);
        else if (arg == "--pretokenized_meta") opts.pretokenized_meta = get_val(arg);
        else if (arg == "--output_dir") opts.output_dir = get_val(arg);
        else if (arg == "--epochs") opts.epochs = std::stoi(get_val(arg));
        else if (arg == "--steps" || arg == "--max_steps") opts.steps = std::stoi(get_val(arg));
        else if (arg == "--seq_len") opts.seq_len = std::stoi(get_val(arg));
        else if (arg == "--batch" || arg == "--batch_size") opts.batch = std::stoi(get_val(arg));
        else if (arg == "--grad_accum" || arg == "--grad_accum_steps") opts.grad_accum = std::stoi(get_val(arg));
        else if (arg == "--learning_rate" || arg == "--lr") opts.learning_rate = std::stof(get_val(arg));
        else if (arg == "--warmup_steps") opts.warmup_steps = std::stoi(get_val(arg));
        else if (arg == "--lr_scheduler") opts.lr_scheduler = get_val(arg);
        else if (arg == "--max_grad_norm") opts.max_grad_norm = std::stof(get_val(arg));
        else if (arg == "--data_fraction") opts.data_fraction = std::stof(get_val(arg));
        else if (arg == "--weight_decay") opts.weight_decay = std::stof(get_val(arg));
        else if (arg == "--optimizer") opts.optimizer = get_val(arg);
        else if (arg == "--target_mode") opts.target_mode = get_val(arg);
        else if (arg == "--qv_only") opts.target_mode = "qv";
        else if (arg == "--qkvo" || arg == "--full") opts.target_mode = "full";
        else if (arg == "--lora_r") opts.lora_r = std::stoi(get_val(arg));
        else if (arg == "--lora_alpha") opts.lora_alpha = std::stof(get_val(arg));
        else if (arg == "--lora_dropout") opts.lora_dropout = std::stof(get_val(arg));
        else if (arg == "--logging_steps" || arg == "--log_interval") opts.logging_steps = std::stoi(get_val(arg));
        else if (arg == "--eval_steps" || arg == "--eval_interval") opts.eval_steps = std::stoi(get_val(arg));
        else if (arg == "--eval_batches") opts.eval_batches = std::stoi(get_val(arg));
        else if (arg == "--save_every") opts.save_every = std::stoi(get_val(arg));
        else if (arg == "--seed") opts.seed = static_cast<uint64_t>(std::stoull(get_val(arg)));
        else if (arg == "--resume_from") opts.resume_from = get_val(arg);
        else if (arg == "--shuffle") opts.shuffle_train = true;
        else if (arg == "--no_shuffle") opts.shuffle_train = false;
        else if (arg == "--shuffle_train") opts.shuffle_train = parse_bool_value(get_val(arg));
        else if (arg == "--dump_first_batch") opts.dump_first_batch = true;
        else if (arg == "--dump_first_tokens") opts.dump_first_tokens = std::stoi(get_val(arg));
        else if (arg == "--dump_first_batch_path") opts.dump_first_batch_path = get_val(arg);
        else if (arg == "--exit_after_dump") opts.exit_after_dump = true;
        else if (arg == "--fixed_batch_bin") opts.fixed_batch_bin = get_val(arg);
        else if (arg == "--fixed_batch_steps") opts.fixed_batch_steps = std::stoi(get_val(arg));
        else if (arg == "--print_step0") opts.print_step0 = parse_bool_value(get_val(arg));
        else if (arg == "--tokenizer_json") opts.tokenizer_json = get_val(arg);
        else if (arg == "--eval_out") opts.eval_out = get_val(arg);
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (opts.steps <= 0 && opts.epochs <= 0) {
        opts.epochs = 1;
    }
    if (opts.optimizer != "adam" && opts.optimizer != "adamw") {
        throw std::runtime_error("Unsupported optimizer: " + opts.optimizer + " (expected adam or adamw)");
    }
    if (opts.eval_out.empty() && !opts.output_dir.empty()) {
        opts.eval_out = opts.output_dir + "/eval.jsonl";
    }
    return opts;
}

void dump_batch(const Batch& batch,
                int dump_first_tokens,
                const std::string& dump_path) {
    auto ids = batch.input_ids->data<int32_t>();
    const int64_t B = batch.input_ids->size(0);
    const int64_t S = batch.input_ids->size(1);
    const int tok_n = std::max(0, std::min(dump_first_tokens, static_cast<int>(S)));

    std::cout << "[Dump] first_batch shape=(" << B << "," << S << ")\n";
    std::cout << "[Dump] sample0 first" << tok_n << "=[";
    for (int i = 0; i < tok_n; ++i) {
        if (i) std::cout << ",";
        std::cout << ids[i];
    }
    std::cout << "]\n";
    if (B > 1) {
        std::cout << "[Dump] sample1 first" << tok_n << "=[";
        const int64_t base = S;
        for (int i = 0; i < tok_n; ++i) {
            if (i) std::cout << ",";
            std::cout << ids[base + i];
        }
        std::cout << "]\n";
    }
    const uint64_t hash = fnv1a_hash_int32(ids, static_cast<size_t>(B * S));
    std::cout << "[Dump] batch_hash_fnv1a64=" << hash << "\n";

    if (!dump_path.empty()) {
        const auto parent = std::filesystem::path(dump_path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        auto attn = batch.attention_mask->data<float>();
        std::ofstream ofs(dump_path, std::ios::binary);
        if (!ofs) {
            std::cerr << "[Dump] Failed to open " << dump_path << std::endl;
            return;
        }
        const int32_t b32 = static_cast<int32_t>(B);
        const int32_t s32 = static_cast<int32_t>(S);
        ofs.write(reinterpret_cast<const char*>(&b32), sizeof(int32_t));
        ofs.write(reinterpret_cast<const char*>(&s32), sizeof(int32_t));
        ofs.write(reinterpret_cast<const char*>(ids), sizeof(int32_t) * static_cast<size_t>(B * S));
        ofs.write(reinterpret_cast<const char*>(attn), sizeof(float) * static_cast<size_t>(B * S));
        std::cout << "[Dump] Wrote first batch to " << dump_path << std::endl;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto cli = parse_cli(argc, argv);
        const bool use_jsonl = !cli.jsonl_train.empty();
        const bool use_pretokenized = !cli.pretokenized_path.empty();
        const bool qv_only = (cli.target_mode == "qv");

        std::cout << "\n========== Qwen LoRA Finetune ==========\n";
        std::cout << "[Config]\n";
        std::cout << "  model_dir       : " << cli.model_dir << "\n";
        if (use_pretokenized) {
            std::cout << "  data_source     : pretokenized stream\n";
            std::cout << "  pretokenized    : " << cli.pretokenized_path << "\n";
            if (!cli.pretokenized_meta.empty()) {
                std::cout << "  pretokenized_meta: " << cli.pretokenized_meta << "\n";
            }
        } else if (use_jsonl) {
            std::cout << "  data_source     : JSONL(masked)\n";
            std::cout << "  jsonl_train     : " << cli.jsonl_train << "\n";
            std::cout << "  jsonl_valid     : " << (cli.jsonl_valid.empty() ? cli.jsonl_train : cli.jsonl_valid) << "\n";
        } else {
            std::cout << "  data_source     : WikiText-2 raw\n";
            std::cout << "  data_dir        : " << cli.data_dir << "\n";
        }
        std::cout << "  output_dir      : " << cli.output_dir << "\n";
        std::cout << "  epochs/steps    : " << cli.epochs << "/" << cli.steps << "\n";
        std::cout << "  batch/accum     : " << cli.batch << "/" << cli.grad_accum << "\n";
        std::cout << "  seq_len         : " << cli.seq_len << "\n";
        std::cout << "  lr/scheduler    : " << cli.learning_rate << "/" << cli.lr_scheduler << "\n";
        std::cout << "  warmup_steps    : " << cli.warmup_steps << "\n";
        std::cout << "  max_grad_norm   : " << cli.max_grad_norm << "\n";
        std::cout << "  optimizer       : " << cli.optimizer << "\n";
        std::cout << "  lora r/a/drop   : " << cli.lora_r << "/" << cli.lora_alpha << "/" << cli.lora_dropout << "\n";
        std::cout << "  target_mode     : " << cli.target_mode << "\n";
        std::cout << "  shuffle_train   : " << (cli.shuffle_train ? "true" : "false") << "\n";
        std::cout << "  logging/eval    : " << cli.logging_steps << "/" << cli.eval_steps << "\n";
        std::cout << "  save_every      : " << cli.save_every << "\n";
        std::cout << "  tokenizer       : hf(native)\n";

        std::filesystem::create_directories(cli.output_dir);
        if (!cli.eval_out.empty()) {
            std::filesystem::create_directories(std::filesystem::path(cli.eval_out).parent_path());
        }

        std::unique_ptr<HFTokenizer> hf_tok;
        std::function<std::vector<int32_t>(const std::string&)> encode_fn;

        QwenConfig qcfg = QwenConfig::from_pretrained(cli.model_dir + "/config.json");
        int eos_id = qcfg.eos_token_id;
        int pad_id = qcfg.pad_token_id;
        const bool use_model_dir = cli.tokenizer_json.empty();
        const std::string tok_path = use_model_dir ? cli.model_dir : cli.tokenizer_json;
        hf_tok = std::make_unique<HFTokenizer>(tok_path, use_model_dir);
        if (!hf_tok->load()) {
            throw std::runtime_error(
                "Failed to load HF tokenizer natively. Native HF tokenizer is now the standard path; "
                "rebuild with -DQWEN_USE_HF_TOKENIZERS=ON.");
        }
        std::cout << "[Tokenizer] HF native tokenizer loaded"
                  << (use_model_dir ? " from model_dir" : " from tokenizer_json")
                  << "\n";
        if (eos_id < 0) {
            for (auto t : {"<|endoftext|>", "<|eos|>", "</s>"}) {
                int id = hf_tok->token_to_id(t);
                if (id >= 0) {
                    eos_id = id;
                    break;
                }
            }
        }
        if (pad_id < 0) pad_id = eos_id;
        encode_fn = [tok = hf_tok.get()](const std::string& text) {
            return tok->encode(text);
        };
        QwenModel model(qcfg);

        SafeTensorsReader reader(cli.model_dir + "/model.safetensors");
        reader.parse_header();
        auto mapping = QwenKeyMapper::generate_qwen_mapping(qcfg.num_hidden_layers);
        SafeTensorsLoadOptions load_opts;
        load_opts.verbose = false;
        load_opts.transpose_linear = true;
        auto tensors = reader.load_tensors_mapped(mapping, load_opts);
        for (auto& kv : tensors) {
            model.assign_weight(kv.first, kv.second);
        }

        QwenLoraMetadata lora_meta;
        lora_meta.rank = cli.lora_r;
        lora_meta.alpha = cli.lora_alpha;
        lora_meta.dropout = cli.lora_dropout;
        lora_meta.qv_only = qv_only;

        if (!cli.resume_from.empty()) {
            QwenLoraMetadata file_meta = lora_meta;
            if (QwenLoraSaver::parse_metadata(cli.resume_from, file_meta)) {
                lora_meta = file_meta;
            }
        }

        model.init_lora(lora_meta.rank, lora_meta.alpha, lora_meta.dropout, lora_meta.qv_only);
        model.freeze_base();
        if (!cli.resume_from.empty()) {
            std::cout << "[Resume] loading " << cli.resume_from << std::endl;
            QwenLoraSaver::load_safetensors(cli.resume_from, model, lora_meta);
        }
        auto trainable = model.get_lora_parameters();
        std::cout << "[LoRA] trainable tensors: " << trainable.size() << std::endl;

        auto make_dataset = [&](Split split) -> std::unique_ptr<WikiText2Dataset> {
            WT2Config cfg;
            cfg.seq_len = cli.seq_len;
            cfg.stride = -1;
            cfg.eos_id = eos_id;
            cfg.pad_id = pad_id;
            cfg.insert_eos_between_lines = true;
            cfg.drop_last = (split == Split::Train);
            cfg.seed = cli.seed;
            cfg.shuffle_train = (split == Split::Train) ? cli.shuffle_train : false;
            cfg.streaming_mode = false;
            cfg.data_fraction = (split == Split::Train) ? cli.data_fraction : 1.0f;
            if (use_pretokenized) {
                cfg.pretokenized_path = cli.pretokenized_path;
                cfg.pretokenized_meta = cli.pretokenized_meta;
            } else if (use_jsonl) {
                cfg.jsonl_train = cli.jsonl_train;
                cfg.jsonl_valid = cli.jsonl_valid.empty() ? cli.jsonl_train : cli.jsonl_valid;
                cfg.jsonl_test = cfg.jsonl_valid;
            } else {
                cfg.train_path = cli.data_dir + "/wiki.train.raw";
                cfg.valid_path = cli.data_dir + "/wiki.valid.raw";
                cfg.test_path = cli.data_dir + "/wiki.test.raw";
            }

            auto ds = std::make_unique<WikiText2Dataset>(cfg, encode_fn);
            ds->load(split);
            if (split == Split::Train) {
                if (cli.shuffle_train) ds->shuffle();
                else ds->reset_cursor();
            } else {
                ds->reset_cursor();
            }
            return ds;
        };

        auto train_dataset = make_dataset(Split::Train);
        auto valid_dataset = make_dataset(Split::Valid);

        FixedBatch fixed_storage;
        Batch fixed_batch;
        bool use_fixed_batch = false;
        if (!cli.fixed_batch_bin.empty()) {
            fixed_storage = load_fixed_batch_bin(cli.fixed_batch_bin);
            use_fixed_batch = true;
            fixed_batch.input_ids = from_blob(
                fixed_storage.ids.data(),
                {static_cast<int64_t>(fixed_storage.B), static_cast<int64_t>(fixed_storage.S)},
                kInt32, kCPU);
            fixed_batch.attention_mask = from_blob(
                fixed_storage.attn.data(),
                {static_cast<int64_t>(fixed_storage.B), static_cast<int64_t>(fixed_storage.S)},
                kFloat32, kCPU);
            fixed_batch.labels = from_blob(
                fixed_storage.ids.data(),
                {static_cast<int64_t>(fixed_storage.B), static_cast<int64_t>(fixed_storage.S)},
                kInt32, kCPU);
            if (cli.fixed_batch_steps > 0) {
                cli.steps = cli.fixed_batch_steps;
            }
        }

        if (cli.dump_first_batch) {
            Batch batch0 = use_fixed_batch ? fixed_batch : train_dataset->get_batch(0, static_cast<size_t>(cli.batch));
            dump_batch(batch0, cli.dump_first_tokens, cli.dump_first_batch_path);
            if (cli.exit_after_dump) {
                std::cout << "[Dump] exit_after_dump=1, stopping before training." << std::endl;
                return 0;
            }
        }

        if (cli.print_step0) {
            NoGradGuard guard;
            Batch batch0 = use_fixed_batch ? fixed_batch : train_dataset->get_batch(0, static_cast<size_t>(cli.batch));
            auto logits0 = model.forward(batch0.input_ids, batch0.attention_mask);
            auto loss0 = lm_cross_entropy(logits0, batch0.labels, -100, "mean");
            const float loss_val = loss0->data<float>()[0];
            const float ppl = perplexity_from_loss(loss_val);
            std::cout << "[Step0] loss " << std::fixed << std::setprecision(6) << loss_val
                      << " ppl " << std::setprecision(2) << ppl
                      << " tokens " << count_tokens(batch0.attention_mask) << std::endl;
        }

        const size_t num_seqs = train_dataset->num_sequences();
        const size_t total_micro_batches = use_fixed_batch
                                               ? 1
                                               : (num_seqs / static_cast<size_t>(std::max(1, cli.batch)));
        const size_t steps_per_epoch = (total_micro_batches + static_cast<size_t>(cli.grad_accum) - 1) / static_cast<size_t>(cli.grad_accum);

        int total_steps = cli.steps;
        if (total_steps <= 0) {
            total_steps = static_cast<int>(steps_per_epoch) * std::max(1, cli.epochs);
        }
        std::cout << "[Dataset] train_sequences=" << num_seqs
                  << " valid_sequences=" << valid_dataset->num_sequences() << "\n";
        std::cout << "[Schedule] total_steps=" << total_steps
                  << " steps_per_epoch=" << steps_per_epoch
                  << " effective_batch=" << (cli.batch * cli.grad_accum) << std::endl;

        std::unique_ptr<Optimizer> optimizer;
        if (cli.optimizer == "adamw") {
            AdamWConfig opt_cfg;
            opt_cfg.learning_rate = cli.learning_rate;
            opt_cfg.beta1 = 0.9f;
            opt_cfg.beta2 = 0.999f;
            opt_cfg.epsilon = 1e-8f;
            opt_cfg.weight_decay = cli.weight_decay;
            opt_cfg.clip_grad_norm = cli.max_grad_norm;
            optimizer = std::make_unique<AdamW>(opt_cfg);
        } else {
            AdamConfig opt_cfg;
            opt_cfg.learning_rate = cli.learning_rate;
            opt_cfg.beta1 = 0.9f;
            opt_cfg.beta2 = 0.999f;
            opt_cfg.epsilon = 1e-8f;
            opt_cfg.weight_decay = cli.weight_decay;
            opt_cfg.clip_grad_norm = cli.max_grad_norm;
            optimizer = std::make_unique<Adam>(opt_cfg);
        }

        auto clip_and_get_grad_norm = [&](float max_norm) -> float {
            double norm_sq = 0.0;
            for (const auto& p : trainable) {
                auto g = p ? p->grad() : nullptr;
                if (!g) continue;
                const float* gd = g->data<float>();
                for (int64_t i = 0; i < g->numel(); ++i) {
                    norm_sq += static_cast<double>(gd[i]) * static_cast<double>(gd[i]);
                }
            }
            float total_norm = static_cast<float>(std::sqrt(norm_sq));
            if (max_norm > 0.0f && total_norm > max_norm) {
                const float scale = max_norm / (total_norm + 1e-6f);
                for (const auto& p : trainable) {
                    auto g = p ? p->grad() : nullptr;
                    if (!g) continue;
                    float* gd = g->data<float>();
                    for (int64_t i = 0; i < g->numel(); ++i) gd[i] *= scale;
                }
                total_norm = max_norm;
            }
            return total_norm;
        };

        auto evaluate_valid = [&]() -> float {
            NoGradGuard guard;
            valid_dataset->reset_cursor();
            double loss_sum = 0.0;
            int batches = 0;
            const int max_batches = cli.eval_batches > 0
                                        ? cli.eval_batches
                                        : static_cast<int>((valid_dataset->num_sequences() + static_cast<size_t>(cli.batch) - 1) / static_cast<size_t>(cli.batch));
            while (batches < max_batches) {
                auto batch = valid_dataset->next_batch(static_cast<size_t>(cli.batch), false);
                if (!batch.input_ids) break;
                auto logits = model.forward(batch.input_ids, batch.attention_mask);
                auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
                loss_sum += static_cast<double>(loss->data<float>()[0]);
                batches++;
            }
            valid_dataset->reset_cursor();
            if (batches == 0) return std::numeric_limits<float>::infinity();
            return std::exp(static_cast<float>(loss_sum / static_cast<double>(batches)));
        };

        auto save_adapter = [&](const std::string& path) {
            std::filesystem::create_directories(std::filesystem::path(path).parent_path());
            QwenLoraSaver::save_safetensors(path, model, lora_meta);
        };

        float ema_loss = 0.0f;
        bool ema_init = false;
        int64_t total_tokens = 0;
        float best_valid_ppl = std::numeric_limits<float>::infinity();

        std::cout << "[Train] start" << std::endl;
        for (int step = 0; step < total_steps; ++step) {
            double accum_loss = 0.0;
            int64_t accum_tokens = 0;
            optimizer->zero_grad(trainable);

            for (int acc = 0; acc < std::max(1, cli.grad_accum); ++acc) {
                auto batch = use_fixed_batch
                                 ? fixed_batch
                                 : train_dataset->next_batch(static_cast<size_t>(cli.batch), true);
                auto logits = model.forward(batch.input_ids, batch.attention_mask);
                auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
                accum_loss += static_cast<double>(loss->data<float>()[0]);
                accum_tokens += count_tokens(batch.attention_mask);
                auto scaled_loss = mul(loss, 1.0f / static_cast<float>(std::max(1, cli.grad_accum)));
                scaled_loss->backward();
            }

            const float grad_norm = clip_and_get_grad_norm(cli.max_grad_norm);
            const float cur_lr = make_scheduler(step, total_steps, cli.warmup_steps, cli.learning_rate, cli.lr_scheduler);
            optimizer->set_learning_rate(cur_lr);

            std::vector<TensorPtr> grads;
            grads.reserve(trainable.size());
            for (const auto& p : trainable) grads.push_back(p->grad());
            optimizer->step(trainable, grads);
            for (auto& p : trainable) p->zero_grad();

            const float avg_loss = static_cast<float>(accum_loss / static_cast<double>(std::max(1, cli.grad_accum)));
            total_tokens += accum_tokens;
            if (!ema_init) {
                ema_loss = avg_loss;
                ema_init = true;
            } else {
                ema_loss = 0.9f * ema_loss + 0.1f * avg_loss;
            }

            if ((step + 1) % std::max(1, cli.logging_steps) == 0) {
                std::cout << "[Train] step " << (step + 1) << "/" << total_steps
                          << " lr " << std::fixed << std::setprecision(6) << cur_lr
                          << " loss " << std::setprecision(4) << avg_loss
                          << " ppl " << std::setprecision(2) << perplexity_from_loss(avg_loss)
                          << " ema_loss " << std::setprecision(4) << ema_loss
                          << " grad_norm " << std::setprecision(3) << grad_norm
                          << " tokens " << accum_tokens
                          << std::endl;
            }

            if (cli.eval_steps > 0 && (step + 1) % cli.eval_steps == 0) {
                const float valid_ppl = evaluate_valid();
                std::cout << "[Eval] step " << (step + 1) << "/" << total_steps
                          << " valid_ppl " << std::fixed << std::setprecision(2) << valid_ppl
                          << " ema_loss " << std::setprecision(4) << ema_loss
                          << " total_tokens " << total_tokens
                          << std::endl;
                std::ostringstream js;
                js << "{\"step\":" << (step + 1)
                   << ",\"valid_ppl\":" << valid_ppl
                   << ",\"ema_loss\":" << ema_loss
                   << ",\"total_tokens\":" << total_tokens
                   << "}";
                append_jsonl(cli.eval_out, js.str());
                if (valid_ppl < best_valid_ppl) {
                    best_valid_ppl = valid_ppl;
                    const std::string best_path = cli.output_dir + "/lora_best.safetensors";
                    save_adapter(best_path);
                    std::cout << "[Best] updated " << best_path << std::endl;
                }
            }

            if (cli.save_every > 0 && (step + 1) % cli.save_every == 0) {
                const std::string ckpt_path = make_checkpoint_path(cli.output_dir, step + 1);
                save_adapter(ckpt_path);
                std::cout << "[Checkpoint] saved " << ckpt_path << std::endl;
            }

            MemoryManager::instance().force_cleanup();
            if ((step + 1) % 50 == 0) {
                MemoryManager::instance().clear_unused_memory();
            }
        }

        const std::string final_path = cli.output_dir + "/lora.safetensors";
        save_adapter(final_path);
        std::cout << "\n🎉 Qwen LoRA training done. Saved adapter to " << final_path << std::endl;
        std::cout << "Total steps " << total_steps
                  << ", total tokens " << total_tokens
                  << ", final EMA loss " << std::fixed << std::setprecision(4) << ema_loss
                  << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}
