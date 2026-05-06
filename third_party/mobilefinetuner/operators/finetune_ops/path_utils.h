#pragma once

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace ops {
namespace path_utils {

struct Gpt2WikiText2Paths {
    std::string model_dir;
    std::string model_file;
    std::string data_dir;
    std::string train_file;
    std::string valid_file;
    std::string test_file;
};

inline std::string pick_existing_dir(const std::vector<std::string>& candidates,
                                     const std::string& must_have_file) {
    namespace fs = std::filesystem;
    for (const auto& dir : candidates) {
        if (dir.empty()) continue;
        if (must_have_file.empty()) {
            if (fs::exists(dir)) return dir;
        } else if (fs::exists(dir + "/" + must_have_file)) {
            return dir;
        }
    }
    return std::string();
}

inline std::string env_or_empty(const char* key) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : std::string();
}

inline std::string resolve_gpt2_pretrained_dir() {
    std::string from_env = env_or_empty("GPT2_PRETRAINED_DIR");
    if (!from_env.empty()) return from_env;

    std::vector<std::string> candidates = {
        "gpt2_lora_finetune/pretrained/gpt2",
        "../gpt2_lora_finetune/pretrained/gpt2",
        "../../gpt2_lora_finetune/pretrained/gpt2"
    };

    std::string existing = pick_existing_dir(candidates, "model.safetensors");
    return existing.empty() ? candidates.front() : existing;
}

inline std::string resolve_wikitext2_dir() {
    std::string from_env = env_or_empty("WIKITEXT2_DIR");
    if (!from_env.empty()) return from_env;

    std::vector<std::string> candidates = {
        "data/wikitext2/wikitext-2-raw",
        "../data/wikitext2/wikitext-2-raw",
        "../../data/wikitext2/wikitext-2-raw"
    };
    std::string existing = pick_existing_dir(candidates, "wiki.train.raw");
    return existing.empty() ? candidates.front() : existing;
}

inline std::string resolve_mmlu_dir() {
    std::string from_env = env_or_empty("MMLU_DIR");
    if (!from_env.empty()) return from_env;

    std::vector<std::string> candidates = {
        "data/mmlu/data",
        "../data/mmlu/data",
        "../../data/mmlu/data"
    };
    std::string existing = pick_existing_dir(candidates, "");
    return existing.empty() ? candidates.front() : existing;
}

inline Gpt2WikiText2Paths resolve_gpt2_wikitext2_paths(bool require_exists = true) {
    namespace fs = std::filesystem;
    Gpt2WikiText2Paths paths;
    paths.model_dir = resolve_gpt2_pretrained_dir();
    paths.data_dir = resolve_wikitext2_dir();
    paths.model_file = paths.model_dir + "/model.safetensors";
    paths.train_file = paths.data_dir + "/wiki.train.raw";
    paths.valid_file = paths.data_dir + "/wiki.valid.raw";
    paths.test_file = paths.data_dir + "/wiki.test.raw";

    if (require_exists) {
        if (!fs::exists(paths.model_file)) {
            throw std::runtime_error(
                "Missing GPT-2 model.safetensors at: " + paths.model_file +
                " (set GPT2_PRETRAINED_DIR)");
        }
        if (!fs::exists(paths.train_file) || !fs::exists(paths.valid_file)) {
            throw std::runtime_error(
                "Missing WikiText2 files under: " + paths.data_dir +
                " (set WIKITEXT2_DIR)");
        }
    }
    return paths;
}

}  // namespace path_utils
}  // namespace ops

