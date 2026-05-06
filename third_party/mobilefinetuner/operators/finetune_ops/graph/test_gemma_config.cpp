#include "gemma_model.h"
#include "safetensors_loader.h"
#include "../path_utils.h"

#include <cassert>
#include <filesystem>
#include <iostream>

using namespace ops;

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    std::string model_dir;
    if (argc > 1 && argv[1] != nullptr) {
        model_dir = argv[1];
    } else {
        std::string env_dir = path_utils::env_or_empty("GEMMA_270M_DIR");
        if (!env_dir.empty()) {
            model_dir = env_dir;
        } else {
            const std::vector<std::string> candidates = {
                "gemma-3-270m",
                "../gemma-3-270m",
                "../../gemma-3-270m",
                "../../../gemma-3-270m"
            };
            for (const auto& dir : candidates) {
                if (fs::exists(dir + "/config.json")) {
                    model_dir = dir;
                    break;
                }
            }
        }
    }

    if (model_dir.empty() || !fs::exists(model_dir + "/config.json")) {
        std::cerr << "[FAIL] Cannot locate Gemma-3-270M config.json." << std::endl;
        std::cerr << "Pass model dir as argv[1] or set GEMMA_270M_DIR." << std::endl;
        return 2;
    }

    std::cout << "[Test] GemmaTextConfig::from_pretrained\n";
    auto cfg = GemmaTextConfig::from_pretrained(model_dir);

    assert(cfg.vocab_size == 262144);
    assert(cfg.hidden_size == 640);
    assert(cfg.intermediate_size == 2048);
    assert(cfg.num_hidden_layers == 18);
    assert(cfg.num_attention_heads == 4);
    assert(cfg.num_key_value_heads == 1);
    assert(cfg.head_dim == 256);
    assert(cfg.sliding_window == 512);
    assert(cfg.layer_types.size() == static_cast<size_t>(cfg.num_hidden_layers));
    assert(cfg.layer_types[0] == "sliding_attention");

    std::cout << "  ✓ core fields match Gemma-3-270M config\n";

    std::cout << "\n[Test] GemmaKeyMapper::generate_gemma_mapping\n";
    auto mapping = GemmaKeyMapper::generate_gemma_mapping(cfg.num_hidden_layers);

    size_t expected_per_layer = 13;  // 4 norms + 4 attn proj + 2 attn norms + 3 mlp
    size_t expected_total = 3 + cfg.num_hidden_layers * expected_per_layer;  // top embed/norm/lm_head
    assert(mapping.size() == expected_total);

    auto it = mapping.find("layers.0.self_attn.q_proj.weight");
    assert(it != mapping.end());
    assert(it->second == "model.layers.0.self_attn.q_proj.weight");

    std::cout << "  ✓ mapping size = " << mapping.size() << ", sample entry verified\n";
    std::cout << "\nAll Gemma config/mapping tests passed\n";
    return 0;
}
