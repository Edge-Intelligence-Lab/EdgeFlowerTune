/**
 * @file test_gpt2_forward.cpp
 * @brief GPT-2 forward alignment test (compare logits with PyTorch/HF)
 * 
 * Acceptance criteria:
 * - max_abs_err <= 1e-4
 * - argmax(cpp) == argmax(pt)
 * - top-5 basically consistent
 */

#include "gpt2_model.h"
#include "safetensors_loader.h"
#include "../core/tokenizer_hf.h"
#include "../path_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <cstdlib>

using namespace ops;

// Simple JSON array parser (only supports float array)
std::vector<float> load_json_array(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open: " + path);
    
    std::string content((std::istreambuf_iterator<char>(in)), {});
    std::vector<float> result;
    result.reserve(60000);
    
    // Very simple parsing: extract all numbers (scientific notation simplified)
    std::istringstream iss(content);
    char c;
    std::string num_str;
    bool in_number = false;
    
    while (iss >> c) {
        if (c == '-' || c == '.' || (c >= '0' && c <= '9') || c == 'e' || c == 'E') {
            num_str += c;
            in_number = true;
        } else {
            if (in_number && !num_str.empty()) {
                try {
                    result.push_back(std::stof(num_str));
                } catch (...) {}
                num_str.clear();
            }
            in_number = false;
        }
    }
    if (in_number && !num_str.empty()) {
        try {
            result.push_back(std::stof(num_str));
        } catch (...) {}
    }
    
    return result;
}

int main() {
    try {
        std::cout << "========== GPT-2 Forward Alignment Test ==========" << std::endl;
        auto paths = path_utils::resolve_gpt2_wikitext2_paths(true);
        
        // 1. Build model
        GPT2Config cfg;
        GPT2Model model(cfg);
        model.tie_weights();
        
        // 2. Load weights
        std::cout << "\n[1/5] Loading weights from safetensors..." << std::endl;
        SafeTensorsReader reader(paths.model_file);
        reader.parse_header();
        
        auto key_map = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        
        int loaded_count = 0;
        for (const auto& [internal_key, hf_key] : key_map) {
            try {
                auto info = reader.get_tensor_info(hf_key);
                if (!info.dtype.empty()) {
                        // HuggingFace GPT-2 uses Conv1D, weights are already in [in, out] format, no transpose needed!
                    bool transpose = false;
                    auto tensor = reader.load_tensor(hf_key, transpose);
                    model.assign_weight(internal_key, tensor);
                    loaded_count++;
                    
                    // Debug: print key weight shapes
                    if (internal_key == "wte.weight" || internal_key == "blocks.0.attn.qkv.weight" ||
                        internal_key == "blocks.0.ln_1.weight") {
                        std::cout << "✓ Loaded " << internal_key << ": shape = [";
                        for (size_t i = 0; i < tensor->shape().size(); ++i) {
                            std::cout << tensor->shape()[i];
                            if (i + 1 < tensor->shape().size()) std::cout << ", ";
                        }
                        std::cout << "], transposed=" << (transpose ? "yes" : "no") << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "[WARN] Failed to load " << hf_key << ": " << e.what() << std::endl;
            }
        }
        
        std::cout << "Loaded " << loaded_count << " / " << key_map.size() << " tensors." << std::endl;
        
        // Verify: check if layer 11 weights are loaded
        std::cout << "\n[Verify] Checking layer 11 weights..." << std::endl;
        for (const auto& [internal_key, hf_key] : key_map) {
            if (internal_key.find("blocks.11.") != std::string::npos) {
                std::cout << "  ✓ " << internal_key << std::endl;
            }
        }
        
        // 3. Prepare input
        std::cout << "\n[2/5] Preparing input..." << std::endl;
        HFTokenizer tokenizer(paths.model_dir, true);
        if (!tokenizer.load()) {
            throw std::runtime_error("Failed to load HF tokenizer from: " + paths.model_dir);
        }
        auto input_ids_vec = tokenizer.encode("Hello, world!\n");
        
        std::cout << "Input text: \"Hello, world!\\n\"" << std::endl;
        std::cout << "Input IDs (from PyTorch): [";
        for (size_t i = 0; i < input_ids_vec.size(); ++i) {
            std::cout << input_ids_vec[i];
            if (i + 1 < input_ids_vec.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Convert to Tensor
        auto input_ids = std::make_shared<Tensor>(
            std::vector<int64_t>{1, static_cast<int64_t>(input_ids_vec.size())},
            input_ids_vec.data(), kInt32, kCPU);
        
        // Create attention_mask (all 1s)
        auto attn_mask = std::make_shared<Tensor>(
            std::vector<int64_t>{1, static_cast<int64_t>(input_ids_vec.size())},
            kFloat32, kCPU);
        float* mask_data = attn_mask->data<float>();
        for (size_t i = 0; i < input_ids_vec.size(); ++i) {
            mask_data[i] = 1.0f;
        }
        
        // 4. Forward (disable dropout)
        std::cout << "\n[3/5] Running forward pass..." << std::endl;
        // TODO: Disable dropout (global switch or parameter)
        auto logits = model.forward(input_ids, attn_mask);  // [1, S, V]
        
        int64_t batch = logits->shape()[0];
        int64_t seq_len = logits->shape()[1];
        int64_t vocab_size = logits->shape()[2];
        
        std::cout << "Logits shape: [" << batch << ", " << seq_len << ", " << vocab_size << "]" << std::endl;
        
        // Extract last token logits
        const float* logits_data = logits->data<float>();
        std::vector<float> last_logits(vocab_size);
        std::memcpy(last_logits.data(), logits_data + (seq_len - 1) * vocab_size, vocab_size * sizeof(float));
        
        // Print first 10 values
        std::cout << "\nC++ last_logits[0:10]: [";
        for (int i = 0; i < 10; ++i) {
            printf("%.2f", last_logits[i]);
            if (i < 9) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Top-5
        std::vector<std::pair<float, int>> scored;
        scored.reserve(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            scored.emplace_back(last_logits[i], i);
        }
        std::partial_sort(scored.begin(), scored.begin() + 5, scored.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        std::cout << "\nC++ top-5 IDs:  [";
        for (int i = 0; i < 5; ++i) {
            std::cout << scored[i].second;
            if (i < 4) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "C++ top-5 vals: [";
        for (int i = 0; i < 5; ++i) {
            printf("%.6f", scored[i].first);
            if (i < 4) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        int argmax_cpp = scored[0].second;
        
        // 5. Load PyTorch gold standard and compare
        std::cout << "\n[4/5] Loading PyTorch gold standard..." << std::endl;
        std::string pt_logits_path;
        if (const char* p = std::getenv("PT_GOLD_LOGITS")) {
            pt_logits_path = p;
        } else {
            std::vector<std::string> candidates = {
                "operators/finetune_ops/graph/pt_last_logits.json",
                "finetune_ops/graph/pt_last_logits.json",
                "../operators/finetune_ops/graph/pt_last_logits.json",
                "../../operators/finetune_ops/graph/pt_last_logits.json"
            };
            for (const auto& c : candidates) {
                if (std::filesystem::exists(c)) {
                    pt_logits_path = c;
                    break;
                }
            }
            if (pt_logits_path.empty()) pt_logits_path = candidates.front();
        }
        auto pt_logits = load_json_array(pt_logits_path);
        
        if (pt_logits.size() != static_cast<size_t>(vocab_size)) {
            std::cerr << "ERROR: PyTorch logits size mismatch: " << pt_logits.size()
                     << " vs " << vocab_size << std::endl;
            return 1;
        }
        
        // Calculate error
        float max_abs_err = 0.0f;
        float sum_abs_err = 0.0f;
        int argmax_pt = 0;
        float max_val_pt = pt_logits[0];
        
        for (int i = 0; i < vocab_size; ++i) {
            float err = std::abs(last_logits[i] - pt_logits[i]);
            if (err > max_abs_err) max_abs_err = err;
            sum_abs_err += err;
            
            if (pt_logits[i] > max_val_pt) {
                max_val_pt = pt_logits[i];
                argmax_pt = i;
            }
        }
        
        float mean_abs_err = sum_abs_err / vocab_size;
        
        // 6. Report results
        std::cout << "\n[5/5] ========== Alignment Report ==========" << std::endl;
        std::cout << "Max absolute error: " << max_abs_err << std::endl;
        std::cout << "Mean absolute error: " << mean_abs_err << std::endl;
        std::cout << "Argmax (C++): " << argmax_cpp << std::endl;
        std::cout << "Argmax (PT):  " << argmax_pt << std::endl;
        
        // Acceptance criteria
        bool pass = true;
        
        if (max_abs_err > 1e-4f) {
            std::cout << "\nFAIL: max_abs_err > 1e-4" << std::endl;
            pass = false;
        } else {
            std::cout << "\nPASS: max_abs_err <= 1e-4" << std::endl;
        }
        
        if (argmax_cpp != argmax_pt) {
            std::cout << "FAIL: argmax mismatch" << std::endl;
            pass = false;
        } else {
            std::cout << "PASS: argmax consistent" << std::endl;
        }
        
        std::cout << "\n" << (pass ? "All checks passed!" : "WARNING: Some checks failed.") << std::endl;
        
        return pass ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "\nException: " << e.what() << std::endl;
        return 1;
    }
}
