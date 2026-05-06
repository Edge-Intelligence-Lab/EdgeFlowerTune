/**
 * @file test_lora_correctness.cpp
 * @brief LoRA correctness test (zero impact + merge/unmerge idempotence)
 * 
 * Acceptance criteria:
 * 1. Zero impact: after injection A~N(0,1/r), B=0 -> output element-wise identical to non-injected
 * 2. merge idempotence: merge() twice -> weights unchanged; unmerge() back to W0
 * 3. unmerge idempotence: unmerge() twice -> weights unchanged
 */

#include "gpt2_model.h"
#include "lora_injector.h"
#include "safetensors_loader.h"
#include "../core/tokenizer_hf.h"
#include "../path_utils.h"
#include <iostream>
#include <cmath>
#include <vector>

using namespace ops;

// Calculate maximum absolute error between two Tensors
float max_abs_diff(const TensorPtr& a, const TensorPtr& b) {
    if (a->shape() != b->shape()) {
        throw std::runtime_error("Shape mismatch in max_abs_diff");
    }
    
    int64_t n = a->numel();
    const float* a_data = a->data<float>();
    const float* b_data = b->data<float>();
    
    float max_err = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float err = std::abs(a_data[i] - b_data[i]);
        if (err > max_err) max_err = err;
    }
    
    return max_err;
}

// Clone weights (deep copy)
TensorPtr clone_tensor(const TensorPtr& src) {
    auto dst = std::make_shared<Tensor>(src->shape(), src->dtype(), src->device());
    std::memcpy(dst->data_ptr(), src->data_ptr(), src->numel() * sizeof(float));
    return dst;
}

int main() {
    try {
        std::cout << "========== LoRA Correctness Tests ==========" << std::endl;
        auto paths = path_utils::resolve_gpt2_wikitext2_paths(true);
        
        // Prepare model (load weights)
        std::cout << "\n[Setup] Loading GPT-2 model..." << std::endl;
        GPT2Config cfg;
        cfg.n_layer = 2;  // Simplify test, only use 2 layers
        GPT2Model model(cfg);
        model.tie_weights();
        
        SafeTensorsReader reader(paths.model_file);
        reader.parse_header();
        
        auto key_map = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        
        for (const auto& [internal_key, hf_key] : key_map) {
            auto info = reader.get_tensor_info(hf_key);
            if (!info.dtype.empty()) {
                // HuggingFace GPT-2 uses Conv1D, weights are already in [in, out] format, no transpose needed!
                bool transpose = false;
                auto tensor = reader.load_tensor(hf_key, transpose);
                model.assign_weight(internal_key, tensor);
            }
        }
        
        // Prepare fixed input
        HFTokenizer tokenizer(paths.model_dir, true);
        if (!tokenizer.load()) {
            throw std::runtime_error("Failed to load HF tokenizer from: " + paths.model_dir);
        }
        
        auto input_ids_vec = tokenizer.encode("Test LoRA.");
        
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
        
        // ============================================================
        // Test 1: LoRA zero impact (B=0 initialization -> output unchanged)
        // ============================================================
        std::cout << "\n[Test 1/3] LoRA Zero Impact (B=0 init)" << std::endl;
        
        // Base forward
        auto logits_base = model.forward(input_ids, attn_mask);
        
        // Inject LoRA (split_qkv=true, rank=8)
        LoraSpec spec = LoraSpec::default_config();
        spec.rank = 8;
        spec.split_qkv = true;
        
        LoraInjector injector;
        injector.inject(model, spec);
        
        // Forward after LoRA injection (B=0, should output same)
        auto logits_lora = model.forward(input_ids, attn_mask);
        
        float err_zero_impact = max_abs_diff(logits_base, logits_lora);
        std::cout << "  Max abs diff (base vs LoRA B=0): " << err_zero_impact << std::endl;
        
        bool pass_zero_impact = (err_zero_impact < 1e-6f);
        if (pass_zero_impact) {
            std::cout << "  PASS: LoRA zero impact" << std::endl;
        } else {
            std::cout << "  FAIL: LoRA zero impact (expected < 1e-6)" << std::endl;
        }
        
        // ============================================================
        // Test 2: merge idempotence
        // ============================================================
        std::cout << "\n[Test 2/3] Merge Idempotency" << std::endl;
        
        // Manually set A/B to non-zero (for merge verification)
        auto lora_params = injector.collect_lora_parameters();
        for (auto& p : lora_params) {
            float* data = p->data<float>();
            for (int64_t i = 0; i < p->numel(); ++i) {
                data[i] = 0.01f * (i % 10 - 5);  // Small random values
            }
        }
        
        // Save original weight W0 (layer 0 attn.qkv.weight)
        auto W0 = clone_tensor(*model.attn_qkv_params(0).first);
        
        // First merge
        injector.merge();
        auto W1 = clone_tensor(*model.attn_qkv_params(0).first);
        
        // Second merge (should have no change)
        injector.merge();
        auto W2 = clone_tensor(*model.attn_qkv_params(0).first);
        
        float err_merge_idempotent = max_abs_diff(W1, W2);
        std::cout << "  Max abs diff (merge 1 vs merge 2): " << err_merge_idempotent << std::endl;
        
        bool pass_merge_idempotent = (err_merge_idempotent < 1e-8f);
        if (pass_merge_idempotent) {
            std::cout << "  PASS: merge idempotent" << std::endl;
        } else {
            std::cout << "  FAIL: merge not idempotent" << std::endl;
        }
        
        // unmerge back to W0
        injector.unmerge();
        auto W_back = clone_tensor(*model.attn_qkv_params(0).first);
        
        float err_unmerge_restore = max_abs_diff(W0, W_back);
        std::cout << "  Max abs diff (W0 vs unmerge): " << err_unmerge_restore << std::endl;
        
        bool pass_unmerge_restore = (err_unmerge_restore < 1e-6f);
        if (pass_unmerge_restore) {
            std::cout << "  PASS: unmerge restores W0" << std::endl;
        } else {
            std::cout << "  FAIL: unmerge does not restore W0" << std::endl;
        }
        
        // ============================================================
        // Test 3: unmerge idempotence
        // ============================================================
        std::cout << "\n[Test 3/3] Unmerge Idempotency" << std::endl;
        
        // Unmerge again (should have no change)
        injector.unmerge();
        auto W_back2 = clone_tensor(*model.attn_qkv_params(0).first);
        
        float err_unmerge_idempotent = max_abs_diff(W_back, W_back2);
        std::cout << "  Max abs diff (unmerge 1 vs unmerge 2): " << err_unmerge_idempotent << std::endl;
        
        bool pass_unmerge_idempotent = (err_unmerge_idempotent < 1e-8f);
        if (pass_unmerge_idempotent) {
            std::cout << "  PASS: unmerge idempotent" << std::endl;
        } else {
            std::cout << "  FAIL: unmerge not idempotent" << std::endl;
        }
        
        // ============================================================
        // Summary
        // ============================================================
        std::cout << "\n========== Summary ==========" << std::endl;
        
        bool all_pass = pass_zero_impact && pass_merge_idempotent 
                       && pass_unmerge_restore && pass_unmerge_idempotent;
        
        if (all_pass) {
            std::cout << "All LoRA correctness tests passed!" << std::endl;
            return 0;
        } else {
            std::cout << "WARNING: Some LoRA correctness tests failed." << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\nException: " << e.what() << std::endl;
        return 1;
    }
}
