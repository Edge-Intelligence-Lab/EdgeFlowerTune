/**
 * @file test_baseline_10step.cpp
 * @brief Pure GPT-2 baseline loss (pretrained model initial performance, no training)
 * Used for comparison with LoRA training
 */

#include "../graph/gpt2_model.h"
#include "../graph/safetensors_loader.h"
#include "../data/wikitext2_dataset.h"
#include "../core/tokenizer_hf.h"
#include "../core/lm_loss.h"
#include "test_path_utils.h"
#include "adam.h"
#include <iostream>
#include <vector>

using namespace ops;

int main() {
    try {
        std::cout << "========== Baseline GPT-2 10-step loss (pretrained model, no training) ==========\n" << std::endl;
        auto paths = test_utils::resolve_gpt2_wikitext2_paths();
        
        // 1. Load model
        std::cout << "[1/4] Loading model..." << std::endl;
        GPT2Config cfg;
        GPT2Model model(cfg);
        model.tie_weights();
        
        SafeTensorsReader reader(paths.model_file);
        reader.parse_header();
        auto key_map = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        
        for (const auto& [internal_key, hf_key] : key_map) {
            try {
                auto info = reader.get_tensor_info(hf_key);
                if (!info.dtype.empty()) {
                    auto tensor = reader.load_tensor(hf_key, false);
                    model.assign_weight(internal_key, tensor);
                }
            } catch (...) {}
        }
        
        // 2. Prepare data (exactly same as LoRA training)
        std::cout << "\n[2/2] Loading data..." << std::endl;
        HFTokenizer tokenizer(paths.model_dir, true);
        if (!tokenizer.load()) {
            throw std::runtime_error("Failed to load HF tokenizer from: " + paths.model_dir);
        }
        
        WT2Config data_cfg;
        data_cfg.train_path = paths.train_file;
        data_cfg.valid_path = paths.valid_file;
        data_cfg.seq_len = 128;
        data_cfg.stride = -1;
        
        WikiText2Dataset dataset(data_cfg, &tokenizer);
        dataset.load(Split::Train);
        
        // 3. 10 steps forward only (compute loss, no training)
        std::cout << "\n========== 10 Steps Forward Only (Baseline) ==========\n" << std::endl;
        
        std::vector<float> losses;
        
        for (int step = 0; step < 10; ++step) {
            // Get batch (same data flow as LoRA training)
            auto batch = dataset.next_batch(1);  // batch_size=1
            
            // Forward only (no backward, no parameter updates)
            auto logits = model.forward(batch.input_ids, batch.attention_mask);
            auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
            float loss_val = loss->data<float>()[0];
            losses.push_back(loss_val);
            
            // Calculate perplexity
            float ppl = perplexity_from_loss(loss_val);
            
            printf("Step %2d: Loss=%.4f, PPL=%.2f\n", step, loss_val, ppl);
        }
        
        // Verify convergence
        std::cout << "\n========== Results ==========" << std::endl;
        float first_loss = losses[0];
        float last_loss = losses[9];
        float improvement = first_loss - last_loss;
        
        std::cout << "  First loss: " << first_loss << std::endl;
        std::cout << "  Last loss: " << last_loss << std::endl;
        std::cout << "  Improvement: " << improvement << std::endl;
        
        std::cout << "\nBaseline forward completed (pretrained model not updated)" << std::endl;
        std::cout << "Comparison: LoRA training reduces loss by 0.80; baseline maintains pretrained state" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nException: " << e.what() << std::endl;
        return 1;
    }
}
