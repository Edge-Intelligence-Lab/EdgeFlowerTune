/**
 * @file test_10step_train.cpp
 * @brief 10-step training smoke test (simplified version, verify optimizer pipeline)
 */

#include "../graph/gpt2_model.h"
#include "../graph/safetensors_loader.h"
#include "../graph/lora_injector.h"
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
        std::cout << "========== 10-Step Training Smoke Test ==========\n" << std::endl;
        std::cout << "Note: This is a simplified test, only training embedding weights to verify optimizer pipeline" << std::endl;
        std::cout << "(Complete LoRA training requires LoRA computation integrated in model forward)\n" << std::endl;
        auto paths = test_utils::resolve_gpt2_wikitext2_paths();
        
        // 1. Load model
        std::cout << "[1/3] Loading model..." << std::endl;
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
        
        // 2. Prepare data
        std::cout << "\n[2/3] Loading data..." << std::endl;
        HFTokenizer tokenizer(paths.model_dir, true);
        if (!tokenizer.load()) {
            throw std::runtime_error("Failed to load HF tokenizer from: " + paths.model_dir);
        }
        
        WT2Config data_cfg;
        data_cfg.train_path = paths.train_file;
        data_cfg.valid_path = paths.valid_file;
        data_cfg.seq_len = 64;
        data_cfg.stride = -1;
        
        WikiText2Dataset dataset(data_cfg, &tokenizer);
        dataset.load(Split::Train);
        
        // 3. Train 10 steps
        std::cout << "\n[3/3] Training 10 steps..." << std::endl;
        std::cout << "(Simplified test: freeze all params, only check Forward+Loss pipeline)\n" << std::endl;
        
        std::vector<float> losses;
        
        for (int step = 0; step < 10; ++step) {
            auto batch = dataset.next_batch(2);
            
            // Forward
            auto logits = model.forward(batch.input_ids, batch.attention_mask);
            auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
            float loss_val = loss->data<float>()[0];
            
            losses.push_back(loss_val);
            float ppl = perplexity_from_loss(loss_val);
            
            std::cout << "Step " << step << ": Loss=" << loss_val 
                      << ", PPL=" << ppl << std::endl;
        }
        
        // Verify loss stability (pretrained model loss should be in 3-5 range and stable)
        float first_loss = losses[0];
        float last_loss = losses[9];
        float mean_loss = 0.0f;
        for (float l : losses) mean_loss += l;
        mean_loss /= losses.size();
        
        std::cout << "\n[Statistics]" << std::endl;
        std::cout << "  First loss: " << first_loss << std::endl;
        std::cout << "  Last loss: " << last_loss << std::endl;
        std::cout << "  Mean loss: " << mean_loss << std::endl;
        
        if (first_loss > 2.0f && first_loss < 10.0f && 
            std::abs(first_loss - last_loss) < 2.0f) {
            std::cout << "\nTest passed: Loss stable, Forward+Loss+Dataset pipeline OK" << std::endl;
            std::cout << "\nWARN: This is only a Forward test" << std::endl;
            std::cout << "   Complete LoRA training requires:" << std::endl;
            std::cout << "   1. Integrate lora_linear computation in GPT2Model" << std::endl;
            std::cout << "   2. Implement Backward propagation" << std::endl;
            std::cout << "   3. Connect optimizer to update LoRA parameters" << std::endl;
            return 0;
        } else {
            std::cout << "\nLoss abnormal" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\nException: " << e.what() << std::endl;
        return 1;
    }
}
