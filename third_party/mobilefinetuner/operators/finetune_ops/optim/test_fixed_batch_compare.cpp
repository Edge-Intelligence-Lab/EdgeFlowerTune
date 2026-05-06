/**
 * @file test_fixed_batch_compare.cpp
 * @brief Fixed validation batch comparison: Baseline vs LoRA after training
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
#include <fstream>
#include <vector>

using namespace ops;

int main() {
    try {
        std::cout << "========== Fixed Batch Comparison: Baseline vs LoRA ==========\n" << std::endl;
        auto paths = test_utils::resolve_gpt2_wikitext2_paths();
        
        // 1. Prepare tokenizer and dataset
        std::cout << "[1/5] Loading tokenizer and data..." << std::endl;
        HFTokenizer tokenizer(paths.model_dir, true);
        if (!tokenizer.load()) {
            throw std::runtime_error("Failed to load HF tokenizer from: " + paths.model_dir);
        }
        
        WT2Config data_cfg;
        data_cfg.train_path = paths.train_file;
        data_cfg.valid_path = paths.valid_file;
        data_cfg.seq_len = 128;
        data_cfg.stride = -1;
        data_cfg.seed = 2025;  // Fixed seed
        
        WikiText2Dataset train_dataset(data_cfg, &tokenizer);
        train_dataset.load(Split::Train);
        
        WikiText2Dataset valid_dataset(data_cfg, &tokenizer);
        valid_dataset.load(Split::Valid);
        
        // 2. Extract 10 fixed validation batches
        std::cout << "\n[2/5] Extracting 10 fixed validation batches..." << std::endl;
        std::vector<Batch> fixed_batches;
        for (int i = 0; i < 10; ++i) {
            auto batch = valid_dataset.get_batch(i, 1);
            fixed_batches.push_back(batch);
        }
        std::cout << "Fixed 10 validation batches" << std::endl;
        
        // 3. Baseline (pretrained model) loss on fixed batches
        std::cout << "\n[3/5] Testing Baseline (pretrained)..." << std::endl;
        GPT2Config cfg;
        GPT2Model baseline_model(cfg);
        baseline_model.tie_weights();
        
        SafeTensorsReader reader(paths.model_file);
        reader.parse_header();
        auto key_map = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        
        for (const auto& [internal_key, hf_key] : key_map) {
            try {
                auto info = reader.get_tensor_info(hf_key);
                if (!info.dtype.empty()) {
                    auto tensor = reader.load_tensor(hf_key, false);
                    baseline_model.assign_weight(internal_key, tensor);
                }
            } catch (...) {}
        }
        
        std::vector<float> baseline_losses;
        for (size_t i = 0; i < fixed_batches.size(); ++i) {
            const auto& batch = fixed_batches[i];
            auto logits = baseline_model.forward(batch.input_ids, batch.attention_mask);
            auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
            float loss_val = loss->data<float>()[0];
            baseline_losses.push_back(loss_val);
        }
        
        // 4. LoRA training for 50 steps
        std::cout << "\n[4/5] Training LoRA for 50 steps..." << std::endl;
        GPT2Model lora_model(cfg);
        lora_model.tie_weights();
        
        // Reload weights
        SafeTensorsReader reader2(paths.model_file);
        reader2.parse_header();
        for (const auto& [internal_key, hf_key] : key_map) {
            try {
                auto info = reader2.get_tensor_info(hf_key);
                if (!info.dtype.empty()) {
                    auto tensor = reader2.load_tensor(hf_key, false);
                    lora_model.assign_weight(internal_key, tensor);
                }
            } catch (...) {}
        }
        
        // Inject LoRA
        lora_model.init_lora_modules();
        LoraSpec spec;
        spec.rank = 8;
        spec.alpha = 16.0f;
        spec.dropout = 0.0f;
        spec.split_qkv = true;
        
        LoraInjector injector;
        injector.inject(lora_model, spec);
        auto trainable = lora_model.get_lora_parameters();
        
        // Optimizer
        AdamConfig opt_cfg;
        opt_cfg.learning_rate = 1e-4f;
        opt_cfg.beta1 = 0.9f;
        opt_cfg.beta2 = 0.999f;
        opt_cfg.epsilon = 1e-8f;
        opt_cfg.weight_decay = 0.0f;
        opt_cfg.clip_grad_norm = 1.0f;
        Adam optimizer(opt_cfg);
        
        // Train 50 steps
        for (int step = 0; step < 50; ++step) {
            auto batch = train_dataset.next_batch(1);
            auto logits = lora_model.forward(batch.input_ids, batch.attention_mask);
            auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
            
            loss->backward();
            std::vector<TensorPtr> grads;
            for (const auto& p : trainable) grads.push_back(p->grad());
            optimizer.step(trainable, grads);
            for (auto& p : trainable) p->zero_grad();
            
            if (step % 10 == 0 || step == 49) {
                float loss_val = loss->data<float>()[0];
                printf("  Train step %2d: Loss=%.4f\n", step, loss_val);
            }
        }
        
        // 5. LoRA loss on fixed batches after training
        std::cout << "\n[5/5] Testing LoRA (after 50 steps training)..." << std::endl;
        std::vector<float> lora_losses;
        for (size_t i = 0; i < fixed_batches.size(); ++i) {
            const auto& batch = fixed_batches[i];
            auto logits = lora_model.forward(batch.input_ids, batch.attention_mask);
            auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
            float loss_val = loss->data<float>()[0];
            lora_losses.push_back(loss_val);
        }
        
        // 6. Comparison results
        std::cout << "\n========== Fixed Validation Batch Comparison ==========" << std::endl;
        std::cout << "\nBatch | Baseline Loss | LoRA Loss | Improvement" << std::endl;
        std::cout << "------|---------------|-----------|------------" << std::endl;
        
        float total_baseline = 0.0f;
        float total_lora = 0.0f;
        
        for (size_t i = 0; i < 10; ++i) {
            float improvement = baseline_losses[i] - lora_losses[i];
            printf("%5zu | %13.4f | %9.4f | %+11.4f\n", 
                   i, baseline_losses[i], lora_losses[i], improvement);
            total_baseline += baseline_losses[i];
            total_lora += lora_losses[i];
        }
        
        float avg_baseline = total_baseline / 10.0f;
        float avg_lora = total_lora / 10.0f;
        float avg_improvement = avg_baseline - avg_lora;
        
        std::cout << "\n========== Summary ==========" << std::endl;
        printf("Baseline average: %.4f\n", avg_baseline);
        printf("LoRA average:     %.4f\n", avg_lora);
        printf("Improvement:      %.4f (%.2f%%)\n", avg_improvement, (avg_improvement/avg_baseline)*100.0f);
        
        std::cout << "\nComparison completed" << std::endl;
        
        // Export CSV for plotting
        std::cout << "\nWriting loss_comparison.csv..." << std::endl;
        std::ofstream csv("loss_comparison.csv");
        csv << "Batch,Baseline_Loss,LoRA_Loss,Improvement\n";
        for (size_t i = 0; i < 10; ++i) {
            csv << i << "," << baseline_losses[i] << "," << lora_losses[i] 
                << "," << (baseline_losses[i] - lora_losses[i]) << "\n";
        }
        csv.close();
        std::cout << "Saved to loss_comparison.csv" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nException: " << e.what() << std::endl;
        return 1;
    }
}
