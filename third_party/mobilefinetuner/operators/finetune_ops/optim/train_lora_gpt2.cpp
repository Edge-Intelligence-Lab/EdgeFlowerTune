/**
 * @file train_lora_gpt2.cpp
 * @brief GPT-2 LoRA fine-tuning main entry
 * 
 * Usage:
 *   cd operators/build
 *   ./train_lora_gpt2 --data_dir ../data/wikitext2/wikitext-2-raw \
 *                      --model_dir ../pretrained/gpt2 \
 *                      --output_dir ./lora_output \
 *                      --epochs 3 --batch_size 4 --lr 2e-4
 */

#include "trainer.h"
#include "../graph/gpt2_model.h"
#include "../graph/lora_injector.h"
#include "../graph/safetensors_loader.h"
#include "../data/wikitext2_dataset.h"
#include "../core/tokenizer_hf.h"
#include "test_path_utils.h"
#include <iostream>

using namespace ops;

int main(int argc, char** argv) {
    try {
        std::cout << "========== GPT-2 LoRA Fine-tuning ==========\n" << std::endl;
        auto paths = test_utils::resolve_gpt2_wikitext2_paths();
        
        // 1. Load model
        std::cout << "[1/5] Loading GPT-2 model..." << std::endl;
        GPT2Config model_cfg;
        GPT2Model model(model_cfg);
        model.tie_weights();
        
        // Load pretrained weights
        SafeTensorsReader reader(paths.model_file);
        reader.parse_header();
        auto key_map = GPT2KeyMapper::generate_gpt2_mapping(model_cfg.n_layer);
        
        for (const auto& [internal_key, hf_key] : key_map) {
            try {
                auto info = reader.get_tensor_info(hf_key);
                if (!info.dtype.empty()) {
                    auto tensor = reader.load_tensor(hf_key, false);
                    model.assign_weight(internal_key, tensor);
                }
            } catch (...) {}
        }
        std::cout << "[OK] Loaded pretrained weights" << std::endl;
        
        // 2. Inject LoRA
        std::cout << "\n[2/5] Injecting LoRA..." << std::endl;
        LoraSpec lora_spec;
        lora_spec.rank = 8;
        lora_spec.alpha = 16.0f;
        lora_spec.dropout = 0.05f;
        // Align with PyTorch/PEFT defaults: do not split QKV and only target attention
        lora_spec.split_qkv = false;
        lora_spec.targets = {
            LoraTarget::AttnQKV,
            LoraTarget::AttnProj
        };
        
        LoraInjector lora;
        lora.inject(model, lora_spec);
        lora.print_info();
        
        // 3. Prepare data
        std::cout << "\n[3/5] Loading datasets..." << std::endl;
        HFTokenizer tokenizer(paths.model_dir, true);
        if (!tokenizer.load()) {
            throw std::runtime_error("Failed to load HF tokenizer from: " + paths.model_dir);
        }
        
        WT2Config data_cfg;
        data_cfg.train_path = paths.train_file;
        data_cfg.valid_path = paths.valid_file;
        data_cfg.seq_len = 128;  // Use short sequences for testing
        data_cfg.stride = -1;
        
        WikiText2Dataset train_data(data_cfg, &tokenizer);
        train_data.load(Split::Train);
        
        WikiText2Dataset eval_data(data_cfg, &tokenizer);
        eval_data.load(Split::Valid);
        
        std::cout << "[OK] Train sequences: " << train_data.num_sequences() << std::endl;
        std::cout << "[OK] Eval sequences: " << eval_data.num_sequences() << std::endl;
        
        // 4. Create Trainer
        std::cout << "\n[4/5] Initializing trainer..." << std::endl;
        TrainerConfig trainer_cfg;
        trainer_cfg.learning_rate = 2e-4f;
        trainer_cfg.num_epochs = 3;
        trainer_cfg.gradient_accumulation_steps = 4;
        trainer_cfg.output_dir = "./lora_checkpoints";
        
        LoRATrainer trainer(model, lora, train_data, eval_data, trainer_cfg);
        
        // 5. Start training
        std::cout << "\n[5/5] Starting training...\n" << std::endl;
        trainer.train();
        
        // 6. Save LoRA
        std::cout << "\n[6/6] Saving LoRA weights..." << std::endl;
        trainer.save_lora(trainer_cfg.output_dir + "/lora_final.safetensors");
        
        std::cout << "\n[DONE] Training completed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Error: " << e.what() << std::endl;
        return 1;
    }
}
