/**
 * @file gpt2_model.h
 * @brief GPT-2 model lightweight wrapper (assembles gpt2_components, supports tie-weights and LoRA injection)
 */

#pragma once

#include "../core/tensor.h"
#include "../nn/lora_linear.h"
#include "../../opt_ops/sharding/parameter_sharder.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <algorithm>

namespace ops {

/**
 * @brief GPT-2 Transformer Block weights (includes LoRALinear modules)
 */
struct GPT2BlockWeights {
    // LayerNorm
    TensorPtr ln_1_weight, ln_1_bias;
    TensorPtr ln_2_weight, ln_2_bias;
    
    // Attention fused weights (for initialization)
    TensorPtr attn_qkv_weight;   // [C, 3C]
    TensorPtr attn_qkv_bias;     // [3C]
    TensorPtr attn_proj_weight;  // [C, C]
    TensorPtr attn_proj_bias;    // [C]
    
    // MLP weights
    TensorPtr mlp_fc_in_weight;   // [C, 4C]
    TensorPtr mlp_fc_in_bias;     // [4C]
    TensorPtr mlp_fc_out_weight;  // [4C, C]
    TensorPtr mlp_fc_out_bias;    // [C]
    
    // LoRA-enhanced linear layers
    std::unique_ptr<LoRALinear> qkv_lin;
    std::unique_ptr<LoRALinear> proj_lin;
    std::unique_ptr<LoRALinear> fc_in_lin;
    std::unique_ptr<LoRALinear> fc_out_lin;
    
    bool lora_initialized = false;
};

/**
 * @brief GPT-2 configuration (aligned with HuggingFace)
 */
struct GPT2Config {
    int vocab_size = 50257;
    int n_positions = 1024;
    int n_embd = 768;
    int n_layer = 12;
    int n_head = 12;
    float layernorm_eps = 1e-5f;      // Consistent with HF
    bool tie_word_embeddings = true;  // lm_head <-> wte
    bool use_cache = false;           // Disable KV cache during training
    
    // Memory optimization options
    bool use_bf16_activations = false;            // Downcast activations to FP16 (enable when needed)
    int checkpoint_every = 0;                    // Activation checkpointing: 0=off, N=every N blocks
    bool checkpoint_mlp = false;                 // Checkpoint MLP subgraph to reduce activation memory
    int mlp_chunk_size = 0;                      // MLP chunk size along seq (0=off)
    bool defer_parameter_allocation = false;     // Sharding mode: avoid eager full-weight allocation
    
    // Load from config.json (future implementation)
    static GPT2Config from_pretrained(const std::string& config_path);
};

/**
 * @brief GPT-2 model (lightweight wrapper)
 * 
 * Assembles Embedding + TransformerBlock + LayerNorm + lm_head
 * Supports:
 * - tie_weights (lm_head <-> wte)
 * - assign_weight (for safetensors_loader to fill)
 * - Forward inference (input_ids + attention_mask -> logits)
 */
class GPT2Model {
public:
    explicit GPT2Model(const GPT2Config& config);
    ~GPT2Model() = default;
    
    /**
     * @brief Forward propagation
     * @param input_ids [batch, seq_len] int32
     * @param attention_mask [batch, seq_len] int32/float (1=valid, 0=pad)
     * @return logits [batch, seq_len, vocab_size]
     */
    TensorPtr forward(const TensorPtr& input_ids,
                     const TensorPtr& attention_mask = nullptr);
    
    /**
     * @brief Bind lm_head.weight <-> wte.weight (same memory)
     */
    void tie_weights();
    
    /**
     * @brief For safetensors_loader to fill weights
     * @param key Internal key name (e.g., "wte.weight", "blocks.0.ln_1.weight")
     * @param tensor Weight tensor
     */
    void assign_weight(const std::string& key, const TensorPtr& tensor);
    TensorPtr* parameter_ref(const std::string& key);
    
    /**
     * @brief Get all parameters (for optimizer)
     */
    std::vector<TensorPtr> parameters();
    
    /**
     * @brief Get trainable parameters (after LoRA injection, only return LoRA parameters)
     */
    std::vector<TensorPtr> trainable_parameters();
    
    /**
     * @brief Freeze all parameters except LoRA
     */
    void freeze_base_parameters();
    
    /**
     * @brief Print model info
     */
    void print_model_info() const;
    
    const GPT2Config& config() const { return config_; }
    void set_mlp_chunk_size(int chunk_size) { config_.mlp_chunk_size = std::max(0, chunk_size); }
    int mlp_chunk_size() const { return config_.mlp_chunk_size; }

    // ========================= Getters required for LoRA binding =========================
    // Return writable pointers (TensorPtr*) for LoRA injector to bind to actual layer parameters
    // i in [0, config_.n_layer)
    std::pair<TensorPtr*, TensorPtr*> attn_qkv_params(int i);
    std::pair<TensorPtr*, TensorPtr*> attn_proj_params(int i);
    std::pair<TensorPtr*, TensorPtr*> mlp_fc_in_params(int i);
    std::pair<TensorPtr*, TensorPtr*> mlp_fc_out_params(int i);
    
    /**
     * @brief Access block (for use by LoRA injector)
     */
    GPT2BlockWeights& get_block(int i) { return blocks_[i]; }
    const GPT2BlockWeights& get_block(int i) const { return blocks_[i]; }
    
    /**
     * @brief Collect all LoRA trainable parameters
     */
    std::vector<TensorPtr> get_lora_parameters();
    
    /**
     * @brief Initialize LoRA modules (call after loading weights)
     */
    void init_lora_modules();

    // Optional sharding management
    void set_parameter_sharder(sharding::ParameterSharder* sharder) { sharder_ = sharder; }
    TensorPtr& wte_weight_ref() { return wte_weight_; }
    TensorPtr& wpe_weight_ref() { return wpe_weight_; }
    TensorPtr& ln_f_weight_ref() { return ln_f_weight_; }
    TensorPtr& ln_f_bias_ref() { return ln_f_bias_; }

private:
    GPT2Config config_;
    
    // Embeddings
    TensorPtr wte_weight_;  // [vocab_size, n_embd]
    TensorPtr wpe_weight_;  // [n_positions, n_embd]
    
    // Transformer blocks (using externally defined GPT2BlockWeights)
    std::vector<GPT2BlockWeights> blocks_;
    
    // Final LayerNorm
    TensorPtr ln_f_weight_;
    TensorPtr ln_f_bias_;
    
    // lm_head (tied to wte_weight_, not stored separately)
    bool weights_tied_ = false;
    TensorPtr wte_weight_t_;  // Cached transpose (needs rebuild when sharding)

    // Optional sharder
    sharding::ParameterSharder* sharder_ = nullptr;
    
    // Internal utilities
    TensorPtr build_causal_mask(int seq_len);
    TensorPtr build_padding_mask(const TensorPtr& attention_mask);
    TensorPtr layer_norm(const TensorPtr& x, const TensorPtr& weight, const TensorPtr& bias);
    TensorPtr gelu_new(const TensorPtr& x);
    TensorPtr embedding_lookup(const TensorPtr& weight, const TensorPtr& indices);
    TensorPtr forward_block(const TensorPtr& x, int block_idx,
                           const TensorPtr& causal_mask,
                           const TensorPtr& pad_mask);
};

}  // namespace ops
