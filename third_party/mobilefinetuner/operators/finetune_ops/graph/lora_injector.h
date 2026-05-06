/**
 * @file lora_injector.h
 * @brief LoRA injector (supports split_qkv, merge/unmerge, save/load)
 */

#pragma once

#include "../core/tensor.h"
#include "gpt2_model.h"
#include <vector>
#include <string>
#include <unordered_map>

namespace ops {

/**
 * @brief LoRA target layer type
 */
enum class LoraTarget {
    AttnQKV,    // Attention QKV (supports splitting into q/k/v three groups)
    AttnProj,   // Attention output projection
    MlpFcIn,    // MLP fc_in (C -> 4C)
    MlpFcOut    // MLP fc_out (4C -> C)
};

/**
 * @brief LoRA configuration
 */
struct LoraSpec {
    int rank = 8;
    float alpha = 16.0f;
    float dropout = 0.05f;
    bool split_qkv = true;  // Whether QKV is split into q/k/v three groups (recommended)
    
    // By default, only apply LoRA to attention layers (consistent with PyTorch PEFT tradition)
    // To include MLP, manually add LoraTarget::MlpFcIn and LoraTarget::MlpFcOut
    std::vector<LoraTarget> targets = {
        LoraTarget::AttnQKV,   // Attention Q/K/V projection
        LoraTarget::AttnProj   // Attention output projection
    };
    
    std::vector<int> layers;  // empty = all layers
    
    LoraSpec() = default;
    
    // Default configuration (out-of-box, attention layers only)
    static LoraSpec default_config() {
        LoraSpec spec;
        spec.rank = 8;
        spec.alpha = 16.0f;
        spec.dropout = 0.05f;
        spec.split_qkv = true;
        return spec;
    }
    
    // Configuration with MLP (broader coverage)
    static LoraSpec full_config() {
        LoraSpec spec;
        spec.rank = 8;
        spec.alpha = 16.0f;
        spec.dropout = 0.05f;
        spec.split_qkv = true;
        spec.targets = {
            LoraTarget::AttnQKV,
            LoraTarget::AttnProj,
            LoraTarget::MlpFcIn,
            LoraTarget::MlpFcOut
        };
        return spec;
    }
};

/**
 * @brief Single LoRA state (A/B matrices + metadata)
 */
struct LoraState {
    TensorPtr A;  // [in, r]
    TensorPtr B;  // [r, out]
    float scale;  // alpha / r
    float dropout_p;
    bool enabled = true;
    
    // Initialize (A ~ N(0, 1/r), B = 0)
    void init(int64_t in_features, int64_t out_features, int rank, float alpha, float dropout);
};

/**
 * @brief LoRA injector
 */
class LoraInjector {
public:
    LoraInjector() = default;
    ~LoraInjector() = default;
    
    /**
     * @brief Inject LoRA into model and freeze base weights
     * @param model GPT2Model instance
     * @param spec LoRA configuration
     */
    void inject(GPT2Model& model, const LoraSpec& spec);
    
    /**
     * @brief Merge LoRA into base weights (before inference)
     * W' = W + B @ A * scale
     */
    void merge();
    
    /**
     * @brief Restore LoRA (continue training)
     * W = W' - B @ A * scale
     */
    void unmerge();
    
    /**
     * @brief Merge all LoRALinear's LoRA into base in model
     */
    void merge_all(GPT2Model& model);
    
    /**
     * @brief Restore all LoRALinear's LoRA in model
     */
    void unmerge_all(GPT2Model& model);
    
    /**
     * @brief Collect trainable parameters (LoRA A/B only)
     * @return LoRA parameter list
     */
    std::vector<TensorPtr> collect_lora_parameters() const;
    
    /**
     * @brief Save LoRA weights to safetensors
     * @param path Output path
     */
    void save_lora_safetensors(const std::string& path) const;
    
    /**
     * @brief Load LoRA weights from safetensors
     * @param path Input path
     */
    void load_lora_safetensors(const std::string& path);
    
    /**
     * @brief Print LoRA injection info
     */
    void print_info() const;
    
    /**
     * @brief Get all LoRA trainable parameters (A and B matrices)
     * @return List of all LoRA parameters
     */
    std::vector<TensorPtr> get_trainable_params();
    
    /**
     * @brief LoRA-enhanced linear forward (wrapper function)
     * @param x Input [*, in]
     * @param W Base weight [in, out]
     * @param bias Base bias [out] (optional)
     * @param lora LoRA state (optional)
     * @param training Whether training mode (affects dropout)
     * @return Output [*, out]
     */
    static TensorPtr lora_linear_forward(const TensorPtr& x,
                                        const TensorPtr& W,
                                        const TensorPtr& bias,
                                        const LoraState* lora,
                                        bool training = false);

private:
    struct Hook {
        std::string name;
        TensorPtr* W_ptr;
        TensorPtr* bias_ptr;
        LoraState state;
        // When acting only on partial columns of weights (e.g., Q/K/V split), specify column range
        // Column range is [col_offset, col_offset + col_size)
        int64_t col_offset = 0;
        int64_t col_size = -1;  // -1 means covering entire out dimension
    };
    
    std::vector<Hook> hooks_;
    LoraSpec spec_;
    bool merged_ = false;
    int num_layers_ = 0;
    
    // Internal helpers
    void inject_qkv_split(GPT2Model& model, int layer_idx, int rank, float alpha, float dropout);
    void inject_qkv_fused(GPT2Model& model, int layer_idx, int rank, float alpha, float dropout);
    void inject_layer(GPT2Model& model, int layer_idx, const std::string& layer_name,
                     int64_t in_features, int64_t out_features,
                     int rank, float alpha, float dropout);
};

}  // namespace ops

