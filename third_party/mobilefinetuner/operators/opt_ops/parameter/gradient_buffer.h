/**
 * @file gradient_buffer.h
 * @brief In-place gradient accumulation buffer to avoid clone-related leaks
 * 
 * Core idea:
 * - Pre-allocate a fixed-size gradient buffer for each trainable parameter.
 * - Accumulate gradients in place with axpy semantics and no temporary objects.
 * - Zero buffers in place after updates without reset/clone churn.
 */

#pragma once

#include "../core/tensor.h"
#include <vector>
#include <memory>

namespace ops {
namespace memory {

class InPlaceGradientBuffer {
private:
    struct GradBuffer {
        float* data = nullptr;
        size_t size = 0;
        bool owned = true;
    };
    
    std::vector<GradBuffer> buffers_;
    size_t total_bytes_ = 0;
    
public:
    InPlaceGradientBuffer() = default;
    ~InPlaceGradientBuffer();
    
    // Initialize fixed gradient buffers for each parameter.
    void initialize(const std::vector<TensorPtr>& params);
    
    // Accumulate gradients in place (buffer += grad).
    void accumulate(size_t param_idx, const TensorPtr& grad);
    
    // Return the accumulated gradient as a zero-copy wrapped tensor.
    TensorPtr get_gradient(size_t param_idx, const std::vector<int64_t>& shape);
    
    // Zero all gradient buffers.
    void zero();
    
    // Stats.
    size_t total_bytes() const { return total_bytes_; }
    void print_stats() const;
    
    // Non-copyable.
    InPlaceGradientBuffer(const InPlaceGradientBuffer&) = delete;
    InPlaceGradientBuffer& operator=(const InPlaceGradientBuffer&) = delete;
};

} // namespace memory
} // namespace ops
