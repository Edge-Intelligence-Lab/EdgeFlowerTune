/**
 * @file checkpoint.cpp
 * @brief Activation checkpointing implementation
 */

#include "checkpoint.h"
#include "autograd_engine.h"
#include "backward_functions.h"
#include "ops.h"

#include <iostream>

namespace ops {

namespace {

bool ckpt_debug_enabled() {
    const char* v = std::getenv("OPS_CKPT_DEBUG");
    if (!v) return false;
    return std::string(v) == "1";
}

class CheckpointBackward : public BackwardFunction {
public:
    CheckpointBackward(std::function<TensorPtr(const TensorPtr&)> fn,
                       TensorPtr input)
        : fn_(std::move(fn)), input_(std::move(input)) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override {
        if (!input_) {
            throw TensorError("CheckpointBackward: input is null");
        }

        if (ckpt_debug_enabled()) {
            std::cout << "[CKPT] backward begin, input=" << input_.get()
                      << " shape=" << input_->to_string() << std::endl;
        }

        // Create a safe detached copy for recompute to avoid aliasing issues
        auto input_view = input_->clone();
        input_view->set_requires_grad(true);

        auto& eng = autograd::Engine::instance();
        const bool prev_enabled = eng.is_enabled();
        const bool prev_suppress = eng.suppress_external_grad_nodes();

        // Swap out engine state to allow nested backward safely
        auto saved_state = eng.take_state();
        eng.set_enabled(true);
        eng.set_suppress_external_grad_nodes(true);

        // Recompute forward with autograd enabled
        if (ckpt_debug_enabled()) {
            std::cout << "[CKPT] recompute forward..." << std::endl;
        }
        auto output = fn_(input_view);
        if (output && !output->requires_grad()) {
            output->set_requires_grad(true);
        }

        // Run backward on the recomputed graph
        if (ckpt_debug_enabled()) {
            std::cout << "[CKPT] run backward..." << std::endl;
        }
        if (output) {
            output->backward(grad_output);
        }

        // Restore engine state
        eng.set_enabled(prev_enabled);
        eng.set_suppress_external_grad_nodes(prev_suppress);
        eng.restore_state(std::move(saved_state));

        auto grad_input = input_view->grad();
        if (!grad_input) {
            grad_input = zeros(input_->shape(), input_->dtype(), input_->device());
        }

        if (ckpt_debug_enabled()) {
            std::cout << "[CKPT] backward done, grad_input=" << grad_input.get() << std::endl;
        }

        return {grad_input};
    }

private:
    std::function<TensorPtr(const TensorPtr&)> fn_;
    TensorPtr input_;
};

}  // namespace

TensorPtr checkpoint(const std::function<TensorPtr(const TensorPtr&)>& fn,
                     const TensorPtr& input) {
    if (!fn || !input) {
        throw TensorError("checkpoint: fn or input is null");
    }

    auto& eng = autograd::Engine::instance();
    const bool prev_enabled = eng.is_enabled();

    // Forward with autograd disabled to avoid saving activations
    if (ckpt_debug_enabled()) {
        std::cout << "[CKPT] forward begin, input=" << input.get()
                  << " shape=" << input->to_string() << std::endl;
    }
    eng.set_enabled(false);
    TensorPtr output = fn(input);
    eng.set_enabled(prev_enabled);

    // Attach custom backward if needed
    if (input->requires_grad()) {
        output->set_requires_grad(true);
        auto backward_fn = std::make_shared<CheckpointBackward>(fn, input);
        eng.register_node(output, {input}, backward_fn);
    }

    if (ckpt_debug_enabled()) {
        std::cout << "[CKPT] forward end, output=" << output.get()
                  << " shape=" << output->to_string() << std::endl;
    }

    return output;
}

}  // namespace ops
