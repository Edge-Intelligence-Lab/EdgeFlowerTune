/**
 * @file checkpoint.h
 * @brief Activation checkpointing utilities (recompute forward during backward)
 *
 * This module provides a lightweight checkpoint wrapper that:
 * - runs forward with autograd disabled (no activation saving)
 * - attaches a custom backward that recomputes forward and runs backward
 *
 * This is critical for reducing peak RSS during training on mobile devices.
 */

#pragma once

#include "tensor.h"
#include <functional>

namespace ops {

/**
 * @brief Checkpoint a function of a single tensor input
 *
 * @param fn Forward function (takes input tensor, returns output tensor)
 * @param input Input tensor
 * @return Output tensor with custom backward that recomputes forward
 */
TensorPtr checkpoint(const std::function<TensorPtr(const TensorPtr&)>& fn,
                     const TensorPtr& input);

}  // namespace ops
