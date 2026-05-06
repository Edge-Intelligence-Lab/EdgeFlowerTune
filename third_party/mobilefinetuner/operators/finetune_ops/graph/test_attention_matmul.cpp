/**
 * @file test_attention_matmul.cpp
 * @brief Definitive test: 3 tests to locate batched matmul stride/averaging/axis errors
 * 
 * Critical: Must follow exactly same path as gpt2_model.cpp:
 *   reshape -> permute -> matmul -> permute -> reshape
 */

#include "../core/tensor.h"
#include "../core/ops.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

using namespace ops;

// Helper function: compute statistics
struct Stats {
    float mean, std, max_abs;
    
    Stats(const TensorPtr& t) {
        const float* data = t->data<float>();
        int64_t n = t->numel();
        double sum = 0.0, sum_sq = 0.0;
        max_abs = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            sum += data[i];
            sum_sq += data[i] * data[i];
            max_abs = std::max(max_abs, std::abs(data[i]));
        }
        mean = sum / n;
        std = std::sqrt(sum_sq / n - mean * mean);
    }
};

void test1_ones_v() {
    std::cout << "\n========== Test 1: Ones-V (detect /K or /S averaging bug) ==========" << std::endl;
    
    const int64_t B = 1, h = 3, S = 17, Hd = 7;
    
    // Construct valid probs (each row softmax, sum to 1)
    auto probs_4d = zeros({B, h, S, S}, kFloat32, kCPU);
    float* probs_data = probs_4d->data<float>();
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hh = 0; hh < h; ++hh) {
            for (int64_t i = 0; i < S; ++i) {
                // Generate random weights and normalize
                float sum = 0.0f;
                std::vector<float> row(S);
                for (int64_t j = 0; j < S; ++j) {
                    row[j] = dist(gen);
                    sum += row[j];
                }
                // Normalize
                for (int64_t j = 0; j < S; ++j) {
                    int64_t idx = ((b * h + hh) * S + i) * S + j;
                    probs_data[idx] = row[j] / sum;
                }
            }
        }
    }
    
    // V all ones
    auto V_4d = full({B, h, S, Hd}, 1.0f, kFloat32, kCPU);
    
    // Follow complete path: consistent with gpt2_model.cpp
    // context = probs @ V  -> [B,h,S,Hd]
    auto context = matmul(probs_4d, V_4d);
    
    // Verify: should all be 1
    const float* ctx_data = context->data<float>();
    float max_err = 0.0f;
    double sum = 0.0;
    for (int64_t i = 0; i < context->numel(); ++i) {
        sum += ctx_data[i];
        max_err = std::max(max_err, std::abs(ctx_data[i] - 1.0f));
    }
    float mean = sum / context->numel();
    
    std::cout << "Expected: all 1s" << std::endl;
    std::cout << "Actual: mean=" << mean << ", max_err=" << max_err << std::endl;
    
    if (max_err < 1e-5f) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL: context is not 1" << std::endl;
        if (std::abs(mean - 0.2f) < 0.05f) {
            std::cout << "DETECTED /S averaging bug (mean approx 1/S=" << 1.0/S << ")" << std::endl;
        } else if (std::abs(mean - 1.0f/Hd) < 0.05f) {
            std::cout << "DETECTED /Hd averaging bug (mean approx 1/Hd=" << 1.0/Hd << ")" << std::endl;
        }
        std::cout << "Sample values[0:10]: ";
        for (int64_t i = 0; i < std::min(static_cast<int64_t>(10), context->numel()); ++i) {
            printf("%.4f ", ctx_data[i]);
        }
        std::cout << std::endl;
    }
}

void test2_identity_probs() {
    std::cout << "\n========== Test 2: Identity-probs (detect axis/stride error) ==========" << std::endl;
    
    const int64_t B = 1, h = 3, S = 17, Hd = 7;
    
    // probs: each head is identity matrix I_S
    auto probs_4d = zeros({B, h, S, S}, kFloat32, kCPU);
    float* probs_data = probs_4d->data<float>();
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hh = 0; hh < h; ++hh) {
            for (int64_t i = 0; i < S; ++i) {
                int64_t idx = ((b * h + hh) * S + i) * S + i;
                probs_data[idx] = 1.0f;  // Diagonal
            }
        }
    }
    
    // V: random values
    std::vector<float> v_data(B * h * S * Hd);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& val : v_data) val = dist(gen);
    
    auto V_4d = std::make_shared<Tensor>(std::vector<int64_t>{B, h, S, Hd}, 
                                         v_data.data(), kFloat32, kCPU);
    
    // context = I @ V = V
    auto context = matmul(probs_4d, V_4d);
    
    // Verify: context should equal V
    const float* ctx_data = context->data<float>();
    const float* v_ptr = V_4d->data<float>();
    float max_err = 0.0f;
    for (int64_t i = 0; i < context->numel(); ++i) {
        max_err = std::max(max_err, std::abs(ctx_data[i] - v_ptr[i]));
    }
    
    std::cout << "Expected: context == V" << std::endl;
    std::cout << "Actual: max_err=" << max_err << std::endl;
    
    if (max_err < 1e-5f) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL: axis error/stride error/batch confusion" << std::endl;
        std::cout << "Sample comparison:" << std::endl;
        std::cout << "V[0:10]:       ";
        for (int i = 0; i < 10; ++i) printf("%.4f ", v_ptr[i]);
        std::cout << "\ncontext[0:10]: ";
        for (int i = 0; i < 10; ++i) printf("%.4f ", ctx_data[i]);
        std::cout << std::endl;
    }
}

void test3_full_reference() {
    std::cout << "\n========== Test 3: Full comparison (reference implementation) ==========" << std::endl;
    
    const int64_t B = 1, h = 3, S = 17, Hd = 7;
    
    // Random probs (validated)
    std::vector<float> probs_data(B * h * S * S);
    std::mt19937 gen(456);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hh = 0; hh < h; ++hh) {
            for (int64_t i = 0; i < S; ++i) {
                float sum = 0.0f;
                std::vector<float> row(S);
                for (int64_t j = 0; j < S; ++j) {
                    row[j] = dist(gen);
                    sum += row[j];
                }
                for (int64_t j = 0; j < S; ++j) {
                    int64_t idx = ((b * h + hh) * S + i) * S + j;
                    probs_data[idx] = row[j] / sum;
                }
            }
        }
    }
    auto probs_4d = std::make_shared<Tensor>(std::vector<int64_t>{B, h, S, S}, 
                                              probs_data.data(), kFloat32, kCPU);
    
    // Random V
    std::vector<float> v_data(B * h * S * Hd);
    std::uniform_real_distribution<float> v_dist(-5.0f, 5.0f);
    for (auto& val : v_data) val = v_dist(gen);
    auto V_4d = std::make_shared<Tensor>(std::vector<int64_t>{B, h, S, Hd}, 
                                         v_data.data(), kFloat32, kCPU);
    
    // Reference implementation: triple for-loop
    std::vector<float> context_ref(B * h * S * Hd, 0.0f);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hh = 0; hh < h; ++hh) {
            for (int64_t i = 0; i < S; ++i) {
                for (int64_t d = 0; d < Hd; ++d) {
                    float sum = 0.0f;
                    for (int64_t k = 0; k < S; ++k) {
                        int64_t probs_idx = ((b * h + hh) * S + i) * S + k;
                        int64_t v_idx = ((b * h + hh) * S + k) * Hd + d;
                        sum += probs_data[probs_idx] * v_data[v_idx];
                    }
                    int64_t ctx_idx = ((b * h + hh) * S + i) * Hd + d;
                    context_ref[ctx_idx] = sum;
                }
            }
        }
    }
    
    // Matmul implementation
    auto context = matmul(probs_4d, V_4d);
    
    // Element-wise comparison
    const float* ctx_data = context->data<float>();
    float max_err = 0.0f;
    for (int64_t i = 0; i < context->numel(); ++i) {
        max_err = std::max(max_err, std::abs(ctx_data[i] - context_ref[i]));
    }
    
    // Statistics comparison
    Stats ref_stats(std::make_shared<Tensor>(std::vector<int64_t>{B, h, S, Hd}, 
                                             context_ref.data(), kFloat32, kCPU));
    Stats ctx_stats(context);
    
    std::cout << "Reference: mean=" << ref_stats.mean << ", std=" << ref_stats.std << std::endl;
    std::cout << "Matmul:    mean=" << ctx_stats.mean << ", std=" << ctx_stats.std << std::endl;
    std::cout << "max_err=" << max_err << std::endl;
    
    if (max_err < 1e-5f) {
        std::cout << "PASS: matmul fully consistent with reference" << std::endl;
    } else {
        std::cout << "FAIL: matmul kernel has bug" << std::endl;
        std::cout << "Sample comparison[0:10]:" << std::endl;
        std::cout << "Ref:     ";
        for (int i = 0; i < 10; ++i) printf("%.4f ", context_ref[i]);
        std::cout << "\nMatmul:  ";
        for (int i = 0; i < 10; ++i) printf("%.4f ", ctx_data[i]);
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "========== Batched Matmul Diagnostic Test ==========" << std::endl;
    std::cout << "Dimensions: B=1, h=3, S=17, Hd=7 (non-power-of-2, most likely to expose stride bugs)\n" << std::endl;
    
    try {
        test1_ones_v();
        test2_identity_probs();
        test3_full_reference();
        
        std::cout << "\n========== Diagnostic Conclusions ==========" << std::endl;
        std::cout << "• If Test 1 FAIL and mean approx 1/S -> matmul is averaging (/K bug)" << std::endl;
        std::cout << "• If Test 2 FAIL -> K-axis reduction error or stride confusion" << std::endl;
        std::cout << "• If Test 3 FAIL -> batch stepping/index calculation error" << std::endl;
        std::cout << "• If all PASS -> problem is in permute/reshape, not matmul" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nException: " << e.what() << std::endl;
        return 1;
    }
}

