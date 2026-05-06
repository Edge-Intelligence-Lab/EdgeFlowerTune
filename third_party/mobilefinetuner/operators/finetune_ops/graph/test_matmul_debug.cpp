/**
 * @file test_matmul_debug.cpp
 * @brief Three unit tests to locate matmul bugs in attention
 */

#include "../core/tensor.h"
#include "../core/ops.h"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace ops;

void test_ones_v() {
    std::cout << "\n========== Test 1: Ones-V (detect /K bug) ==========" << std::endl;
    
    // probs: [1, 1, 5, 5] - valid probability matrix (each row sum=1)
    std::vector<float> probs_data = {
        // Each row sum=1
        0.2, 0.2, 0.2, 0.2, 0.2,  // row 0
        0.3, 0.1, 0.4, 0.1, 0.1,  // row 1
        0.5, 0.1, 0.1, 0.2, 0.1,  // row 2
        0.1, 0.3, 0.2, 0.3, 0.1,  // row 3
        0.2, 0.3, 0.1, 0.1, 0.3   // row 4
    };
    auto probs = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 5, 5}, 
                                          probs_data.data(), kFloat32, kCPU);
    
    // V: [1, 1, 5, 4] - all 1s
    auto v = full({1, 1, 5, 4}, 1.0f, kFloat32, kCPU);
    
    // context = probs @ V
    auto context = matmul(probs, v);
    
    // Verify: should all be 1
    const float* ctx_data = context->data<float>();
    bool pass = true;
    float max_err = 0.0f;
    for (int64_t i = 0; i < context->numel(); ++i) {
        float err = std::abs(ctx_data[i] - 1.0f);
        max_err = std::max(max_err, err);
        if (err > 1e-5f) {
            pass = false;
        }
    }
    
    std::cout << "Expected: all 1s" << std::endl;
    std::cout << "Actual: mean=" << (double)std::accumulate(ctx_data, ctx_data + context->numel(), 0.0) / context->numel()
              << ", max_err=" << max_err << std::endl;
    
    if (pass) {
        std::cout << "PASS: context all 1s" << std::endl;
    } else {
        std::cout << "FAIL: context not 1, possibly /K or /S bug" << std::endl;
        std::cout << "Sample values: ";
        for (int64_t i = 0; i < std::min(static_cast<int64_t>(10), context->numel()); ++i) {
            printf("%.4f ", ctx_data[i]);
        }
        std::cout << std::endl;
    }
}

void test_identity_probs() {
    std::cout << "\n========== Test 2: Identity-probs (detect axis/stride error) ==========" << std::endl;
    
    // probs: [1, 1, 5, 5] - identity matrix
    auto probs = zeros({1, 1, 5, 5}, kFloat32, kCPU);
    float* probs_data = probs->data<float>();
    for (int i = 0; i < 5; ++i) {
        probs_data[i * 5 + i] = 1.0f;  // Diagonal is 1
    }
    
    // V: [1, 1, 5, 4] - random values
    std::vector<float> v_data = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0
    };
    auto v = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 5, 4}, 
                                      v_data.data(), kFloat32, kCPU);
    
    // context = I @ V = V
    auto context = matmul(probs, v);
    
    // Verify: context should equal V
    const float* ctx_data = context->data<float>();
    const float* v_ptr = v->data<float>();
    bool pass = true;
    float max_err = 0.0f;
    for (int64_t i = 0; i < context->numel(); ++i) {
        float err = std::abs(ctx_data[i] - v_ptr[i]);
        max_err = std::max(max_err, err);
        if (err > 1e-5f) {
            pass = false;
        }
    }
    
    std::cout << "Expected: context == V" << std::endl;
    std::cout << "Actual: max_err=" << max_err << std::endl;
    
    if (pass) {
        std::cout << "PASS: context == V" << std::endl;
    } else {
        std::cout << "FAIL: dimension reduction or stride error" << std::endl;
        std::cout << "V[0:5]:       ";
        for (int i = 0; i < 5; ++i) printf("%.2f ", v_ptr[i]);
        std::cout << "\ncontext[0:5]: ";
        for (int i = 0; i < 5; ++i) printf("%.2f ", ctx_data[i]);
        std::cout << std::endl;
    }
}

void test_manual_dotprod() {
    std::cout << "\n========== Test 3: Manual dot product comparison ==========" << std::endl;
    
    // probs: [1, 1, 5, 5]
    std::vector<float> probs_data = {
        0.1, 0.2, 0.3, 0.2, 0.2,
        0.3, 0.1, 0.4, 0.1, 0.1,
        0.5, 0.1, 0.1, 0.2, 0.1,  // Row 2 (index 2)
        0.1, 0.3, 0.2, 0.3, 0.1,
        0.2, 0.3, 0.1, 0.1, 0.3
    };
    auto probs = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 5, 5}, 
                                          probs_data.data(), kFloat32, kCPU);
    
    // V: [1, 1, 5, 4]
    std::vector<float> v_data = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0
    };
    auto v = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 5, 4}, 
                                      v_data.data(), kFloat32, kCPU);
    
    // matmul
    auto context = matmul(probs, v);
    
    // Manually calculate context[0, 0, 2, :] = sum_k probs[0,0,2,k] * v[0,0,k,:]
    int row = 2;
    std::vector<float> expected(4, 0.0f);
    for (int k = 0; k < 5; ++k) {
        float weight = probs_data[row * 5 + k];  // probs[2, k]
        for (int d = 0; d < 4; ++d) {
            expected[d] += weight * v_data[k * 4 + d];  // V[k, d]
        }
    }
    
    // Compare
    const float* ctx_data = context->data<float>();
    const float* ctx_row = ctx_data + row * 4;  // context[0, 0, 2, :]
    
    std::cout << "Manual calculation context[2,:]: ";
    for (int d = 0; d < 4; ++d) printf("%.4f ", expected[d]);
    std::cout << "\nmatmul output context[2,:]: ";
    for (int d = 0; d < 4; ++d) printf("%.4f ", ctx_row[d]);
    std::cout << std::endl;
    
    bool pass = true;
    float max_err = 0.0f;
    for (int d = 0; d < 4; ++d) {
        float err = std::abs(ctx_row[d] - expected[d]);
        max_err = std::max(max_err, err);
        if (err > 1e-5f) {
            pass = false;
        }
    }
    
    std::cout << "max_err=" << max_err << std::endl;
    if (pass) {
        std::cout << "PASS: manual calculation consistent with matmul" << std::endl;
    } else {
        std::cout << "FAIL: matmul kernel implementation has error" << std::endl;
    }
}

int main() {
    try {
        test_ones_v();
        test_identity_probs();
        test_manual_dotprod();
        
        std::cout << "\n========== Summary ==========" << std::endl;
        std::cout << "If Test 1 FAIL and output approx 0.2 -> matmul is averaging (/K or /S)" << std::endl;
        std::cout << "If Test 2 FAIL -> K-axis misalignment or stride error" << std::endl;
        std::cout << "If Test 3 FAIL -> kernel implementation has bug" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}

