#include "../core/ops.h"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace ops;

static float max_abs_diff(const TensorPtr& a, const TensorPtr& b) {
    if (!a || !b || a->shape() != b->shape()) return 1e9f;
    const float* ad = a->data<float>();
    const float* bd = b->data<float>();
    float m = 0.0f;
    for (int64_t i = 0; i < a->numel(); ++i) {
        m = std::max(m, std::abs(ad[i] - bd[i]));
    }
    return m;
}

static TensorPtr build_causal_mask(int64_t s) {
    auto m = zeros({s, s}, kFloat32, kCPU);
    float* md = m->data<float>();
    for (int64_t i = 0; i < s; ++i) {
        for (int64_t j = i + 1; j < s; ++j) {
            md[i * s + j] = -1e10f;
        }
    }
    return m;
}

int main() {
    try {
        const int64_t B = 2;
        const int64_t H = 3;
        const int64_t S = 4;

        auto causal = build_causal_mask(S);

        // pad mask [B,1,1,S]
        auto pad_4d = zeros({B, 1, 1, S}, kFloat32, kCPU);
        float* p4 = pad_4d->data<float>();
        p4[0 * S + 3] = -1e10f;
        p4[1 * S + 2] = -1e10f;

        auto make_scores = [&]() -> TensorPtr {
            auto scores = zeros({B, H, S, S}, kFloat32, kCPU);
            float* sd = scores->data<float>();
            for (int64_t i = 0; i < scores->numel(); ++i) {
                sd[i] = std::sin(0.01f * static_cast<float>(i)) + 0.1f * static_cast<float>(i % 7);
            }
            return scores;
        };

        // Forward equivalence on detached graph.
        auto ref_logits = add(add(make_scores(), causal), pad_4d);
        auto ref_probs = softmax(ref_logits, -1);
        auto fused_probs = masked_softmax(make_scores(), causal, pad_4d, -1);

        const float fwd_err = max_abs_diff(ref_probs, fused_probs);
        if (fwd_err > 1e-6f) {
            std::cerr << "[FAIL] masked_softmax forward mismatch, max_abs=" << fwd_err << std::endl;
            return 1;
        }

        // Backward equivalence. Build and backward each graph independently:
        // current autograd engine clears graph after backward().
        auto scores_ref = make_scores();
        scores_ref->set_requires_grad(true);
        auto ref_loss = mean(softmax(add(add(scores_ref, causal), pad_4d), -1));
        ref_loss->backward();

        auto scores_fused = make_scores();
        scores_fused->set_requires_grad(true);
        auto fused_loss = mean(masked_softmax(scores_fused, causal, pad_4d, -1));
        fused_loss->backward();

        if (!scores_ref->grad() || !scores_fused->grad()) {
            std::cerr << "[FAIL] missing grad: ref=" << (scores_ref->grad() ? "ok" : "null")
                      << " fused=" << (scores_fused->grad() ? "ok" : "null") << std::endl;
            return 2;
        }

        const float grad_err = max_abs_diff(scores_ref->grad(), scores_fused->grad());
        if (grad_err > 1e-5f) {
            std::cerr << "[FAIL] masked_softmax grad mismatch, max_abs=" << grad_err << std::endl;
            return 2;
        }

        // Also validate [B,S] pad-mask format.
        auto pad_2d = zeros({B, S}, kFloat32, kCPU);
        float* p2 = pad_2d->data<float>();
        p2[0 * S + 3] = -1e10f;
        p2[1 * S + 2] = -1e10f;
        auto fused_probs_2d = masked_softmax(scores_fused->detach(), causal, pad_2d, -1);
        const float pad_fmt_err = max_abs_diff(fused_probs, fused_probs_2d);
        if (pad_fmt_err > 1e-6f) {
            std::cerr << "[FAIL] masked_softmax pad format mismatch, max_abs=" << pad_fmt_err << std::endl;
            return 3;
        }

        std::cout << "masked_softmax forward/grad test passed\n";
        std::cout << "  forward max_abs_diff: " << fwd_err << "\n";
        std::cout << "  grad max_abs_diff   : " << grad_err << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] " << e.what() << std::endl;
        return 10;
    }
}
