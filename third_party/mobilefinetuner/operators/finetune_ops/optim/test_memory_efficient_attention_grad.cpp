#include "../core/ops.h"
#include "../core/autograd_engine.h"
#include "../core/memory_efficient_attention.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace ops;

static TensorPtr make_scalar_loss(const TensorPtr& context, const TensorPtr& upstream) {
    auto prod = mul(context, upstream);
    auto s1 = sum(prod, -1, false);
    auto s2 = sum(s1, -1, false);
    auto s3 = sum(s2, -1, false);
    return sum(s3, -1, false);
}

static float tensor_l2_norm(const TensorPtr& t) {
    const float* p = t->data<float>();
    double acc = 0.0;
    for (int64_t i = 0; i < t->numel(); ++i) {
        acc += static_cast<double>(p[i]) * static_cast<double>(p[i]);
    }
    return static_cast<float>(std::sqrt(acc));
}

static float eval_loss_no_grad(const TensorPtr& q, const TensorPtr& k, const TensorPtr& v, const TensorPtr& upstream) {
    MemoryEfficientAttentionConfig cfg;
    cfg.use_causal_mask = true;
    cfg.scale = 0.0f;  // auto: 1/sqrt(head_dim)
    auto ctx = memory_efficient_attention(q, k, v, nullptr, nullptr, cfg);
    auto loss = make_scalar_loss(ctx, upstream);
    return loss->data<float>()[0];
}

static std::vector<int64_t> sample_indices(int64_t n) {
    std::vector<int64_t> idx;
    if (n <= 0) return idx;
    idx.push_back(0);
    idx.push_back(n / 5);
    idx.push_back((2 * n) / 5);
    idx.push_back((3 * n) / 5);
    idx.push_back((4 * n) / 5);
    idx.push_back(n - 1);
    std::sort(idx.begin(), idx.end());
    idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
    return idx;
}

static bool check_numeric_subset(const TensorPtr& ana_grad,
                                 const TensorPtr& base_q,
                                 const TensorPtr& base_k,
                                 const TensorPtr& base_v,
                                 const TensorPtr& upstream,
                                 char target_name,
                                 float eps,
                                 float tol_rel) {
    const auto idx = sample_indices(ana_grad->numel());
    const float* g = ana_grad->data<float>();
    float max_rel = 0.0f;

    for (int64_t i : idx) {
        auto q_pos = base_q->clone(); q_pos->set_requires_grad(false);
        auto q_neg = base_q->clone(); q_neg->set_requires_grad(false);
        auto k_pos = base_k->clone(); k_pos->set_requires_grad(false);
        auto k_neg = base_k->clone(); k_neg->set_requires_grad(false);
        auto v_pos = base_v->clone(); v_pos->set_requires_grad(false);
        auto v_neg = base_v->clone(); v_neg->set_requires_grad(false);

        switch (target_name) {
            case 'q':
                q_pos->data<float>()[i] += eps;
                q_neg->data<float>()[i] -= eps;
                break;
            case 'k':
                k_pos->data<float>()[i] += eps;
                k_neg->data<float>()[i] -= eps;
                break;
            case 'v':
                v_pos->data<float>()[i] += eps;
                v_neg->data<float>()[i] -= eps;
                break;
            default:
                return false;
        }

        float lp = eval_loss_no_grad(q_pos, k_pos, v_pos, upstream);
        float ln = eval_loss_no_grad(q_neg, k_neg, v_neg, upstream);
        float gn = (lp - ln) / (2.0f * eps);

        float denom = std::max(1e-4f, std::fabs(gn));
        float rel = std::fabs(g[i] - gn) / denom;
        if (rel > max_rel) max_rel = rel;
    }

    std::cout << "  [" << target_name << "] numeric subset max_rel = " << max_rel << std::endl;
    return std::isfinite(max_rel) && max_rel <= tol_rel;
}

int main() {
    try {
        std::cout << "========== Memory-Efficient Attention Grad Test ==========\n" << std::endl;

        const int64_t B = 1, H = 2, S = 4, D = 8;
        auto q = zeros({B, H, S, D}, kFloat32, kCPU);
        auto k = zeros({B, H, S, D}, kFloat32, kCPU);
        auto v = zeros({B, H, S, D}, kFloat32, kCPU);
        q->set_requires_grad(true);
        k->set_requires_grad(true);
        v->set_requires_grad(true);

        // Deterministic initialization.
        for (int64_t i = 0; i < q->numel(); ++i) q->data<float>()[i] = 0.01f * static_cast<float>((i % 17) - 8);
        for (int64_t i = 0; i < k->numel(); ++i) k->data<float>()[i] = 0.012f * static_cast<float>((i % 13) - 6);
        for (int64_t i = 0; i < v->numel(); ++i) v->data<float>()[i] = 0.014f * static_cast<float>((i % 19) - 9);

        auto upstream = zeros({B, H, S, D}, kFloat32, kCPU);
        for (int64_t i = 0; i < upstream->numel(); ++i) {
            upstream->data<float>()[i] = 0.02f * static_cast<float>((i % 11) - 5);
        }

        MemoryEfficientAttentionConfig cfg;
        cfg.use_causal_mask = true;
        cfg.scale = 0.0f;  // auto
        auto ctx = memory_efficient_attention(q, k, v, nullptr, nullptr, cfg);
        auto loss = make_scalar_loss(ctx, upstream);
        std::cout << "  loss = " << loss->data<float>()[0] << std::endl;

        using namespace ops::autograd;
        Engine::instance().run_backward({loss}, {nullptr});

        auto gq = q->grad();
        auto gk = k->grad();
        auto gv = v->grad();
        if (!gq || !gk || !gv) {
            std::cerr << "[FAIL] missing q/k/v gradient" << std::endl;
            return 1;
        }

        float n_q = tensor_l2_norm(gq);
        float n_k = tensor_l2_norm(gk);
        float n_v = tensor_l2_norm(gv);
        std::cout << "  ||gq|| = " << n_q << ", ||gk|| = " << n_k << ", ||gv|| = " << n_v << std::endl;

        bool ok = true;
        auto in_range = [](float x) { return std::isfinite(x) && x > 1e-7f && x < 1e6f; };
        if (!in_range(n_q) || !in_range(n_k) || !in_range(n_v)) {
            std::cerr << "[FAIL] q/k/v gradient norm out of expected range" << std::endl;
            ok = false;
        }

        // Numeric subset checks to guard autograd correctness regressions.
        const float eps = 1e-3f;
        const float tol_rel = 0.2f;
        if (!check_numeric_subset(gq, q, k, v, upstream, 'q', eps, tol_rel)) ok = false;
        if (!check_numeric_subset(gk, q, k, v, upstream, 'k', eps, tol_rel)) ok = false;
        if (!check_numeric_subset(gv, q, k, v, upstream, 'v', eps, tol_rel)) ok = false;

        std::cout << (ok ? "[PASS]" : "[FAIL]") << std::endl;
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}

