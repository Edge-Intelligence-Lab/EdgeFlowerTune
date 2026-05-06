#include "../core/ops.h"
#include "../core/autograd_engine.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ops;

static float rel_diff(const TensorPtr& a, const TensorPtr& b) {
    const float* pa = a->data<float>();
    const float* pb = b->data<float>();
    double nb = 0.0;
    double nd = 0.0;
    int64_t n = a->numel();
    for (int64_t i = 0; i < n; ++i) {
        const double va = static_cast<double>(pa[i]);
        const double vb = static_cast<double>(pb[i]);
        nd += (va - vb) * (va - vb);
        nb += vb * vb;
    }
    if (nb == 0.0) return static_cast<float>(std::sqrt(nd));
    return static_cast<float>(std::sqrt(nd / nb));
}

struct BucketStats {
    int64_t count = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    double sum_abs = 0.0;
    double sum_rel = 0.0;
};

struct GradMetrics {
    double rms_rel = std::numeric_limits<double>::infinity();
    double max_abs = 0.0;
    double max_rel = 0.0;
    bool finite = true;
    BucketStats buckets[4];  // tiny, small, medium, large
};

static int bucket_idx(double ref_abs) {
    if (ref_abs < 1e-6) return 0;   // tiny
    if (ref_abs < 1e-4) return 1;   // small
    if (ref_abs < 1e-2) return 2;   // medium
    return 3;                       // large
}

static GradMetrics calc_metrics(const TensorPtr& analytic, const TensorPtr& numeric, double rel_floor = 1e-7) {
    GradMetrics m;
    const float* pa = analytic->data<float>();
    const float* pn = numeric->data<float>();
    const int64_t n = analytic->numel();

    double rel_sq = 0.0;
    int64_t rel_cnt = 0;
    for (int64_t i = 0; i < n; ++i) {
        const double a = static_cast<double>(pa[i]);
        const double r = static_cast<double>(pn[i]);
        const double abs_err = std::fabs(a - r);
        const double denom = std::max(std::fabs(r), rel_floor);
        const double rel_err = abs_err / denom;

        if (!std::isfinite(abs_err) || !std::isfinite(rel_err)) {
            m.finite = false;
        }

        m.max_abs = std::max(m.max_abs, abs_err);
        m.max_rel = std::max(m.max_rel, rel_err);

        rel_sq += rel_err * rel_err;
        rel_cnt++;

        const int b = bucket_idx(std::fabs(r));
        auto& bs = m.buckets[b];
        bs.count++;
        bs.max_abs = std::max(bs.max_abs, abs_err);
        bs.max_rel = std::max(bs.max_rel, rel_err);
        bs.sum_abs += abs_err;
        bs.sum_rel += rel_err;
    }
    m.rms_rel = (rel_cnt > 0) ? std::sqrt(rel_sq / static_cast<double>(rel_cnt)) : 0.0;
    return m;
}

static void print_metrics(const std::string& tag, float eps, const GradMetrics& m) {
    std::cout << "[" << tag << "][eps=" << eps << "]"
              << " rms_rel=" << m.rms_rel
              << " max_rel=" << m.max_rel
              << " max_abs=" << m.max_abs
              << " | tiny(max_abs=" << m.buckets[0].max_abs
              << ",max_rel=" << m.buckets[0].max_rel
              << ",n=" << m.buckets[0].count << ")"
              << " small(max_rel=" << m.buckets[1].max_rel
              << ",n=" << m.buckets[1].count << ")"
              << " mid(max_rel=" << m.buckets[2].max_rel
              << ",n=" << m.buckets[2].count << ")"
              << " large(max_rel=" << m.buckets[3].max_rel
              << ",n=" << m.buckets[3].count << ")"
              << std::endl;
}

static double metrics_score(const GradMetrics& m) {
    // Lower is better. Favor RMS relative error while penalizing outliers.
    return m.rms_rel + 0.15 * m.max_rel + 5e2 * std::min(m.max_abs, 1e-4);
}

template <typename LossFn>
static TensorPtr finite_diff_grad(const TensorPtr& base, float eps, LossFn&& loss_fn) {
    auto out = zeros(base->shape(), kFloat32, kCPU);
    for (int64_t i = 0; i < base->numel(); ++i) {
        auto p = base->clone();
        auto n = base->clone();
        p->data<float>()[i] += eps;
        n->data<float>()[i] -= eps;
        const float lp = loss_fn(p);
        const float ln = loss_fn(n);
        out->data<float>()[i] = (lp - ln) / (2.0f * eps);
    }
    return out;
}

template <typename LossFn>
static std::pair<float, GradMetrics> find_best_eps(const std::string& tag,
                                                   const TensorPtr& analytic,
                                                   const TensorPtr& base,
                                                   const std::vector<float>& eps_list,
                                                   LossFn&& loss_fn,
                                                   TensorPtr* best_numeric_out) {
    float best_eps = eps_list.front();
    GradMetrics best_metrics;
    double best_score = std::numeric_limits<double>::infinity();
    TensorPtr best_numeric = nullptr;

    for (float eps : eps_list) {
        auto numeric = finite_diff_grad(base, eps, loss_fn);
        auto m = calc_metrics(analytic, numeric);
        print_metrics(tag, eps, m);
        const double score = metrics_score(m);
        if (score < best_score) {
            best_score = score;
            best_eps = eps;
            best_metrics = m;
            best_numeric = numeric;
        }
    }
    if (best_numeric_out) *best_numeric_out = best_numeric;
    std::cout << "[" << tag << "] best_eps=" << best_eps
              << " best_rms_rel=" << best_metrics.rms_rel
              << " best_max_rel=" << best_metrics.max_rel
              << " best_max_abs=" << best_metrics.max_abs << std::endl;
    return {best_eps, best_metrics};
}

struct Criteria {
    double tiny_abs_tol;
    double small_rel_tol;
    double mid_rel_tol;
    double large_rel_tol;
    double rms_rel_tol;
};

static bool pass_by_buckets(const std::string& tag, const GradMetrics& m, const Criteria& c) {
    bool ok = true;
    if (!m.finite) {
        std::cout << "[" << tag << "] FAIL: non-finite metric" << std::endl;
        return false;
    }
    if (m.rms_rel > c.rms_rel_tol) {
        std::cout << "[" << tag << "] FAIL: rms_rel " << m.rms_rel
                  << " > " << c.rms_rel_tol << std::endl;
        ok = false;
    }
    if (m.buckets[0].max_abs > c.tiny_abs_tol) {
        std::cout << "[" << tag << "] FAIL: tiny max_abs " << m.buckets[0].max_abs
                  << " > " << c.tiny_abs_tol << std::endl;
        ok = false;
    }
    if (m.buckets[1].count > 0 && m.buckets[1].max_rel > c.small_rel_tol) {
        std::cout << "[" << tag << "] FAIL: small max_rel " << m.buckets[1].max_rel
                  << " > " << c.small_rel_tol << std::endl;
        ok = false;
    }
    if (m.buckets[2].count > 0 && m.buckets[2].max_rel > c.mid_rel_tol) {
        std::cout << "[" << tag << "] FAIL: mid max_rel " << m.buckets[2].max_rel
                  << " > " << c.mid_rel_tol << std::endl;
        ok = false;
    }
    if (m.buckets[3].count > 0 && m.buckets[3].max_rel > c.large_rel_tol) {
        std::cout << "[" << tag << "] FAIL: large max_rel " << m.buckets[3].max_rel
                  << " > " << c.large_rel_tol << std::endl;
        ok = false;
    }
    if (ok) std::cout << "[" << tag << "] PASS" << std::endl;
    return ok;
}

int main() {
    // Small-dim attention: B=1, H=1, S=3, Hd=4
    const int64_t B = 1, H = 1, S = 3, Hd = 4;
    const float scale = std::pow(256.0f, -0.5f);  // Gemma3 default query_pre_attn_scalar=256

    // Prepare q, k with requires_grad
    auto q = zeros({B, H, S, Hd}, kFloat32, kCPU);
    auto k = zeros({B, H, S, Hd}, kFloat32, kCPU);
    q->set_requires_grad(true);
    k->set_requires_grad(true);

    // Fill deterministic values
    {
        float* qd = q->data<float>();
        float* kd = k->data<float>();
        for (int i = 0; i < q->numel(); ++i) qd[i] = 0.1f * (i % 7 - 3);
        for (int i = 0; i < k->numel(); ++i) kd[i] = 0.07f * (i % 5 - 2);
    }

    // scores = (q @ k^T) * scale
    auto k_t = transpose(k, 2, 3);      // [B,H,Hd,S]
    auto scores_mat = matmul(q, k_t);   // [B,H,S,S]
    scores_mat->retain_grad();
    auto scores = mul(scores_mat, scale);
    scores->retain_grad();

    // Softmax over last dim
    auto probs = softmax(scores, -1);   // [B,H,S,S]

    // Upstream grad on probs (deterministic)
    auto g_probs = zeros(probs->shape(), kFloat32, kCPU);
    {
        float* gp = g_probs->data<float>();
        for (int i = 0; i < g_probs->numel(); ++i) gp[i] = 0.03f * ((i % 11) - 5);
    }

    // Run backward through engine: dL/dq, dL/dk
    {
        using namespace ops::autograd;
        Engine::instance().run_backward({probs}, {g_probs});
    }

    auto gq_engine = q->grad();
    auto gk_engine = k->grad();
    auto gscores_engine = scores->grad();      // dL/dscores
    auto gmat_engine = scores_mat->grad();     // dL/d(scores_mat)
    if (!gscores_engine) gscores_engine = zeros(scores->shape(), kFloat32, kCPU);
    if (!gmat_engine) gmat_engine = zeros(scores_mat->shape(), kFloat32, kCPU);

    // Reference grads (analytic)
    auto go_y = mul(g_probs, probs);
    int last_dim = static_cast<int>(go_y->shape().size()) - 1;
    auto sum_go = sum(go_y, last_dim, true);
    auto diff = sub(g_probs, sum_go);
    auto grad_scores = mul(probs, diff);
    auto grad_mat = mul(grad_scores, scale);   // dL/d(scores_mat)
    auto grad_q_ref = matmul(grad_mat, k);                   // [B,H,S,Hd]
    auto grad_k_ref = matmul(transpose(grad_mat, -2, -1), q);    // [B,H,S,Hd]

    float rds = rel_diff(gscores_engine, grad_scores);
    float rdm = rel_diff(gmat_engine, grad_mat);
    float rdq = rel_diff(gq_engine, grad_q_ref);
    float rdk = rel_diff(gk_engine, grad_k_ref);
    std::cout << "[T5.QKT][analytic] rel_diff dScores=" << rds
              << " dMat=" << rdm
              << " dq=" << rdq
              << " dk=" << rdk << std::endl;

    bool ok = (rdq < 1e-6f) && (rdk < 1e-6f) && (rds < 1e-6f) && (rdm < 1e-6f);
    std::cout << (ok ? "[PASS]" : "[FAIL]") << std::endl;

    // Additional: standalone softmax backward check
    auto x = zeros({2, 3}, kFloat32, kCPU);
    x->set_requires_grad(true);
    {
        float* xd = x->data<float>();
        for (int i = 0; i < x->numel(); ++i) xd[i] = 0.1f * (i - 2);
    }
    auto y = softmax(x, -1);
    auto gy = zeros(y->shape(), kFloat32, kCPU);
    {
        float* gd = gy->data<float>();
        for (int i = 0; i < gy->numel(); ++i) gd[i] = 0.05f * ((i % 5) - 2);
    }
    {
        using namespace ops::autograd;
        Engine::instance().run_backward({y}, {gy});
    }
    auto gx_engine = x->grad();
    auto go_y2 = mul(gy, y);
    int last_dim2 = static_cast<int>(go_y2->shape().size()) - 1;
    auto sum_go2 = sum(go_y2, last_dim2, true);
    auto diff2 = sub(gy, sum_go2);
    auto gx_ref = mul(y, diff2);
    float rdsf = rel_diff(gx_engine, gx_ref);
    std::cout << "[SoftmaxOnly][analytic] rel_diff dX=" << rdsf << std::endl;
    ok = ok && (rdsf < 1e-6f);

    const std::vector<float> eps_list = {1e-1f, 3e-2f, 1e-2f, 3e-3f, 1e-3f, 3e-4f, 1e-4f};

    // Softmax-only FD with eps sweep
    TensorPtr numeric_softmax = nullptr;
    auto softmax_loss_fn = [&](const TensorPtr& xvar) -> float {
        auto yy = softmax(xvar, -1);
        auto l = sum(mul(yy, gy));
        return l->data<float>()[0];
    };
    auto [best_eps_softmax, m_softmax] =
        find_best_eps("SoftmaxOnly", gx_engine, x, eps_list, softmax_loss_fn, &numeric_softmax);
    (void)best_eps_softmax;
    ok = pass_by_buckets("SoftmaxOnly", m_softmax,
                         {/*tiny_abs_tol=*/5e-4, /*small_rel_tol=*/0.35,
                          /*mid_rel_tol=*/0.18, /*large_rel_tol=*/0.08,
                          /*rms_rel_tol=*/0.18}) && ok;

    // Numeric check for full chain wrt q and k (small dims)
    auto G = zeros(probs->shape(), kFloat32, kCPU);
    {
        float* gd = G->data<float>();
        for (int i = 0; i < G->numel(); ++i) gd[i] = 0.25f * ((i % 7) - 3);
    }

    // Clear previous grads and re-run backward with G.
    {
        using namespace ops::autograd;
        if (q->grad()) q->zero_grad();
        if (k->grad()) k->zero_grad();
        auto k_t2 = transpose(k, 2, 3);
        auto scores_mat2 = matmul(q, k_t2);
        auto scores2 = mul(scores_mat2, scale);
        auto probs2 = softmax(scores2, -1);
        Engine::instance().run_backward({probs2}, {G});
    }
    gq_engine = q->grad();
    gk_engine = k->grad();

    TensorPtr numeric_q = nullptr;
    auto q_loss_fn = [&](const TensorPtr& qv) -> float {
        auto s = mul(matmul(qv, transpose(k, 2, 3)), scale);
        auto yy = softmax(s, -1);
        auto l = sum(mul(yy, G));
        return l->data<float>()[0];
    };
    auto [best_eps_q, m_q] = find_best_eps("T5.QKT:dq", gq_engine, q, eps_list, q_loss_fn, &numeric_q);
    (void)best_eps_q;
    ok = pass_by_buckets("T5.QKT:dq", m_q,
                         {/*tiny_abs_tol=*/8e-4, /*small_rel_tol=*/0.55,
                          /*mid_rel_tol=*/0.30, /*large_rel_tol=*/0.12,
                          /*rms_rel_tol=*/0.22}) && ok;

    TensorPtr numeric_k = nullptr;
    auto k_loss_fn = [&](const TensorPtr& kv) -> float {
        auto s = mul(matmul(q, transpose(kv, 2, 3)), scale);
        auto yy = softmax(s, -1);
        auto l = sum(mul(yy, G));
        return l->data<float>()[0];
    };
    auto [best_eps_k, m_k] = find_best_eps("T5.QKT:dk", gk_engine, k, eps_list, k_loss_fn, &numeric_k);
    (void)best_eps_k;
    ok = pass_by_buckets("T5.QKT:dk", m_k,
                         {/*tiny_abs_tol=*/8e-4, /*small_rel_tol=*/0.55,
                          /*mid_rel_tol=*/0.30, /*large_rel_tol=*/0.12,
                          /*rms_rel_tol=*/0.22}) && ok;

    // Isolate matmul backward (no softmax): C = A @ B, L = sum(C * M)
    auto A = zeros(q->shape(), kFloat32, kCPU);
    auto Bten = zeros(k_t->shape(), kFloat32, kCPU);
    A->set_requires_grad(true);
    Bten->set_requires_grad(true);
    {
        float* ad = A->data<float>();
        float* bd = Bten->data<float>();
        for (int i = 0; i < A->numel(); ++i) ad[i] = 0.1f * (i % 9 - 4);
        for (int i = 0; i < Bten->numel(); ++i) bd[i] = 0.07f * (i % 7 - 3);
    }
    auto C = matmul(A, Bten);
    auto M = zeros(C->shape(), kFloat32, kCPU);
    {
        float* md = M->data<float>();
        for (int i = 0; i < M->numel(); ++i) md[i] = 0.03f * (i % 5 - 2);
    }
    {
        using namespace ops::autograd;
        Engine::instance().run_backward({C}, {M});
    }
    auto gA_engine = A->grad();
    auto gB_engine = Bten->grad();

    TensorPtr gA_num = nullptr;
    auto A_loss_fn = [&](const TensorPtr& Av) -> float {
        auto Cv = matmul(Av, Bten);
        auto l = sum(mul(Cv, M));
        return l->data<float>()[0];
    };
    auto [best_eps_A, m_A] = find_best_eps("MatmulOnly:dA", gA_engine, A, eps_list, A_loss_fn, &gA_num);
    (void)best_eps_A;
    ok = pass_by_buckets("MatmulOnly:dA", m_A,
                         {/*tiny_abs_tol=*/3e-4, /*small_rel_tol=*/0.18,
                          /*mid_rel_tol=*/0.10, /*large_rel_tol=*/0.05,
                          /*rms_rel_tol=*/0.10}) && ok;

    TensorPtr gB_num = nullptr;
    auto B_loss_fn = [&](const TensorPtr& Bv) -> float {
        auto Cv = matmul(A, Bv);
        auto l = sum(mul(Cv, M));
        return l->data<float>()[0];
    };
    auto [best_eps_B, m_B] = find_best_eps("MatmulOnly:dB", gB_engine, Bten, eps_list, B_loss_fn, &gB_num);
    (void)best_eps_B;
    ok = pass_by_buckets("MatmulOnly:dB", m_B,
                         {/*tiny_abs_tol=*/3e-4, /*small_rel_tol=*/0.18,
                          /*mid_rel_tol=*/0.10, /*large_rel_tol=*/0.05,
                          /*rms_rel_tol=*/0.10}) && ok;

    std::cout << (ok ? "[FINAL PASS]" : "[FINAL FAIL]") << std::endl;
    return ok ? 0 : 1;
}

