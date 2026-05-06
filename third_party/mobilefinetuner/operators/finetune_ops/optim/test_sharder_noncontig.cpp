#include "opt_ops/sharding/parameter_sharder.h"
#include "finetune_ops/core/ops.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

using namespace ops;
using namespace ops::sharding;

int main() {
    std::filesystem::remove_all("/tmp/test_sharder_noncontig");

    auto base = std::make_shared<Tensor>(std::vector<int64_t>{3, 4}, kFloat32, kCPU);
    float* p = base->data<float>();
    for (int i = 0; i < 12; ++i) p[i] = static_cast<float>(i + 1);

    auto t = transpose(base, 0, 1);  // Non-contiguous logical tensor [4,3]
    assert(!t->is_contiguous());

    ShardConfig cfg;
    cfg.offload_dir = "/tmp/test_sharder_noncontig";
    cfg.max_resident_bytes = 64 * 1024 * 1024;
    cfg.quantize_fp16_on_disk = false;
    ParameterSharder sharder(cfg);

    TensorPtr owner = t;
    sharder.register_parameter("transpose_weight", t, true, &owner);
    sharder.offload_all();
    assert(owner == nullptr);

    auto reloaded = sharder.require("transpose_weight");
    assert(reloaded != nullptr);
    assert(reloaded->shape() == std::vector<int64_t>({4, 3}));

    const float* out = reloaded->data<float>();
    const float* src = base->data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            float expect = src[j * 4 + i];
            float got = out[i * 3 + j];
            if (std::abs(expect - got) > 1e-6f) {
                std::cerr << "[FAIL] mismatch at (" << i << "," << j
                          << "), expect=" << expect << ", got=" << got << std::endl;
                return 1;
            }
        }
    }

    std::cout << "[OK] Sharder non-contiguous tensor roundtrip passed." << std::endl;
    return 0;
}
