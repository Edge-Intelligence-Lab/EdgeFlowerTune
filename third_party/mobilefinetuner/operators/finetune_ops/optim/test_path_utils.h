#pragma once

#include "../path_utils.h"

namespace ops {
namespace test_utils {

using Gpt2WikiText2Paths = ops::path_utils::Gpt2WikiText2Paths;

inline Gpt2WikiText2Paths resolve_gpt2_wikitext2_paths() {
    return ops::path_utils::resolve_gpt2_wikitext2_paths(true);
}

}  // namespace test_utils
}  // namespace ops
