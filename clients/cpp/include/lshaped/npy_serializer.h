#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <string>
#include <vector>

namespace lshaped {

enum class NpyDType {
    kFloat32,
    kInt32,
    kUInt8,
};

std::vector<std::uint8_t> SerializeNpy(
    const void* data,
    std::size_t element_count,
    const std::vector<std::int64_t>& shape,
    NpyDType dtype);

struct ParsedNpyArray {
    NpyDType dtype;
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> raw_bytes;
};

ParsedNpyArray ParseNpy(std::string_view blob);

inline std::size_t ElementSize(NpyDType dtype) {
    switch (dtype) {
        case NpyDType::kFloat32:
            return sizeof(float);
        case NpyDType::kInt32:
            return sizeof(std::int32_t);
        case NpyDType::kUInt8:
            return sizeof(std::uint8_t);
    }
    return 0;
}

}  // namespace lshaped
