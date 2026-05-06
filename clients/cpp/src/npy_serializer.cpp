#include "lshaped/npy_serializer.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace lshaped {

namespace {

std::string Trim(std::string value) {
    value.erase(
        value.begin(),
        std::find_if(value.begin(), value.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    value.erase(
        std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
        value.end());
    return value;
}

NpyDType ParseDescr(const std::string& header) {
    if (header.find("<f4") != std::string::npos) {
        return NpyDType::kFloat32;
    }
    if (header.find("<i4") != std::string::npos || header.find("|i4") != std::string::npos) {
        return NpyDType::kInt32;
    }
    if (header.find("|u1") != std::string::npos || header.find("<u1") != std::string::npos) {
        return NpyDType::kUInt8;
    }
    throw std::runtime_error("Unsupported NPY dtype in header");
}

std::vector<std::int64_t> ParseShape(const std::string& header) {
    const std::size_t left = header.find('(');
    const std::size_t right = header.find(')', left);
    if (left == std::string::npos || right == std::string::npos || right <= left + 1) {
        throw std::runtime_error("Failed to parse NPY shape");
    }
    std::vector<std::int64_t> shape;
    std::stringstream ss(header.substr(left + 1, right - left - 1));
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = Trim(token);
        if (token.empty()) {
            continue;
        }
        shape.push_back(std::stoll(token));
    }
    if (shape.empty()) {
        throw std::runtime_error("NPY shape must not be empty");
    }
    return shape;
}

}  // namespace

std::vector<std::uint8_t> SerializeNpy(
    const void* data,
    std::size_t element_count,
    const std::vector<std::int64_t>& shape,
    NpyDType dtype) {
    assert(data != nullptr);
    assert(!shape.empty());

    std::string descr;
    switch (dtype) {
        case NpyDType::kFloat32:
            descr = "<f4";
            break;
        case NpyDType::kInt32:
            descr = "<i4";
            break;
        case NpyDType::kUInt8:
            descr = "|u1";
            break;
        default:
            throw std::runtime_error("Unsupported NPY dtype");
    }

    std::string shape_repr = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        shape_repr += std::to_string(shape[i]);
        if (shape.size() == 1) {
            shape_repr += ",";
        }
        if (i + 1 < shape.size()) {
            shape_repr += ", ";
        }
    }
    shape_repr += ")";

    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape_repr + ", }";
    constexpr char kMagic[] = "\x93NUMPY";
    constexpr std::uint8_t kVersionMajor = 1;
    constexpr std::uint8_t kVersionMinor = 0;

    const std::size_t preamble = 6 + 2 + 2;
    std::size_t header_len = header.size() + 1;
    std::size_t padding = 16 - ((preamble + header_len) % 16);
    if (padding == 16) {
        padding = 0;
    }
    header += std::string(padding, ' ');
    header.push_back('\n');

    const std::uint16_t header_size = static_cast<std::uint16_t>(header.size());
    std::vector<std::uint8_t> output;
    output.resize(preamble + header.size() + element_count * ElementSize(dtype));

    std::size_t offset = 0;
    std::memcpy(output.data() + offset, kMagic, 6);
    offset += 6;
    output[offset++] = kVersionMajor;
    output[offset++] = kVersionMinor;
    std::memcpy(output.data() + offset, &header_size, sizeof(header_size));
    offset += sizeof(header_size);
    std::memcpy(output.data() + offset, header.data(), header.size());
    offset += header.size();
    std::memcpy(output.data() + offset, data, element_count * ElementSize(dtype));
    return output;
}

ParsedNpyArray ParseNpy(std::string_view blob) {
    if (blob.size() < 10 || std::memcmp(blob.data(), "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("Invalid NPY blob");
    }

    const auto* header_size_ptr = reinterpret_cast<const std::uint16_t*>(blob.data() + 8);
    const std::uint16_t header_size = *header_size_ptr;
    const std::size_t preamble = 10;
    if (blob.size() < preamble + header_size) {
        throw std::runtime_error("Truncated NPY header");
    }

    const std::string header(blob.data() + preamble, header_size);
    const NpyDType dtype = ParseDescr(header);
    const std::vector<std::int64_t> shape = ParseShape(header);

    std::size_t element_count = 1;
    for (const std::int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("NPY dimensions must be positive");
        }
        element_count *= static_cast<std::size_t>(dim);
    }

    const std::size_t expected_bytes = element_count * ElementSize(dtype);
    const std::size_t data_offset = preamble + header_size;
    if (blob.size() != data_offset + expected_bytes) {
        throw std::runtime_error("NPY payload size mismatch");
    }

    ParsedNpyArray parsed{dtype, shape, std::vector<std::uint8_t>(expected_bytes)};
    std::memcpy(parsed.raw_bytes.data(), blob.data() + data_offset, expected_bytes);
    return parsed;
}

}  // namespace lshaped
