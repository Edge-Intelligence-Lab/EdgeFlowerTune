/**
 * @file safetensors_loader.cpp
 * @brief SafeTensors format weight loader implementation
 */

#include "safetensors_loader.h"
#include "../core/ops.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <regex>
#include <cstring>
#include <cmath>

namespace ops {

namespace {

constexpr uint32_t kShardFileMagic = 0x31524853u;  // "SHR1"

struct QuantizedShardHeader {
    uint32_t magic = kShardFileMagic;
    uint8_t quantization = 0;
    uint8_t dtype = 0;
    uint16_t reserved = 0;
    uint64_t numel = 0;
    float scale = 1.0f;
};

float fp16_to_fp32_inline_local(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    uint32_t f32_bits;
    if (exponent == 0) {
        f32_bits = (mantissa == 0) ? (sign << 31) : 0;
    } else if (exponent == 0x1F) {
        f32_bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        f32_bits = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

float bf16_to_fp32_inline_local(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

float compute_abs_max(const float* src, size_t elems) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < elems; ++i) {
        max_abs = std::max(max_abs, std::fabs(src[i]));
    }
    return max_abs;
}

inline int8_t quantize_int8_value(float value, float scale) {
    const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    const float q = std::round(value * inv_scale);
    const float clamped = std::max(-127.0f, std::min(127.0f, q));
    return static_cast<int8_t>(clamped);
}

inline int8_t quantize_int4_value(float value, float scale) {
    const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    const float q = std::round(value * inv_scale);
    const int clamped = std::max(-8, std::min(7, static_cast<int>(q)));
    return static_cast<int8_t>(clamped);
}

inline void write_int4_index(uint8_t* dst, size_t idx, int8_t q) {
    const uint8_t packed = static_cast<uint8_t>(q + 8) & 0x0Fu;
    const size_t byte_idx = idx / 2;
    if ((idx & 1U) == 0U) {
        dst[byte_idx] = static_cast<uint8_t>((dst[byte_idx] & 0xF0u) | packed);
    } else {
        dst[byte_idx] = static_cast<uint8_t>((dst[byte_idx] & 0x0Fu) | (packed << 4));
    }
}

}  // namespace

// ============================================================================
// SafeTensorsReader implementation
// ============================================================================

SafeTensorsReader::SafeTensorsReader(const std::string& filepath)
    : filepath_(filepath), header_len_(0), data_offset_(0) {
    file_.open(filepath, std::ios::binary);
    if (!file_.is_open()) {
        throw std::runtime_error("Cannot open safetensors file: " + filepath);
    }
}

SafeTensorsReader::~SafeTensorsReader() {
    if (file_.is_open()) {
        file_.close();
    }
}

void SafeTensorsReader::parse_header() {
    // 1. Read header_len (first 8 bytes, little-endian uint64)
    uint64_t header_len_raw;
    file_.read(reinterpret_cast<char*>(&header_len_raw), 8);
    if (!file_) {
        throw std::runtime_error("Failed to read header_len");
    }
    header_len_ = static_cast<size_t>(header_len_raw);
    
    // 2. Read JSON header
    std::vector<char> header_bytes(header_len_);
    file_.read(header_bytes.data(), header_len_);
    if (!file_) {
        throw std::runtime_error("Failed to read JSON header");
    }
    
    std::string header_json(header_bytes.begin(), header_bytes.end());
    data_offset_ = 8 + header_len_;
    
    // 3. Simple JSON parsing (manual extraction of tensor metadata)
    parse_tensor_metadata(header_json);
}

void SafeTensorsReader::parse_tensor_metadata(const std::string& json_str) {
    // Simple regex extraction: find all "tensor_name": {...}
    std::regex tensor_pattern(R"#("([^"]+)"\s*:\s*\{[^}]+\})#");
    auto tensor_begin = std::sregex_iterator(json_str.begin(), json_str.end(), tensor_pattern);
    auto tensor_end = std::sregex_iterator();
    
    for (auto it = tensor_begin; it != tensor_end; ++it) {
        std::string name = (*it)[1].str();
        std::string block = (*it)[0].str();
        
        // Extract dtype
        std::regex dtype_pattern(R"#("dtype"\s*:\s*"([^"]+)")#");
        std::smatch dtype_match;
        std::string dtype = "F32";
        if (std::regex_search(block, dtype_match, dtype_pattern)) {
            dtype = dtype_match[1].str();
        }
        
        // Extract shape
        std::regex shape_pattern(R"#("shape"\s*:\s*\[([^\]]+)\])#");
        std::smatch shape_match;
        std::vector<int64_t> shape;
        if (std::regex_search(block, shape_match, shape_pattern)) {
            std::string shape_str = shape_match[1].str();
            std::istringstream iss(shape_str);
            std::string val;
            while (std::getline(iss, val, ',')) {
                // Remove spaces
                val.erase(std::remove_if(val.begin(), val.end(), ::isspace), val.end());
                if (!val.empty()) {
                    shape.push_back(std::stoll(val));
                }
            }
        }
        
        // Extract data_offsets
        std::regex offsets_pattern(R"#("data_offsets"\s*:\s*\[(\d+)\s*,\s*(\d+)\])#");
        std::smatch offsets_match;
        std::vector<size_t> offsets;
        if (std::regex_search(block, offsets_match, offsets_pattern)) {
            offsets.push_back(std::stoull(offsets_match[1].str()));
            offsets.push_back(std::stoull(offsets_match[2].str()));
        }
        
        SafeTensorInfo info;
        info.dtype = dtype;
        info.shape = shape;
        info.data_offsets = offsets;
        
        tensor_map_[name] = info;
    }
}

std::vector<std::string> SafeTensorsReader::get_tensor_names() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : tensor_map_) {
        names.push_back(name);
    }
    return names;
}

SafeTensorInfo SafeTensorsReader::get_tensor_info(const std::string& name) const {
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

TensorPtr SafeTensorsReader::load_tensor(const std::string& name, bool transpose) {
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    
    return read_tensor_data(it->second, transpose);
}

std::vector<uint8_t> SafeTensorsReader::load_tensor_quantized(
    const std::string& name,
    sharding::DiskQuantizationMode quant_mode,
    bool transpose,
    std::vector<int64_t>* out_shape,
    DType* out_dtype) {
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    if (quant_mode != sharding::DiskQuantizationMode::INT8 &&
        quant_mode != sharding::DiskQuantizationMode::INT4) {
        throw std::runtime_error("load_tensor_quantized only supports INT8/INT4");
    }

    const SafeTensorInfo& info = it->second;
    if (info.data_offsets.size() != 2) {
        throw std::runtime_error("Invalid data_offsets");
    }
    if (!(info.dtype == "F32" || info.dtype == "F16" || info.dtype == "BF16")) {
        throw std::runtime_error("Unsupported dtype for quantized load: " + info.dtype);
    }

    int64_t numel = 1;
    for (auto dim : info.shape) {
        numel *= dim;
    }
    const size_t elems = static_cast<size_t>(numel);

    std::vector<int64_t> target_shape = info.shape;
    const bool need_transpose = transpose && info.shape.size() == 2;
    if (need_transpose) {
        std::swap(target_shape[0], target_shape[1]);
    }
    if (out_shape) {
        *out_shape = target_shape;
    }
    if (out_dtype) {
        *out_dtype = kFloat32;
    }

    const size_t start = info.data_offsets[0];
    const size_t end = info.data_offsets[1];
    const size_t byte_size = end - start;
    size_t element_size = 4;
    if (info.dtype == "F16" || info.dtype == "BF16") {
        element_size = 2;
    }
    if (byte_size != elems * element_size) {
        throw std::runtime_error("Size mismatch for tensor");
    }

    auto seek_tensor = [&]() {
        file_.clear();
        file_.seekg(data_offset_ + start, std::ios::beg);
        if (!file_) {
            throw std::runtime_error("Failed to seek tensor data");
        }
    };

    auto decode_chunk = [&](const char* src_bytes, size_t cur, float* dst) {
        if (info.dtype == "F32") {
            std::memcpy(dst, src_bytes, cur * sizeof(float));
            return;
        }
        const uint16_t* src16 = reinterpret_cast<const uint16_t*>(src_bytes);
        if (info.dtype == "F16") {
            for (size_t i = 0; i < cur; ++i) {
                dst[i] = fp16_to_fp32_inline_local(src16[i]);
            }
            return;
        }
        for (size_t i = 0; i < cur; ++i) {
            dst[i] = bf16_to_fp32_inline_local(src16[i]);
        }
    };

    float abs_max = 0.0f;
    {
        seek_tensor();
        const size_t chunk_elems = 1 << 18;
        std::vector<char> raw(std::min(chunk_elems, elems) * element_size);
        std::vector<float> decoded(std::min(chunk_elems, elems));
        size_t offset = 0;
        while (offset < elems) {
            const size_t cur = std::min(chunk_elems, elems - offset);
            file_.read(raw.data(), static_cast<std::streamsize>(cur * element_size));
            if (!file_) {
                throw std::runtime_error("Failed to read tensor bytes");
            }
            decode_chunk(raw.data(), cur, decoded.data());
            abs_max = std::max(abs_max, compute_abs_max(decoded.data(), cur));
            offset += cur;
        }
    }

    QuantizedShardHeader header;
    header.quantization = static_cast<uint8_t>(quant_mode);
    header.dtype = static_cast<uint8_t>(kFloat32);
    header.numel = static_cast<uint64_t>(elems);
    if (abs_max <= 1e-12f) {
        header.scale = 1.0f;
    } else if (quant_mode == sharding::DiskQuantizationMode::INT8) {
        header.scale = abs_max / 127.0f;
    } else {
        header.scale = abs_max / 7.0f;
    }

    const size_t payload_bytes =
        (quant_mode == sharding::DiskQuantizationMode::INT8) ? elems : ((elems + 1) / 2);
    std::vector<uint8_t> encoded(sizeof(header) + payload_bytes, 0);
    std::memcpy(encoded.data(), &header, sizeof(header));
    uint8_t* payload = encoded.data() + sizeof(header);

    if (!need_transpose) {
        seek_tensor();
        const size_t chunk_elems = 1 << 18;
        std::vector<char> raw(std::min(chunk_elems, elems) * element_size);
        std::vector<float> decoded(std::min(chunk_elems, elems));
        size_t offset = 0;
        while (offset < elems) {
            const size_t cur = std::min(chunk_elems, elems - offset);
            file_.read(raw.data(), static_cast<std::streamsize>(cur * element_size));
            if (!file_) {
                throw std::runtime_error("Failed to read tensor bytes");
            }
            decode_chunk(raw.data(), cur, decoded.data());
            if (quant_mode == sharding::DiskQuantizationMode::INT8) {
                int8_t* dst = reinterpret_cast<int8_t*>(payload) + offset;
                for (size_t i = 0; i < cur; ++i) {
                    dst[i] = quantize_int8_value(decoded[i], header.scale);
                }
            } else {
                for (size_t i = 0; i < cur; ++i) {
                    write_int4_index(payload, offset + i, quantize_int4_value(decoded[i], header.scale));
                }
            }
            offset += cur;
        }
        return encoded;
    }

    const size_t rows = static_cast<size_t>(info.shape[0]);
    const size_t cols = static_cast<size_t>(info.shape[1]);
    const size_t row_block = std::max<size_t>(1, (1 << 18) / std::max<size_t>(1, cols));
    std::vector<char> raw(row_block * cols * element_size);
    std::vector<float> decoded(row_block * cols);

    seek_tensor();
    size_t row0 = 0;
    while (row0 < rows) {
        const size_t cur_rows = std::min(row_block, rows - row0);
        const size_t cur_elems = cur_rows * cols;
        file_.read(raw.data(), static_cast<std::streamsize>(cur_elems * element_size));
        if (!file_) {
            throw std::runtime_error("Failed to read tensor bytes");
        }
        decode_chunk(raw.data(), cur_elems, decoded.data());
        for (size_t r = 0; r < cur_rows; ++r) {
            const size_t src_row = row0 + r;
            const float* row_ptr = decoded.data() + r * cols;
            for (size_t c = 0; c < cols; ++c) {
                const size_t dst_index = c * rows + src_row;
                if (quant_mode == sharding::DiskQuantizationMode::INT8) {
                    reinterpret_cast<int8_t*>(payload)[dst_index] =
                        quantize_int8_value(row_ptr[c], header.scale);
                } else {
                    write_int4_index(payload, dst_index, quantize_int4_value(row_ptr[c], header.scale));
                }
            }
        }
        row0 += cur_rows;
    }

    return encoded;
}

TensorPtr SafeTensorsReader::read_tensor_data(const SafeTensorInfo& info, bool transpose) {
    auto fp16_to_fp32_inline = [](uint16_t h) -> float {
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exponent = (h >> 10) & 0x1F;
        uint32_t mantissa = h & 0x3FF;
        uint32_t f32_bits;
        if (exponent == 0) {
            f32_bits = (mantissa == 0) ? (sign << 31) : 0;
        } else if (exponent == 0x1F) {
            f32_bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        } else {
            f32_bits = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        }
        float result;
        std::memcpy(&result, &f32_bits, sizeof(float));
        return result;
    };

    if (info.data_offsets.size() != 2) {
        throw std::runtime_error("Invalid data_offsets");
    }

    size_t start = info.data_offsets[0];
    size_t end = info.data_offsets[1];
    size_t byte_size = end - start;

    int64_t numel = 1;
    for (auto dim : info.shape) {
        numel *= dim;
    }

    size_t element_size = 4;
    DType target_dtype = kFloat32;

    if (info.dtype == "F32") {
        element_size = 4;
        target_dtype = kFloat32;
    } else if (info.dtype == "F16" || info.dtype == "BF16") {
        element_size = 2;
        target_dtype = kFloat32;
    } else if (info.dtype == "I32") {
        element_size = 4;
        target_dtype = kInt32;
    } else {
        throw std::runtime_error("Unsupported dtype: " + info.dtype);
    }

    if (byte_size != static_cast<size_t>(numel) * element_size) {
        throw std::runtime_error("Size mismatch for tensor");
    }

    file_.seekg(data_offset_ + start, std::ios::beg);
    if (!file_) {
        throw std::runtime_error("Failed to seek tensor data");
    }

    std::vector<int64_t> shape = info.shape;
    bool need_transpose = transpose && shape.size() == 2;
    if (need_transpose) {
        std::swap(shape[0], shape[1]);
    }

    TensorPtr tensor = std::make_shared<Tensor>(shape, target_dtype, kCPU);
    auto transpose_buffer = [&](float* buffer) {
        if (!need_transpose) return;
        int64_t rows = info.shape[0];
        int64_t cols = info.shape[1];
        std::vector<float> temp(numel);
        std::memcpy(temp.data(), buffer, numel * sizeof(float));
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                buffer[j * rows + i] = temp[i * cols + j];
            }
        }
    };

    if (info.dtype == "F32") {
        if (!need_transpose) {
            file_.read(reinterpret_cast<char*>(tensor->data<float>()), byte_size);
            if (!file_) throw std::runtime_error("Failed to read F32 tensor data");
        } else {
            std::vector<float> temp(numel);
            file_.read(reinterpret_cast<char*>(temp.data()), byte_size);
            if (!file_) throw std::runtime_error("Failed to read F32 tensor data");
            std::memcpy(tensor->data<float>(), temp.data(), byte_size);
            transpose_buffer(tensor->data<float>());
        }
    } else if (info.dtype == "F16" || info.dtype == "BF16") {
        float* fp32_data = tensor->data<float>();
        if (!need_transpose) {
            const size_t chunk = 1 << 18;  // 256K elements
            std::vector<uint16_t> tmp(std::min<size_t>(chunk, static_cast<size_t>(numel)));
            size_t offset = 0;
            while (offset < static_cast<size_t>(numel)) {
                size_t cur = std::min<size_t>(tmp.size(), static_cast<size_t>(numel) - offset);
                file_.read(reinterpret_cast<char*>(tmp.data()), cur * sizeof(uint16_t));
                if (!file_) throw std::runtime_error("Failed to read FP16/BF16 tensor data");
                if (info.dtype == "F16") {
                    for (size_t i = 0; i < cur; ++i) {
                        fp32_data[offset + i] = fp16_to_fp32_inline(tmp[i]);
                    }
                } else {
                    for (size_t i = 0; i < cur; ++i) {
                        uint32_t bits = static_cast<uint32_t>(tmp[i]) << 16;
                        std::memcpy(&fp32_data[offset + i], &bits, sizeof(float));
                    }
                }
                offset += cur;
            }
        } else {
            std::vector<float> temp(numel);
            const size_t chunk = 1 << 18;  // 256K elements
            std::vector<uint16_t> tmp(std::min<size_t>(chunk, static_cast<size_t>(numel)));
            size_t offset = 0;
            while (offset < static_cast<size_t>(numel)) {
                size_t cur = std::min<size_t>(tmp.size(), static_cast<size_t>(numel) - offset);
                file_.read(reinterpret_cast<char*>(tmp.data()), cur * sizeof(uint16_t));
                if (!file_) throw std::runtime_error("Failed to read FP16/BF16 tensor data");
                if (info.dtype == "F16") {
                    for (size_t i = 0; i < cur; ++i) {
                        temp[offset + i] = fp16_to_fp32_inline(tmp[i]);
                    }
                } else {
                    for (size_t i = 0; i < cur; ++i) {
                        uint32_t bits = static_cast<uint32_t>(tmp[i]) << 16;
                        std::memcpy(&temp[offset + i], &bits, sizeof(float));
                    }
                }
                offset += cur;
            }
            std::memcpy(fp32_data, temp.data(), static_cast<size_t>(numel) * sizeof(float));
            transpose_buffer(fp32_data);
        }
    } else if (info.dtype == "I32") {
        file_.read(reinterpret_cast<char*>(tensor->data<int32_t>()), byte_size);
        if (!file_) throw std::runtime_error("Failed to read I32 tensor data");
    }

    return tensor;
}

std::unordered_map<std::string, TensorPtr> 
SafeTensorsReader::load_tensors_mapped(
    const std::unordered_map<std::string, std::string>& key_mapping,
    const SafeTensorsLoadOptions& options) {
    
    std::unordered_map<std::string, TensorPtr> result;
    
    for (const auto& [internal_key, hf_key] : key_mapping) {
        auto it = tensor_map_.find(hf_key);
        std::string alt_key;
        // GPT-2 safetensors often has "transformer." prefix, compatible with old mapping without prefix
        if (it == tensor_map_.end() && hf_key.rfind("transformer.", 0) != 0) {
            alt_key = "transformer." + hf_key;
            it = tensor_map_.find(alt_key);
        }
        if (it == tensor_map_.end()) {
            if (options.verbose) {
                std::cerr << "[WARN] HF key not found: " << hf_key << std::endl;
            }
            continue;
        }
        
        // Determine if transpose is needed (linear layer weights)
        bool is_embedding = (hf_key.find("wte") != std::string::npos) ||
                            (hf_key.find("wpe") != std::string::npos) ||
                            (hf_key.find("embed_tokens") != std::string::npos);
        bool transpose = options.transpose_linear && 
                        (hf_key.find("weight") != std::string::npos) &&
                        (hf_key.find("ln") == std::string::npos) &&
                        !is_embedding &&
                        it->second.shape.size() == 2;
        
        auto tensor = read_tensor_data(it->second, transpose);
        result[internal_key] = tensor;
        
        if (options.verbose) {
            std::cout << "[Loaded] " << internal_key << " <- " << hf_key 
                      << " shape=[";
            for (size_t i = 0; i < tensor->shape().size(); ++i) {
                std::cout << tensor->shape()[i];
                if (i < tensor->shape().size() - 1) std::cout << ", ";
            }
            std::cout << "]";
            if (transpose) std::cout << " (transposed)";
            std::cout << std::endl;
        }
    }
    
    return result;
}

// ============================================================================
// GPT2KeyMapper implementation
// ============================================================================

std::unordered_map<std::string, std::string> 
GPT2KeyMapper::generate_gpt2_mapping(int num_layers) {
    std::unordered_map<std::string, std::string> mapping;
    
    // Embeddings
    mapping["wte.weight"] = "wte.weight";
    mapping["wpe.weight"] = "wpe.weight";
    
    // Transformer blocks
    for (int i = 0; i < num_layers; ++i) {
        std::string hf_prefix = "h." + std::to_string(i) + ".";
        std::string internal_prefix = "blocks." + std::to_string(i) + ".";
        
        // LayerNorm 1
        mapping[internal_prefix + "ln_1.weight"] = hf_prefix + "ln_1.weight";
        mapping[internal_prefix + "ln_1.bias"] = hf_prefix + "ln_1.bias";
        
        // Attention
        mapping[internal_prefix + "attn.qkv.weight"] = hf_prefix + "attn.c_attn.weight";
        mapping[internal_prefix + "attn.qkv.bias"] = hf_prefix + "attn.c_attn.bias";
        mapping[internal_prefix + "attn.proj.weight"] = hf_prefix + "attn.c_proj.weight";
        mapping[internal_prefix + "attn.proj.bias"] = hf_prefix + "attn.c_proj.bias";
        
        // LayerNorm 2
        mapping[internal_prefix + "ln_2.weight"] = hf_prefix + "ln_2.weight";
        mapping[internal_prefix + "ln_2.bias"] = hf_prefix + "ln_2.bias";
        
        // MLP
        mapping[internal_prefix + "mlp.fc_in.weight"] = hf_prefix + "mlp.c_fc.weight";
        mapping[internal_prefix + "mlp.fc_in.bias"] = hf_prefix + "mlp.c_fc.bias";
        mapping[internal_prefix + "mlp.fc_out.weight"] = hf_prefix + "mlp.c_proj.weight";
        mapping[internal_prefix + "mlp.fc_out.bias"] = hf_prefix + "mlp.c_proj.bias";
    }
    
    // Final LayerNorm
    mapping["ln_f.weight"] = "ln_f.weight";
    mapping["ln_f.bias"] = "ln_f.bias";
    
    // lm_head (note: usually tied with wte, not loaded separately; can uncomment if independent loading needed)
    // mapping["lm_head.weight"] = "lm_head.weight";
    
    return mapping;
}

void GPT2KeyMapper::print_mapping(const std::unordered_map<std::string, std::string>& mapping) {
    std::cout << "\n[GPT2 Key Mapping] Total: " << mapping.size() << " entries\n";
    for (const auto& [internal, hf] : mapping) {
        std::cout << "  " << internal << " <- " << hf << std::endl;
    }
}

std::unordered_map<std::string, std::string> 
GemmaKeyMapper::generate_gemma_mapping(int num_layers) {
    std::unordered_map<std::string, std::string> mapping;
    mapping["embed_tokens.weight"] = "model.embed_tokens.weight";
    mapping["norm.weight"] = "model.norm.weight";
    mapping["lm_head.weight"] = "lm_head.weight";

    for (int i = 0; i < num_layers; ++i) {
        std::string internal_prefix = "layers." + std::to_string(i);
        std::string hf_prefix = "model.layers." + std::to_string(i);

        mapping[internal_prefix + ".input_layernorm.weight"] = hf_prefix + ".input_layernorm.weight";
        mapping[internal_prefix + ".post_attention_layernorm.weight"] = hf_prefix + ".post_attention_layernorm.weight";
        mapping[internal_prefix + ".pre_feedforward_layernorm.weight"] =
            hf_prefix + ".pre_feedforward_layernorm.weight";
        mapping[internal_prefix + ".post_feedforward_layernorm.weight"] =
            hf_prefix + ".post_feedforward_layernorm.weight";

        mapping[internal_prefix + ".self_attn.q_proj.weight"] = hf_prefix + ".self_attn.q_proj.weight";
        mapping[internal_prefix + ".self_attn.k_proj.weight"] = hf_prefix + ".self_attn.k_proj.weight";
        mapping[internal_prefix + ".self_attn.v_proj.weight"] = hf_prefix + ".self_attn.v_proj.weight";
        mapping[internal_prefix + ".self_attn.o_proj.weight"] = hf_prefix + ".self_attn.o_proj.weight";
        mapping[internal_prefix + ".self_attn.q_norm.weight"] = hf_prefix + ".self_attn.q_norm.weight";
        mapping[internal_prefix + ".self_attn.k_norm.weight"] = hf_prefix + ".self_attn.k_norm.weight";

        mapping[internal_prefix + ".mlp.gate_proj.weight"] = hf_prefix + ".mlp.gate_proj.weight";
        mapping[internal_prefix + ".mlp.up_proj.weight"] = hf_prefix + ".mlp.up_proj.weight";
        mapping[internal_prefix + ".mlp.down_proj.weight"] = hf_prefix + ".mlp.down_proj.weight";
    }

    return mapping;
}

void GemmaKeyMapper::print_mapping(const std::unordered_map<std::string, std::string>& mapping) {
    for (const auto& kv : mapping) {
        std::cout << kv.first << " -> " << kv.second << std::endl;
    }
}

// ============================================================================
// QwenKeyMapper implementation
// ============================================================================

std::unordered_map<std::string, std::string>
QwenKeyMapper::generate_qwen_mapping(int num_layers) {
    std::unordered_map<std::string, std::string> mapping;
    mapping["embed_tokens.weight"] = "model.embed_tokens.weight";
    mapping["final_norm.weight"] = "model.norm.weight";

    for (int i = 0; i < num_layers; ++i) {
        std::string internal_prefix = "layers." + std::to_string(i);
        std::string hf_prefix = "model.layers." + std::to_string(i);

        mapping[internal_prefix + ".input_norm.weight"] = hf_prefix + ".input_layernorm.weight";
        mapping[internal_prefix + ".post_norm.weight"] = hf_prefix + ".post_attention_layernorm.weight";

        mapping[internal_prefix + ".self_attn.q_proj.weight"] = hf_prefix + ".self_attn.q_proj.weight";
        mapping[internal_prefix + ".self_attn.q_proj.bias"] = hf_prefix + ".self_attn.q_proj.bias";
        mapping[internal_prefix + ".self_attn.k_proj.weight"] = hf_prefix + ".self_attn.k_proj.weight";
        mapping[internal_prefix + ".self_attn.k_proj.bias"] = hf_prefix + ".self_attn.k_proj.bias";
        mapping[internal_prefix + ".self_attn.v_proj.weight"] = hf_prefix + ".self_attn.v_proj.weight";
        mapping[internal_prefix + ".self_attn.v_proj.bias"] = hf_prefix + ".self_attn.v_proj.bias";
        mapping[internal_prefix + ".self_attn.o_proj.weight"] = hf_prefix + ".self_attn.o_proj.weight";

        mapping[internal_prefix + ".mlp.gate_proj.weight"] = hf_prefix + ".mlp.gate_proj.weight";
        mapping[internal_prefix + ".mlp.up_proj.weight"] = hf_prefix + ".mlp.up_proj.weight";
        mapping[internal_prefix + ".mlp.down_proj.weight"] = hf_prefix + ".mlp.down_proj.weight";
    }

    return mapping;
}

void QwenKeyMapper::print_mapping(const std::unordered_map<std::string, std::string>& mapping) {
    for (const auto& kv : mapping) {
        std::cout << kv.first << " -> " << kv.second << std::endl;
    }
}

}  // namespace ops
