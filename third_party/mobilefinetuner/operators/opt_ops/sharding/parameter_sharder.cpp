/**
 * @file parameter_sharder.cpp
 * @brief Implementation of ZeRO-inspired single-device parameter sharding/offload.
 */

#include "parameter_sharder.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace ops {
namespace sharding {

namespace {

constexpr uint32_t kShardFileMagic = 0x31524853u;  // "SHR1"

struct DiskTensorHeader {
    uint32_t magic = kShardFileMagic;
    uint8_t quantization = 0;
    uint8_t dtype = 0;
    uint16_t reserved = 0;
    uint64_t numel = 0;
    float scale = 1.0f;
};

bool FileExists(const std::string& path) {
    return ::access(path.c_str(), F_OK) == 0;
}

bool IsMemoryStorageDir(const std::string& dir) {
    return dir == "__memory__";
}

std::string JoinPath(const std::string& dir, const std::string& child) {
    if (dir.empty() || dir == ".") {
        return child;
    }
    if (dir.back() == '/') {
        return dir + child;
    }
    return dir + "/" + child;
}

void EnsureDirectories(const std::string& dir) {
    if (dir.empty()) {
        return;
    }
    std::string path;
    path.reserve(dir.size());
    for (size_t i = 0; i < dir.size(); ++i) {
        path.push_back(dir[i]);
        if (dir[i] == '/' && !path.empty()) {
            if (path.size() > 1) {
                ::mkdir(path.c_str(), 0775);
            }
        }
    }
    if (::mkdir(dir.c_str(), 0775) != 0 && errno != EEXIST) {
        throw std::runtime_error("ParameterSharder: mkdir failed for " + dir);
    }
}

const char* QuantModeName(DiskQuantizationMode mode) {
    switch (mode) {
        case DiskQuantizationMode::None: return "fp32";
        case DiskQuantizationMode::FP16: return "fp16";
        case DiskQuantizationMode::INT8: return "int8";
        case DiskQuantizationMode::INT4: return "int4";
        default: return "unknown";
    }
}

size_t EncodedPayloadBytes(DiskQuantizationMode mode, DType dtype, size_t elems) {
    switch (mode) {
        case DiskQuantizationMode::FP16:
            return elems * sizeof(uint16_t);
        case DiskQuantizationMode::INT8:
            return elems * sizeof(int8_t);
        case DiskQuantizationMode::INT4:
            return ((elems + 1) / 2) * sizeof(uint8_t);
        case DiskQuantizationMode::None:
        default:
            return elems * ((dtype == kFloat16) ? sizeof(uint16_t) : sizeof(float));
    }
}

DiskQuantizationMode EffectiveQuantMode(const ShardConfig& cfg, DType dtype) {
    if (dtype != kFloat32) {
        return DiskQuantizationMode::None;
    }
    if (cfg.disk_quantization != DiskQuantizationMode::None) {
        return cfg.disk_quantization;
    }
    return cfg.quantize_fp16_on_disk ? DiskQuantizationMode::FP16 : DiskQuantizationMode::None;
}

void WriteHeader(std::ofstream& out, const DiskTensorHeader& header) {
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
}

bool TryReadHeader(std::ifstream& in, DiskTensorHeader& header) {
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || header.magic != kShardFileMagic) {
        in.clear();
        in.seekg(0, std::ios::beg);
        return false;
    }
    return true;
}

float ComputeAbsMax(const float* src, size_t elems) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < elems; ++i) {
        max_abs = std::max(max_abs, std::fabs(src[i]));
    }
    return max_abs;
}

void QuantizeInt8Chunk(const float* src, int8_t* dst, size_t elems, float scale) {
    const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    for (size_t i = 0; i < elems; ++i) {
        const float q = std::round(src[i] * inv_scale);
        const float clamped = std::max(-127.0f, std::min(127.0f, q));
        dst[i] = static_cast<int8_t>(clamped);
    }
}

void DequantizeInt8Chunk(const int8_t* src, float* dst, size_t elems, float scale) {
    for (size_t i = 0; i < elems; ++i) {
        dst[i] = static_cast<float>(src[i]) * scale;
    }
}

void QuantizeInt4Chunk(const float* src, uint8_t* dst, size_t elems, float scale) {
    const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    for (size_t i = 0; i < elems; i += 2) {
        const float q0f = std::round(src[i] * inv_scale);
        const int q0 = std::max(-8, std::min(7, static_cast<int>(q0f)));
        int q1 = 0;
        if (i + 1 < elems) {
            const float q1f = std::round(src[i + 1] * inv_scale);
            q1 = std::max(-8, std::min(7, static_cast<int>(q1f)));
        }
        const uint8_t packed0 = static_cast<uint8_t>(q0 + 8) & 0x0Fu;
        const uint8_t packed1 = static_cast<uint8_t>(q1 + 8) & 0x0Fu;
        dst[i / 2] = static_cast<uint8_t>(packed0 | (packed1 << 4));
    }
}

void DequantizeInt4Chunk(const uint8_t* src, float* dst, size_t elems, float scale) {
    for (size_t i = 0; i < elems; i += 2) {
        const uint8_t packed = src[i / 2];
        const int q0 = static_cast<int>(packed & 0x0Fu) - 8;
        dst[i] = static_cast<float>(q0) * scale;
        if (i + 1 < elems) {
            const int q1 = static_cast<int>((packed >> 4) & 0x0Fu) - 8;
            dst[i + 1] = static_cast<float>(q1) * scale;
        }
    }
}

}  // namespace

// Helper: float32 <-> fp16 (same codepath as core ops, kept local to avoid exposing internals)
static uint16_t float32_to_fp16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        uint32_t shifted = mantissa >> (1 - exponent + 13);
        return static_cast<uint16_t>(sign | shifted);
    }

    if (exponent >= 31) {
        uint16_t inf_nan = (mantissa == 0) ? 0x7C00u : static_cast<uint16_t>(0x7C00u | (mantissa >> 13));
        return static_cast<uint16_t>(sign | inf_nan);
    }

    uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
    return half;
}

static float fp16_to_float32(uint16_t value) {
    uint32_t sign = (value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1Fu;
    uint32_t mantissa = value & 0x3FFu;

    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3FFu;
            exponent = exponent - 1 + 127 - 15;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        exponent = exponent - 15 + 127;
        bits = sign | (exponent << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

size_t ShardEntry::num_bytes_fp32() const {
    size_t elems = 1;
    for (auto d : shape) elems *= static_cast<size_t>(d);
    size_t bytes = elems * sizeof(float);
    if (dtype == kFloat16) bytes = elems * sizeof(uint16_t);
    return bytes;
}

ParameterSharder::ParameterSharder(const ShardConfig& cfg)
    : cfg_(cfg), memory_storage_(IsMemoryStorageDir(cfg.offload_dir)), resident_bytes_(0), clock_(0) {
    if (cfg_.offload_dir.empty()) {
        throw std::runtime_error("ParameterSharder: offload_dir must not be empty");
    }
    if (!memory_storage_) {
        EnsureDirectories(cfg_.offload_dir);
    }
}

void ParameterSharder::register_parameter(const std::string& name, const TensorPtr& tensor, bool keep_in_memory, TensorPtr* owner_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!tensor) throw std::runtime_error("register_parameter: tensor is null");
    if (entries_.count(name)) throw std::runtime_error("register_parameter: duplicated name " + name);

    ShardEntry e;
    e.name = name;
    e.shape = tensor->shape();
    e.dtype = tensor->dtype();
    e.disk_quantization = EffectiveQuantMode(cfg_, e.dtype);
    if (!memory_storage_) {
        e.path = JoinPath(cfg_.offload_dir, sanitize_filename(name) + ".bin");
    }
    e.tensor = tensor;
    e.owner_ptr = owner_ptr;
    e.state = ShardState::InMemory;
    e.last_used = ++clock_;

    // First write a copy to disk (ensure disk as primary storage)
    offload_entry(e);
    // Return to memory state (optionally keep, otherwise release)
    if (keep_in_memory) {
        load_entry(e);
    } else {
        if (e.owner_ptr) *e.owner_ptr = nullptr;
        e.tensor.reset();
        e.state = ShardState::Offloaded;
    }

    if (e.tensor) {
        resident_bytes_ += e.num_bytes_fp32();
    }
    entries_.emplace(name, std::move(e));
}

void ParameterSharder::register_encoded(const std::string& name,
                                        const std::vector<int64_t>& shape,
                                        DType dtype,
                                        std::vector<uint8_t>&& encoded,
                                        TensorPtr* owner_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (entries_.count(name)) {
        throw std::runtime_error("register_encoded: duplicated name " + name);
    }
    if (encoded.size() < sizeof(DiskTensorHeader)) {
        throw std::runtime_error("register_encoded: malformed encoded payload for " + name);
    }

    DiskTensorHeader header;
    std::memcpy(&header, encoded.data(), sizeof(header));
    if (header.magic != kShardFileMagic) {
        throw std::runtime_error("register_encoded: bad shard header for " + name);
    }

    ShardEntry e;
    e.name = name;
    e.shape = shape;
    e.dtype = dtype;
    e.disk_quantization = static_cast<DiskQuantizationMode>(header.quantization);
    e.owner_ptr = owner_ptr;
    e.state = ShardState::Offloaded;
    e.last_used = ++clock_;
    if (!memory_storage_) {
        e.path = JoinPath(cfg_.offload_dir, sanitize_filename(name) + ".bin");
        std::ofstream out(e.path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("register_encoded: cannot open " + e.path);
        }
        out.write(reinterpret_cast<const char*>(encoded.data()),
                  static_cast<std::streamsize>(encoded.size()));
        if (!out) {
            throw std::runtime_error("register_encoded: short write to " + e.path);
        }
    } else {
        e.encoded = std::move(encoded);
    }
    if (e.owner_ptr) {
        *e.owner_ptr = nullptr;
    }
    entries_.emplace(name, std::move(e));
}

TensorPtr ParameterSharder::require(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(name);
    if (it == entries_.end()) {
        throw std::runtime_error("require: unknown parameter " + name);
    }
    ShardEntry& e = it->second;
    size_t need = e.num_bytes_fp32();
    ensure_budget(need, name);
    if (e.state == ShardState::Offloaded) {
        load_entry(e);
        resident_bytes_ += need;
    }
    e.last_used = ++clock_;
    return e.tensor;
}

void ParameterSharder::mark_dirty(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(name);
    if (it == entries_.end()) return;
    it->second.dirty = true;
}

void ParameterSharder::offload_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& kv : entries_) {
        offload_entry(kv.second);
        kv.second.tensor.reset();
        kv.second.state = ShardState::Offloaded;
        kv.second.dirty = false;
    }
    resident_bytes_ = 0;
}

std::string ParameterSharder::debug_string() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    oss << "ParameterSharder{resident=" << resident_bytes_ / 1024.0 / 1024.0
        << "MB, entries=" << entries_.size()
        << ", storage=" << (memory_storage_ ? "memory" : "disk") << "}\n";
    for (const auto& kv : entries_) {
        const auto& e = kv.second;
        oss << "  [" << kv.first << "] state=" << (e.state == ShardState::InMemory ? "RAM" : (memory_storage_ ? "MEMQ" : "Disk"))
            << ", bytes=" << e.num_bytes_fp32()
            << ", quant=" << QuantModeName(e.disk_quantization)
            << ", dirty=" << (e.dirty ? "1" : "0")
            << "\n";
    }
    return oss.str();
}

void ParameterSharder::ensure_budget(size_t incoming_bytes, const std::string& keep_name) {
    if (incoming_bytes > cfg_.max_resident_bytes) {
        // Allow one oversized parameter to be loaded by evicting everything else.
        for (auto& kv : entries_) {
            if (kv.first == keep_name || kv.second.state == ShardState::Offloaded) {
                continue;
            }
            ShardEntry& victim = kv.second;
            offload_entry(victim);
            resident_bytes_ -= victim.num_bytes_fp32();
            victim.tensor.reset();
            victim.state = ShardState::Offloaded;
            victim.dirty = false;
        }
        return;
    }
    // If exceeds budget, evict by LRU (ignore currently needed keep_name)
    while (resident_bytes_ + incoming_bytes > cfg_.max_resident_bytes) {
        auto victim_it = std::min_element(entries_.begin(), entries_.end(),
            [&](const auto& a, const auto& b) {
                // Skip keep_name and those already not in memory
                const bool a_valid = (a.first != keep_name && a.second.state == ShardState::InMemory);
                const bool b_valid = (b.first != keep_name && b.second.state == ShardState::InMemory);
                if (a_valid != b_valid) return a_valid; // true < false
                if (!a_valid && !b_valid) return false;
                return a.second.last_used < b.second.last_used;
            });
        if (victim_it == entries_.end()) break;
        if (victim_it->first == keep_name || victim_it->second.state == ShardState::Offloaded) {
            break;
        }
        ShardEntry& victim = victim_it->second;
        offload_entry(victim);
        resident_bytes_ -= victim.num_bytes_fp32();
        victim.tensor.reset();
        victim.state = ShardState::Offloaded;
        victim.dirty = false;
    }
}

void ParameterSharder::offload_entry(ShardEntry& e) {
    if (memory_storage_ && e.tensor && !e.dirty && !e.encoded.empty()) {
        // Already has latest memory copy
    } else if (!memory_storage_ && e.tensor && !e.dirty && FileExists(e.path)) {
        // Already has latest disk copy
    } else if (e.tensor) {
        TensorPtr src_tensor = e.tensor;
        if (!src_tensor->is_contiguous()) {
            // Sharder must serialize logical tensor values, not raw strided view memory.
            src_tensor = src_tensor->contiguous();
        }
        const size_t elems = src_tensor ? static_cast<size_t>(src_tensor->numel()) : 0;
        DiskTensorHeader header;
        header.quantization = static_cast<uint8_t>(e.disk_quantization);
        header.dtype = static_cast<uint8_t>(e.dtype);
        header.numel = static_cast<uint64_t>(elems);
        if ((e.disk_quantization == DiskQuantizationMode::INT8 ||
             e.disk_quantization == DiskQuantizationMode::INT4) &&
            elems > 0) {
            header.scale = ComputeAbsMax(src_tensor->data<float>(), elems);
            if (header.scale <= 1e-12f) {
                header.scale = 1.0f;
            } else if (e.disk_quantization == DiskQuantizationMode::INT8) {
                header.scale /= 127.0f;
            } else {
                header.scale /= 7.0f;
            }
        }

        auto write_bytes = [&](const void* ptr, size_t count) {
            if (count == 0) {
                return;
            }
            if (memory_storage_) {
                const auto* begin = reinterpret_cast<const uint8_t*>(ptr);
                e.encoded.insert(e.encoded.end(), begin, begin + count);
            }
        };

        std::ofstream out;
        if (memory_storage_) {
            e.encoded.clear();
            e.encoded.reserve(sizeof(header) + EncodedPayloadBytes(e.disk_quantization, e.dtype, elems));
        } else {
            out.open(e.path, std::ios::binary | std::ios::trunc);
            if (!out) {
                throw std::runtime_error("offload_entry: cannot open " + e.path);
            }
        }

        if (memory_storage_) {
            write_bytes(&header, sizeof(header));
        } else {
            WriteHeader(out, header);
        }

        if (e.disk_quantization == DiskQuantizationMode::FP16) {
            // Streaming quantization, avoid allocating equally-sized buffer for entire parameter
            const float* src = src_tensor->data<float>();
            const size_t chunk = 1 << 18; // 256K elements (~512KB)
            std::vector<uint16_t> tmp(std::min(chunk, elems));
            size_t offset = 0;
            while (offset < elems) {
                size_t cur = std::min(chunk, elems - offset);
                for (size_t i = 0; i < cur; ++i) {
                    tmp[i] = float32_to_fp16(src[offset + i]);
                }
                if (memory_storage_) {
                    write_bytes(tmp.data(), cur * sizeof(uint16_t));
                } else {
                    out.write(reinterpret_cast<const char*>(tmp.data()), cur * sizeof(uint16_t));
                }
                offset += cur;
            }
        } else if (e.disk_quantization == DiskQuantizationMode::INT8) {
            const float* src = src_tensor->data<float>();
            const size_t chunk = 1 << 18;
            std::vector<int8_t> tmp(std::min(chunk, elems));
            size_t offset = 0;
            while (offset < elems) {
                size_t cur = std::min(chunk, elems - offset);
                QuantizeInt8Chunk(src + offset, tmp.data(), cur, header.scale);
                if (memory_storage_) {
                    write_bytes(tmp.data(), cur * sizeof(int8_t));
                } else {
                    out.write(reinterpret_cast<const char*>(tmp.data()), cur * sizeof(int8_t));
                }
                offset += cur;
            }
        } else if (e.disk_quantization == DiskQuantizationMode::INT4) {
            const float* src = src_tensor->data<float>();
            const size_t chunk = 1 << 18;
            std::vector<uint8_t> tmp((std::min(chunk, elems) + 1) / 2);
            size_t offset = 0;
            while (offset < elems) {
                size_t cur = std::min(chunk, elems - offset);
                tmp.resize((cur + 1) / 2);
                QuantizeInt4Chunk(src + offset, tmp.data(), cur, header.scale);
                if (memory_storage_) {
                    write_bytes(tmp.data(), tmp.size());
                } else {
                    out.write(reinterpret_cast<const char*>(tmp.data()), tmp.size());
                }
                offset += cur;
            }
        } else {
            const void* src = src_tensor->data_ptr();
            size_t bytes = elems * ((e.dtype == kFloat16) ? sizeof(uint16_t) : sizeof(float));
            if (memory_storage_) {
                write_bytes(src, bytes);
            } else {
                out.write(reinterpret_cast<const char*>(src), bytes);
            }
        }
        e.dirty = false;
    }
    // Release memory and sync external pointer
    if (e.owner_ptr) *e.owner_ptr = nullptr;
    e.tensor.reset();
}

void ParameterSharder::load_entry(ShardEntry& e) {
    // If already has memory tensor, return directly
    if (e.tensor) {
        e.state = ShardState::InMemory;
        return;
    }
    size_t elems = 1;
    for (auto d : e.shape) elems *= static_cast<size_t>(d);
    DiskTensorHeader header;
    bool has_header = false;
    std::ifstream in;
    const uint8_t* encoded_ptr = nullptr;
    size_t encoded_size = 0;
    size_t encoded_offset = 0;

    auto read_bytes = [&](void* dst, size_t count) {
        if (count == 0) {
            return;
        }
        if (memory_storage_) {
            if (!encoded_ptr || encoded_offset + count > encoded_size) {
                throw std::runtime_error("load_entry: in-memory shard buffer underrun for " + e.name);
            }
            std::memcpy(dst, encoded_ptr + encoded_offset, count);
            encoded_offset += count;
        } else {
            in.read(reinterpret_cast<char*>(dst), count);
            if (!in) {
                throw std::runtime_error("load_entry: short read for " + e.path);
            }
        }
    };

    if (memory_storage_) {
        if (e.encoded.empty()) {
            throw std::runtime_error("load_entry: missing in-memory shard for " + e.name);
        }
        encoded_ptr = e.encoded.data();
        encoded_size = e.encoded.size();
        if (encoded_size >= sizeof(header)) {
            std::memcpy(&header, encoded_ptr, sizeof(header));
            if (header.magic == kShardFileMagic) {
                has_header = true;
                encoded_offset = sizeof(header);
            }
        }
    } else {
        in.open(e.path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("load_entry: cannot open " + e.path);
        }
        has_header = TryReadHeader(in, header);
    }

    const DiskQuantizationMode stored_quant = has_header
        ? static_cast<DiskQuantizationMode>(header.quantization)
        : e.disk_quantization;
    DType target_dtype = (stored_quant == DiskQuantizationMode::None) ? e.dtype : kFloat32;
    TensorPtr t = std::make_shared<Tensor>(e.shape, target_dtype, kCPU);

    if (stored_quant == DiskQuantizationMode::FP16) {
        const size_t chunk = 1 << 18; // 256K elements
        std::vector<uint16_t> tmp(std::min(chunk, elems));
        float* dst = t->data<float>();
        size_t offset = 0;
        while (offset < elems) {
            size_t cur = std::min(chunk, elems - offset);
            read_bytes(tmp.data(), cur * sizeof(uint16_t));
            for (size_t i = 0; i < cur; ++i) {
                dst[offset + i] = fp16_to_float32(tmp[i]);
            }
            offset += cur;
        }
    } else if (stored_quant == DiskQuantizationMode::INT8) {
        const size_t chunk = 1 << 18;
        std::vector<int8_t> tmp(std::min(chunk, elems));
        float* dst = t->data<float>();
        size_t offset = 0;
        while (offset < elems) {
            size_t cur = std::min(chunk, elems - offset);
            read_bytes(tmp.data(), cur * sizeof(int8_t));
            DequantizeInt8Chunk(tmp.data(), dst + offset, cur, header.scale);
            offset += cur;
        }
    } else if (stored_quant == DiskQuantizationMode::INT4) {
        const size_t chunk = 1 << 18;
        std::vector<uint8_t> tmp((std::min(chunk, elems) + 1) / 2);
        float* dst = t->data<float>();
        size_t offset = 0;
        while (offset < elems) {
            size_t cur = std::min(chunk, elems - offset);
            tmp.resize((cur + 1) / 2);
            read_bytes(tmp.data(), tmp.size());
            DequantizeInt4Chunk(tmp.data(), dst + offset, cur, header.scale);
            offset += cur;
        }
    } else {
        size_t bytes = elems * ((e.dtype == kFloat16) ? sizeof(uint16_t) : sizeof(float));
        read_bytes(t->data_ptr(), bytes);
    }
    e.tensor = t;
    if (e.owner_ptr) *e.owner_ptr = e.tensor;
    e.state = ShardState::InMemory;
}

std::string ParameterSharder::sanitize_filename(const std::string& name) {
    std::string out = name;
    for (auto& ch : out) {
        if (ch == '/' || ch == ' ') ch = '_';
    }
    return out;
}

} // namespace sharding
} // namespace ops
