/**
 * @file arena_allocator.h
 * @brief Partitioned memory management system for physical-footprint control
 * 
 * Core idea:
 * 1. StepScratchArena resets at the start of each step and reclaims all step
 *    activations at the end with one reset.
 * 2. StaticWeightArena stores long-lived read-mostly weights and avoids cache/trim logic.
 * 3. DirectLargeAllocation sends large tensors (>= 8 MB) through a dedicated path.
 * 
 * Goal: prevent Activity Monitor Memory/Footprint from growing linearly with steps.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>

#ifdef __APPLE__
#include <sys/mman.h>
#elif defined(__linux__)
#include <sys/mman.h>
#endif

namespace ops {
namespace memory {

// ============================================================================
// StepScratchArena - per-step scratch space (reset every step)
// ============================================================================

class StepScratchArena {
public:
    void* base_ptr_ = nullptr;  // Public for ArenaManager access.
    
private:
    size_t capacity_ = 0;
    size_t offset_ = 0;
    size_t peak_usage_ = 0;
    size_t num_allocations_ = 0;
    size_t num_resets_ = 0;
    
    static constexpr size_t ALIGNMENT = 64;
    
public:
    explicit StepScratchArena(size_t capacity_mb = 128);
    ~StepScratchArena();
    
    // Allocate memory aligned to 64 bytes.
    void* allocate(size_t size);
    
    // Reclaim all memory at the end of the step.
    void reset();
    
    // Fully recreate the arena and reset virtual address space state.
    void recreate();
    
    // Stats.
    size_t current_usage() const { return offset_; }
    size_t peak_usage() const { return peak_usage_; }
    size_t capacity() const { return capacity_; }
    void print_stats() const;
    
    // Non-copyable.
    StepScratchArena(const StepScratchArena&) = delete;
    StepScratchArena& operator=(const StepScratchArena&) = delete;
};

// ============================================================================
// StaticWeightArena - static weight region (read-mostly, outside cache logic)
// ============================================================================

class StaticWeightArena {
private:
    struct WeightBlock {
        void* ptr = nullptr;
        size_t size = 0;
        std::string name;
    };
    
    std::vector<WeightBlock> blocks_;
    size_t total_size_ = 0;
    mutable std::mutex mutex_;
    
public:
    StaticWeightArena() = default;
    ~StaticWeightArena();
    
    // Allocate static weight storage.
    void* allocate_static(size_t size, const std::string& name = "");
    
    // Stats.
    size_t total_size() const { return total_size_; }
    void print_stats() const;
    
    // Non-copyable.
    StaticWeightArena(const StaticWeightArena&) = delete;
    StaticWeightArena& operator=(const StaticWeightArena&) = delete;
};

// ============================================================================
// DirectLargeAllocator - direct allocation path for large tensors (>= 8 MB)
// ============================================================================

class DirectLargeAllocator {
private:
    struct LargeBlock {
        void* ptr = nullptr;
        size_t size = 0;
    };
    
    std::unordered_map<void*, LargeBlock> allocations_;
    size_t total_allocated_ = 0;
    size_t num_allocations_ = 0;
    mutable std::mutex mutex_;
    
    static constexpr size_t LARGE_THRESHOLD = 16 * 1024 * 1024;  // 16 MB threshold.
    
public:
    DirectLargeAllocator() = default;
    ~DirectLargeAllocator();
    
    // Decide whether a request should use the large-allocation path.
    static bool is_large(size_t size) { return size >= LARGE_THRESHOLD; }
    
    // Allocate a large tensor with direct mmap/malloc.
    void* allocate(size_t size);
    
    // Release a large tensor with madvise + munmap when applicable.
    void free(void* ptr);
    
    // Stats.
    size_t total_allocated() const { return total_allocated_; }
    void print_stats() const;
};

// ============================================================================
// ArenaManager - unified manager (thread-local + global singleton)
// ============================================================================

class ArenaManager {
private:
    // Global singleton state.
    std::unique_ptr<StaticWeightArena> static_arena_;
    std::unique_ptr<DirectLargeAllocator> large_allocator_;
    
    // Thread-local current-step arena. nullptr means disabled.
    static thread_local StepScratchArena* current_step_arena_;
    
    mutable std::mutex mutex_;
    
    ArenaManager();
    
public:
    ~ArenaManager();
    
    // Singleton access.
    static ArenaManager& instance();
    
    // Step-level arena control.
    void set_current_step_arena(StepScratchArena* arena);
    StepScratchArena* get_current_step_arena();
    void clear_current_step_arena();
    
    // Static weight arena access.
    StaticWeightArena& static_weights() { return *static_arena_; }
    
    // Direct large-allocation access.
    DirectLargeAllocator& large_alloc() { return *large_allocator_; }
    
    // Unified allocation entry point with size/context-based routing.
    void* allocate(size_t size);
    void free(void* ptr, size_t size);
    
    // Statistics and diagnostics.
    void print_all_stats() const;
    
    // Non-copyable.
    ArenaManager(const ArenaManager&) = delete;
    ArenaManager& operator=(const ArenaManager&) = delete;
};

// ============================================================================
// RAII helper for step-arena lifetime management
// ============================================================================

class StepArenaGuard {
private:
    StepScratchArena arena_;
    
public:
    explicit StepArenaGuard(size_t capacity_mb = 128) 
        : arena_(capacity_mb) {
        ArenaManager::instance().set_current_step_arena(&arena_);
    }
    
    ~StepArenaGuard() {
        ArenaManager::instance().clear_current_step_arena();
        arena_.reset();  // Reclaim everything in one reset.
    }
    
    StepScratchArena& get_arena() { return arena_; }
    
    // Proactively recreate the arena to limit macOS footprint buildup.
    void regenerate() {
        arena_.recreate();
    }
    
    // Non-copyable.
    StepArenaGuard(const StepArenaGuard&) = delete;
    StepArenaGuard& operator=(const StepArenaGuard&) = delete;
};

} // namespace memory
} // namespace ops
