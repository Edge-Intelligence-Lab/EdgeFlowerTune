/**
 * @file arena_allocator.cpp
 * @brief Partitioned memory management implementation
 */

#include "arena_allocator.h"
#include "../core/logger.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#ifdef __APPLE__
#include <sys/mman.h>
#include <unistd.h>
#elif defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace ops {
namespace memory {

// ============================================================================
// StepScratchArena implementation
// ============================================================================

StepScratchArena::StepScratchArena(size_t capacity_mb) 
    : capacity_(capacity_mb * 1024 * 1024), offset_(0), peak_usage_(0), 
      num_allocations_(0), num_resets_(0) {
    
    #if defined(__APPLE__) || defined(__linux__)
    // Reserve address space with mmap (MAP_ANON + MAP_PRIVATE).
    base_ptr_ = mmap(nullptr, capacity_, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_ptr_ == MAP_FAILED) {
        base_ptr_ = nullptr;
        throw std::bad_alloc();
    }
    
    // Hint the expected access pattern to the kernel.
    #ifdef __APPLE__
    madvise(base_ptr_, capacity_, MADV_SEQUENTIAL);  // Sequential access.
    #endif
    
    #else
    // Windows or other platforms: use malloc directly.
    base_ptr_ = std::malloc(capacity_);
    if (!base_ptr_) {
        throw std::bad_alloc();
    }
    #endif
    
    // quiet log: StepScratchArena initialized
}

StepScratchArena::~StepScratchArena() {
    if (base_ptr_) {
        #if defined(__APPLE__) || defined(__linux__)
        munmap(base_ptr_, capacity_);
        #else
        std::free(base_ptr_);
        #endif
    }
}

void* StepScratchArena::allocate(size_t size) {
    if (size == 0) return nullptr;
    
    // Alignment.
    size_t aligned_offset = (offset_ + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    
    if (aligned_offset + size > capacity_) {
        // Arena exhausted. The configured budget is too small.
        OPS_LOG_ERROR_F("StepScratchArena exhausted: need %zu MB, used %zu MB / %zu MB",
                       size / (1024 * 1024), aligned_offset / (1024 * 1024), 
                       capacity_ / (1024 * 1024));
        throw std::bad_alloc();
    }
    
    void* ptr = static_cast<char*>(base_ptr_) + aligned_offset;
    offset_ = aligned_offset + size;
    num_allocations_++;
    
    peak_usage_ = std::max(peak_usage_, offset_);
    
    // Zero-initialize the allocation.
    std::memset(ptr, 0, size);
    
    return ptr;
}

void StepScratchArena::reset() {
    #ifdef __APPLE__
    // On macOS, prefer MADV_DONTNEED for immediate physical-page reclamation.
    // It may trigger future page faults, but it prevents footprint accumulation.
    if (offset_ > 0) {
        madvise(base_ptr_, offset_, MADV_DONTNEED);
    }
    #elif defined(__linux__)
    // On Linux, MADV_DONTNEED immediately releases physical pages.
    if (offset_ > 0) {
        madvise(base_ptr_, offset_, MADV_DONTNEED);
    }
    #endif
    
    offset_ = 0;
    num_resets_++;
}

void StepScratchArena::recreate() {
    // Recreate the arena and reset the virtual address space.
    // This is the most reliable way to force real footprint reduction on macOS.
    
    #if defined(__APPLE__) || defined(__linux__)
    // 1. Release the current virtual address space with munmap.
    if (base_ptr_) {
        munmap(base_ptr_, capacity_);
        base_ptr_ = nullptr;
    }
    
    // 2. Recreate the arena with a fresh mmap region.
    base_ptr_ = mmap(nullptr, capacity_, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_ptr_ == MAP_FAILED) {
        base_ptr_ = nullptr;
        OPS_LOG_ERROR("Arena recreate failed: mmap failed");
        throw std::bad_alloc();
    }
    
    #ifdef __APPLE__
    madvise(base_ptr_, capacity_, MADV_SEQUENTIAL);
    #endif
    
    #else
    // Windows or other platforms: reallocate with malloc.
    if (base_ptr_) {
        std::free(base_ptr_);
    }
    base_ptr_ = std::malloc(capacity_);
    if (!base_ptr_) {
        throw std::bad_alloc();
    }
    #endif
    
    // 3. Reset runtime state.
    offset_ = 0;
    peak_usage_ = 0;
    num_allocations_ = 0;
    // Keep num_resets_ for cumulative statistics.
    
    // Recreate quietly to avoid log noise.
}

void StepScratchArena::print_stats() const {
    std::cout << "StepScratchArena Stats:\n";
    std::cout << "  Capacity: " << capacity_ / (1024 * 1024) << " MB\n";
    std::cout << "  Current usage: " << offset_ / (1024 * 1024) << " MB\n";
    std::cout << "  Peak usage: " << peak_usage_ / (1024 * 1024) << " MB\n";
    std::cout << "  Utilization: " << (100.0 * peak_usage_ / capacity_) << "%\n";
    std::cout << "  Total allocations: " << num_allocations_ << "\n";
    std::cout << "  Total resets: " << num_resets_ << "\n";
}

// ============================================================================
// StaticWeightArena implementation
// ============================================================================

StaticWeightArena::~StaticWeightArena() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& block : blocks_) {
        if (block.ptr) {
            #if defined(__APPLE__) || defined(__linux__)
            munmap(block.ptr, block.size);
            #else
            std::free(block.ptr);
            #endif
        }
    }
}

void* StaticWeightArena::allocate_static(size_t size, const std::string& name) {
    if (size == 0) return nullptr;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    void* ptr = nullptr;
    
    #if defined(__APPLE__) || defined(__linux__)
    // Allocate with mmap.
    ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        OPS_LOG_ERROR_F("Failed to allocate static weight: %zu MB", size / (1024 * 1024));
        return nullptr;
    }
    
    // Hint a random-access pattern to the kernel.
    madvise(ptr, size, MADV_RANDOM);
    
    #else
    ptr = std::malloc(size);
    if (!ptr) {
        return nullptr;
    }
    #endif
    
    // Zero-initialize the allocation.
    std::memset(ptr, 0, size);
    
    // Record the block.
    blocks_.push_back({ptr, size, name});
    total_size_ += size;
    
    // Log only at coarse milestones to avoid spamming output.
    static size_t last_logged_mb = 0;
    size_t current_mb = total_size_ / (1024 * 1024);
    if (current_mb >= last_logged_mb + 500) {  // Log every additional 500 MB.
        OPS_LOG_INFO_F("StaticWeightArena total: %zu MB", current_mb);
        last_logged_mb = current_mb;
    }
    
    return ptr;
}

void StaticWeightArena::print_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "StaticWeightArena Stats:\n";
    std::cout << "  Total blocks: " << blocks_.size() << "\n";
    std::cout << "  Total size: " << total_size_ / (1024 * 1024) << " MB\n";
    
    for (const auto& block : blocks_) {
        std::cout << "    - " << block.name << ": " 
                  << block.size / (1024 * 1024) << " MB\n";
    }
}

// ============================================================================
// DirectLargeAllocator implementation
// ============================================================================

DirectLargeAllocator::~DirectLargeAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& pair : allocations_) {
        if (pair.second.ptr) {
            #if defined(__APPLE__) || defined(__linux__)
            munmap(pair.second.ptr, pair.second.size);
            #else
            std::free(pair.second.ptr);
            #endif
        }
    }
}

void* DirectLargeAllocator::allocate(size_t size) {
    if (size == 0) return nullptr;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    void* ptr = nullptr;
    
    #if defined(__APPLE__) || defined(__linux__)
    // Use mmap for large tensors.
    ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        OPS_LOG_ERROR_F("Failed to allocate large tensor: %zu MB", size / (1024 * 1024));
        return nullptr;
    }
    #else
    ptr = std::malloc(size);
    if (!ptr) {
        return nullptr;
    }
    #endif
    
    // Zero-initialize the allocation.
    std::memset(ptr, 0, size);
    
    allocations_[ptr] = {ptr, size};
    total_allocated_ += size;
    num_allocations_++;
    
    // quiet log for DirectLarge allocations
    
    return ptr;
}

void DirectLargeAllocator::free(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        OPS_LOG_WARNING("Attempted to free unknown large pointer");
        return;
    }
    
    auto& block = it->second;
    
    #if defined(__APPLE__) || defined(__linux__)
    // First madvise the range so physical pages can be reclaimed.
    #ifdef __APPLE__
    madvise(block.ptr, block.size, MADV_FREE);
    #elif defined(__linux__)
    madvise(block.ptr, block.size, MADV_DONTNEED);
    #endif
    
    // Then unmap the region.
    munmap(block.ptr, block.size);
    #else
    std::free(block.ptr);
    #endif
    
    total_allocated_ -= block.size;
    allocations_.erase(it);
}

void DirectLargeAllocator::print_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "DirectLargeAllocator Stats:\n";
    std::cout << "  Active allocations: " << allocations_.size() << "\n";
    std::cout << "  Total allocated: " << total_allocated_ / (1024 * 1024) << " MB\n";
    std::cout << "  Total count: " << num_allocations_ << "\n";
}

// ============================================================================
// ArenaManager implementation
// ============================================================================

thread_local StepScratchArena* ArenaManager::current_step_arena_ = nullptr;

ArenaManager::ArenaManager() {
    static_arena_ = std::make_unique<StaticWeightArena>();
    large_allocator_ = std::make_unique<DirectLargeAllocator>();
    
    OPS_LOG_INFO("ArenaManager initialized (StaticWeight + DirectLarge)");
}

ArenaManager::~ArenaManager() = default;

ArenaManager& ArenaManager::instance() {
    static ArenaManager instance;
    return instance;
}

void ArenaManager::set_current_step_arena(StepScratchArena* arena) {
    current_step_arena_ = arena;
}

StepScratchArena* ArenaManager::get_current_step_arena() {
    return current_step_arena_;
}

void ArenaManager::clear_current_step_arena() {
    current_step_arena_ = nullptr;
}

void* ArenaManager::allocate(size_t size) {
    if (size == 0) return nullptr;
    
    // Routing policy:
    // 1. Large tensors -> DirectLargeAllocator
    // 2. Within a training step -> StepScratchArena
    // 3. Otherwise -> malloc fallback
    
    if (DirectLargeAllocator::is_large(size)) {
        return large_allocator_->allocate(size);
    }
    
    if (current_step_arena_) {
        try {
            return current_step_arena_->allocate(size);
        } catch (const std::bad_alloc&) {
            // Arena exhausted, fall back to malloc.
            OPS_LOG_WARNING("StepArena exhausted, fallback to malloc");
            void* ptr = std::malloc(size);
            if (ptr) std::memset(ptr, 0, size);
            return ptr;
        }
    }
    
    // Default fallback: malloc.
    void* ptr = std::malloc(size);
    if (ptr) {
        std::memset(ptr, 0, size);
    }
    return ptr;
}

void ArenaManager::free(void* ptr, size_t size) {
    if (!ptr) return;
    
    // Check whether this is a large tensor allocation.
    if (DirectLargeAllocator::is_large(size)) {
        large_allocator_->free(ptr);
        return;
    }
    
    // Step arena allocations are reclaimed by reset(), so they are not freed here.
    if (current_step_arena_) {
        // Simple heuristic: allocations made during a step belong to the arena.
        return;
    }
    
    // Otherwise this was a plain malloc allocation.
    std::free(ptr);
}

void ArenaManager::print_all_stats() const {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Arena Memory Management Statistics\n";
    std::cout << std::string(60, '=') << "\n";
    
    static_arena_->print_stats();
    std::cout << "\n";
    
    large_allocator_->print_stats();
    std::cout << "\n";
    
    if (current_step_arena_) {
        current_step_arena_->print_stats();
    } else {
        std::cout << "StepScratchArena: Not active\n";
    }
    
    std::cout << std::string(60, '=') << "\n";
}

} // namespace memory
} // namespace ops
