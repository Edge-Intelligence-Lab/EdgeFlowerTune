/**
 * @file memory_governor.h
 * @brief Memory governor with hard limits and automatic degradation
 * 
 * Core functionality:
 * 1. Monitor RSS and footprint in real time.
 * 2. Automatically reduce configuration when memory crosses thresholds.
 * 3. Force-stop when the hard limit is reached.
 * 4. Provide a detailed memory report.
 */

#pragma once

#include <cstddef>
#include <string>
#include <functional>

namespace ops {
namespace memory {

struct MemoryBudget {
    size_t soft_limit_mb = 2048;  // 2 GB soft limit: trigger degradation.
    size_t hard_limit_mb = 4096;  // 4 GB hard limit: force stop.
    size_t warning_threshold_mb = 1536;  // 1.5 GB warning threshold.
};

enum class MemoryPressureLevel {
    NORMAL,      // < warning_threshold
    WARNING,     // >= warning_threshold
    CRITICAL,    // >= soft_limit
    EMERGENCY    // >= hard_limit
};

struct MemoryStatus {
    size_t rss_mb = 0;
    size_t footprint_mb = 0;
    size_t vsz_mb = 0;
    MemoryPressureLevel pressure = MemoryPressureLevel::NORMAL;
    bool should_reduce_config = false;
    bool should_stop = false;
};

class MemoryGovernor {
private:
    MemoryBudget budget_;
    size_t peak_rss_mb_ = 0;
    size_t num_warnings_ = 0;
    size_t num_reductions_ = 0;
    
    // Reduction callback.
    using ReductionCallback = std::function<void(MemoryPressureLevel)>;
    ReductionCallback reduction_callback_;
    
public:
    explicit MemoryGovernor(const MemoryBudget& budget = MemoryBudget());
    
    // Set the reduction callback.
    void set_reduction_callback(ReductionCallback callback);
    
    // Monitoring checkpoint, typically called every step.
    MemoryStatus check_and_act();
    
    // Get the current memory status.
    MemoryStatus get_status() const;
    
    // Force a status check, useful for debugging.
    void force_check();
    
    // Stats.
    void print_report() const;
    size_t peak_rss() const { return peak_rss_mb_; }
    
private:
    size_t get_current_rss_mb() const;
    size_t get_current_footprint_mb() const;
    MemoryPressureLevel assess_pressure(size_t rss_mb) const;
};

} // namespace memory
} // namespace ops
