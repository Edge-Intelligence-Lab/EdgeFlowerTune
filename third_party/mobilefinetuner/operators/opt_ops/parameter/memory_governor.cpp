/**
 * @file memory_governor.cpp
 * @brief Memory governor implementation
 */

#include "memory_governor.h"
#include "../core/logger.h"
#include <iostream>
#include <iomanip>

#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <fstream>
#include <string>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif

namespace ops {
namespace memory {

MemoryGovernor::MemoryGovernor(const MemoryBudget& budget)
    : budget_(budget), peak_rss_mb_(0), num_warnings_(0), num_reductions_(0) {
    
    OPS_LOG_INFO_F("MemoryGovernor initialized: soft=%zuMB, hard=%zuMB",
                  budget_.soft_limit_mb, budget_.hard_limit_mb);
}

void MemoryGovernor::set_reduction_callback(ReductionCallback callback) {
    reduction_callback_ = callback;
}

size_t MemoryGovernor::get_current_rss_mb() const {
#ifdef __APPLE__
    struct task_basic_info info;
    mach_msg_type_number_t size = TASK_BASIC_INFO_COUNT;
    kern_return_t kerr = task_info(mach_task_self(), TASK_BASIC_INFO,
                                   (task_info_t)&info, &size);
    if (kerr == KERN_SUCCESS) {
        return info.resident_size / (1024 * 1024);
    }
#elif defined(__linux__)
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("VmRSS:") == 0) {
            size_t kb = std::stoul(line.substr(6));
            return kb / 1024;
        }
    }
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024 * 1024);
    }
#endif
    return 0;
}

size_t MemoryGovernor::get_current_footprint_mb() const {
    // Approximate footprint with RSS. On macOS the true footprint can be higher.
    // A more precise value would require vmmap parsing or additional task_info fields.
    return get_current_rss_mb();
}

MemoryPressureLevel MemoryGovernor::assess_pressure(size_t rss_mb) const {
    if (rss_mb >= budget_.hard_limit_mb) {
        return MemoryPressureLevel::EMERGENCY;
    } else if (rss_mb >= budget_.soft_limit_mb) {
        return MemoryPressureLevel::CRITICAL;
    } else if (rss_mb >= budget_.warning_threshold_mb) {
        return MemoryPressureLevel::WARNING;
    } else {
        return MemoryPressureLevel::NORMAL;
    }
}

MemoryStatus MemoryGovernor::check_and_act() {
    size_t rss_mb = get_current_rss_mb();
    size_t footprint_mb = get_current_footprint_mb();
    
    peak_rss_mb_ = std::max(peak_rss_mb_, rss_mb);
    
    MemoryPressureLevel pressure = assess_pressure(rss_mb);
    
    MemoryStatus status;
    status.rss_mb = rss_mb;
    status.footprint_mb = footprint_mb;
    status.pressure = pressure;
    status.should_reduce_config = false;
    status.should_stop = false;
    
    switch (pressure) {
        case MemoryPressureLevel::EMERGENCY:
            OPS_LOG_ERROR_F("EMERGENCY: RSS=%zuMB >= hard_limit=%zuMB, stopping",
                           rss_mb, budget_.hard_limit_mb);
            std::cout << "Memory exceeded the hard limit (" << rss_mb << "MB >= "
                      << budget_.hard_limit_mb << "MB). Stopping training." << std::endl;
            status.should_stop = true;
            break;
            
        case MemoryPressureLevel::CRITICAL:
            num_reductions_++;
            OPS_LOG_WARNING("CRITICAL: RSS exceeds soft_limit, triggering reduction");
            std::cout << "Memory pressure reached the critical threshold (" << rss_mb
                      << "MB >= " << budget_.soft_limit_mb
                      << "MB). Triggering automatic degradation." << std::endl;
            status.should_reduce_config = true;
            
            if (reduction_callback_) {
                reduction_callback_(pressure);
            }
            break;
            
        case MemoryPressureLevel::WARNING:
            if (num_warnings_++ % 10 == 0) {  // Log once every 10 warnings.
                OPS_LOG_WARNING("WARNING: RSS exceeds warning threshold");
            }
            break;
            
        case MemoryPressureLevel::NORMAL:
            // Normal state, no action required.
            break;
    }
    
    return status;
}

MemoryStatus MemoryGovernor::get_status() const {
    size_t rss_mb = get_current_rss_mb();
    
    MemoryStatus status;
    status.rss_mb = rss_mb;
    status.footprint_mb = get_current_footprint_mb();
    status.pressure = assess_pressure(rss_mb);
    status.should_reduce_config = (status.pressure == MemoryPressureLevel::CRITICAL);
    status.should_stop = (status.pressure == MemoryPressureLevel::EMERGENCY);
    
    return status;
}

void MemoryGovernor::force_check() {
    auto status = check_and_act();
    
    std::cout << "Memory Governor Status:\n";
    std::cout << "  RSS: " << status.rss_mb << " MB\n";
    std::cout << "  Footprint: " << status.footprint_mb << " MB\n";
    std::cout << "  Pressure: ";
    
    switch (status.pressure) {
        case MemoryPressureLevel::NORMAL:
            std::cout << "NORMAL ✅\n";
            break;
        case MemoryPressureLevel::WARNING:
            std::cout << "WARNING ⚠️\n";
            break;
        case MemoryPressureLevel::CRITICAL:
            std::cout << "CRITICAL ⚠️⚠️\n";
            break;
        case MemoryPressureLevel::EMERGENCY:
            std::cout << "EMERGENCY 🚨\n";
            break;
    }
}

void MemoryGovernor::print_report() const {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Memory Governor Report\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Budget:\n";
    std::cout << "  Warning threshold: " << budget_.warning_threshold_mb << " MB\n";
    std::cout << "  Soft limit: " << budget_.soft_limit_mb << " MB\n";
    std::cout << "  Hard limit: " << budget_.hard_limit_mb << " MB\n";
    std::cout << "\nStatistics:\n";
    std::cout << "  Peak RSS: " << peak_rss_mb_ << " MB\n";
    std::cout << "  Warnings triggered: " << num_warnings_ << "\n";
    std::cout << "  Reductions triggered: " << num_reductions_ << "\n";
    
    auto status = get_status();
    std::cout << "\nCurrent Status:\n";
    std::cout << "  RSS: " << status.rss_mb << " MB\n";
    std::cout << "  Utilization: " << std::fixed << std::setprecision(1)
              << (100.0 * status.rss_mb / budget_.soft_limit_mb) << "%\n";
    std::cout << std::string(60, '=') << "\n";
}

} // namespace memory
} // namespace ops
