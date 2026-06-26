#include "device/AllocationTracker.h"
#include <algorithm>
#include <chrono>
#include <iomanip>

namespace OwnTensor {

static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

AllocationTracker& AllocationTracker::instance() {
    static AllocationTracker tracker;
    return tracker;
}

void AllocationTracker::set_location(const char* loc) {
    current_location() = (loc != nullptr) ? loc : "UNKNOWN";
}

void AllocationTracker::clear_location() {
    current_location() = "UNKNOWN";
}

const char* AllocationTracker::get_location() {
    return current_location().c_str();
}

void AllocationTracker::init(const char* csv_path) {
    std::lock_guard<std::mutex> lock(mtx_);

    if (initialized_)
        return;

    csv_.open(csv_path, std::ios::out | std::ios::trunc);
    if (!csv_.is_open()) {
        return; // Silent fail - no console output
    }

    csv_ << "id,event,device,address,requested_bytes,allocated_bytes,timestamp_ns,location\n";
    csv_.flush();

    initialized_ = true;
}

void AllocationTracker::shutdown() {
    std::lock_guard<std::mutex> lock(mtx_);

    if (!initialized_)
        return;

    csv_.flush();
    csv_.close();
    initialized_ = false;
}

void AllocationTracker::on_alloc(void* ptr, size_t bytes, size_t allocated_bytes, int device) {
    if (!initialized_) {
        return;
    }

    uint64_t ts = now_ns();
    uint64_t id = alloc_counter_.fetch_add(1);

    std::lock_guard<std::mutex> lock(mtx_);

    std::string location = current_location();

    AllocationRecord record;
    record.id = id;
    record.device = device;
    record.address = ptr;
    record.bytes = bytes;
    record.allocated_bytes = allocated_bytes;
    record.timestamp_ns = ts;
    record.location = location;

    live_allocs_[ptr] = record;

    current_bytes_[device] += allocated_bytes;
    if (current_bytes_[device] > peak_bytes_[device]) {
        peak_bytes_[device] = current_bytes_[device];
    }

    csv_ << id << ",ALLOC," << device << "," << ptr << "," << bytes << ","
         << allocated_bytes << "," << ts << "," << location << "\n";
    csv_.flush();
}

void AllocationTracker::on_free(void* ptr, int device) {
    if (!initialized_) {
        return;
    }

    uint64_t ts = now_ns();

    std::lock_guard<std::mutex> lock(mtx_);

    uint64_t id = 0;
    size_t bytes = 0;
    size_t allocated_bytes = 0;
    std::string location = "UNKNOWN";

    auto it = live_allocs_.find(ptr);
    if (it != live_allocs_.end()) {
        id = it->second.id;
        bytes = it->second.bytes;
        allocated_bytes = it->second.allocated_bytes;
        location = it->second.location;
        live_allocs_.erase(it);

        if (current_bytes_.count(device) && current_bytes_[device] >= allocated_bytes) {
            current_bytes_[device] -= allocated_bytes;
        }
    }

    dealloc_counter_.fetch_add(1);

    csv_ << id << ",FREE," << device << "," << ptr << "," << bytes << ","
         << allocated_bytes << "," << ts << "," << location << "\n";
    csv_.flush();
}

std::vector<LeakInfo> AllocationTracker::get_leaks() const {
    uint64_t now = now_ns();
    std::vector<LeakInfo> leaks;

    std::lock_guard<std::mutex> lock(mtx_);

    for (const auto& [ptr, record] : live_allocs_) {
        LeakInfo info;
        info.id = record.id;
        info.bytes = record.bytes;
        info.device = record.device;
        info.age_ns = now - record.timestamp_ns;
        info.location = record.location;
        leaks.push_back(info);
    }

    std::sort(leaks.begin(), leaks.end(),
              [](const LeakInfo& a, const LeakInfo& b) {
                  return a.id < b.id;
              });

    return leaks;
}

void AllocationTracker::write_leak_report(const char* output_path) const {
    auto leaks = get_leaks();

    std::ofstream out(output_path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        return;
    }

    out << "id,device,bytes,age_ms,location\n";

    for (const auto& leak : leaks) {
        double age_ms = leak.age_ns / 1e6;
        out << leak.id << "," << leak.device << "," << leak.bytes << ","
            << std::fixed << std::setprecision(2) << age_ms << ","
            << leak.location << "\n";
    }

    out.close();
}

size_t AllocationTracker::get_total_allocations() const {
    return alloc_counter_.load();
}

size_t AllocationTracker::get_total_deallocations() const {
    return dealloc_counter_.load();
}

size_t AllocationTracker::get_current_allocated(int device) const {
    std::lock_guard<std::mutex> lock(mtx_);

    if (device == -1) {
        size_t total = 0;
        for (const auto& [dev, bytes] : current_bytes_) {
            total += bytes;
        }
        return total;
    }

    auto it = current_bytes_.find(device);
    return (it != current_bytes_.end()) ? it->second : 0;
}

size_t AllocationTracker::get_peak_allocated(int device) const {
    std::lock_guard<std::mutex> lock(mtx_);

    if (device == -1) {
        size_t total = 0;
        for (const auto& [dev, bytes] : peak_bytes_) {
            total += bytes;
        }
        return total;
    }

    auto it = peak_bytes_.find(device);
    return (it != peak_bytes_.end()) ? it->second : 0;
}

void AllocationTracker::reset_peak() {
    std::lock_guard<std::mutex> lock(mtx_);

    for (auto& [device, peak] : peak_bytes_) {
        peak = current_bytes_[device];
    }
}

} // namespace OwnTensor