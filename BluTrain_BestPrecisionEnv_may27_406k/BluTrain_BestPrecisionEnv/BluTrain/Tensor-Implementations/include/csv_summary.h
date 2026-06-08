#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cstdint>

namespace csv_summary {

struct Record {
    uint64_t id;
    std::string event;
    int        device;
    uint64_t   address;
    uint64_t   requested_bytes;
    uint64_t   allocated_bytes;
    uint64_t   timestamp_ns;
    std::string location;
};

struct ScopeStat {
    uint64_t allocations   = 0;
    uint64_t deallocations = 0;
};

struct Summary {
    uint64_t totalAllocs   = 0;
    uint64_t totalDeallocs = 0;
    std::map<std::string, ScopeStat> perScope;
    std::unordered_map<uint64_t, uint64_t> liveAddresses;
    uint64_t liveTensors = 0;
};

inline std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

inline std::vector<std::string> splitCSV(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool inQuote = false;
    for (char c : line) {
        if (c == '"') {
            inQuote = !inQuote;
        } else if (c == ',' && !inQuote) {
            fields.push_back(trim(field));
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(trim(field));
    return fields;
}

inline int colIndex(const std::vector<std::string>& headers, const std::string& name) {
    for (int i = 0; i < (int)headers.size(); ++i)
        if (headers[i] == name) return i;
    return -1;
}

inline std::vector<Record> parseCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open file '" << filename << "'\n";
        std::exit(1);
    }

    std::string line;
    if (!std::getline(file, line)) {
        std::cerr << "Error: empty file\n";
        std::exit(1);
    }
    std::vector<std::string> headers = splitCSV(line);

    int ci_id       = colIndex(headers, "id");
    int ci_event    = colIndex(headers, "event");
    int ci_device   = colIndex(headers, "device");
    int ci_address  = colIndex(headers, "address");
    int ci_req_bytes = colIndex(headers, "requested_bytes");
    int ci_alloc_bytes = colIndex(headers, "allocated_bytes");
    int ci_ts       = colIndex(headers, "timestamp_ns");
    int ci_location = colIndex(headers, "location");

    auto require = [&](int idx, const std::string& name) {
        if (idx < 0) {
            std::cerr << "Error: required column '" << name << "' not found in CSV header\n";
            std::exit(1);
        }
    };
    require(ci_id,          "id");
    require(ci_event,       "event");
    require(ci_device,      "device");
    require(ci_address,     "address");
    require(ci_req_bytes,   "requested_bytes");
    require(ci_alloc_bytes, "allocated_bytes");
    require(ci_ts,          "timestamp_ns");
    require(ci_location,    "location");

    std::vector<Record> records;
    while (std::getline(file, line)) {
        if (trim(line).empty()) continue;
        auto f = splitCSV(line);
        if ((int)f.size() <= std::max({ci_id, ci_event, ci_device,
                                       ci_address, ci_req_bytes, ci_alloc_bytes, ci_ts, ci_location}))
            continue;

        Record r;
        try {
            r.id              = std::stoull(f[ci_id]);
            r.event           = f[ci_event];
            r.device          = std::stoi(f[ci_device]);
            r.address         = std::stoull(f[ci_address], nullptr, 0);
            r.requested_bytes = std::stoull(f[ci_req_bytes]);
            r.allocated_bytes = std::stoull(f[ci_alloc_bytes]);
            r.timestamp_ns    = std::stoull(f[ci_ts]);
            r.location        = f[ci_location].empty() ? "UNKNOWN" : f[ci_location];
        } catch (...) {
            continue;
        }
        records.push_back(r);
    }
    return records;
}

inline Summary analyse(const std::vector<Record>& records) {
    Summary s;
    for (const auto& r : records) {
        if (r.event == "ALLOC") {
            ++s.totalAllocs;
            ++s.perScope[r.location].allocations;
            s.liveAddresses[r.address] = r.allocated_bytes;
        } else if (r.event == "FREE") {
            ++s.totalDeallocs;
            ++s.perScope[r.location].deallocations;
            s.liveAddresses.erase(r.address);
        }
    }
    s.liveTensors = s.liveAddresses.size();
    return s;
}

inline void printSeparator(int width = 118) {
    std::cout << std::string(width, '-') << "|\n";
}

inline void printEqSeparator(int width = 90) {
    std::cout << std::string(width, '=') << "\n";
}

inline void printSummary(const Summary& s) {
    std::cout << "\n\n";
    printEqSeparator();
    std::cout << "                             ALLOCATION ANALYSIS SUMMARY                                \n";
    printEqSeparator();
    std::cout << '\n';

    std::cout << "Total Allocations:   " << s.totalAllocs   << '\n';
    std::cout << "Total Deallocations: " << s.totalDeallocs << '\n';
    std::cout << "Live Tensors:        " << s.liveTensors   << '\n';
    std::cout << '\n';

    std::cout << "Allocations per Scope:\n\n";

    std::cout << std::left
              << std::setw(60) << "           Location"
              << std::setw(18) << "|   Allocations   "
              << std::setw(18+2) << "|   Deallocations   "
              << "|   Alive Tensors   "
              << "|\n";
    printSeparator();

    for (const auto& [loc, st] : s.perScope) {
        int64_t diff = (int64_t)st.allocations - (int64_t)st.deallocations;
        std::cout << std::left
                  << std::setw(60) << loc << "|    "
                  << std::setw(10) << st.allocations << "   |    "
                  << std::setw(12) << st.deallocations << "   |    "
                  << std::setw(15) << diff
                  << "|\n";
    }
    std::cout << '\n';
}

// Single entry point: pass a CSV path and it prints the summary
inline void printCSVSummary(const std::string& csvPath) {
    auto records = parseCSV(csvPath);
    auto summary = analyse(records);
    printSummary(summary);
}

} // namespace csv_summary
