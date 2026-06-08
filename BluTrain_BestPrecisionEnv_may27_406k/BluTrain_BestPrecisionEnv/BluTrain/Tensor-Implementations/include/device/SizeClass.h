#pragma once

#include <cstddef>
#include <vector>
#include <mutex>
#include <iostream>
#include <iomanip>
#include "device/SizeClassTuner.h"
#include "device/TuningConfig.h"

namespace OwnTensor
{
    class SizeClass {
        public:
            // rounding upto appropriate size grp
            static size_t round_size(size_t size) {

                // If tuned boundaries are active, use them
                if (tuned_active()) {
                    // Small allocations still use 512-byte alignment
                    if (size < MB_1) {
                        return ((size + 511) / 512) * 512;
                    }
                    const auto& b = boundaries();
                    for (size_t bound : b) {
                        if (size <= bound) return bound;
                    }
                    // Larger than all boundaries -- 2 MB alignment
                    return ((size + ALIGN_MB_2 - 1) / ALIGN_MB_2) * ALIGN_MB_2;
                }

                // DEFAULT:
                // if (size < kSmallSize) {
                //    // Rounding to 512 byte allocations for small allocations
                //     return ((size + 511) / 512) * 512;
                // } else if (size < kSmallBuffer) {
                //     // Round to 2Mb for medium allocations
                //     return kSmallBuffer;
                // } else if (size < kLargeBuffer) {
                //     // Round to 20MB for large allocations
                //     return kLargeBuffer;
                // } else {
                //     // Round to 2MB alignment for very large allocations
                //     return ((size + kRoundLarge - 1) / kRoundLarge) * kRoundLarge;
                // }

                // KURRA's OPTIMIZATION
                // if (size < 512) return 512;
                // if (size <= kSmallSize) {
                //     return ((size + 511) / 512) * 512;
                // }

                // size_t rounded = 1;
                // while (rounded < size) rounded <<= 1;

                // // If it's a power of 2, we are done
                // if (rounded == size) return size;

                // // Otherwise, divide the range [rounded/2, rounded] into 8 parts
                // size_t prev_pow2 = rounded >> 1;
                // size_t step = (rounded - prev_pow2) / 8;

                // return prev_pow2 + ((size - prev_pow2 + step - 1) / step) * step;

                // CUSTOM SPLIT - WORKED BETTER ON PRE OPTIM 44M
                if (size < 512) return 512;
                if (size < MB_1) {
                    return ((size + 511) / 512) * 512;
                } else if (size <= MB_1_5) {
                    return MB_1_5;
                }
                else if (size <= MB_2) {
                    return MB_2;
                }
                else if (size <= MB_3) {
                    return MB_3;
                }
                else if (size <= MB_4) {
                    return MB_4;
                }
                else if (size <= MB_7) {
                    return MB_7;
                }
                else if (size <= MB_9) {
                    return MB_9;
                }
                else if (size < MB_20) {
                    return MB_20;
                }
                else {
                    return ((size + ALIGN_MB_1 - 1) / ALIGN_MB_1) * ALIGN_MB_1;
                }
            }

            /**
             * Round of 2 divisions
             *
             * Basically instead of the standard 512 bytes alignment with the allocation,
             * we can use rounding based on the power of 2's
             *
             * We can split stuffs in this ratio
             * [256: 1, 512: 2, 1024:4, >:8]
             *
             * Example:
             *      For a 1200 byte allocation, since it is between 1024 (2^10) and 2048 (2^11)
             *      we do 4 divisions for this region so 1280, 1536, 1792 bytes instead of the usual
             *      512 bytes alignment
             *
             *      So for 1200 bytes - instead of aligning to 1536 bytes, using this we can allocate just
             *      1024 + 256 = 1280 bytes which gives less unused memory and reduce fragmentation
             */

            // to use small pool whenever possible
            static bool is_small(size_t size) {
                return size <= kSmallSize;
            }

            // ---------------------------------------------------------------
            // Runtime boundary management
            // ---------------------------------------------------------------

            // Replace the current boundaries with new ones.
            // boundaries must be sorted ascending.
            static void set_boundaries(const std::vector<size_t>& new_boundaries) {
                std::lock_guard<std::mutex> lock(boundaries_mutex());
                boundaries() = new_boundaries;
                tuned_active() = true;

                std::cerr << "[SizeClass] Boundaries updated (tuned rounding active): [";
                for (size_t i = 0; i < new_boundaries.size(); i++) {
                    if (i > 0) std::cerr << ", ";
                    std::cerr << std::fixed << std::setprecision(3)
                              << (double)new_boundaries[i] / (1024.0*1024.0) << "MB";
                }
                std::cerr << "]\n";
            }

            static const std::vector<size_t>& get_boundaries() {
                // No lock on read path for performance -- boundaries are only
                // written once (after warm-up, before any allocations on new config).
                return boundaries();
            }

            static void reset_to_default() {
                std::lock_guard<std::mutex> lock(boundaries_mutex());
                boundaries() = default_boundaries();
                tuned_active() = false;
            }

            // Thresholds
            static constexpr size_t kSmallSize = 1048576;           // 1MB - small pool threshold
            static constexpr size_t kSmallBuffer = 1048576 * 2;     // 2MB
            static constexpr size_t kLargeBuffer = 1048576 * 20;    // 20MB
            static constexpr size_t kMinLargeAlloc = 1048576 * 10;  // 10MB
            static constexpr size_t kRoundLarge = 2097152;          // 2MB alignment

            static constexpr size_t MB_1 = 1048576;
            static constexpr size_t MB_1_5 = 1572864;
            static constexpr size_t MB_2 = 2097152;
            static constexpr size_t MB_3 = 3145728;
            static constexpr size_t MB_4 = 4194304;
            static constexpr size_t MB_6 = 6291456;
            static constexpr size_t MB_7 = 7340032;
            static constexpr size_t MB_9 = 9437184;
            static constexpr size_t MB_10 = 10485760;
            static constexpr size_t MB_13 = 13631488;
            static constexpr size_t MB_20 = 20971520;
            static constexpr size_t ALIGN_MB_1 = 1048576;
            static constexpr size_t ALIGN_MB_2 = 2097152;

        private:
            static std::mutex& boundaries_mutex() {
                static std::mutex mtx;
                return mtx;
            }

            static bool& tuned_active() {
                static bool active = false;
                return active;
            }

            static std::vector<size_t>& boundaries() {
                static std::vector<size_t> b = default_boundaries();
                return b;
            }

            static std::vector<size_t> default_boundaries() {
                return {kSmallBuffer, kLargeBuffer};
            }

    };
}