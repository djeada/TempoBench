// What every C++ implementation in examples/cross_language shares: the input
// generator, the timing loop and the output.  harness.rs and harness.py are
// the same code in Rust and Python, so all three languages do the same work.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace harness {

using u64 = std::uint64_t;
using i64 = std::int64_t;

constexpr u64 kSeed = 42;
constexpr u64 kHashMod = 1000000007;

// The input generator: SplitMix64, identical in every language.
struct SplitMix64 {
    u64 state;
    u64 next() {
        state += 0x9E3779B97F4A7C15ULL;
        u64 z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    i64 below(u64 bound) { return static_cast<i64>((next() >> 11) % bound); }
};

// Optimisation barriers in the style of Google Benchmark.  `escape` makes the
// compiler assume the pointed-to memory is read and may be modified by someone
// else, and `opaque` hides a scalar's value, so a kernel whose inputs did not
// change between calls is still executed on every call rather than hoisted.
inline void escape(const void* p) { asm volatile("" : : "g"(p) : "memory"); }
template <typename T>
inline T opaque(T value) {
    asm volatile("" : "+r"(value));
    return value;
}

inline i64 hash_sequence(const std::vector<i64>& values) {
    u64 h = 0;
    for (i64 v : values) h = (h * 31 + static_cast<u64>(v)) % kHashMod;
    return static_cast<i64>(h);
}

inline std::vector<i64> random_values(SplitMix64& rng, std::size_t n, u64 bound) {
    std::vector<i64> values(n);
    for (auto& v : values) v = rng.below(bound);
    return values;
}

// Copy `input` rotated by a step that changes every call.  Sorting the very
// same array over and over lets the branch predictor learn its comparisons,
// which makes small inputs unrealistically fast; a rotation gives every call
// a different arrangement while the sorted result stays the same.
inline void rotated_copy(const std::vector<i64>& input, std::vector<i64>& out, std::size_t& shift) {
    shift = (shift + 7919) % input.size();
    std::rotate_copy(input.begin(), input.begin() + static_cast<std::ptrdiff_t>(shift), input.end(), out.begin());
}

// Repeat `kernel` until at least `min_ms` of it has been timed and return the
// mean duration of one call.  `prepare` runs untimed before each call.  Fast
// kernels are thereby timed over many calls instead of one call near the
// clock's resolution.
template <typename Prepare, typename Kernel>
double measure(double min_ms, Prepare prepare, Kernel kernel) {
    using Clock = std::chrono::steady_clock;
    double total = 0.0;
    long calls = 0;
    do {
        prepare();
        escape(&calls);
        auto start = Clock::now();
        kernel();
        escape(&calls);
        total += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        ++calls;
    } while (total < min_ms);
    return total / static_cast<double>(calls);
}

inline void nothing() {}

struct Args {
    long long n = -1;
    double min_ms = 20.0;
};

[[noreturn]] inline void usage(const char* message) {
    std::fprintf(stderr, "error: %s\nusage: PROGRAM --n N [--min-ms MS]\n", message);
    std::exit(2);
}

inline Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) usage("missing value for an option");
        if (!std::strcmp(argv[i], "--n")) args.n = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--min-ms")) args.min_ms = std::atof(argv[++i]);
        else usage("unknown option");
    }
    if (args.n < 1) usage("a positive --n is required");
    return args;
}

// The two lines TempoBench reads: what was computed, and how long it took.
inline void report(i64 checksum, double ms) {
    std::printf("CHECKSUM: %lld\n", static_cast<long long>(checksum));
    std::printf("TEMPOBENCH_MS: %.6f\n", ms);
}

}  // namespace harness
