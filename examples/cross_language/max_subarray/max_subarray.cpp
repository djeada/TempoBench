// Maximum subarray (Kadane): O(n).  The largest sum of a contiguous run of n values.
//
// The same algorithm, on the same input, as max_subarray.rs and max_subarray.py: all three
// print the same CHECKSUM.  Only the algorithm is timed.
//
//   g++ -O2 -std=c++17 -o max_subarray max_subarray.cpp && ./max_subarray --n 1000

#include "../common/harness.hpp"

using namespace harness;

// Kadane's algorithm: the best sum ending here either extends the best sum
// ending one step earlier or starts afresh.
i64 max_subarray(const std::vector<i64>& values) {
    i64 best = values[0], current = values[0];
    for (std::size_t i = 1; i < values.size(); ++i) {
        current = std::max(values[i], current + values[i]);
        best = std::max(best, current);
    }
    return best;
}

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    SplitMix64 rng{kSeed};
    std::vector<i64> values = random_values(rng, static_cast<std::size_t>(args.n), 2001);
    for (auto& v : values) v -= 1000;

    i64 best = 0;
    escape(&best);
    const double ms = measure(args.min_ms, nothing, [&] {
        escape(values.data());
        best = max_subarray(values);
    });
    report(best, ms);
}
