// Binary search: O(log n).  65,536 lookups in a sorted array of n elements.
//
// The same algorithm, on the same input, as binary_search.rs and binary_search.py: all three
// print the same CHECKSUM.  Only the algorithm is timed.
//
//   g++ -O2 -std=c++17 -o binary_search binary_search.cpp && ./binary_search --n 1000

#include "../common/harness.hpp"

using namespace harness;

constexpr std::size_t kQueries = 1 << 16;

// How many of `queries` are in `sorted`, each found by halving the range.
i64 binary_search_count(const std::vector<i64>& sorted, const std::vector<i64>& queries) {
    i64 found = 0;
    for (i64 q : queries) {
        std::size_t lo = 0, hi = sorted.size();
        while (lo < hi) {
            std::size_t mid = lo + (hi - lo) / 2;
            if (sorted[mid] < q) lo = mid + 1; else hi = mid;
        }
        if (lo < sorted.size() && sorted[lo] == q) ++found;
    }
    return found;
}

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    const auto n = static_cast<std::size_t>(args.n);
    SplitMix64 rng{kSeed};
    std::vector<i64> sorted(n);
    for (std::size_t i = 0; i < n; ++i) sorted[i] = 2 * static_cast<i64>(i);
    const std::vector<i64> queries = random_values(rng, kQueries, 2 * static_cast<u64>(n));

    i64 found = 0;
    escape(&found);
    const double ms = measure(args.min_ms, nothing, [&] {
        escape(sorted.data());
        escape(queries.data());
        found = binary_search_count(sorted, queries);
    });
    report(found, ms);
}
