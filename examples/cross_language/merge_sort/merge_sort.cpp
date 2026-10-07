// Merge sort: O(n log n).  Top-down merge sort of n random values.
//
// The same algorithm, on the same input, as merge_sort.rs and merge_sort.py: all three
// print the same CHECKSUM.  Only the algorithm is timed.
//
//   g++ -O2 -std=c++17 -o merge_sort merge_sort.cpp && ./merge_sort --n 1000

#include "../common/harness.hpp"

using namespace harness;

// Sort a[lo, hi): sort each half, then merge the halves through `tmp`.
void merge_sort_range(std::vector<i64>& a, std::vector<i64>& tmp, std::size_t lo, std::size_t hi) {
    if (hi - lo < 2) return;
    std::size_t mid = lo + (hi - lo) / 2;
    merge_sort_range(a, tmp, lo, mid);
    merge_sort_range(a, tmp, mid, hi);
    std::size_t i = lo, j = mid, k = lo;
    while (i < mid && j < hi) tmp[k++] = (a[j] < a[i]) ? a[j++] : a[i++];
    while (i < mid) tmp[k++] = a[i++];
    while (j < hi) tmp[k++] = a[j++];
    for (k = lo; k < hi; ++k) a[k] = tmp[k];
}

void merge_sort(std::vector<i64>& a) {
    std::vector<i64> tmp(a.size());
    merge_sort_range(a, tmp, 0, a.size());
}

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    SplitMix64 rng{kSeed};
    const std::vector<i64> input = random_values(rng, static_cast<std::size_t>(args.n), u64{1} << 30);
    std::vector<i64> work(input.size());
    std::size_t shift = 0;
    const double ms = measure(args.min_ms, [&] { rotated_copy(input, work, shift); }, [&] {
        merge_sort(work);
        escape(work.data());
    });
    report(hash_sequence(work), ms);
}
