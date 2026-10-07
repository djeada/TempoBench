// Insertion sort: O(n²).  Insertion sort of n random values.
//
// The same algorithm, on the same input, as insertion_sort.rs and insertion_sort.py: all three
// print the same CHECKSUM.  Only the algorithm is timed.
//
//   g++ -O2 -std=c++17 -o insertion_sort insertion_sort.cpp && ./insertion_sort --n 1000

#include "../common/harness.hpp"

using namespace harness;

// Grow a sorted prefix one element at a time, shifting larger ones right.
void insertion_sort(std::vector<i64>& a) {
    for (std::size_t i = 1; i < a.size(); ++i) {
        i64 key = a[i];
        std::size_t j = i;
        while (j > 0 && a[j - 1] > key) {
            a[j] = a[j - 1];
            --j;
        }
        a[j] = key;
    }
}

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    SplitMix64 rng{kSeed};
    const std::vector<i64> input = random_values(rng, static_cast<std::size_t>(args.n), u64{1} << 30);
    std::vector<i64> work(input.size());
    std::size_t shift = 0;
    const double ms = measure(args.min_ms, [&] { rotated_copy(input, work, shift); }, [&] {
        insertion_sort(work);
        escape(work.data());
    });
    report(hash_sequence(work), ms);
}
