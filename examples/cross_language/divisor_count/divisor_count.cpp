// Counting divisors: O(√n).  Count the divisors of n by trial division up to √n.
//
// The same algorithm, on the same input, as divisor_count.rs and divisor_count.py: all three
// print the same CHECKSUM.  Only the algorithm is timed.
//
//   g++ -O2 -std=c++17 -o divisor_count divisor_count.cpp && ./divisor_count --n 1000

#include "../common/harness.hpp"

using namespace harness;

// The number of divisors of n: they come in pairs (d, n/d) with d < √n.
i64 divisor_count(i64 n) {
    i64 count = 0;
    i64 d = 1;
    for (; d * d < n; ++d) {
        if (n % d == 0) count += 2;
    }
    if (d * d == n) ++count;
    return count;
}

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    i64 count = 0;
    escape(&count);
    const double ms = measure(args.min_ms, nothing, [&] { count = divisor_count(opaque(args.n)); });
    report(count, ms);
}
