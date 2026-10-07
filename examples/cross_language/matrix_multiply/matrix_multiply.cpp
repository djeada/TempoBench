// Matrix multiplication: O(n³).  The naive product of two n×n matrices.
//
// The same algorithm, on the same input, as matrix_multiply.rs and matrix_multiply.py: all three
// print the same CHECKSUM.  Only the algorithm is timed.
//
//   g++ -O2 -std=c++17 -o matrix_multiply matrix_multiply.cpp && ./matrix_multiply --n 1000

#include "../common/harness.hpp"

using namespace harness;

// c = a × b for n×n matrices stored row by row, in i-k-j order so the
// innermost loop walks memory contiguously.
void matrix_multiply(const std::vector<i64>& a, const std::vector<i64>& b, std::vector<i64>& c, std::size_t n) {
    std::fill(c.begin(), c.end(), 0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < n; ++k) {
            const i64 aik = a[i * n + k];
            for (std::size_t j = 0; j < n; ++j) c[i * n + j] += aik * b[k * n + j];
        }
}

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    const auto n = static_cast<std::size_t>(args.n);
    SplitMix64 rng{kSeed};
    const std::vector<i64> a = random_values(rng, n * n, 10);
    const std::vector<i64> b = random_values(rng, n * n, 10);
    std::vector<i64> c(n * n);
    const double ms = measure(args.min_ms, nothing, [&] {
        escape(a.data());
        escape(b.data());
        matrix_multiply(a, b, c, n);
        escape(c.data());
    });
    report(hash_sequence(c), ms);
}
