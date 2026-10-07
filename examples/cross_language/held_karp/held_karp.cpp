// Travelling salesman (Held–Karp): O(n² 2ⁿ).  The shortest tour through n cities, by dynamic programming.
//
// The same algorithm, on the same input, as held_karp.rs and held_karp.py: all three
// print the same CHECKSUM.  Only the algorithm is timed.
//
//   g++ -O2 -std=c++17 -o held_karp held_karp.cpp && ./held_karp --n 1000

#include "../common/harness.hpp"

using namespace harness;

constexpr long long kMaxCities = 24;  // the table has n·2ⁿ entries

// The shortest tour from city 0 through every city and back.  dp[S][j] is
// the cheapest path that starts at 0, visits exactly the cities in S, and
// ends at j; each entry is extended by one more city at a time.
i64 held_karp(const std::vector<i64>& dist, std::size_t n) {
    if (n == 1) return 0;
    const i64 inf = INT64_MAX / 4;
    const std::size_t subsets = std::size_t{1} << n;
    std::vector<i64> dp(subsets * n, inf);
    dp[1 * n + 0] = 0;
    for (std::size_t mask = 1; mask < subsets; mask += 2) {
        for (std::size_t last = 0; last < n; ++last) {
            const i64 cost = dp[mask * n + last];
            if (cost >= inf || !(mask >> last & 1)) continue;
            for (std::size_t next = 0; next < n; ++next) {
                if (mask >> next & 1) continue;
                const std::size_t to = (mask | (std::size_t{1} << next)) * n + next;
                const i64 candidate = cost + dist[last * n + next];
                if (candidate < dp[to]) dp[to] = candidate;
            }
        }
    }
    i64 best = inf;
    for (std::size_t last = 1; last < n; ++last)
        best = std::min(best, dp[(subsets - 1) * n + last] + dist[last * n]);
    return best;
}

// Copy the distance matrix with the cities shuffled.  The shortest tour is the
// same whatever the cities are called, but solving the very same matrix over
// and over lets the branch predictor learn it.
void relabel_cities(const std::vector<i64>& dist, std::vector<i64>& out, std::size_t n, SplitMix64& rng) {
    std::vector<std::size_t> perm(n);
    for (std::size_t i = 0; i < n; ++i) perm[i] = i;
    for (std::size_t i = n - 1; i > 0; --i) std::swap(perm[i], perm[static_cast<std::size_t>(rng.below(i + 1))]);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) out[i * n + j] = dist[perm[i] * n + perm[j]];
}

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    if (args.n > kMaxCities) usage("--n must be at most 24");
    const auto n = static_cast<std::size_t>(args.n);
    SplitMix64 rng{kSeed};
    std::vector<i64> dist = random_values(rng, n * n, 100);
    for (auto& d : dist) d += 1;
    std::vector<i64> relabelled(dist.size());

    i64 tour = 0;
    escape(&tour);
    const double ms = measure(args.min_ms, [&] { relabel_cities(dist, relabelled, n, rng); }, [&] {
        escape(relabelled.data());
        tour = held_karp(relabelled, opaque(n));
    });
    report(tour, ms);
}
