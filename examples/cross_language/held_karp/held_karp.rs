//! Travelling salesman (Held–Karp): O(n² 2ⁿ).  The shortest tour through n cities, by dynamic programming.
//!
//! The same algorithm, on the same input, as held_karp.cpp and held_karp.py: all
//! three print the same CHECKSUM.  Only the algorithm is timed.
//!
//!   rustc -C opt-level=3 held_karp.rs && ./held_karp --n 1000

#[path = "../common/harness.rs"]
mod harness;

use harness::*;
use std::hint::black_box;

const MAX_CITIES: i64 = 24; // the table has n·2ⁿ entries

/// The shortest tour from city 0 through every city and back.  dp[S][j] is
/// the cheapest path that starts at 0, visits exactly the cities in S, and
/// ends at j; each entry is extended by one more city at a time.
fn held_karp(dist: &[i64], n: usize) -> i64 {
    if n == 1 {
        return 0;
    }
    let inf = i64::MAX / 4;
    let subsets = 1usize << n;
    let mut dp = vec![inf; subsets * n];
    dp[n] = 0; // subset {0}, ending at city 0
    for mask in (1..subsets).step_by(2) {
        for last in 0..n {
            let cost = dp[mask * n + last];
            if cost >= inf || (mask >> last) & 1 == 0 {
                continue;
            }
            for next in 0..n {
                if (mask >> next) & 1 == 1 {
                    continue;
                }
                let to = (mask | (1 << next)) * n + next;
                let candidate = cost + dist[last * n + next];
                if candidate < dp[to] {
                    dp[to] = candidate;
                }
            }
        }
    }
    (1..n)
        .map(|last| dp[(subsets - 1) * n + last] + dist[last * n])
        .min()
        .unwrap_or(inf)
}

/// Copy the distance matrix with the cities shuffled.  The shortest tour is
/// the same whatever the cities are called, but solving the very same matrix
/// over and over lets the branch predictor learn it.
fn relabel_cities(dist: &[i64], n: usize, rng: &mut SplitMix64) -> Vec<i64> {
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        perm.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let mut out = vec![0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = dist[perm[i] * n + perm[j]];
        }
    }
    out
}

fn main() {
    let args = parse_args();
    if args.n > MAX_CITIES {
        usage("--n must be at most 24");
    }
    let n = args.n as usize;
    let mut rng = SplitMix64 { state: SEED };
    let dist: Vec<i64> = random_values(&mut rng, n * n, 100)
        .into_iter()
        .map(|d| d + 1)
        .collect();
    let (ms, tour) = measure(
        args.min_ms,
        || relabel_cities(&dist, n, &mut rng),
        |relabelled| held_karp(&relabelled, black_box(n)),
    );
    report(tour, ms);
}
