//! Binary search: O(log n).  65,536 lookups in a sorted array of n elements.
//!
//! The same algorithm, on the same input, as binary_search.cpp and binary_search.py: all
//! three print the same CHECKSUM.  Only the algorithm is timed.
//!
//!   rustc -C opt-level=3 binary_search.rs && ./binary_search --n 1000

#[path = "../common/harness.rs"]
mod harness;

use harness::*;
use std::hint::black_box;

const QUERIES: usize = 1 << 16;

/// How many of `queries` are in `sorted`, each found by halving the range.
fn binary_search_count(sorted: &[i64], queries: &[i64]) -> i64 {
    let mut found = 0;
    for &q in queries {
        let (mut lo, mut hi) = (0usize, sorted.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if sorted[mid] < q {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo < sorted.len() && sorted[lo] == q {
            found += 1;
        }
    }
    found
}

fn main() {
    let args = parse_args();
    let mut rng = SplitMix64 { state: SEED };
    let sorted: Vec<i64> = (0..args.n).map(|i| 2 * i).collect();
    let queries = random_values(&mut rng, QUERIES, 2 * args.n as u64);
    let (ms, found) = measure(
        args.min_ms,
        || (),
        |()| binary_search_count(black_box(&sorted), black_box(&queries)),
    );
    report(found, ms);
}
