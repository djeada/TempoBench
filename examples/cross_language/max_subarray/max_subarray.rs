//! Maximum subarray (Kadane): O(n).  The largest sum of a contiguous run of n values.
//!
//! The same algorithm, on the same input, as max_subarray.cpp and max_subarray.py: all
//! three print the same CHECKSUM.  Only the algorithm is timed.
//!
//!   rustc -C opt-level=3 max_subarray.rs && ./max_subarray --n 1000

#[path = "../common/harness.rs"]
mod harness;

use harness::*;
use std::hint::black_box;

/// Kadane's algorithm: the best sum ending here either extends the best sum
/// ending one step earlier or starts afresh.
fn max_subarray(values: &[i64]) -> i64 {
    let (mut best, mut current) = (values[0], values[0]);
    for &v in &values[1..] {
        current = v.max(current + v);
        best = best.max(current);
    }
    best
}

fn main() {
    let args = parse_args();
    let mut rng = SplitMix64 { state: SEED };
    let values: Vec<i64> = random_values(&mut rng, args.n as usize, 2001)
        .into_iter()
        .map(|v| v - 1000)
        .collect();
    let (ms, best) = measure(args.min_ms, || (), |()| max_subarray(black_box(&values)));
    report(best, ms);
}
