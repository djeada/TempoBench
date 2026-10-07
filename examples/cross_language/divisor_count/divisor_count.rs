//! Counting divisors: O(√n).  Count the divisors of n by trial division up to √n.
//!
//! The same algorithm, on the same input, as divisor_count.cpp and divisor_count.py: all
//! three print the same CHECKSUM.  Only the algorithm is timed.
//!
//!   rustc -C opt-level=3 divisor_count.rs && ./divisor_count --n 1000

#[path = "../common/harness.rs"]
mod harness;

use harness::*;
use std::hint::black_box;

/// The number of divisors of n: they come in pairs (d, n/d) with d < √n.
fn divisor_count(n: i64) -> i64 {
    let mut count = 0;
    let mut d = 1;
    while d * d < n {
        if n % d == 0 {
            count += 2;
        }
        d += 1;
    }
    if d * d == n {
        count += 1;
    }
    count
}

fn main() {
    let args = parse_args();
    let (ms, count) = measure(args.min_ms, || (), |()| divisor_count(black_box(args.n)));
    report(count, ms);
}
