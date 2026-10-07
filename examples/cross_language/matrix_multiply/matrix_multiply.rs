//! Matrix multiplication: O(n³).  The naive product of two n×n matrices.
//!
//! The same algorithm, on the same input, as matrix_multiply.cpp and matrix_multiply.py: all
//! three print the same CHECKSUM.  Only the algorithm is timed.
//!
//!   rustc -C opt-level=3 matrix_multiply.rs && ./matrix_multiply --n 1000

#[path = "../common/harness.rs"]
mod harness;

use harness::*;
use std::hint::black_box;

/// c = a × b for n×n matrices stored row by row, in i-k-j order so the
/// innermost loop walks memory contiguously.
fn matrix_multiply(a: &[i64], b: &[i64], c: &mut [i64], n: usize) {
    c.fill(0);
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
}

fn main() {
    let args = parse_args();
    let n = args.n as usize;
    let mut rng = SplitMix64 { state: SEED };
    let a = random_values(&mut rng, n * n, 10);
    let b = random_values(&mut rng, n * n, 10);
    let mut c = vec![0; n * n];
    let (ms, ()) = measure(
        args.min_ms,
        || (),
        |()| matrix_multiply(black_box(&a), black_box(&b), &mut c, black_box(n)),
    );
    report(hash_sequence(&c), ms);
}
