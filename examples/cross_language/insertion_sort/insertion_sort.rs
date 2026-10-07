//! Insertion sort: O(n²).  Insertion sort of n random values.
//!
//! The same algorithm, on the same input, as insertion_sort.cpp and insertion_sort.py: all
//! three print the same CHECKSUM.  Only the algorithm is timed.
//!
//!   rustc -C opt-level=3 insertion_sort.rs && ./insertion_sort --n 1000

#[path = "../common/harness.rs"]
mod harness;

use harness::*;

/// Grow a sorted prefix one element at a time, shifting larger ones right.
fn insertion_sort(a: &mut [i64]) {
    for i in 1..a.len() {
        let key = a[i];
        let mut j = i;
        while j > 0 && a[j - 1] > key {
            a[j] = a[j - 1];
            j -= 1;
        }
        a[j] = key;
    }
}

fn main() {
    let args = parse_args();
    let mut rng = SplitMix64 { state: SEED };
    let input = random_values(&mut rng, args.n as usize, 1 << 30);
    let mut shift = 0;
    let (ms, sorted) = measure(
        args.min_ms,
        || rotated_copy(&input, &mut shift),
        |mut work| {
            insertion_sort(&mut work);
            work
        },
    );
    report(hash_sequence(&sorted), ms);
}
