//! Merge sort: O(n log n).  Top-down merge sort of n random values.
//!
//! The same algorithm, on the same input, as merge_sort.cpp and merge_sort.py: all
//! three print the same CHECKSUM.  Only the algorithm is timed.
//!
//!   rustc -C opt-level=3 merge_sort.rs && ./merge_sort --n 1000

#[path = "../common/harness.rs"]
mod harness;

use harness::*;

/// Sort a[lo..hi]: sort each half, then merge the halves through `tmp`.
fn merge_sort_range(a: &mut [i64], tmp: &mut [i64], lo: usize, hi: usize) {
    if hi - lo < 2 {
        return;
    }
    let mid = lo + (hi - lo) / 2;
    merge_sort_range(a, tmp, lo, mid);
    merge_sort_range(a, tmp, mid, hi);
    let (mut i, mut j, mut k) = (lo, mid, lo);
    while i < mid && j < hi {
        if a[j] < a[i] {
            tmp[k] = a[j];
            j += 1;
        } else {
            tmp[k] = a[i];
            i += 1;
        }
        k += 1;
    }
    while i < mid {
        tmp[k] = a[i];
        i += 1;
        k += 1;
    }
    while j < hi {
        tmp[k] = a[j];
        j += 1;
        k += 1;
    }
    a[lo..hi].copy_from_slice(&tmp[lo..hi]);
}

fn merge_sort(a: &mut [i64]) {
    let mut tmp = vec![0; a.len()];
    let len = a.len();
    merge_sort_range(a, &mut tmp, 0, len);
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
            merge_sort(&mut work);
            work
        },
    );
    report(hash_sequence(&sorted), ms);
}
