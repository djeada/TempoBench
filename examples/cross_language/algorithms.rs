//! Seven textbook algorithms, one per complexity class TempoBench fits.
//!
//! A line-for-line port of algorithms.cpp: the same algorithms on the same
//! SplitMix64 inputs, so all three languages print the same CHECKSUM for a
//! given --algo and --n.  Only the hot section is timed.
//!
//!   rustc -C opt-level=3 -o algorithms algorithms.rs
//!   ./algorithms --algo merge_sort --n 100000

use std::hint::black_box;
use std::process::exit;
use std::time::Instant;

const SEED: u64 = 42;
const HASH_MOD: u64 = 1_000_000_007;
const QUERIES: usize = 1 << 16;
const MAX_CITIES: i64 = 24; // Held-Karp needs n * 2^n table entries

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> i64 {
        ((self.next() >> 11) % bound) as i64
    }
}

fn hash_sequence(values: &[i64]) -> i64 {
    values
        .iter()
        .fold(0u64, |h, &v| (h * 31 + v as u64) % HASH_MOD) as i64
}

fn random_values(rng: &mut SplitMix64, n: usize, bound: u64) -> Vec<i64> {
    (0..n).map(|_| rng.below(bound)).collect()
}

/// Copy `input` rotated by a step that changes every call.  Sorting the very
/// same array over and over lets the branch predictor learn its comparisons,
/// which makes small inputs unrealistically fast; a rotation gives every call
/// different subarrays to merge while the sorted result stays the same.
fn rotated_copy(input: &[i64], shift: &mut usize) -> Vec<i64> {
    *shift = (*shift + 7919) % input.len();
    [&input[*shift..], &input[..*shift]].concat()
}

// --- O(log n): binary search, a fixed batch of lookups ----------------------

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

// --- O(sqrt n): count divisors by trial division -----------------------------

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

// --- O(n): Kadane's maximum subarray sum ------------------------------------

fn max_subarray(values: &[i64]) -> i64 {
    let (mut best, mut current) = (values[0], values[0]);
    for &v in &values[1..] {
        current = v.max(current + v);
        best = best.max(current);
    }
    best
}

// --- O(n log n): top-down merge sort ----------------------------------------

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

// --- O(n^2): insertion sort --------------------------------------------------

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

// --- O(n^3): naive matrix multiplication -------------------------------------

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

// --- O(n^2 2^n): Held-Karp travelling salesman -------------------------------

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
/// over and over lets the branch predictor learn it, as for the sorts above.
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

// --- Harness -----------------------------------------------------------------

/// Repeat `kernel` until at least `min_ms` of it has been timed; return the
/// mean duration of one call and the last call's result.  `prepare` runs
/// untimed before each call and hands the kernel its input, e.g. a fresh copy
/// of the array a sort works on in place.
fn measure<I, R>(min_ms: f64, mut prepare: impl FnMut() -> I, mut kernel: impl FnMut(I) -> R) -> (f64, R) {
    let mut total = 0.0;
    let mut calls = 0u32;
    loop {
        let input = black_box(prepare());
        let start = Instant::now();
        let result = black_box(kernel(input));
        total += start.elapsed().as_secs_f64() * 1000.0;
        calls += 1;
        if total >= min_ms {
            return (total / f64::from(calls), result);
        }
    }
}

fn usage(message: &str) -> ! {
    eprintln!("error: {message}\nusage: algorithms --algo NAME --n N [--min-ms MS]");
    exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut algo, mut n, mut min_ms) = (String::new(), 0i64, 20.0f64);
    for pair in args.chunks(2) {
        let [option, value] = pair else {
            usage("missing value for an option")
        };
        match option.as_str() {
            "--algo" => algo = value.clone(),
            "--n" => n = value.parse().unwrap_or_else(|_| usage("--n must be an integer")),
            "--min-ms" => min_ms = value.parse().unwrap_or_else(|_| usage("--min-ms must be a number")),
            _ => usage("unknown option"),
        }
    }
    if algo.is_empty() || n < 1 {
        usage("--algo and a positive --n are required");
    }
    if algo == "held_karp" && n > MAX_CITIES {
        usage("held_karp needs --n <= 24");
    }

    let mut rng = SplitMix64 { state: SEED };
    let size = n as usize;
    let nothing = || ();

    // black_box hides each kernel's inputs and result from the optimiser, so
    // every call is executed.  Sequence checksums are hashed afterwards,
    // outside the timed section.
    let (ms, checksum) = match algo.as_str() {
        "binary_search" => {
            let sorted: Vec<i64> = (0..n).map(|i| 2 * i).collect();
            let queries = random_values(&mut rng, QUERIES, 2 * n as u64);
            measure(min_ms, nothing, |()| binary_search_count(black_box(&sorted), black_box(&queries)))
        }
        "divisor_count" => measure(min_ms, nothing, |()| divisor_count(black_box(n))),
        "max_subarray" => {
            let values: Vec<i64> = random_values(&mut rng, size, 2001).into_iter().map(|v| v - 1000).collect();
            measure(min_ms, nothing, |()| max_subarray(black_box(&values)))
        }
        "merge_sort" | "insertion_sort" => {
            let input = random_values(&mut rng, size, 1 << 30);
            let sort = if algo == "merge_sort" { merge_sort } else { insertion_sort };
            let mut shift = 0;
            let (ms, sorted) = measure(min_ms, || rotated_copy(&input, &mut shift), |mut work| {
                sort(&mut work);
                work
            });
            (ms, hash_sequence(&sorted))
        }
        "matrix_multiply" => {
            let a = random_values(&mut rng, size * size, 10);
            let b = random_values(&mut rng, size * size, 10);
            let mut c = vec![0; size * size];
            let (ms, ()) = measure(min_ms, nothing, |()| matrix_multiply(black_box(&a), black_box(&b), &mut c, black_box(size)));
            (ms, hash_sequence(&c))
        }
        "held_karp" => {
            let dist: Vec<i64> = random_values(&mut rng, size * size, 100).into_iter().map(|d| d + 1).collect();
            measure(min_ms, || relabel_cities(&dist, size, &mut rng), |relabelled| held_karp(&relabelled, black_box(size)))
        }
        _ => usage("unknown --algo"),
    };

    println!("CHECKSUM: {checksum}");
    println!("TEMPOBENCH_MS: {ms:.6}");
}
