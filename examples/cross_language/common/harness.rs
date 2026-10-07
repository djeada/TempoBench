//! What every Rust implementation in examples/cross_language shares: the input
//! generator, the timing loop and the output.  harness.hpp and harness.py are
//! the same code in C++ and Python, so all three languages do the same work.
#![allow(dead_code)] // each program uses only the parts it needs

use std::hint::black_box;
use std::process::exit;
use std::time::Instant;

pub const SEED: u64 = 42;
const HASH_MOD: u64 = 1_000_000_007;

/// The input generator: SplitMix64, identical in every language.
pub struct SplitMix64 {
    pub state: u64,
}

impl SplitMix64 {
    pub fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, bound: u64) -> i64 {
        ((self.next() >> 11) % bound) as i64
    }
}

pub fn hash_sequence(values: &[i64]) -> i64 {
    values
        .iter()
        .fold(0u64, |h, &v| (h * 31 + v as u64) % HASH_MOD) as i64
}

pub fn random_values(rng: &mut SplitMix64, n: usize, bound: u64) -> Vec<i64> {
    (0..n).map(|_| rng.below(bound)).collect()
}

/// Copy `input` rotated by a step that changes every call.  Sorting the very
/// same array over and over lets the branch predictor learn its comparisons,
/// which makes small inputs unrealistically fast; a rotation gives every call
/// a different arrangement while the sorted result stays the same.
pub fn rotated_copy(input: &[i64], shift: &mut usize) -> Vec<i64> {
    *shift = (*shift + 7919) % input.len();
    [&input[*shift..], &input[..*shift]].concat()
}

/// Repeat `kernel` until at least `min_ms` of it has been timed; return the
/// mean duration of one call and the last call's result.  `prepare` runs
/// untimed before each call and hands the kernel its input.  `black_box`
/// hides the input and result from the optimiser, so every call is executed.
pub fn measure<I, R>(
    min_ms: f64,
    mut prepare: impl FnMut() -> I,
    mut kernel: impl FnMut(I) -> R,
) -> (f64, R) {
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

pub struct Args {
    pub n: i64,
    pub min_ms: f64,
}

pub fn usage(message: &str) -> ! {
    eprintln!("error: {message}\nusage: PROGRAM --n N [--min-ms MS]");
    exit(2);
}

pub fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut args = Args { n: 0, min_ms: 20.0 };
    for pair in argv.chunks(2) {
        let [option, value] = pair else {
            usage("missing value for an option")
        };
        match option.as_str() {
            "--n" => {
                args.n = value
                    .parse()
                    .unwrap_or_else(|_| usage("--n must be an integer"))
            }
            "--min-ms" => {
                args.min_ms = value
                    .parse()
                    .unwrap_or_else(|_| usage("--min-ms must be a number"))
            }
            _ => usage("unknown option"),
        }
    }
    if args.n < 1 {
        usage("a positive --n is required");
    }
    args
}

/// The two lines TempoBench reads: what was computed, and how long it took.
pub fn report(checksum: i64, ms: f64) {
    println!("CHECKSUM: {checksum}");
    println!("TEMPOBENCH_MS: {ms:.6}");
}
