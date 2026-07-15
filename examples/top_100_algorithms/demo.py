"""Run one of the Top 100 algorithm demonstrations."""

from __future__ import annotations

import argparse

from .catalog import ALGORITHMS
from .probes import count_python_steps, median_runtime_ns


def run_demo(algorithm: str) -> None:
    """CLI used by each standalone module in ``demos``."""
    parser = argparse.ArgumentParser(description=f"Pure-Python {algorithm} demo")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--mode", choices=("steps", "time", "result"), default="result")
    args = parser.parse_args()
    spec = ALGORITHMS[algorithm]
    if args.n < 1:
        parser.error("--n must be positive")
    if args.mode == "steps":
        result, value = count_python_steps(spec.run, args.n)
        print(value, result)
    elif args.mode == "time":
        print(median_runtime_ns(spec.run, args.n))
    else:
        print(spec.run(args.n))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=sorted(ALGORITHMS))
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--mode", choices=("steps", "time", "result"), default="result")
    args = parser.parse_args()
    spec = ALGORITHMS[args.algorithm]
    if args.n < 1:
        parser.error("--n must be positive")
    if args.mode == "steps":
        result, value = count_python_steps(spec.run, args.n)
        print(value, result)
    elif args.mode == "time":
        print(median_runtime_ns(spec.run, args.n))
    else:
        print(spec.run(args.n))


if __name__ == "__main__":
    main()
