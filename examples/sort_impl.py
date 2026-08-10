import argparse
import random
import time


def make_data(n: int, impl: str):
    if impl == "random":
        return [random.random() for _ in range(n)]
    if impl == "sorted":
        return list(range(n))
    if impl == "reversed":
        return list(range(n, 0, -1))
    raise ValueError(f"unknown impl {impl}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--impl", type=str, required=True)
    args = p.parse_args()
    data = make_data(args.n, args.impl)

    started = time.perf_counter()
    data.sort()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # print small digest to avoid optimization removal
    print(data[0], data[-1])
    # TempoBench reads this marker and measures the sort rather than the process.
    print(f"TEMPOBENCH_MS: {elapsed_ms:.6f}")


if __name__ == "__main__":
    main()
