import argparse
import random


def make_data(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    # Keep value range smaller than n so duplicates are common.
    max_value = max(2, n // 2)
    return [rng.randrange(max_value) for _ in range(n)]


def unique_count_quadratic(data: list[int]) -> int:
    uniques: list[int] = []
    for value in data:
        seen = False
        for current in uniques:
            if current == value:
                seen = True
                break
        if not seen:
            uniques.append(value)
    return len(uniques)


def unique_count_sort_scan(data: list[int]) -> int:
    if not data:
        return 0
    sorted_data = sorted(data)
    count = 1
    prev = sorted_data[0]
    for value in sorted_data[1:]:
        if value != prev:
            count += 1
            prev = value
    return count


def unique_count_hash_set(data: list[int]) -> int:
    return len(set(data))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument(
        "--impl",
        type=str,
        required=True,
        choices=["quadratic", "sort_scan", "hash_set"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = make_data(args.n, args.seed)

    implementations = {
        "quadratic": unique_count_quadratic,
        "sort_scan": unique_count_sort_scan,
        "hash_set": unique_count_hash_set,
    }
    result = implementations[args.impl](data)

    # Print result so all implementations are forced to run fully.
    print(result)


if __name__ == "__main__":
    main()
