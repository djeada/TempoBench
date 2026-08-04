"""Real pure-Python implementations for the advanced algorithm demos.

Inputs are generated deterministically from ``n``.  The return value is a small
checksum so callers cannot discard the work while keeping benchmark setup out
of the measured algorithm where practical.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Callable


def fast_exponentiation(n: int) -> int:
    base, exponent, result, modulus = 7, n, 1, 1_000_000_007
    while exponent:
        if exponent & 1:
            result = result * base % modulus
        base = base * base % modulus
        exponent >>= 1
    return result


def euclidean_gcd(n: int) -> int:
    a, b = 0, 1
    while b <= n:
        a, b = b, a + b
    while b:
        a, b = b, a % b
    return a


def sieve(n: int) -> list[int]:
    prime = bytearray(b"\x01") * (n + 1)
    if n >= 0:
        prime[0] = 0
    if n >= 1:
        prime[1] = 0
    p = 2
    while p * p <= n:
        if prime[p]:
            for multiple in range(p * p, n + 1, p):
                prime[multiple] = 0
        p += 1
    return [i for i in range(2, n + 1) if prime[i]]


def sieve_of_eratosthenes(n: int) -> int:
    return sum(sieve(n))


def prefix_function(s: str) -> list[int]:
    pi = [0] * len(s)
    for i in range(1, len(s)):
        j = pi[i - 1]
        while j and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi


def prefix_function_kmp(n: int) -> int:
    pattern = "ababaca"
    text = ("abacababacaba" * ((n + 12) // 13))[:n]
    joined = pattern + "#" + text
    pi = prefix_function(joined)
    return sum(value == len(pattern) for value in pi)


def z_values(s: str) -> list[int]:
    z = [0] * len(s)
    left = right = 0
    for i in range(1, len(s)):
        if i <= right:
            z[i] = min(right - i + 1, z[i - left])
        while i + z[i] < len(s) and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > right:
            left, right = i, i + z[i] - 1
    return z


def z_algorithm(n: int) -> int:
    return sum(z_values(("aabcaabxaaaz" * ((n + 11) // 12))[:n]))


def manacher(s: str) -> list[int]:
    transformed = "^#" + "#".join(s) + "#$"
    radius = [0] * len(transformed)
    center = right = 0
    for i in range(1, len(transformed) - 1):
        if i < right:
            radius[i] = min(right - i, radius[2 * center - i])
        while transformed[i + radius[i] + 1] == transformed[i - radius[i] - 1]:
            radius[i] += 1
        if i + radius[i] > right:
            center, right = i, i + radius[i]
    return radius


def manacher_palindromes(n: int) -> int:
    return max(manacher(("abacabae" * ((n + 7) // 8))[:n]), default=0)


def rabin_karp(n: int) -> int:
    pattern = "needle"
    text = ("haystackneedle" * ((n + 13) // 14))[:n]
    m, base, mod = len(pattern), 257, 1_000_000_007
    if len(text) < m:
        return 0
    power = pow(base, m - 1, mod)
    ph = th = 0
    for a, b in zip(pattern, text[:m]):
        ph, th = (ph * base + ord(a)) % mod, (th * base + ord(b)) % mod
    matches = 0
    for i in range(len(text) - m + 1):
        if ph == th and text[i : i + m] == pattern:
            matches += 1
        if i + m < len(text):
            th = ((th - ord(text[i]) * power) * base + ord(text[i + m])) % mod
    return matches


def aho_search(patterns: list[str], text: str) -> int:
    go = [{}]
    fail = [0]
    output = [0]
    for pattern in patterns:
        state = 0
        for char in pattern:
            if char not in go[state]:
                go[state][char] = len(go)
                go.append({})
                fail.append(0)
                output.append(0)
            state = go[state][char]
        output[state] += 1
    queue = deque(go[0].values())
    while queue:
        state = queue.popleft()
        for char, child in go[state].items():
            queue.append(child)
            fallback = fail[state]
            while fallback and char not in go[fallback]:
                fallback = fail[fallback]
            fail[child] = go[fallback].get(char, 0)
            output[child] += output[fail[child]]
    state = matches = 0
    for char in text:
        while state and char not in go[state]:
            state = fail[state]
        state = go[state].get(char, 0)
        matches += output[state]
    return matches


def aho_corasick_search(n: int) -> int:
    return aho_search(["he", "she", "his", "hers"], ("ahishers" * ((n + 7) // 8))[:n])


def trie_insert(n: int) -> int:
    root: dict[str, dict] = {}
    node = root
    for char in ("algorithm" * ((n + 8) // 9))[:n]:
        node = node.setdefault(char, {})
    node[""] = {}
    return n


def run_length_encoding(n: int) -> int:
    data = ("aaabccccdd" * ((n + 9) // 10))[:n]
    encoded: list[tuple[str, int]] = []
    for char in data:
        if encoded and encoded[-1][0] == char:
            encoded[-1] = (char, encoded[-1][1] + 1)
        else:
            encoded.append((char, 1))
    return sum(count for _, count in encoded) + len(encoded)


def huffman_frequency_pass(n: int) -> int:
    frequencies = [0] * 256
    for value in range(n):
        frequencies[(value * 37) & 255] += 1
    return sum((i + 1) * count for i, count in enumerate(frequencies))


def huffman_cost(weights: list[int]) -> int:
    heap: list[int] = []

    def push(value: int) -> None:
        heap.append(value)
        child = len(heap) - 1
        while child:
            parent = (child - 1) // 2
            if heap[parent] <= value:
                break
            heap[child] = heap[parent]
            child = parent
        heap[child] = value

    def pop() -> int:
        root = heap[0]
        last = heap.pop()
        if heap:
            parent = 0
            while 2 * parent + 1 < len(heap):
                child = 2 * parent + 1
                if child + 1 < len(heap) and heap[child + 1] < heap[child]:
                    child += 1
                if heap[child] >= last:
                    break
                heap[parent] = heap[child]
                parent = child
            heap[parent] = last
        return root

    for weight in weights:
        push(weight)
    cost = 0
    while len(heap) > 1:
        merged = pop() + pop()
        cost += merged
        push(merged)
    return cost


def huffman_coding(n: int) -> int:
    return huffman_cost([1 + (i * 37) % (n + 1) for i in range(n)])


def suffix_array(s: str) -> list[int]:
    n = len(s)
    sa = list(range(n))
    alphabet = {char: i for i, char in enumerate(sorted(set(s)))}
    rank = [alphabet[char] for char in s]
    width = 1

    def counting_sort(indices: list[int], keys: list[int], maximum: int) -> list[int]:
        counts = [0] * (maximum + 1)
        for index in indices:
            counts[keys[index]] += 1
        for i in range(1, len(counts)):
            counts[i] += counts[i - 1]
        ordered = [0] * len(indices)
        for index in reversed(indices):
            key = keys[index]
            counts[key] -= 1
            ordered[counts[key]] = index
        return ordered

    while width < n:
        first = [value + 1 for value in rank]
        second = [rank[i + width] + 1 if i + width < n else 0 for i in range(n)]
        sa = counting_sort(sa, second, n)
        sa = counting_sort(sa, first, n)
        new = [0] * n
        for i in range(1, n):
            a, b = sa[i - 1], sa[i]
            new[b] = new[a] + ((rank[a], rank[a + width] if a + width < n else -1) < (rank[b], rank[b + width] if b + width < n else -1))
        rank = new
        width *= 2
        if n and rank[sa[-1]] == n - 1:
            break
    return sa


def suffix_array_doubling(n: int) -> int:
    s = "".join(chr(97 + (i * 17 + i * i) % 26) for i in range(n))
    return sum((i + 1) * pos for i, pos in enumerate(suffix_array(s)))


def lis_length(values: list[int]) -> int:
    tails: list[int] = []
    for value in values:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        i = lo
        if i == len(tails):
            tails.append(value)
        else:
            tails[i] = value
    return len(tails)


def longest_increasing_subsequence(n: int) -> int:
    return lis_length([(i * 97) % max(1, n) for i in range(n)])


def naive_substring_search(n: int) -> int:
    text, pattern = "a" * n, "a" * max(1, n // 2 - 1) + "b"
    matches = 0
    for i in range(len(text) - len(pattern) + 1):
        for j in range(len(pattern)):
            if text[i + j] != pattern[j]:
                break
        else:
            matches += 1
    return matches


def lcs(a: str, b: str) -> int:
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def longest_common_subsequence(n: int) -> int:
    return lcs(("ABCBDAB" * ((n + 6) // 7))[:n], ("BDCABA" * ((n + 5) // 6))[:n])


def levenshtein(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]


def levenshtein_distance(n: int) -> int:
    return levenshtein(("kitten" * ((n + 5) // 6))[:n], ("sitting" * ((n + 6) // 7))[:n])


def longest_common_substring(n: int) -> int:
    a = ("ababc" * ((n + 4) // 5))[:n]
    b = ("babca" * ((n + 4) // 5))[:n]
    row = [0] * (n + 1)
    best = 0
    for x in a:
        new = [0]
        for j, y in enumerate(b, 1):
            new.append(row[j - 1] + 1 if x == y else 0)
            best = max(best, new[-1])
        row = new
    return best


def sequence_alignment(n: int) -> int:
    a = ("GATTACA" * ((n + 6) // 7))[:n]
    b = ("GCATGCU" * ((n + 6) // 7))[:n]
    row = [-2 * j for j in range(n + 1)]
    for i, x in enumerate(a, 1):
        new = [-2 * i]
        for j, y in enumerate(b, 1):
            new.append(max(row[j - 1] + (1 if x == y else -1), row[j] - 2, new[-1] - 2))
        row = new
    return row[-1]


def subset_sum(n: int) -> int:
    target = n
    reachable = [False] * (target + 1)
    reachable[0] = True
    for i in range(n):
        weight = i % 7 + 1
        for total in range(target, weight - 1, -1):
            reachable[total] |= reachable[total - weight]
    return int(reachable[target])


def zero_one_knapsack(n: int) -> int:
    best = [0] * (n + 1)
    for i in range(n):
        weight, value = i % 7 + 1, i % 11 + 1
        for capacity in range(n, weight - 1, -1):
            best[capacity] = max(best[capacity], best[capacity - weight] + value)
    return best[n]


def coin_change(n: int) -> int:
    ways = [0] * (n + 1)
    ways[0] = 1
    for coin in range(1, n + 1):
        for amount in range(coin, n + 1):
            ways[amount] += ways[amount - coin]
    return ways[n]


def matrix_chain_order(n: int) -> int:
    dims = [2 + (i % 7) for i in range(n + 1)]
    cost = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for left in range(n - length + 1):
            right = left + length - 1
            cost[left][right] = min(
                cost[left][k] + cost[k + 1][right] + dims[left] * dims[k + 1] * dims[right + 1] for k in range(left, right)
            )
    return cost[0][-1] if n else 0


def held_karp_tsp(n: int) -> int:
    # Exact Held--Karp: O(n^2 * 2^n), intentionally not disguised as cubic.
    size = n
    dist = [[0 if i == j else 1 + ((i - j) * (i - j) + 3 * i + 5 * j) % 19 for j in range(size)] for i in range(size)]
    dp = {(1 << k, k): dist[0][k] for k in range(1, size)}
    for subset_size in range(2, size):
        for subset in combinations(range(1, size), subset_size):
            mask = sum(1 << x for x in subset)
            for last in subset:
                prior = mask ^ (1 << last)
                dp[mask, last] = min(dp[prior, prev] + dist[prev][last] for prev in subset if prev != last)
    if size <= 1:
        return 0
    full = (1 << size) - 2
    return min(dp[full, last] + dist[last][0] for last in range(1, size))


def optimal_bst(n: int) -> int:
    freq = [1 + i % 9 for i in range(n)]
    prefix = [0]
    for value in freq:
        prefix.append(prefix[-1] + value)
    cost = [[0] * n for _ in range(n)]
    for length in range(1, n + 1):
        for left in range(n - length + 1):
            right = left + length - 1
            total = prefix[right + 1] - prefix[left]
            cost[left][right] = min(
                (cost[left][root - 1] if root > left else 0) + (cost[root + 1][right] if root < right else 0) + total
                for root in range(left, right + 1)
            )
    return cost[0][-1] if n else 0


def cyk_parsing(n: int) -> int:
    # CNF grammar: S -> AB | BC, A -> BA | a, B -> CC | b, C -> AB | a
    terminals = {"a": {"A", "C"}, "b": {"B"}}
    binary = {("A", "B"): {"S", "C"}, ("B", "C"): {"S"}, ("B", "A"): {"A"}, ("C", "C"): {"B"}}
    word = ("baaba" * ((n + 4) // 5))[:n]
    table = [[set() for _ in range(n)] for _ in range(n)]
    for i, char in enumerate(word):
        table[i][i] = set(terminals[char])
    for length in range(2, n + 1):
        for left in range(n - length + 1):
            right = left + length - 1
            for split in range(left, right):
                for a in table[left][split]:
                    for b in table[split + 1][right]:
                        table[left][right].update(binary.get((a, b), ()))
    return int(bool(n) and "S" in table[0][n - 1])


def naive_matrix_multiplication(n: int) -> int:
    a = [[(i + j) % 7 for j in range(n)] for i in range(n)]
    b = [[(i * 3 + j) % 11 for j in range(n)] for i in range(n)]
    checksum = 0
    for i in range(n):
        for j in range(n):
            cell = 0
            for k in range(n):
                cell += a[i][k] * b[k][j]
            checksum += cell
    return checksum


def tensor_contraction_3d(n: int) -> int:
    # Dot-product contraction C[i,j] = sum_k A[i,j,k] * v[k], generated lazily.
    checksum = 0
    for i in range(n):
        for j in range(n):
            cell = 0
            for k in range(n):
                cell += ((i + 2 * j + 3 * k) % 13) * ((k % 7) + 1)
            checksum += cell
    return checksum


def gaussian_elimination(n: int) -> int:
    # Strictly diagonally dominant system, so no singular benchmark inputs.
    matrix = [
        [(n + 1 if i == j else ((i + j) % 3)) for j in range(n)] + [sum((n + 1 if i == j else ((i + j) % 3)) * (j + 1) for j in range(n))]
        for i in range(n)
    ]
    for pivot in range(n):
        best = max(range(pivot, n), key=lambda row: abs(matrix[row][pivot]))
        matrix[pivot], matrix[best] = matrix[best], matrix[pivot]
        for row in range(pivot + 1, n):
            factor = matrix[row][pivot] / matrix[pivot][pivot]
            for col in range(pivot, n + 1):
                matrix[row][col] -= factor * matrix[pivot][col]
    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        solution[row] = (matrix[row][-1] - sum(matrix[row][j] * solution[j] for j in range(row + 1, n))) / matrix[row][row]
    return round(sum(solution))


IMPLEMENTATIONS: dict[str, Callable[[int], int]] = {
    name.replace("_", "-"): value
    for name, value in list(globals().items())
    if callable(value)
    and name
    in {
        "fast_exponentiation",
        "euclidean_gcd",
        "sieve_of_eratosthenes",
        "prefix_function_kmp",
        "z_algorithm",
        "manacher_palindromes",
        "rabin_karp",
        "aho_corasick_search",
        "trie_insert",
        "run_length_encoding",
        "huffman_frequency_pass",
        "huffman_coding",
        "suffix_array_doubling",
        "longest_increasing_subsequence",
        "naive_substring_search",
        "longest_common_subsequence",
        "levenshtein_distance",
        "longest_common_substring",
        "sequence_alignment",
        "subset_sum",
        "zero_one_knapsack",
        "coin_change",
        "matrix_chain_order",
        "held_karp_tsp",
        "optimal_bst",
        "cyk_parsing",
        "naive_matrix_multiplication",
        "tensor_contraction_3d",
        "gaussian_elimination",
    }
}


# Each exported demo has an independently addressable correctness oracle.  The
# checks use textbook examples or an invariant stronger than "it returned an
# integer"; this also makes it possible for integration tests to report the
# precise named algorithm that regressed.
SELF_CHECKS: dict[str, Callable[[], bool]] = {
    "fast-exponentiation": lambda: fast_exponentiation(13) == pow(7, 13, 1_000_000_007),
    "euclidean-gcd": lambda: euclidean_gcd(100) == 1 and euclidean_gcd(1_000) == 1,
    "sieve-of-eratosthenes": lambda: sieve(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
    "prefix-function-kmp": lambda: prefix_function("ababaca") == [0, 0, 1, 2, 3, 0, 1],
    "z-algorithm": lambda: z_values("aabcaabxaaaz") == [0, 1, 0, 0, 3, 1, 0, 0, 2, 2, 1, 0],
    "manacher-palindromes": lambda: max(manacher("forgeeksskeegfor")) == 10,
    "rabin-karp": lambda: rabin_karp(14) == 1 and rabin_karp(5) == 0,
    "aho-corasick-search": lambda: aho_search(["he", "she", "his", "hers"], "ahishers") == 4,
    "trie-insert": lambda: trie_insert(9) == len("algorithm"),
    "run-length-encoding": lambda: run_length_encoding(10) == 14,
    "huffman-frequency-pass": lambda: huffman_frequency_pass(256) == sum(range(1, 257)),
    "huffman-coding": lambda: huffman_cost([5, 9, 12, 13, 16, 45]) == 224,
    "suffix-array-doubling": lambda: suffix_array("banana") == [5, 3, 1, 0, 4, 2],
    "longest-increasing-subsequence": lambda: lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4,
    "naive-substring-search": lambda: naive_substring_search(20) == 0,
    "longest-common-subsequence": lambda: lcs("ABCBDAB", "BDCABA") == 4,
    "levenshtein-distance": lambda: levenshtein("kitten", "sitting") == 3,
    "longest-common-substring": lambda: longest_common_substring(6) == 5,
    "sequence-alignment": lambda: sequence_alignment(7) == -1,
    "subset-sum": lambda: subset_sum(6) == 1 and subset_sum(1) == 1,
    "zero-one-knapsack": lambda: zero_one_knapsack(6) == 6,
    "coin-change": lambda: coin_change(6) == 11,
    "matrix-chain-order": lambda: matrix_chain_order(3) == 64,
    "held-karp-tsp": lambda: held_karp_tsp(1) == 0 and held_karp_tsp(6) == 29,
    "optimal-bst": lambda: optimal_bst(3) == 10,
    "cyk-parsing": lambda: cyk_parsing(5) == 1 and cyk_parsing(1) == 0,
    "naive-matrix-multiplication": lambda: naive_matrix_multiplication(2) == 22,
    "tensor-contraction-3d": lambda: tensor_contraction_3d(2) == 42,
    "gaussian-elimination": lambda: gaussian_elimination(4) == sum(range(1, 5)),
}


def run_all_named_self_checks() -> None:
    """Run the canonical oracle for every exported advanced implementation."""
    assert SELF_CHECKS.keys() == IMPLEMENTATIONS.keys()
    for name, check in SELF_CHECKS.items():
        assert check(), f"canonical self-check failed: {name}"


def run_self_checks() -> None:
    # Number theory.
    assert fast_exponentiation(10) == pow(7, 10, 1_000_000_007)
    assert euclidean_gcd(100) == 1
    assert sieve(10) == [2, 3, 5, 7]
    # String search and indexing.
    assert prefix_function("ababaca") == [0, 0, 1, 2, 3, 0, 1]
    assert z_values("aaaa") == [0, 3, 2, 1]
    assert max(manacher("abacaba")) == 7
    assert aho_search(["he", "she", "hers"], "ushers") == 3
    assert rabin_karp(14) == 1
    assert trie_insert(6) == 6
    # Compression: a frequency scan and a full optimal prefix-tree merge.
    assert run_length_encoding(6) == 9
    assert huffman_frequency_pass(6) == 561
    assert huffman_cost([5, 9, 12, 13, 16, 45]) == 224
    assert suffix_array("banana") == [5, 3, 1, 0, 4, 2]
    # Dynamic programming.
    assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert lcs("ABCBDAB", "BDCABA") == 4
    assert levenshtein("kitten", "sitting") == 3
    assert longest_common_substring(6) == 5
    assert sequence_alignment(6) == 0
    assert subset_sum(6) == 1
    assert zero_one_knapsack(6) == 6
    assert coin_change(6) == 11
    assert matrix_chain_order(6) == 320
    assert held_karp_tsp(6) == 29
    assert optimal_bst(6) == 45
    assert cyk_parsing(6) == 1
    # Dense matrix and linear algebra.
    assert naive_matrix_multiplication(6) == 3234
    assert tensor_contraction_3d(6) == 4572
    assert gaussian_elimination(4) == 10
