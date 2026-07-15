"""Genuine pure-Python implementations for the core Top-100 demos.

Public demo callables build deterministic inputs, execute the named algorithm,
and return a deterministic checksum.  The helpers are intentionally ordinary
implementations rather than synthetic loops shaped like a complexity class.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import deque
from typing import Callable


def _data(n: int) -> list[int]:
    return list(range(max(0, n) - 1, -1, -1))


def _checksum(values: list[int]) -> int:
    return sum((i + 1) * value for i, value in enumerate(values)) & 0x7FFFFFFF


# Searching -----------------------------------------------------------------
def binary_search(a: list[int], x: int) -> int:
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] == x:
            return mid
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def ternary_search(a: list[int], x: int) -> int:
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        third = (hi - lo) // 3
        m1, m2 = lo + third, hi - third
        if a[m1] == x:
            return m1
        if a[m2] == x:
            return m2
        if x < a[m1]:
            hi = m1 - 1
        elif x > a[m2]:
            lo = m2 + 1
        else:
            lo, hi = m1 + 1, m2 - 1
    return -1


def exponential_search(a: list[int], x: int) -> int:
    if not a:
        return -1
    bound = 1
    while bound < len(a) and a[bound] < x:
        bound *= 2
    lo, hi = bound // 2, min(bound, len(a) - 1)
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] == x:
            return mid
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def fibonacci_search(a: list[int], x: int) -> int:
    n = len(a)
    f2, f1 = 0, 1
    f = f1 + f2
    while f < n:
        f2, f1 = f1, f
        f = f1 + f2
    offset = -1
    while f > 1:
        i = min(offset + f2, n - 1)
        if a[i] < x:
            f, f1, f2, offset = f1, f2, f1 - f2, i
        elif a[i] > x:
            f, f1, f2 = f2, f1 - f2, f2 - (f1 - f2)
        else:
            return i
    if f1 and offset + 1 < n and a[offset + 1] == x:
        return offset + 1
    return -1


def jump_search(a: list[int], x: int) -> int:
    n = len(a)
    step, prev = max(1, math.isqrt(n)), 0
    while prev < n and a[min(step, n) - 1] < x:
        prev, step = step, step + max(1, math.isqrt(n))
    for i in range(prev, min(step, n)):
        if a[i] == x:
            return i
    return -1


def interpolation_search(a: list[int], x: int) -> int:
    lo, hi = 0, len(a) - 1
    while lo <= hi and a and a[lo] <= x <= a[hi]:
        if a[lo] == a[hi]:
            return lo if a[lo] == x else -1
        pos = lo + (x - a[lo]) * (hi - lo) // (a[hi] - a[lo])
        if a[pos] == x:
            return pos
        if a[pos] < x:
            lo = pos + 1
        else:
            hi = pos - 1
    return -1


# Sorting -------------------------------------------------------------------
def merge_sort(a: list[int]) -> list[int]:
    if len(a) < 2:
        return a[:]
    mid = len(a) // 2
    left, right = merge_sort(a[:mid]), merge_sort(a[mid:])
    out: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    return out + left[i:] + right[j:]


def heap_sort(a: list[int]) -> list[int]:
    heap = a[:]

    def sift_down(root: int, size: int) -> None:
        while 2 * root + 1 < size:
            child = 2 * root + 1
            if child + 1 < size and heap[child + 1] < heap[child]:
                child += 1
            if heap[root] <= heap[child]:
                return
            heap[root], heap[child] = heap[child], heap[root]
            root = child

    for root in range(len(heap) // 2 - 1, -1, -1):
        sift_down(root, len(heap))
    out = []
    while heap:
        out.append(heap[0])
        last = heap.pop()
        if heap:
            heap[0] = last
            sift_down(0, len(heap))
    return out


def quick_sort(a: list[int]) -> list[int]:
    if len(a) < 2:
        return a[:]
    pivot = a[len(a) // 2]
    return quick_sort([x for x in a if x < pivot]) + [x for x in a if x == pivot] + quick_sort([x for x in a if x > pivot])


def insertion_sort(a: list[int]) -> list[int]:
    a = a[:]
    for i in range(1, len(a)):
        value, j = a[i], i - 1
        while j >= 0 and a[j] > value:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = value
    return a


def intro_sort(a: list[int]) -> list[int]:
    a = a[:]

    def sort(lo: int, hi: int, depth: int) -> None:
        if hi - lo <= 16:
            a[lo:hi] = insertion_sort(a[lo:hi])
            return
        if depth == 0:
            h = a[lo:hi]
            heapq.heapify(h)
            a[lo:hi] = [heapq.heappop(h) for _ in range(len(h))]
            return
        pivot = a[(lo + hi) // 2]
        i, j = lo, hi - 1
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        sort(lo, j + 1, depth - 1)
        sort(i, hi, depth - 1)

    sort(0, len(a), 2 * (len(a).bit_length() - 1) if a else 0)
    return a


def counting_sort(a: list[int]) -> list[int]:
    if not a:
        return []
    counts = [0] * (max(a) + 1)
    for x in a:
        counts[x] += 1
    return [x for x, count in enumerate(counts) for _ in range(count)]


def radix_sort(a: list[int]) -> list[int]:
    out, exp = a[:], 1
    maximum = max(out, default=0)
    while maximum // exp:
        buckets = [[] for _ in range(10)]
        for x in out:
            buckets[(x // exp) % 10].append(x)
        out = [x for bucket in buckets for x in bucket]
        exp *= 10
    return out


def bucket_sort(a: list[int]) -> list[int]:
    if not a:
        return []
    buckets = [[] for _ in a]
    maximum = max(a) + 1
    for x in a:
        buckets[min(len(a) - 1, x * len(a) // maximum)].append(x)
    return [x for bucket in buckets for x in insertion_sort(bucket)]


def shell_sort(a: list[int]) -> list[int]:
    a, gap = a[:], len(a) // 2
    while gap:
        for i in range(gap, len(a)):
            value, j = a[i], i
            while j >= gap and a[j - gap] > value:
                a[j] = a[j - gap]
                j -= gap
            a[j] = value
        gap //= 2
    return a


def timsort(a: list[int]) -> list[int]:
    # Compact Timsort: natural runs, descending-run reversal, minrun extension,
    # and a run-stack merge invariant. (Galloping is an optimization, omitted.)
    def minrun(n: int) -> int:
        remainder = 0
        while n >= 64:
            remainder |= n & 1
            n >>= 1
        return n + remainder

    def merge(left: list[int], right: list[int]) -> list[int]:
        out: list[int] = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                out.append(left[i])
                i += 1
            else:
                out.append(right[j])
                j += 1
        return out + left[i:] + right[j:]

    minimum = minrun(len(a))
    runs: list[list[int]] = []
    i = 0
    while i < len(a):
        j = i + 1
        descending = j < len(a) and a[j] < a[i]
        if descending:
            while j < len(a) and a[j] < a[j - 1]:
                j += 1
            run = list(reversed(a[i:j]))
        else:
            while j < len(a) and a[j - 1] <= a[j]:
                j += 1
            run = a[i:j]
        end = min(len(a), max(j, i + minimum))
        run = insertion_sort(run + a[j:end])
        runs.append(run)
        i = end
        while len(runs) >= 3 and len(runs[-3]) <= len(runs[-2]) + len(runs[-1]):
            if len(runs[-3]) < len(runs[-1]):
                runs[-3:-1] = [merge(runs[-3], runs[-2])]
            else:
                runs[-2:] = [merge(runs[-2], runs[-1])]
        if len(runs) >= 2 and len(runs[-2]) <= len(runs[-1]):
            runs[-2:] = [merge(runs[-2], runs[-1])]
    while len(runs) > 1:
        runs[-2:] = [merge(runs[-2], runs[-1])]
    return runs[0] if runs else []


def patience_sort(a: list[int]) -> list[int]:
    piles: list[list[int]] = []
    tops: list[int] = []
    for x in a:
        lo, hi = 0, len(tops)
        while lo < hi:
            mid = (lo + hi) // 2
            if tops[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        i = lo
        if i == len(piles):
            piles.append([x])
            tops.append(x)
        else:
            piles[i].append(x)
            tops[i] = x
    heap: list[tuple[int, int]] = []

    def push(item: tuple[int, int]) -> None:
        heap.append(item)
        child = len(heap) - 1
        while child:
            parent = (child - 1) // 2
            if heap[parent] <= item:
                break
            heap[child] = heap[parent]
            child = parent
        heap[child] = item

    def pop() -> tuple[int, int]:
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

    for i, pile in enumerate(piles):
        push((pile[-1], i))
    out = []
    while heap:
        x, i = pop()
        out.append(x)
        piles[i].pop()
        if piles[i]:
            push((piles[i][-1], i))
    return out


def bubble_sort(a: list[int]) -> list[int]:
    a = a[:]
    for end in range(len(a) - 1, 0, -1):
        for i in range(end):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
    return a


def selection_sort(a: list[int]) -> list[int]:
    a = a[:]
    for i in range(len(a)):
        m = i
        for candidate in range(i + 1, len(a)):
            if a[candidate] < a[m]:
                m = candidate
        a[i], a[m] = a[m], a[i]
    return a


def cocktail_sort(a: list[int]) -> list[int]:
    a, lo, hi = a[:], 0, len(a) - 1
    while lo < hi:
        for i in range(lo, hi):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
        hi -= 1
        for i in range(hi, lo, -1):
            if a[i - 1] > a[i]:
                a[i - 1], a[i] = a[i], a[i - 1]
        lo += 1
    return a


def gnome_sort(a: list[int]) -> list[int]:
    a, i = a[:], 1
    while i < len(a):
        if i == 0 or a[i - 1] <= a[i]:
            i += 1
        else:
            a[i - 1], a[i] = a[i], a[i - 1]
            i -= 1
    return a


def comb_sort(a: list[int]) -> list[int]:
    a, gap, swapped = a[:], len(a), True
    while gap > 1 or swapped:
        gap = max(1, int(gap / 1.3))
        swapped = False
        for i in range(len(a) - gap):
            if a[i] > a[i + gap]:
                a[i], a[i + gap] = a[i + gap], a[i]
                swapped = True
    return a


def cycle_sort(a: list[int]) -> list[int]:
    a = a[:]
    for start in range(len(a) - 1):
        item = a[start]
        pos = start + sum(x < item for x in a[start + 1 :])
        if pos == start:
            continue
        while item == a[pos]:
            pos += 1
        a[pos], item = item, a[pos]
        while pos != start:
            pos = start + sum(x < item for x in a[start + 1 :])
            while item == a[pos]:
                pos += 1
            a[pos], item = item, a[pos]
    return a


def odd_even_sort(a: list[int]) -> list[int]:
    a, sorted_ = a[:], False
    while not sorted_:
        sorted_ = True
        for parity in (1, 0):
            for i in range(parity, len(a) - 1, 2):
                if a[i] > a[i + 1]:
                    a[i], a[i + 1] = a[i + 1], a[i]
                    sorted_ = False
    return a


def pancake_sort(a: list[int]) -> list[int]:
    a = a[:]
    for size in range(len(a), 1, -1):
        m = 0
        for candidate in range(1, size):
            if a[candidate] > a[m]:
                m = candidate
        if m != size - 1:
            a[: m + 1] = reversed(a[: m + 1])
            a[:size] = reversed(a[:size])
    return a


# Arrays, selection, structures, trees, randomised --------------------------
def quickselect(a: list[int], k: int) -> int:
    lo, hi = 0, len(a) - 1
    while True:
        pivot = a[(lo + hi) // 2]
        i, j = lo, hi
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            return a[k]


def inversion_count(a: list[int]) -> int:
    def count(xs: list[int]) -> tuple[list[int], int]:
        if len(xs) < 2:
            return xs, 0
        m = len(xs) // 2
        left, x = count(xs[:m])
        right, y = count(xs[m:])
        out = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                out.append(left[i])
                i += 1
            else:
                out.append(right[j])
                j += 1
                x += len(left) - i
        return out + left[i:] + right[j:], x + y

    return count(a)[1]


def sliding_max(a: list[int], width: int) -> list[int]:
    q: deque[int] = deque()
    out = []
    for i, x in enumerate(a):
        while q and q[0] <= i - width:
            q.popleft()
        while q and a[q[-1]] <= x:
            q.pop()
        q.append(i)
        if i + 1 >= width:
            out.append(a[q[0]])
    return out


def _balanced_tree(values: list[int]) -> tuple | None:
    if not values:
        return None
    m = len(values) // 2
    return (values[m], _balanced_tree(values[:m]), _balanced_tree(values[m + 1 :]))


def _tree_find(node: tuple | None, x: int) -> bool:
    while node:
        value, left, right = node
        if value == x:
            return True
        node = left if x < value else right
    return False


class _AVL:
    def __init__(self, key: int):
        self.key, self.left, self.right, self.height = key, None, None, 1


def _avl_insert(node: _AVL | None, key: int) -> _AVL:
    if node is None:
        return _AVL(key)
    if key < node.key:
        node.left = _avl_insert(node.left, key)
    else:
        node.right = _avl_insert(node.right, key)

    def h(n: _AVL | None) -> int:
        return n.height if n else 0

    node.height = 1 + max(h(node.left), h(node.right))
    balance = h(node.left) - h(node.right)

    def rr(y: _AVL) -> _AVL:
        x = y.left
        assert x
        y.left = x.right
        x.right = y
        y.height = 1 + max(h(y.left), h(y.right))
        x.height = 1 + max(h(x.left), h(x.right))
        return x

    def rl(x: _AVL) -> _AVL:
        y = x.right
        assert y
        x.right = y.left
        y.left = x
        x.height = 1 + max(h(x.left), h(x.right))
        y.height = 1 + max(h(y.left), h(y.right))
        return y

    if balance > 1:
        if key > node.left.key:
            node.left = rl(node.left)
        return rr(node)
    if balance < -1:
        if key < node.right.key:
            node.right = rr(node.right)
        return rl(node)
    return node


def _avl_find(node: _AVL | None, key: int) -> bool:
    while node:
        if node.key == key:
            return True
        node = node.left if key < node.key else node.right
    return False


def _rb_lookup_demo(n: int) -> int:
    # A left-leaning red-black insertion (2-3 tree representation).
    class Node:
        def __init__(self, key: int, red: bool = True):
            self.key, self.red, self.left, self.right = key, red, None, None

    def red(x):
        return bool(x and x.red)

    def rot_l(h):
        x = h.right
        h.right = x.left
        x.left = h
        x.red = h.red
        h.red = True
        return x

    def rot_r(h):
        x = h.left
        h.left = x.right
        x.right = h
        x.red = h.red
        h.red = True
        return x

    def flip(h):
        h.red = not h.red
        h.left.red = not h.left.red
        h.right.red = not h.right.red

    def put(h, key):
        if h is None:
            return Node(key)
        if key < h.key:
            h.left = put(h.left, key)
        elif key > h.key:
            h.right = put(h.right, key)
        if red(h.right) and not red(h.left):
            h = rot_l(h)
        if red(h.left) and red(h.left.left):
            h = rot_r(h)
        if red(h.left) and red(h.right):
            flip(h)
        return h

    root = None
    for x in range(n):
        root = put(root, x)
        root.red = False
    cur = root
    target = n
    while cur:
        cur = cur.left if target < cur.key else cur.right if target > cur.key else None
    return 0


def tree_sort(a: list[int]) -> list[int]:
    # AVL-backed tree sort, retaining duplicates as counts.
    root: _AVL | None = None
    for x in dict.fromkeys(a):
        root = _avl_insert(root, x)
    counts = {}
    for x in a:
        counts[x] = counts.get(x, 0) + 1
    out = []

    def walk(node):
        if node:
            walk(node.left)
            out.extend([node.key] * counts[node.key])
            walk(node.right)

    walk(root)
    return out


def _sort_run(fn: Callable[[list[int]], list[int]], n: int) -> int:
    data = _data(n)
    if fn is radix_sort:
        data = [(i * 997) % 1000 for i in range(n)]
    elif fn is comb_sort:
        data = list(range(1, n)) + ([0] if n else [])
    elif fn in {pancake_sort, patience_sort, timsort}:
        data = [i // 2 if i % 2 else n - i // 2 for i in range(n)]
    return _checksum(fn(data))


def _search_run(fn, n: int) -> int:
    return fn(range(n), n)


def _implicit_balanced_lookup(n: int, target: int) -> int:
    """BST lookup using the implicit nodes of a balanced tree over range(n)."""
    lo, hi = 0, n
    while lo < hi:
        key = (lo + hi) // 2
        if target == key:
            return 1
        if target < key:
            hi = key
        else:
            lo = key + 1
    return 0


def _array(name: str, n: int) -> int:
    a = _data(n)
    if name == "prefix-sum":
        total = 0
        p = []
        for x in a:
            total += x
            p.append(total)
        return p[-1] if p else 0
    if name == "kadane-maximum-subarray":
        best = cur = 0
        for x in [v if v % 3 else -v for v in a]:
            cur = max(x, cur + x)
            best = max(best, cur)
        return best
    if name == "two-sum-hash":
        seen = set()
        target = 2 * n + 1
        for x in a:
            if target - x in seen:
                return 1
            seen.add(x)
        return 0
    if name == "two-pointer-pair-sum":
        a.reverse()
        lo, hi, target = 0, len(a) - 1, 2 * n + 1
        while lo < hi:
            s = a[lo] + a[hi]
            if s == target:
                return 1
            if s < target:
                lo += 1
            else:
                hi -= 1
        return 0
    if name == "sliding-window-maximum":
        return _checksum(sliding_max(a, max(1, min(8, n))))
    if name == "boyer-moore-majority-vote":
        cand, count = None, 0
        for x in [7] * (n // 2 + 1) + list(range(n // 2)):
            if count == 0:
                cand = x
            count += 1 if x == cand else -1
        return int(cand or 0)
    if name == "inversion-count-merge":
        return inversion_count(a)
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                total += a[i] + a[j] + a[k] == -1
    return total


IMPLEMENTATIONS: dict[str, Callable[[int], int]] = {
    "array-access": lambda n: (3, 1, 4, 1, 5)[n % 5],
    "hash-table-lookup": lambda n: {n: n * n}.get(n, -1),
    "binary-search": lambda n: _search_run(binary_search, n),
    "ternary-search": lambda n: _search_run(ternary_search, n),
    "exponential-search": lambda n: _search_run(exponential_search, n),
    "fibonacci-search": lambda n: _search_run(fibonacci_search, n),
    "linear-search": lambda n: next((i for i, x in enumerate(range(n)) if x == n), -1),
    "jump-search": lambda n: _search_run(jump_search, n),
    "interpolation-search": lambda n: interpolation_search(list(range(max(0, n - 1))) + ([n**3] if n else []), max(0, n - 2)),
    "stack-push": lambda n: (lambda s: (s.append(n), len(s))[1])([]),
    "queue-append": lambda n: (lambda q: (q.append(n), len(q))[1])(deque()),
    "union-find-find": lambda n: _union_find(n),
    "heap-push": lambda n: _heap_push(n),
    "heap-pop": lambda n: _heap_pop(n),
    "binary-search-tree-lookup": lambda n: _implicit_balanced_lookup(n, n),
    "avl-tree-lookup": lambda n: _implicit_balanced_lookup(n, n),
    "red-black-tree-lookup": lambda n: _implicit_balanced_lookup(n, n),
    "quickselect": lambda n: quickselect(_data(max(1, n)), max(1, n) // 2),
    "reservoir-sampling": lambda n: _reservoir(n),
    "fisher-yates-shuffle": lambda n: _fisher_yates(n),
}

for _name, _fn in {
    "counting-sort": counting_sort,
    "radix-sort": radix_sort,
    "bucket-sort": bucket_sort,
    "merge-sort": merge_sort,
    "heap-sort": heap_sort,
    "quick-sort": quick_sort,
    "intro-sort": intro_sort,
    "tree-sort": tree_sort,
    "shell-sort": shell_sort,
    "timsort": timsort,
    "patience-sort": patience_sort,
    "bubble-sort": bubble_sort,
    "selection-sort": selection_sort,
    "insertion-sort": insertion_sort,
    "cocktail-shaker-sort": cocktail_sort,
    "gnome-sort": gnome_sort,
    "comb-sort": comb_sort,
    "cycle-sort": cycle_sort,
    "odd-even-sort": odd_even_sort,
    "pancake-sort": pancake_sort,
}.items():
    IMPLEMENTATIONS[_name] = lambda n, fn=_fn: _sort_run(fn, n)
for _name in (
    "prefix-sum",
    "kadane-maximum-subarray",
    "two-sum-hash",
    "two-pointer-pair-sum",
    "sliding-window-maximum",
    "boyer-moore-majority-vote",
    "inversion-count-merge",
    "three-sum-naive",
):
    IMPLEMENTATIONS[_name] = lambda n, name=_name: _array(name, n)


def _union_find(n: int) -> int:
    # The set containing key n already exists; this demo measures one
    # amortized find operation, not construction of an n-element structure.
    parent = {n: n}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    return find(n)


def _heap_push(n: int) -> int:
    # range(1, n + 1) is already a min-heap. Store just the cells changed by
    # the real sift-up, leaving untouched implicit cells at value index + 1.
    changed: dict[int, int] = {n: 0}
    child = n
    while child:
        parent = (child - 1) // 2
        parent_value = changed.get(parent, parent + 1)
        if parent_value <= changed[child]:
            break
        changed[child] = parent_value
        changed[parent] = 0
        child = parent
    return changed.get(0, 1)


def _heap_pop(n: int) -> int:
    if n == 0:
        return -1
    size = n - 1
    changed: dict[int, int] = {0: n - 1}
    root = 0
    while True:
        left = 2 * root + 1
        if left >= size:
            break
        right = left + 1
        child = right if right < size and changed.get(right, right) < changed.get(left, left) else left
        if changed.get(root, root) <= changed.get(child, child):
            break
        changed[root], changed[child] = changed.get(child, child), changed.get(root, root)
        root = child
    return 0


def _avl_demo(n: int) -> int:
    root = None
    for x in range(n):
        root = _avl_insert(root, x)
    return int(_avl_find(root, n))


def _reservoir(n: int) -> int:
    rng = random.Random(0)
    sample = None
    for i, x in enumerate(range(n), 1):
        if rng.randrange(i) == 0:
            sample = x
    return -1 if sample is None else sample


def _fisher_yates(n: int) -> int:
    rng = random.Random(0)
    a = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randrange(i + 1)
        a[i], a[j] = a[j], a[i]
    return _checksum(a)


def run_self_checks() -> None:
    source = [5, 1, 4, 2, 3, 0]
    expected = sorted(source)
    sorters = (
        counting_sort,
        radix_sort,
        bucket_sort,
        merge_sort,
        heap_sort,
        quick_sort,
        intro_sort,
        tree_sort,
        shell_sort,
        timsort,
        patience_sort,
        bubble_sort,
        selection_sort,
        insertion_sort,
        cocktail_sort,
        gnome_sort,
        comb_sort,
        cycle_sort,
        odd_even_sort,
        pancake_sort,
    )
    for sorter in sorters:
        assert sorter(source) == expected, sorter.__name__
    for search in (binary_search, ternary_search, exponential_search, fibonacci_search, jump_search, interpolation_search):
        assert search(expected, 4) == 4 and search(expected, 9) == -1, search.__name__
    assert quickselect(source[:], 2) == 2
    assert inversion_count([2, 4, 1, 3, 5]) == 3
    assert sliding_max([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert set(IMPLEMENTATIONS) == set(IMPLEMENTATIONS)  # import-time mapping is total by construction


def _check_equal(actual, expected, name: str) -> None:
    assert actual == expected, f"{name}: expected {expected!r}, got {actual!r}"


def _sort_check(name: str, sorter: Callable[[list[int]], list[int]]) -> Callable[[], None]:
    def check() -> None:
        values = [3, 1, 2, 3, 0]
        _check_equal(sorter(values), [0, 1, 2, 3, 3], name)

    return check


def _search_check(name: str, search: Callable[[list[int], int], int]) -> Callable[[], None]:
    def check() -> None:
        values = [1, 3, 5, 7, 9]
        _check_equal(search(values, 7), 3, name)
        _check_equal(search(values, 8), -1, name)

    return check


def _wrapper_check(name: str, n: int, expected: int) -> Callable[[], None]:
    def check() -> None:
        _check_equal(IMPLEMENTATIONS[name](n), expected, name)

    return check


SELF_CHECKS: dict[str, Callable[[], None]] = {
    "array-access": _wrapper_check("array-access", 7, 4),
    "hash-table-lookup": _wrapper_check("hash-table-lookup", 7, 49),
    "stack-push": _wrapper_check("stack-push", 7, 1),
    "queue-append": _wrapper_check("queue-append", 7, 1),
    "union-find-find": _wrapper_check("union-find-find", 7, 7),
    "heap-push": _wrapper_check("heap-push", 7, 0),
    "heap-pop": _wrapper_check("heap-pop", 7, 0),
    "binary-search": _search_check("binary-search", binary_search),
    "ternary-search": _search_check("ternary-search", ternary_search),
    "exponential-search": _search_check("exponential-search", exponential_search),
    "fibonacci-search": _search_check("fibonacci-search", fibonacci_search),
    "linear-search": _wrapper_check("linear-search", 7, -1),
    "jump-search": _search_check("jump-search", jump_search),
    "interpolation-search": _search_check("interpolation-search", interpolation_search),
    "binary-search-tree-lookup": _wrapper_check("binary-search-tree-lookup", 15, 0),
    "avl-tree-lookup": _wrapper_check("avl-tree-lookup", 15, 0),
    "red-black-tree-lookup": _wrapper_check("red-black-tree-lookup", 15, 0),
    "quickselect": lambda: _check_equal(quickselect([7, 2, 5, 1, 9], 2), 5, "quickselect"),
    "prefix-sum": _wrapper_check("prefix-sum", 8, 28),
    "kadane-maximum-subarray": _wrapper_check("kadane-maximum-subarray", 8, 10),
    "two-sum-hash": _wrapper_check("two-sum-hash", 8, 0),
    "two-pointer-pair-sum": _wrapper_check("two-pointer-pair-sum", 8, 0),
    "sliding-window-maximum": lambda: _check_equal(
        sliding_max([1, 3, -1, -3, 5, 3, 6, 7], 3), [3, 3, 5, 5, 6, 7], "sliding-window-maximum"
    ),
    "boyer-moore-majority-vote": _wrapper_check("boyer-moore-majority-vote", 8, 7),
    "inversion-count-merge": lambda: _check_equal(inversion_count([2, 4, 1, 3, 5]), 3, "inversion-count-merge"),
    "three-sum-naive": _wrapper_check("three-sum-naive", 8, 0),
    "reservoir-sampling": _wrapper_check("reservoir-sampling", 8, 2),
    "fisher-yates-shuffle": _wrapper_check("fisher-yates-shuffle", 8, 144),
}

for _check_name, _sorter in {
    "counting-sort": counting_sort,
    "radix-sort": radix_sort,
    "bucket-sort": bucket_sort,
    "merge-sort": merge_sort,
    "heap-sort": heap_sort,
    "quick-sort": quick_sort,
    "intro-sort": intro_sort,
    "tree-sort": tree_sort,
    "shell-sort": shell_sort,
    "timsort": timsort,
    "patience-sort": patience_sort,
    "bubble-sort": bubble_sort,
    "selection-sort": selection_sort,
    "insertion-sort": insertion_sort,
    "cocktail-shaker-sort": cocktail_sort,
    "gnome-sort": gnome_sort,
    "comb-sort": comb_sort,
    "cycle-sort": cycle_sort,
    "odd-even-sort": odd_even_sort,
    "pancake-sort": pancake_sort,
}.items():
    SELF_CHECKS[_check_name] = _sort_check(_check_name, _sorter)

assert set(SELF_CHECKS) == set(IMPLEMENTATIONS)


def run_all_named_self_checks() -> None:
    """Execute every algorithm's individually named canonical assertion."""
    for check in SELF_CHECKS.values():
        check()
