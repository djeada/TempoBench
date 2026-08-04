"""Real graph and computational-geometry implementations for the demo catalog.

Each public wrapper constructs deterministic input of the size described by the
catalog, runs the named algorithm, and reduces its genuine result to an integer.
"""

# The compact algorithm listings intentionally keep closely related primitive
# operations together, making the examples easier to compare side by side.
# ruff: noqa: E501, E701, E702

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Callable


class _Priority:
    __slots__ = ("value", "tie")

    def __init__(self, value: int, tie: int):
        self.value = value
        self.tie = tie

    def __lt__(self, other: _Priority) -> bool:
        return (self.value, self.tie) < (other.value, other.tie)


def _merge_sort(items: list, key: Callable) -> list:
    """Stable straight-Python merge sort."""
    if len(items) < 2:
        return items[:]
    mid = len(items) // 2
    left = _merge_sort(items[:mid], key)
    right = _merge_sort(items[mid:], key)
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def _undirected(n: int) -> list[list[int]]:
    """Connected sparse graph: a star plus deterministic leaf chords."""
    g = [[] for _ in range(n)]
    for u in range(1, n):
        g[u].append(0)
        g[0].append(u)
    for u in range(1, max(1, n - 1), 3):
        v = u + 1
        g[u].append(v)
        g[v].append(u)
    return g


def _dag(n: int) -> list[list[int]]:
    return [list(range(1, n))] + [[] for _ in range(n - 1)] if n else []


def _bfs(g: list[list[int]], start: int = 0) -> list[int]:
    if not g:
        return []
    seen = [False] * len(g)
    seen[start] = True
    order = []
    q = deque([start])
    while q:
        u = q.popleft()
        order.append(u)
        for v in g[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return order


def _dfs(g: list[list[int]], start: int = 0) -> list[int]:
    if not g:
        return []
    seen = [False] * len(g)
    order = []
    stack = [start]
    while stack:
        u = stack.pop()
        if seen[u]:
            continue
        seen[u] = True
        order.append(u)
        stack.extend(reversed(g[u]))
    return order


def _topological_dfs(g: list[list[int]]) -> list[int]:
    state = [0] * len(g)
    out = []

    def visit(u: int) -> None:
        if state[u] == 1:
            raise ValueError("cycle")
        if state[u] == 2:
            return
        state[u] = 1
        for v in g[u]:
            visit(v)
        state[u] = 2
        out.append(u)

    for u in range(len(g)):
        visit(u)
    return out[::-1]


def _kahn(g: list[list[int]]) -> list[int]:
    indegree = [0] * len(g)
    for row in g:
        for v in row:
            indegree[v] += 1
    q = deque(i for i, d in enumerate(indegree) if d == 0)
    out = []
    while q:
        u = q.popleft()
        out.append(u)
        for v in g[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)
    if len(out) != len(g):
        raise ValueError("cycle")
    return out


def _components(g: list[list[int]]) -> list[list[int]]:
    seen = set()
    result = []
    for root in range(len(g)):
        if root in seen:
            continue
        part = []
        stack = [root]
        seen.add(root)
        while stack:
            u = stack.pop()
            part.append(u)
            for v in g[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        result.append(part)
    return result


def _is_bipartite(g: list[list[int]]) -> bool:
    color = [-1] * len(g)
    for root in range(len(g)):
        if color[root] >= 0:
            continue
        color[root] = 0
        q = deque([root])
        while q:
            u = q.popleft()
            for v in g[u]:
                if color[v] < 0:
                    color[v] = 1 - color[u]
                    q.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def _tarjan(g: list[list[int]]) -> list[list[int]]:
    index = 0
    indices = [-1] * len(g)
    low = [0] * len(g)
    stack = []
    on_stack = [False] * len(g)
    result = []

    def visit(u: int) -> None:
        nonlocal index
        indices[u] = low[u] = index
        index += 1
        stack.append(u)
        on_stack[u] = True
        for v in g[u]:
            if indices[v] < 0:
                visit(v)
                low[u] = min(low[u], low[v])
            elif on_stack[v]:
                low[u] = min(low[u], indices[v])
        if low[u] == indices[u]:
            part = []
            while True:
                v = stack.pop()
                on_stack[v] = False
                part.append(v)
                if v == u:
                    break
            result.append(part)

    for u in range(len(g)):
        if indices[u] < 0:
            visit(u)
    return result


def _kosaraju(g: list[list[int]]) -> list[list[int]]:
    seen = set()
    order = []

    def first(u: int) -> None:
        seen.add(u)
        for v in g[u]:
            if v not in seen:
                first(v)
        order.append(u)

    for u in range(len(g)):
        if u not in seen:
            first(u)
    rev = [[] for _ in g]
    for u, row in enumerate(g):
        for v in row:
            rev[v].append(u)
    seen.clear()
    result = []
    for root in reversed(order):
        if root in seen:
            continue
        part = []
        stack = [root]
        seen.add(root)
        while stack:
            u = stack.pop()
            part.append(u)
            for v in rev[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        result.append(part)
    return result


def _bridges_and_articulations(g: list[list[int]]) -> tuple[list[tuple[int, int]], set[int]]:
    timer = 0
    tin = [-1] * len(g)
    low = [0] * len(g)
    bridges = []
    cuts = set()

    def visit(u: int, parent: int) -> None:
        nonlocal timer
        tin[u] = low[u] = timer
        timer += 1
        children = 0
        for v in g[u]:
            if v == parent:
                continue
            if tin[v] >= 0:
                low[u] = min(low[u], tin[v])
            else:
                visit(v, u)
                low[u] = min(low[u], low[v])
                children += 1
                if low[v] > tin[u]:
                    bridges.append((min(u, v), max(u, v)))
                if parent >= 0 and low[v] >= tin[u]:
                    cuts.add(u)
        if parent < 0 and children > 1:
            cuts.add(u)

    for u in range(len(g)):
        if tin[u] < 0:
            visit(u, -1)
    return sorted(bridges), cuts


def _flood_fill(grid: list[list[int]], start: tuple[int, int], color: int) -> int:
    if not grid:
        return 0
    old = grid[start[0]][start[1]]
    if old == color:
        return 0
    q = deque([start])
    grid[start[0]][start[1]] = color
    changed = 0
    while q:
        r, c = q.popleft()
        changed += 1
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < len(grid) and 0 <= cc < len(grid[rr]) and grid[rr][cc] == old:
                grid[rr][cc] = color
                q.append((rr, cc))
    return changed


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.p[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        return True


def _weighted_sparse(n: int) -> list[tuple[int, int, int]]:
    # A star makes heap frontiers grow with V; path edges provide alternatives.
    edges = [(0, i, n - i + 10) for i in range(1, n)]
    edges += [(i - 1, i, 1 + i % 7) for i in range(1, n)]
    return edges


def _kruskal(n: int, edges: list[tuple[int, int, int]]) -> int:
    dsu = _DSU(n)
    total = 0
    count = 0
    for u, v, w in _merge_sort(edges, key=lambda e: e[2]):
        if dsu.union(u, v):
            total += w
            count += 1
    if n and count != n - 1:
        raise ValueError("disconnected")
    return total


def _weighted_adj(n: int, edges: list[tuple[int, int, int]]) -> list[list[tuple[int, int]]]:
    g = [[] for _ in range(n)]
    for u, v, w in edges:
        g[u].append((v, w))
        g[v].append((u, w))
    return g


def _prim_heap(g: list[list[tuple[int, int]]]) -> int:
    if not g:
        return 0
    seen = [False] * len(g)
    heap = [(_Priority(0, 0), 0)]
    total = 0
    count = 0
    while heap:
        priority, u = heapq.heappop(heap)
        w = priority.value
        if seen[u]:
            continue
        seen[u] = True
        total += w
        count += 1
        for v, cost in g[u]:
            if not seen[v]:
                heapq.heappush(heap, (_Priority(cost, v), v))
    if count != len(g):
        raise ValueError("disconnected")
    return total


def _dijkstra(g: list[list[tuple[int, int]]], source: int = 0) -> list[float]:
    dist = [math.inf] * len(g)
    if not g:
        return dist
    dist[source] = 0
    heap = [(_Priority(0, source), source)]
    while heap:
        priority, u = heapq.heappop(heap)
        d = priority.value
        if d != dist[u]:
            continue
        for v, w in g[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (_Priority(nd, v), v))
    return dist


def _a_star(width: int, height: int) -> int:
    if width * height == 0:
        return 0
    goal = (height - 1, width - 1)
    best = {(0, 0): 0}
    heap = [(_Priority(sum(goal), 0), 0, (0, 0))]
    while heap:
        _, cost, (r, c) = heapq.heappop(heap)
        if cost != best[(r, c)]:
            continue
        if (r, c) == goal:
            return cost
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < height and 0 <= cc < width:
                new = cost + 1
                if new < best.get((rr, cc), math.inf):
                    best[(rr, cc)] = new
                    h = goal[0] - rr + goal[1] - cc
                    tie = rr * width + cc
                    heapq.heappush(heap, (_Priority(new + h, tie), new, (rr, cc)))
    raise ValueError("no path")


def _matrix(n: int) -> list[list[float]]:
    a = [[math.inf] * n for _ in range(n)]
    for i in range(n):
        a[i][i] = 0
        for j in range(i):
            a[i][j] = a[j][i] = 1 + ((i * 17 + j * 13) % 19)
    return a


def _prim_matrix(a: list[list[float]]) -> int:
    n = len(a)
    key = [math.inf] * n
    used = [False] * n
    if n:
        key[0] = 0
    total = 0
    for _ in range(n):
        u = min((i for i in range(n) if not used[i]), key=lambda i: key[i])
        used[u] = True
        total += int(key[u])
        for v in range(n):
            if not used[v] and a[u][v] < key[v]:
                key[v] = a[u][v]
    return total


def _dijkstra_matrix(a: list[list[float]]) -> list[float]:
    n = len(a)
    dist = [math.inf] * n
    used = [False] * n
    if n:
        dist[0] = 0
    for _ in range(n):
        u = min((i for i in range(n) if not used[i]), key=lambda i: dist[i])
        used[u] = True
        for v in range(n):
            if not used[v] and dist[u] + a[u][v] < dist[v]:
                dist[v] = dist[u] + a[u][v]
    return dist


def _bellman_ford(n: int, edges: list[tuple[int, int, int]]) -> list[float]:
    dist = [math.inf] * n
    if n:
        dist[0] = 0
    directed = edges + [(v, u, w) for u, v, w in edges]
    for _ in range(max(0, n - 1)):
        for u, v, w in directed:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist


def _floyd(a: list[list[float]]) -> list[list[float]]:
    d = [row[:] for row in a]
    n = len(d)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return d


def _closure(g: list[list[int]]) -> list[list[bool]]:
    n = len(g)
    reach = [[i == j for j in range(n)] for i in range(n)]
    for u, row in enumerate(g):
        for v in row:
            reach[u][v] = True
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])
    return reach


Point = tuple[int, int]


def _cross(o: Point, a: Point, b: Point) -> int:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _points(n: int) -> list[Point]:
    # Unique points on two parabolas; deterministic and non-degenerate.
    return [(i, i * i % (2 * n + 1)) for i in range(n)]


def _closest_pair(points: list[Point]) -> int:
    """Squared closest distance, O(n log n) divide and conquer."""
    if len(points) < 2:
        return 0
    px = sorted(points)

    def solve(ps: list[Point]) -> tuple[int, list[Point]]:
        if len(ps) <= 3:
            best = min(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 for i, a in enumerate(ps) for b in ps[i + 1 :]), default=10**30)
            return best, sorted(ps, key=lambda p: p[1])
        mid = len(ps) // 2
        x = ps[mid][0]
        dl, yl = solve(ps[:mid])
        dr, yr = solve(ps[mid:])
        best = min(dl, dr)
        ys = []
        i = j = 0
        while i < len(yl) or j < len(yr):
            if j == len(yr) or (i < len(yl) and yl[i][1] <= yr[j][1]):
                ys.append(yl[i])
                i += 1
            else:
                ys.append(yr[j])
                j += 1
        strip = [p for p in ys if (p[0] - x) ** 2 < best]
        for i, a in enumerate(strip):
            for b in strip[i + 1 : i + 8]:
                best = min(best, (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        return best, ys

    return solve(px)[0]


def _graham(points: list[Point]) -> list[Point]:
    points = _merge_sort(list(set(points)), key=lambda p: p)
    if len(points) <= 1:
        return points
    pivot = min(points, key=lambda p: (p[1], p[0]))
    ordered = _merge_sort(
        [p for p in points if p != pivot],
        key=lambda p: (math.atan2(p[1] - pivot[1], p[0] - pivot[0]), (p[0] - pivot[0]) ** 2 + (p[1] - pivot[1]) ** 2),
    )
    hull = [pivot]
    for p in ordered:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return hull


def _monotonic_hull(points: list[Point]) -> list[Point]:
    ps = _merge_sort(list(set(points)), key=lambda p: p)
    if len(ps) <= 1:
        return ps
    lower = []
    for p in ps:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(ps):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _sum(values: list[float]) -> int:
    return int(sum(values))


def _directed_scc_input(n: int) -> list[list[int]]:
    g = [[] for _ in range(n)]
    for i in range(0, n - 1, 2):
        g[i].append(i + 1)
        g[i + 1].append(i)
    return g


def _a_star_n_states(n: int) -> int:
    """A* worst-case demo with a linear-size frontier and zero heuristic."""
    if n <= 1:
        return 0
    goal = n - 1
    best = [math.inf] * n
    best[0] = 0
    heap = [(_Priority(0, 0), 0)]
    while heap:
        priority, node = heapq.heappop(heap)
        if priority.value != best[node]:
            continue
        if node == goal:
            return int(best[node])
        if node == 0:
            for neighbor in range(1, n):
                cost = n - neighbor + 1
                best[neighbor] = cost
                heapq.heappush(heap, (_Priority(cost, neighbor), neighbor))
    raise ValueError("no path")


IMPLEMENTATIONS: dict[str, Callable[[int], int]] = {
    "breadth-first-search": lambda n: sum(_bfs(_undirected(n))),
    "depth-first-search": lambda n: sum(_dfs(_undirected(n))),
    "topological-sort": lambda n: sum((i + 1) * v for i, v in enumerate(_topological_dfs(_dag(n)))),
    "kahn-topological-sort": lambda n: sum((i + 1) * v for i, v in enumerate(_kahn(_dag(n)))),
    "connected-components": lambda n: sum(len(c) ** 2 for c in _components(_undirected(n))),
    "bipartite-check": lambda n: int(_is_bipartite([[v for v in (i - 1, i + 1) if 0 <= v < n] for i in range(n)])),
    "tarjan-scc": lambda n: sum(len(c) ** 2 for c in _tarjan(_directed_scc_input(n))),
    "kosaraju-scc": lambda n: sum(len(c) ** 2 for c in _kosaraju(_directed_scc_input(n))),
    "bridge-finding": lambda n: sum(u + n * v for u, v in _bridges_and_articulations(_undirected(n))[0]),
    "articulation-points": lambda n: sum(_bridges_and_articulations(_undirected(n))[1]),
    "flood-fill": lambda n: _flood_fill([[0] * n], (0, 0), 1) if n else 0,
    "kruskal-mst": lambda n: _kruskal(n, _weighted_sparse(n)),
    "prim-mst-heap": lambda n: _prim_heap(_weighted_adj(n, _weighted_sparse(n))),
    "dijkstra-heap": lambda n: _sum(_dijkstra(_weighted_adj(n, _weighted_sparse(n)))),
    "a-star": _a_star_n_states,
    "bellman-ford": lambda n: _sum(_bellman_ford(n, _weighted_sparse(n))),
    "prim-mst-matrix": lambda n: _prim_matrix(_matrix(n)),
    "dijkstra-matrix": lambda n: _sum(_dijkstra_matrix(_matrix(n))),
    "floyd-warshall": lambda n: int(sum(map(sum, _floyd(_matrix(n))))),
    "transitive-closure-warshall": lambda n: sum(map(sum, _closure(_dag(n)))),
    "closest-pair-points": lambda n: _closest_pair(_points(n)),
    "convex-hull-graham-scan": lambda n: sum(x * 31 + y for x, y in _graham(_points(n))),
    "convex-hull-monotonic-chain": lambda n: sum(x * 31 + y for x, y in _monotonic_hull(_points(n))),
}


def _assert_equal(actual: object, expected: object) -> None:
    assert actual == expected, (actual, expected)


def _assert_topological(sorter: Callable[[list[list[int]]], list[int]]) -> None:
    graph = [[1, 2], [3], [3], []]
    order = sorter(graph)
    position = {vertex: index for index, vertex in enumerate(order)}
    assert sorted(order) == list(range(len(graph)))
    assert all(position[u] < position[v] for u, row in enumerate(graph) for v in row)


def _assert_hull(hull: list[Point]) -> None:
    assert set(hull) == {(0, 0), (2, 0), (2, 2), (0, 2)}


def _check_flood_fill() -> None:
    grid = [[1, 1, 0], [1, 0, 0]]
    assert _flood_fill(grid, (0, 0), 2) == 3
    assert grid == [[2, 2, 0], [2, 0, 0]]


def _check_bridges() -> None:
    bridges, _ = _bridges_and_articulations([[1], [0, 2, 3], [1], [1]])
    assert bridges == [(0, 1), (1, 2), (1, 3)]


def _check_articulations() -> None:
    _, cuts = _bridges_and_articulations([[1], [0, 2, 3], [1], [1]])
    assert cuts == {1}


def _check_bipartite() -> None:
    assert _is_bipartite([[1], [0, 2], [1]])
    assert not _is_bipartite([[1, 2], [0, 2], [0, 1]])


_WEIGHTED_EXAMPLE = [(0, 1, 1), (1, 2, 2), (0, 2, 5)]
_WEIGHTED_GRAPH = _weighted_adj(3, _WEIGHTED_EXAMPLE)
_MATRIX_EXAMPLE = [[0, 1, 5], [1, 0, 2], [5, 2, 0]]
_SCC_EXAMPLE = [[1], [2], [0, 3], [4], [3]]
_SCC_EXPECTED = [[0, 1, 2], [3, 4]]
_HULL_POINTS = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)]


SELF_CHECKS: dict[str, Callable[[], None]] = {
    "breadth-first-search": lambda: _assert_equal(_bfs([[1, 2], [0, 3], [0], [1]]), [0, 1, 2, 3]),
    "depth-first-search": lambda: _assert_equal(_dfs([[1, 2], [0, 3], [0], [1]]), [0, 1, 3, 2]),
    "topological-sort": lambda: _assert_topological(_topological_dfs),
    "kahn-topological-sort": lambda: _assert_topological(_kahn),
    "connected-components": lambda: _assert_equal(sorted(map(sorted, _components([[1], [0], []]))), [[0, 1], [2]]),
    "bipartite-check": _check_bipartite,
    "tarjan-scc": lambda: _assert_equal(sorted(map(sorted, _tarjan(_SCC_EXAMPLE))), _SCC_EXPECTED),
    "kosaraju-scc": lambda: _assert_equal(sorted(map(sorted, _kosaraju(_SCC_EXAMPLE))), _SCC_EXPECTED),
    "bridge-finding": _check_bridges,
    "articulation-points": _check_articulations,
    "flood-fill": _check_flood_fill,
    "kruskal-mst": lambda: _assert_equal(_kruskal(3, _WEIGHTED_EXAMPLE), 3),
    "prim-mst-heap": lambda: _assert_equal(_prim_heap(_WEIGHTED_GRAPH), 3),
    "dijkstra-heap": lambda: _assert_equal(_dijkstra(_WEIGHTED_GRAPH), [0, 1, 3]),
    "a-star": lambda: _assert_equal(_a_star(3, 2), 3),
    "bellman-ford": lambda: _assert_equal(_bellman_ford(3, _WEIGHTED_EXAMPLE), [0, 1, 3]),
    "prim-mst-matrix": lambda: _assert_equal(_prim_matrix(_MATRIX_EXAMPLE), 3),
    "dijkstra-matrix": lambda: _assert_equal(_dijkstra_matrix(_MATRIX_EXAMPLE), [0, 1, 3]),
    "floyd-warshall": lambda: _assert_equal(_floyd(_MATRIX_EXAMPLE), [[0, 1, 3], [1, 0, 2], [3, 2, 0]]),
    "transitive-closure-warshall": lambda: _assert_equal(
        _closure([[1], [2], []]),
        [[True, True, True], [False, True, True], [False, False, True]],
    ),
    "closest-pair-points": lambda: _assert_equal(_closest_pair(_HULL_POINTS), 2),
    "convex-hull-graham-scan": lambda: _assert_hull(_graham(_HULL_POINTS)),
    "convex-hull-monotonic-chain": lambda: _assert_hull(_monotonic_hull(_HULL_POINTS)),
}


def run_all_named_self_checks() -> None:
    """Run the independently addressable correctness check for every demo."""
    assert set(SELF_CHECKS) == set(IMPLEMENTATIONS)
    for name, check in SELF_CHECKS.items():
        try:
            check()
        except AssertionError as error:
            raise AssertionError(f"self-check failed for {name}") from error


def run_self_checks() -> None:
    """Canonical correctness checks independent of complexity fitting."""
    path = [[1], [0, 2], [1, 3], [2]]
    assert _bfs(path) == [0, 1, 2, 3] and _dfs(path) == [0, 1, 2, 3]
    dag = [[1, 2], [3], [3], []]
    for order in (_topological_dfs(dag), _kahn(dag)):
        pos = {v: i for i, v in enumerate(order)}
        assert all(pos[u] < pos[v] for u, row in enumerate(dag) for v in row)
    assert sorted(map(sorted, _components([[1], [0], []]))) == [[0, 1], [2]]
    assert _is_bipartite(path) and not _is_bipartite([[1, 2], [0, 2], [0, 1]])
    directed = [[1], [2], [0, 3], [4], [3]]
    expected = [[0, 1, 2], [3, 4]]
    assert sorted(map(sorted, _tarjan(directed))) == expected
    assert sorted(map(sorted, _kosaraju(directed))) == expected
    bridges, cuts = _bridges_and_articulations([[1], [0, 2, 3], [1], [1]])
    assert bridges == [(0, 1), (1, 2), (1, 3)] and cuts == {1}
    grid = [[1, 1, 0], [1, 0, 0]]
    assert _flood_fill(grid, (0, 0), 2) == 3 and grid == [[2, 2, 0], [2, 0, 0]]
    edges = [(0, 1, 1), (1, 2, 2), (0, 2, 5)]
    wg = _weighted_adj(3, edges)
    assert _kruskal(3, edges) == 3 and _prim_heap(wg) == 3
    assert _dijkstra(wg) == [0, 1, 3] and _bellman_ford(3, edges) == [0, 1, 3]
    assert _a_star(3, 2) == 3
    matrix = [[0, 1, 5], [1, 0, 2], [5, 2, 0]]
    assert _prim_matrix(matrix) == 3 and _dijkstra_matrix(matrix) == [0, 1, 3]
    assert _floyd(matrix) == [[0, 1, 3], [1, 0, 2], [3, 2, 0]]
    assert _closure([[1], [2], []])[0] == [True, True, True]
    pts = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)]
    assert _closest_pair(pts) == 2
    assert set(_graham(pts)) == {(0, 0), (2, 0), (2, 2), (0, 2)}
    assert set(_monotonic_hull(pts)) == {(0, 0), (2, 0), (2, 2), (0, 2)}
    assert set(IMPLEMENTATIONS) == {
        "breadth-first-search",
        "depth-first-search",
        "topological-sort",
        "kahn-topological-sort",
        "connected-components",
        "bipartite-check",
        "tarjan-scc",
        "kosaraju-scc",
        "bridge-finding",
        "articulation-points",
        "flood-fill",
        "kruskal-mst",
        "prim-mst-heap",
        "dijkstra-heap",
        "a-star",
        "bellman-ford",
        "prim-mst-matrix",
        "dijkstra-matrix",
        "floyd-warshall",
        "transitive-closure-warshall",
        "closest-pair-points",
        "convex-hull-graham-scan",
        "convex-hull-monotonic-chain",
    }
