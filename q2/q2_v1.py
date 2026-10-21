#!/bin/python3

import os

# ===== grid_core.py =====
"""Torus reading, validation and scoring. FROZEN once tests pass: every solver
change is judged through validate() and score(), so these must never drift."""

import math

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def read(grid, r, c, dr, dc, k):
    """Read k letters from (r, c) in direction (dr, dc), wrapping on the torus."""
    n = len(grid)
    return "".join(grid[(r + i * dr) % n][(c + i * dc) % n] for i in range(k))


def all_strings(grid, k):
    """Every length-k string readable anywhere in the grid, any direction."""
    n = len(grid)
    return {read(grid, r, c, dr, dc, k)
            for r in range(n) for c in range(n) for dr, dc in DIRS}


def occurrences(grid, words):
    """List of (word, cells) for every place a word from `words` appears.
    Used by the solver to know which cells to repair."""
    n = len(grid)
    targets = set(words)
    found = []
    for k in {len(w) for w in targets}:
        for r in range(n):
            for c in range(n):
                for dr, dc in DIRS:
                    s = read(grid, r, c, dr, dc, k)
                    if s in targets:
                        cells = [((r + i * dr) % n, (c + i * dc) % n) for i in range(k)]
                        found.append((s, cells))
    return found


def validate(grid, required, forbidden):
    """Return (ok, errors). Deliberately strict so a grid passing here should
    pass the official checker under either reading of the rules:
    - required words longer than N count as missing (no cell reuse)
    - forbidden words are checked at every length, even if longer than N
      (i.e. we also catch occurrences that wrap onto themselves)"""
    errors = []
    n = len(grid)
    if n == 0 or any(len(row) != n for row in grid):
        return False, ["grid is not square"]
    if any(ch not in ALPHABET for row in grid for ch in row):
        return False, ["grid has non A-Z cells"]

    by_len = {}
    for k in {len(w) for w in list(required) + list(forbidden)}:
        by_len[k] = all_strings(grid, k)

    for w in required:
        if len(w) > n or w not in by_len[len(w)]:
            errors.append(f"missing required: {w}")
    for w in forbidden:
        if w in by_len[len(w)]:
            errors.append(f"contains forbidden: {w}")
    return not errors, errors


def score(n, required):
    """score = 100 * 10^-t, t = (N - L) / (U - L).
    The OCR dropped the 't'; this is our reading of it. U - L can be <= 0
    when there are few long words - clamped to 1 here (a guess, ask organisers)."""
    L = max(len(w) for w in required)
    U = len(required)
    t = (n - L) / max(U - L, 1)
    return 100 * math.pow(10, -t)


# ===== sa.py =====
"""Simulated annealing for the fixed-N decision problem: can all words fit?

State = one placement per word. Placement p encodes (r, c, d) as
p = (r*n + c)*8 + d, so warm starts and anchored moves are pure arithmetic.

Cost = number of conflicting letter pairs per cell:
    sum_cells [ C(cover, 2) - sum_letters C(count, 2) ]
Adding letter l to a cell creates (cover - count[l]) conflicts, so every
update is O(1) per cell and a move is O(word length), not O(N^2)."""

import math
import random
import time



_cells_cache = {}


def line_cells(n, k):
    """cells[p] = flat cell indices of a length-k line at placement p."""
    key = (n, k)
    if key not in _cells_cache:
        out = []
        for r in range(n):
            for c in range(n):
                for dr, dc in DIRS:
                    out.append(tuple(((r + i * dr) % n) * n + (c + i * dc) % n
                                     for i in range(k)))
        _cells_cache[key] = out
    return _cells_cache[key]


class State:
    def __init__(self, n, words, pos):
        """pos[i] may be None (word not yet placed) during greedy init."""
        self.n = n
        self.words = [[ord(ch) - 65 for ch in w] for w in words]
        self.cells = [line_cells(n, len(w)) for w in words]
        self.counts = [[0] * 26 for _ in range(n * n)]
        self.cover = [0] * (n * n)
        self.pos = list(pos)
        self.cost = 0
        for i, p in enumerate(self.pos):
            if p is not None:
                self.cost += self._add(i, p)

    def _add(self, i, p):
        d = 0
        counts, cover = self.counts, self.cover
        for idx, l in zip(self.cells[i][p], self.words[i]):
            cnt = counts[idx]
            d += cover[idx] - cnt[l]
            cnt[l] += 1
            cover[idx] += 1
        return d

    def _remove(self, i, p):
        d = 0
        counts, cover = self.counts, self.cover
        for idx, l in zip(self.cells[i][p], self.words[i]):
            cnt = counts[idx]
            cnt[l] -= 1
            cover[idx] -= 1
            d -= cover[idx] - cnt[l]
        return d

    def move(self, i, p):
        """Move word i to placement p (None = unplace). Returns cost delta."""
        d = 0
        if self.pos[i] is not None:
            d += self._remove(i, self.pos[i])
        if p is not None:
            d += self._add(i, p)
        self.pos[i] = p
        self.cost += d
        return d

    def add_cost(self, i, p):
        """Conflicts word i would create at placement p (no mutation)."""
        counts, cover = self.counts, self.cover
        return sum(cover[idx] - counts[idx][l] for idx, l in zip(self.cells[i][p], self.words[i]))

    def word_conflicts(self, i):
        return sum(self.cover[idx] - self.counts[idx][l]
                   for idx, l in zip(self.cells[i][self.pos[i]], self.words[i]))

    def full_cost(self):
        """Recompute from scratch - only for tests."""
        tot = 0
        for idx in range(self.n * self.n):
            c = self.cover[idx]
            tot += c * (c - 1) // 2 - sum(x * (x - 1) // 2 for x in self.counts[idx])
        return tot

    def partial_grid(self):
        """Letters where words sit, None elsewhere (majority letter if conflicted)."""
        n = self.n
        g = [[None] * n for _ in range(n)]
        for idx in range(n * n):
            if self.cover[idx]:
                cnt = self.counts[idx]
                g[idx // n][idx % n] = chr(65 + max(range(26), key=cnt.__getitem__))
        return g


def anchor_pairs(words):
    """For word i: (j, m, j2) where words[i][j] == words[m][j2], m != i.
    An anchored move lays word i through a cell where word m has the same letter."""
    pairs = []
    for i, w in enumerate(words):
        pairs.append([(j, m, j2) for j, ch in enumerate(w)
                      for m, v in enumerate(words) if m != i
                      for j2, ch2 in enumerate(v) if ch2 == ch])
    return pairs


def anchored(st, i, pair, rng):
    j, m, j2 = pair
    n = st.n
    tr, tc = divmod(st.cells[m][st.pos[m]][j2], n)
    d = rng.randrange(8)
    dr, dc = DIRS[d]
    return (((tr - j * dr) % n) * n + (tc - j * dc) % n) * 8 + d


def greedy_init(n, words, rng, samples=300):
    """Place words longest-first at the best of a sample of placements
    (fewest conflicts, then most overlap)."""
    st = State(n, words, [None] * len(words))
    pairs = anchor_pairs(words)
    nplace = n * n * 8
    for i in range(len(words)):
        placed = [pr for pr in pairs[i] if st.pos[pr[1]] is not None]
        cands = [rng.randrange(nplace) for _ in range(samples // 2)]
        cands += [anchored(st, i, rng.choice(placed), rng) for _ in range(samples // 2)] \
            if placed else [rng.randrange(nplace) for _ in range(samples // 2)]
        best, best_key = None, None
        for p in cands:
            d = st.move(i, p)
            overlap = sum(st.cover[idx] > 1 for idx in st.cells[i][p])
            key = (d, -overlap)
            if best_key is None or key < best_key:
                best, best_key = p, key
            st.move(i, None)
        st.move(i, best)
    return st.pos


def warm_start(pos, n_old, n_new):
    """Map placements from an n_old grid onto n_new by wrapping coordinates."""
    out = []
    for p in pos:
        d, s = p % 8, p // 8
        r, c = divmod(s, n_old)
        out.append(((r % n_new) * n_new + c % n_new) * 8 + d)
    return out


def dfs_place(n, words, deadline, rng, branch=3, node_limit=3000):
    """Depth-first placement in the given word order. At each word, try only
    the `branch` conflict-free placements with the most overlap (random
    tie-break), backtracking on dead ends; give up after node_limit nodes
    (the caller restarts with a new order). Returns a partial grid or None."""
    cells = [line_cells(n, len(w)) for w in words]
    codes = [[ord(c) - 65 for c in w] for w in words]
    letter, cover = [-1] * (n * n), [0] * (n * n)
    nodes = 0

    def candidates(i):
        out = []
        code = codes[i]
        for p, cs in enumerate(cells[i]):
            ov = 0
            for idx, l in zip(cs, code):
                x = letter[idx]
                if x == -1:
                    continue
                if x != l:
                    break
                ov += 1
            else:
                out.append((ov, rng.random(), p))
        out.sort(reverse=True)
        return out[:branch]

    def rec(i):
        nonlocal nodes
        if i == len(words):
            return True
        nodes += 1
        if nodes > node_limit or (nodes & 31 == 0 and time.time() > deadline):
            return False
        for _, _, p in candidates(i):
            for idx, l in zip(cells[i][p], codes[i]):
                letter[idx] = l
                cover[idx] += 1
            if rec(i + 1):
                return True
            for idx in cells[i][p]:
                cover[idx] -= 1
                if cover[idx] == 0:
                    letter[idx] = -1
        return False

    if not rec(0):
        return None
    return [[chr(65 + letter[r * n + c]) if letter[r * n + c] >= 0 else None
             for c in range(n)] for r in range(n)]


def eject_search(n, words, deadline, rng, init_pos=None, k=24, p_anchor=0.8, noise=0.05):
    """Keep placed words always conflict-free; some words are unplaced.
    Step: take an unplaced word, try k candidate placements, take the one whose
    displaced words have the least total weight, displace them. Each time a
    word is displaced its weight grows (breakout), so the search stops cycling
    on the same few words. Done when nothing is unplaced.
    Returns (State with cost 0 or None, stats dict)."""
    m = len(words)
    st = State(n, words, [None] * m)
    occ = [[] for _ in range(n * n)]  # cell -> [(word, letter)]
    pairs = anchor_pairs(words)
    nplace = n * n * 8
    weight = [1] * m

    def place(i, p):
        st.move(i, p)
        for idx, l in zip(st.cells[i][p], st.words[i]):
            occ[idx].append((i, l))

    def unplace(i):
        for idx, l in zip(st.cells[i][st.pos[i]], st.words[i]):
            occ[idx].remove((i, l))
        st.move(i, None)

    def clashes(i, p):
        hit = set()
        for idx, l in zip(st.cells[i][p], st.words[i]):
            for w, lw in occ[idx]:
                if lw != l:
                    hit.add(w)
        return hit

    if init_pos is not None:  # warm start: keep a conflict-free subset
        for i in sorted(range(m), key=lambda i: -len(words[i])):
            if not clashes(i, init_pos[i]):
                place(i, init_pos[i])

    unplaced = [i for i in range(m) if st.pos[i] is None]
    start, it = time.time(), 0
    while unplaced:
        it += 1
        if it & 63 == 0 and time.time() >= deadline:
            break
        i = unplaced.pop(rng.randrange(len(unplaced)))
        best, best_cost = [], None
        for _ in range(k):
            pr = rng.choice(pairs[i]) if pairs[i] and rng.random() < p_anchor else None
            if pr is not None and st.pos[pr[1]] is not None:
                p = anchored(st, i, pr, rng)
            else:
                p = rng.randrange(nplace)
            hit = clashes(i, p)
            c = sum(weight[w] for w in hit)
            if best_cost is None or c < best_cost:
                best, best_cost = [(p, hit)], c
            elif c == best_cost:
                best.append((p, hit))
        if rng.random() < noise:  # random walk to escape plateaus
            p = rng.randrange(nplace)
            hit = clashes(i, p)
        else:
            p, hit = rng.choice(best)
        for w in hit:
            unplace(w)
            weight[w] += 1
            unplaced.append(w)
        place(i, p)
    secs = max(time.time() - start, 1e-9)
    stats = {"iters": it, "steps_per_sec": it / secs, "unplaced": len(unplaced)}
    return (st if not unplaced else None), stats


def anneal(n, words, deadline, rng, init_pos=None, t0=0.6, t1=0.05, p_anchor=0.7, k=16):
    """Min-conflicts annealing: lift a conflicted word out, score k candidate
    placements (anchored or random), take the best, accept by Metropolis rule.
    Returns (State with cost 0 or None, stats dict)."""
    if init_pos is None:
        init_pos = greedy_init(n, words, rng)
    st = State(n, words, init_pos)
    pairs = anchor_pairs(words)
    nplace, m = n * n * 8, len(words)
    start = time.time()
    span = max(deadline - start, 1e-3)
    T, it = t0, 0
    while st.cost > 0:
        it += 1
        if it & 255 == 0:
            now = time.time()
            if now >= deadline:
                break
            T = t0 * (t1 / t0) ** ((now - start) / span)
        i = rng.randrange(m)
        for _ in range(3):  # prefer words that are in conflict
            if st.word_conflicts(i):
                break
            i = rng.randrange(m)
        old = st.pos[i]
        d_rem = st.move(i, None)
        best_p, best_c = None, None
        for _ in range(k):
            if pairs[i] and rng.random() < p_anchor:
                p = anchored(st, i, rng.choice(pairs[i]), rng)
            else:
                p = rng.randrange(nplace)
            if p == old:
                continue
            c = st.add_cost(i, p)
            if best_c is None or c < best_c:
                best_p, best_c = p, c
        if best_p is None:
            st.move(i, old)
            continue
        d = d_rem + best_c
        if d <= 0 or rng.random() < math.exp(-d / T):
            st.move(i, best_p)
        else:
            st.move(i, old)
    secs = max(time.time() - start, 1e-9)
    stats = {"iters": it, "moves_per_sec": it / secs, "final_cost": st.cost}
    return (st if st.cost == 0 else None), stats


# ===== solver.py =====
"""Baseline solver: greedy max-overlap placement + random fill + forbidden repair.
Strategy: climb N from L until a valid grid is found (the safe answer), then
spend the remaining time trying to beat it at N - 1.

Improvement hooks are marked HOOK."""

import json
import textwrap
import random
import time




TIME_LIMIT = 5.0  # seconds per test case; unknown, ask organisers


def preprocess(required):
    """Uppercase, dedupe, and drop words that are substrings of a longer
    required word (or its reverse): they appear automatically."""
    words = sorted({w.upper() for w in required}, key=len, reverse=True)
    kept = []
    for w in words:
        if not any(w in k or w[::-1] in k for k in kept):
            kept.append(w)
    return kept


def place_words(n, words, rng):
    """Greedily place each word where it overlaps most existing letters.
    Returns a grid with None for empty cells, or None if a word won't fit.
    HOOK: this is the core of the problem. Better: backtracking, beam search,
    or simulated annealing over placements."""
    grid = [[None] * n for _ in range(n)]
    for w in words:
        best, best_overlap = [], -1
        for r in range(n):
            for c in range(n):
                for dr, dc in DIRS:
                    overlap = 0
                    for i, ch in enumerate(w):
                        cur = grid[(r + i * dr) % n][(c + i * dc) % n]
                        if cur is None:
                            continue
                        if cur != ch:
                            break
                        overlap += 1
                    else:
                        if overlap > best_overlap:
                            best, best_overlap = [(r, c, dr, dc)], overlap
                        elif overlap == best_overlap:
                            best.append((r, c, dr, dc))
        if not best:
            return None
        r, c, dr, dc = rng.choice(best)
        for i, ch in enumerate(w):
            grid[(r + i * dr) % n][(c + i * dc) % n] = ch
    return grid


def fill(grid, forbidden, rng, max_iters=2000):
    """Fill empty cells randomly, then repair forbidden occurrences by
    re-rolling unlocked cells. Returns a full grid or None.
    HOOK: choose fill letters that are rare in the forbidden words."""
    n = len(grid)
    locked = {(r, c) for r in range(n) for c in range(n) if grid[r][c] is not None}
    g = [[ch if ch is not None else rng.choice(ALPHABET) for ch in row] for row in grid]
    if not forbidden:
        return g
    for _ in range(max_iters):
        bad = occurrences(g, forbidden)
        if not bad:
            return g
        for _, cells in bad:
            free = [cell for cell in cells if cell not in locked]
            if not free:
                return None  # forbidden word made entirely of required letters
            r, c = rng.choice(free)
            g[r][c] = rng.choice(ALPHABET)
    return None


def attempt(n, words, forbidden, rng, first=False):
    order = list(words)
    if not first:
        # HOOK: smarter perturbations than a shuffle within length groups
        rng.shuffle(order)
        order.sort(key=len, reverse=True)
    placed = place_words(n, order, rng)
    if placed is None:
        return None
    return fill(placed, forbidden, rng)


# The template writes the grid to OUTPUT_PATH, so stdout is a free channel the
# platform shows as debug output. Print the inputs in a compact, checksummed
# form so they can be transcribed from screenshots and replayed locally.
REPORT = True
PHASE1_TRIES = 1


def generate_grid(required_words, forbidden_words, time_limit=TIME_LIMIT, seed=0):
    t0 = time.time()
    grid = _generate(required_words, forbidden_words, time_limit, seed)
    if REPORT:
        L = max(len(w) for w in required_words)
        print(f"N={len(grid)} L={L} U={len(required_words)} "
              f"letters={sum(map(len, required_words))} F={len(forbidden_words)} "
              f"secs={time.time() - t0:.1f}")
        print(textwrap.fill("REQ " + " ".join(required_words), 60))
        print("FORB " + " ".join(forbidden_words))
    return grid


def _generate(required_words, forbidden_words, time_limit, seed):
    rng = random.Random(seed)
    deadline = time.time() + time_limit
    words = preprocess(required_words)
    forbidden = [f.upper() for f in forbidden_words if f]
    required = [w.upper() for w in required_words]
    L = max(len(w) for w in words)

    # A forbidden word inside a required word (or its reverse) is unavoidable:
    # the case is unsolvable, so don't burn time climbing N.
    if any(f in w or f in w[::-1] for f in forbidden for w in words):
        return ["A" * L] * L

    def ok(g):
        return g is not None and validate(g, required, forbidden)[0]

    # Phase 1: climb until something valid exists (always returns something).
    # One try per size: extra tries at sizes that are too small burned ~75% of
    # the budget on platform-shaped cases; phase 2 makes better use of it.
    best, n = None, L
    cap = sum(len(w) for w in words) + L
    while best is None and n <= cap:
        for i in range(PHASE1_TRIES):
            g = attempt(n, words, forbidden, rng, first=(i == 0))
            if ok(g):
                best = g
                break
        n += 1

    if best is None:  # likely a required word contains a forbidden one
        return ["A" * L] * L

    # Phase 2: portfolio of restarts at N-1. Benchmarks showed no single method
    # dominates: greedy restarts are fastest on tight small grids, limited-
    # backtracking DFS is ~4x better on larger ones, SA occasionally lands one.
    while len(best) > L and time.time() < deadline:
        n = len(best) - 1
        r = rng.random()
        if r < 0.45:
            g = attempt(n, words, forbidden, rng)
        else:
            order = list(words)
            rng.shuffle(order)
            order.sort(key=len, reverse=True)
            if r < 0.9:
                partial = dfs_place(n, order, deadline, rng, branch=2, node_limit=500)
            else:
                st, _ = anneal(n, order, min(deadline, time.time() + 0.1), rng, t0=2, t1=0.3)
                partial = st.partial_grid() if st else None
            g = fill(partial, forbidden, rng) if partial else None
        if ok(g):
            best = g
    return ["".join(row) for row in best]


# ===== platform template main =====
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    required_words_count = int(input().strip())

    required_words = []

    for _ in range(required_words_count):
        required_words_item = input()
        required_words.append(required_words_item)

    forbidden_words_count = int(input().strip())

    forbidden_words = []

    for _ in range(forbidden_words_count):
        forbidden_words_item = input()
        forbidden_words.append(forbidden_words_item)

    result = generate_grid(required_words, forbidden_words)

    fptr.write('\n'.join([' '.join(x) for x in result]))
    fptr.write('\n')

    fptr.close()
