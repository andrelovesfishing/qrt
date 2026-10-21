#!/bin/python3

import math
import os
import random
import re
import sys
import threading
import time


TIME_LIMIT = 4.0  # seconds of search per test case


class _Abort(Exception):
    pass


def _lower_bound(S, m):
    best = None
    w = m
    while True:
        h = max(m, -(-S // w))
        if best is None or w * h < best:
            best = w * h
        if h <= m:
            break
        w += 1
    return best


def _merge(segs):
    out = [segs[0]]
    for s in segs[1:]:
        if s[2] == out[-1][2]:
            out[-1] = [out[-1][0], out[-1][1] + s[1], s[2]]
        else:
            out.append(s)
    return out


def _lowest(segs):
    bi, bh = 0, segs[0][2]
    for j in range(1, len(segs)):
        if segs[j][2] < bh:
            bi, bh = j, segs[j][2]
    return bi


def _greedy(W, desc, cnt0, h_limit, rng=None):
    """Best-fit skyline strip packing of width W. Returns (height, placements) or None."""
    cnt = cnt0[:]
    remaining = sum(cnt)
    segs = [[0, W, 0]]
    place = []
    maxh = 0
    while remaining:
        i = _lowest(segs)
        x, w, h = segs[i]
        pick = -1
        for k, s in enumerate(desc):
            if cnt[k] and s <= w:
                pick = k
                break
        if pick >= 0 and rng is not None and rng.random() < 0.15:
            alts = [k for k in range(pick + 1, len(desc)) if cnt[k]]
            if alts:
                pick = rng.choice(alts[:3])
        lh = segs[i - 1][2] if i > 0 else float('inf')
        rh = segs[i + 1][2] if i + 1 < len(segs) else float('inf')
        if pick < 0:
            segs[i] = [x, w, min(lh, rh)]
            segs = _merge(segs)
            continue
        s = desc[pick]
        if h + s > h_limit:
            return None
        cnt[pick] -= 1
        remaining -= 1
        at_left = lh >= rh
        if rng is not None and lh != float('inf') and rh != float('inf') and rng.random() < 0.1:
            at_left = not at_left
        if at_left:
            new = [[x, s, h + s]] + ([[x + s, w - s, h]] if w > s else [])
            place.append((s, h, x))
        else:
            new = ([[x, w - s, h]] if w > s else []) + [[x + w - s, s, h + s]]
            place.append((s, h, x + w - s))
        segs[i:i + 1] = new
        segs = _merge(segs)
        maxh = max(maxh, h + s)
    return maxh, place


def _subset_sums(desc, cnt, w, lim=None):
    """Bitset of totals <= w reachable with remaining squares (of side <= lim)."""
    reach = 1
    mask = (1 << (w + 1)) - 1
    for k in range(len(desc) - 1, -1, -1):
        s = desc[k]
        if s > w or (lim is not None and s > lim):
            break
        c = cnt[k]
        if c * s > w:
            c = w // s
        p = 1
        while c > 0:
            t = p if p < c else c
            reach = (reach | (reach << (s * t))) & mask
            c -= t
            p <<= 1
        if reach >> w:
            break
    return reach


def _dfs(W, H, desc, cnt0, slack, node_limit, max_disc, left_only=False, sums=None, deadline=None):
    """Limited discrepancy search for a packing of everything into W x H.

    Always fills the lowest skyline gap, placing squares against the taller
    neighbour (or always at its left end with `left_only`). Children are tried
    in heuristic order (largest fitting square first, then wasting the whole
    gap); taking the i-th child costs i discrepancies, at most `max_disc` along
    any path. Empty space may total at most `slack` cells.
    """
    cnt = cnt0[:]
    stack = []
    nodes = [0]
    nd = len(desc)
    if sums is None:
        sums = {}

    def rec(segs, remaining, slack, disc):
        nodes[0] += 1
        if nodes[0] > node_limit:
            raise _Abort
        if deadline is not None and not nodes[0] & 1023 and time.time() > deadline:
            raise _Abort
        if remaining == 0:
            return True
        i = _lowest(segs)
        x, w, h = segs[i]
        lim = min(w, H - h)
        lh = segs[i - 1][2] if i > 0 else H
        rh = segs[i + 1][2] if i + 1 < len(segs) else H
        nh = min(lh, rh)
        cost = w * (nh - h)
        # The bottom row of every well (segment lower than both neighbours)
        # must be covered by squares resting on it, or else left empty.
        ns = len(segs)
        wells = []
        maxw = 0
        for j in range(ns):
            hj = segs[j][2]
            if (j == 0 or segs[j - 1][2] > hj) and (j == ns - 1 or segs[j + 1][2] > hj):
                wells.append(j)
                if segs[j][1] > maxw:
                    maxw = segs[j][1]
        key = tuple(cnt)
        reach = sums.get((maxw, key))
        if reach is None:
            if len(sums) > 300000:
                sums.clear()
            reach = sums[(maxw, key)] = _subset_sums(desc, cnt, maxw)
        waste = 0
        fill = 0
        for j in wells:
            xj, wj, hj = segs[j]
            if H - hj < wj:
                fj = _subset_sums(desc, cnt, wj, H - hj).bit_length() - 1
            else:
                fj = (reach & ((1 << (wj + 1)) - 1)).bit_length() - 1
            if j == i:
                fill = fj
            if fj == 0:
                lj = segs[j - 1][2] if j > 0 else H
                rj = segs[j + 1][2] if j + 1 < ns else H
                waste += wj * ((lj if lj < rj else rj) - hj)
            else:
                waste += wj - fj
            if waste > slack:
                return False
        if fill == 0:
            new = segs[:i] + [[x, w, nh]] + segs[i + 1:]
            return rec(_merge(new), remaining, slack - cost, disc)
        at_left = left_only or lh >= rh
        ci = 0
        for k in range(nd):
            if not cnt[k] or desc[k] > lim:
                continue
            if ci > disc:
                return False
            s = desc[k]
            if at_left:
                mid = [[x, s, h + s]] + ([[x + s, w - s, h]] if w > s else [])
                stack.append((s, h, x))
            else:
                mid = ([[x, w - s, h]] if w > s else []) + [[x + w - s, s, h + s]]
                stack.append((s, h, x + w - s))
            cnt[k] -= 1
            if rec(_merge(segs[:i] + mid + segs[i + 1:]), remaining - 1, slack, disc - ci):
                return True
            cnt[k] += 1
            stack.pop()
            ci += 1
        if ci > disc:
            return False
        if cost <= slack:
            new = segs[:i] + [[x, w, nh]] + segs[i + 1:]
            if rec(_merge(new), remaining, slack - cost, disc - ci):
                return True
        return False

    try:
        if rec([[0, W, 0]], sum(cnt), slack, max_disc):
            return list(stack)
    except _Abort:
        pass
    return None


def _solve(squares):
    n = len(squares)
    if n == 0:
        return []
    start = time.time()
    deadline = start + TIME_LIMIT
    m = max(squares)
    S = sum(a * a for a in squares)
    L = _lower_bound(S, m)
    desc = sorted(set(squares), reverse=True)
    idx = {s: k for k, s in enumerate(desc)}
    cnt = [0] * len(desc)
    for a in squares:
        cnt[idx[a]] += 1

    # Baseline: one row of squares (always valid).
    best_area = sum(squares) * m
    best_place = []
    x = 0
    for a in squares:
        best_place.append((a, 0, x))
        x += a

    def widths(limit_area):
        ws = list(range(m, max(m, math.isqrt(limit_area)) + 1))
        ws.sort(key=lambda w: (w * max(m, -(-S // w)), w))
        return ws

    # Phase 1: greedy skyline over strip widths, most promising first.
    phase1_end = start + TIME_LIMIT * 0.2
    for w in widths(best_area):
        if time.time() > phase1_end and best_place:
            break
        if w * max(m, -(-S // w)) >= best_area:
            continue
        res = _greedy(w, desc, cnt, (best_area - 1) // w)
        if res and res[0] * w < best_area:
            best_area, best_place = res[0] * w, res[1]
        if best_area == L:
            break

    # Phase 2: limited discrepancy search over candidate boxes smaller than the
    # best found, smallest area first, allowing more discrepancies each round.
    def candidates():
        cs = []
        for w in range(m, math.isqrt(best_area - 1) + 1):
            h = max(m, -(-S // w))
            while w * h < best_area and len(cs) < 4000:
                cs.append((w * h, w, h))
                if w != h:
                    cs.append((w * h, h, w))
                h += 1
        cs.sort()
        return cs[:300]

    rng = random.Random(12345)
    sums = {}

    def greedy_burst(seconds):
        nonlocal best_area, best_place
        end = min(deadline, time.time() + seconds)
        ws = widths(best_area)[:40]
        while ws and time.time() < end:
            w = ws[min(len(ws) - 1, int(rng.expovariate(1.0 / 8)))]
            g = _greedy(w, desc, cnt, (best_area - 1) // w, rng)
            if g is not None and g[0] * w < best_area:
                best_area, best_place = g[0] * w, g[1]
                ws = widths(best_area)[:40]

    d = 0
    next_burst = time.time() + 0.1
    while best_area > L and time.time() < deadline:
        limit = 2000 * (d + 1) ** 2
        for area, W, H in candidates():
            if area >= best_area or time.time() > deadline:
                continue
            if time.time() > next_burst:
                greedy_burst(0.05)
                next_burst = time.time() + 0.1
            for left_only in (False, True):
                if area >= best_area or time.time() > deadline:
                    break
                res = _dfs(W, H, desc, cnt, area - S, limit, d, left_only, sums, deadline)
                if res is not None:
                    best_area, best_place = area, res
        d += 1

    # Assign placements back to input indices.
    by_size = {}
    for s, r, c in best_place:
        by_size.setdefault(s, []).append((r, c))
    out = []
    for a in squares:
        r, c = by_size[a].pop()
        out.append([r, c])
    return out


def pack_squares(squares):
    result = [None]
    sys.setrecursionlimit(1000000)
    threading.stack_size(512 * 1024 * 1024)
    t = threading.Thread(target=lambda: result.__setitem__(0, _solve(squares)))
    t.start()
    t.join()
    return result[0]


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    squares_count = int(input().strip())

    squares = []

    for _ in range(squares_count):
        squares_item = int(input().strip())
        squares.append(squares_item)

    result = pack_squares(squares)

    fptr.write('\n'.join([' '.join(map(str, x)) for x in result]))
    fptr.write('\n')

    fptr.close()
