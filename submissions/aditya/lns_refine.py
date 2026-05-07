"""
LNS (Large Neighborhood Search) post-process for macro placement.

After analytical global placement + legalization, run episodes of:
  1. Pick N macros (mix of worst-cost-contributing, connectivity-clustered, random).
  2. Snapshot proxy state.
  3. Run mini-SA on those N macros (with all others frozen) using the
     incremental proxy as fast scoring.
  4. Validate end-of-episode against TILOS PlacementCost.
  5. Accept if real proxy improved; otherwise restore snapshot.

Designed to layer on top of `placer_v4` / `_legalize` output.
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Optional

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost

import sys
from pathlib import Path
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from incremental_proxy import IncrementalProxy


def _build_pin_to_hard(benchmark: Benchmark) -> List[List[int]]:
    """For each hard macro, list of hard macros connected via shared nets."""
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.num_macros
    adj: List[set] = [set() for _ in range(n_hard)]
    for nodes_t in benchmark.net_nodes:
        nodes = nodes_t.numpy().tolist()
        hard_in_net = [i for i in nodes if 0 <= i < n_hard]
        for i in hard_in_net:
            for j in hard_in_net:
                if i != j:
                    adj[i].add(j)
    return [sorted(s) for s in adj]


def _select_subset(incp: IncrementalProxy,
                   adj: List[List[int]],
                   movable: np.ndarray,
                   N: int,
                   strategy: str,
                   rng: random.Random) -> List[int]:
    """Pick N hard macro indices to re-optimize."""
    n_hard = incp.n_hard
    movable_idx = np.where(movable)[0].tolist()
    if N >= len(movable_idx):
        return movable_idx

    if strategy == "random":
        return rng.sample(movable_idx, N)

    if strategy == "worst":
        # Score each macro by sum of density it contributes to
        # high-density bins + sum of net wire-density at top RUDY cells.
        # Cheap proxy: macros with the largest footprint area covering
        # the busiest cells.
        flat_d = incp.density_grid.ravel()
        flat_r = incp.rudy_grid.ravel()
        n_top_d = max(1, flat_d.size // 10)
        n_top_r = max(1, int(flat_r.size * 0.05))
        thr_d = np.partition(flat_d, -n_top_d)[-n_top_d]
        thr_r = np.partition(flat_r, -n_top_r)[-n_top_r]
        hot_d_mask = (incp.density_grid >= thr_d)
        hot_r_mask = (incp.rudy_grid >= thr_r)
        # For each macro, compute its overlap with hot density cells
        scores = np.zeros(n_hard, dtype=np.float64)
        for i in movable_idx:
            x = incp.positions[i, 0]; y = incp.positions[i, 1]
            hw = incp.sizes[i, 0] / 2; hh = incp.sizes[i, 1] / 2
            ov_x = np.maximum(0.0, np.minimum(x + hw, incp.den_col_hi)
                              - np.maximum(x - hw, incp.den_col_lo))
            ov_y = np.maximum(0.0, np.minimum(y + hh, incp.den_row_hi)
                              - np.maximum(y - hh, incp.den_row_lo))
            footprint = np.outer(ov_y, ov_x)
            scores[i] = (footprint * hot_d_mask).sum()
            # Add cheap RUDY contribution: bin RUDY at macro center
            cx = min(int(x / incp.rudy_cell_w), incp.rudy_g - 1)
            cy = min(int(y / incp.rudy_cell_h), incp.rudy_g - 1)
            scores[i] += incp.rudy_grid[cy, cx] * 0.5
        # Top-N by score
        ranked = sorted(movable_idx, key=lambda k: -scores[k])
        return ranked[:N]

    if strategy == "shaw":
        # Pick a random seed, then add nearest-connected macros.
        seed = rng.choice(movable_idx)
        chosen = [seed]
        frontier = list(adj[seed])
        rng.shuffle(frontier)
        for j in frontier:
            if movable[j] and j not in chosen:
                chosen.append(j)
                if len(chosen) >= N:
                    break
        # Pad with random if not enough
        if len(chosen) < N:
            pool = [i for i in movable_idx if i not in chosen]
            chosen.extend(rng.sample(pool, min(N - len(chosen), len(pool))))
        return chosen[:N]

    raise ValueError(f"Unknown strategy: {strategy}")


def _propose_move(rng: random.Random, x: float, y: float,
                  hw: float, hh: float, cw: float, ch: float,
                  step_xy: float) -> tuple:
    """Gaussian shift, clamped to canvas."""
    nx = x + rng.gauss(0, step_xy)
    ny = y + rng.gauss(0, step_xy)
    nx = max(hw, min(cw - hw, nx))
    ny = max(hh, min(ch - hh, ny))
    return nx, ny


def lns_refine(positions: torch.Tensor,
               benchmark: Benchmark,
               plc,
               time_budget: float = 300.0,
               n_episodes: int = 60,
               subset_size: int = 12,
               sa_steps: int = 1500,
               accept_eps: float = 1e-5,
               seed: int = 0,
               sa_mode: str = "sa",  # "sa" or "greedy"
               step_init_frac: float = 0.05,
               step_end_frac: float = 0.0005,
               verbose: bool = False) -> torch.Tensor:
    """
    Run LNS episodes to refine an existing legal placement.

    `positions`: [num_macros, 2] tensor (post-legalize, hard + soft)
    Returns refined positions (same shape).
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    n_hard = benchmark.num_hard_macros
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    scale = max(cw, ch)

    # Build IncrementalProxy from current positions
    # IncrementalProxy reads from benchmark.macro_positions; temporarily set it.
    orig_positions = benchmark.macro_positions.clone()
    benchmark.macro_positions = positions.clone()
    try:
        incp = IncrementalProxy(benchmark)
    finally:
        benchmark.macro_positions = orig_positions

    adj = _build_pin_to_hard(benchmark)

    # Initial real proxy
    initial_real_cost = compute_proxy_cost(positions, benchmark, plc)["proxy_cost"]
    best_cost = initial_real_cost
    best_positions = positions.clone()

    # Operator weights (adaptive)
    op_weights = {"worst": 1.0, "shaw": 1.0, "random": 1.0}
    op_choices = list(op_weights.keys())

    t0 = time.time()
    accepted = 0
    rejected = 0
    real_cost_now = initial_real_cost

    for episode in range(n_episodes):
        if time.time() - t0 > time_budget:
            break

        # Choose operator weighted by its score
        weights = [op_weights[k] for k in op_choices]
        op = rng.choices(op_choices, weights=weights, k=1)[0]

        N = max(2, subset_size)
        subset = _select_subset(incp, adj, movable, N, op, rng)
        if not subset:
            continue

        snapshot = incp.snapshot()

        # Inner SA loop
        # Initial step size proportional to canvas; geometric anneal
        step0 = scale * step_init_frac
        step_end = scale * step_end_frac
        step = step0
        # Decay rate so step_end after sa_steps
        if sa_steps > 1:
            step_decay = (step_end / step0) ** (1.0 / (sa_steps - 1))
        else:
            step_decay = 1.0

        # Approximate proxy at episode start
        cur_proxy = incp.proxy()
        T = max(cur_proxy * 0.05, 1e-6)
        T_decay = (1e-3 / T) ** (1.0 / max(sa_steps - 1, 1))

        sa_accept = 0
        sa_reject = 0
        for sa_step in range(sa_steps):
            i = rng.choice(subset)
            old_x = float(incp.positions[i, 0]); old_y = float(incp.positions[i, 1])
            hw = float(incp.sizes[i, 0] / 2); hh = float(incp.sizes[i, 1] / 2)
            nx, ny = _propose_move(rng, old_x, old_y, hw, hh, cw, ch, step)
            # Quick overlap check before applying
            old_pos_save = (old_x, old_y)
            incp.positions[i, 0] = nx
            incp.positions[i, 1] = ny
            if incp.overlaps_any(i):
                incp.positions[i, 0] = old_x
                incp.positions[i, 1] = old_y
                sa_reject += 1
                continue
            # Apply via move_macro for proper proxy update
            incp.positions[i, 0] = old_x  # restore for proper move_macro semantics
            incp.positions[i, 1] = old_y
            old_proxy = incp.proxy()
            incp.move_macro(i, nx, ny)
            new_proxy = incp.proxy()
            delta = new_proxy - old_proxy
            if sa_mode == "greedy":
                accept = (delta < 0)
            else:
                accept = (delta < 0) or (rng.random() < math.exp(-delta / max(T, 1e-12)))
            if accept:
                cur_proxy = new_proxy
                sa_accept += 1
            else:
                # Reject — move back
                incp.move_macro(i, old_x, old_y)
                sa_reject += 1
            step *= step_decay
            T *= T_decay

        # Validate against TILOS
        # Build positions tensor from incp (updates only hard macros)
        new_positions = positions.clone()
        new_positions[:n_hard, 0] = torch.from_numpy(incp.positions[:n_hard, 0]).float()
        new_positions[:n_hard, 1] = torch.from_numpy(incp.positions[:n_hard, 1]).float()
        real_cost_new = compute_proxy_cost(new_positions, benchmark, plc)["proxy_cost"]

        if real_cost_new < real_cost_now - accept_eps:
            real_cost_now = real_cost_new
            if real_cost_new < best_cost:
                best_cost = real_cost_new
                best_positions = new_positions.clone()
            op_weights[op] = min(op_weights[op] * 1.1, 5.0)
            accepted += 1
            if verbose:
                print(f"  [LNS ep {episode}] op={op} N={N} sa+/-={sa_accept}/{sa_reject} "
                      f"real={real_cost_new:.4f} ACCEPT")
        else:
            incp.restore(snapshot)
            op_weights[op] = max(op_weights[op] * 0.95, 0.1)
            rejected += 1
            if verbose:
                print(f"  [LNS ep {episode}] op={op} N={N} real={real_cost_new:.4f} reject")

    if verbose:
        elapsed = time.time() - t0
        print(f"  [LNS] {accepted} accept / {rejected} reject in {elapsed:.1f}s, "
              f"start={initial_real_cost:.4f} -> best={best_cost:.4f} "
              f"({(best_cost - initial_real_cost) / initial_real_cost * 100:+.2f}%)")

    return best_positions
