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


def basin_hop(positions: torch.Tensor,
              benchmark: Benchmark,
              plc,
              time_budget: float = 300.0,
              n_episodes: int = 200,
              perturb_frac: float = 0.10,    # canvas fraction of large jump
              relax_steps: int = 30,         # local relax: K small steps after jump
              relax_step_frac: float = 0.005,
              T_init_frac: float = 0.02,     # Metropolis temperature on real proxy
              seed: int = 0,
              verbose: bool = False) -> torch.Tensor:
    """
    Basin hopping (Wales/Doye 1997): perturb single macro with a large jump,
    locally relax with K small greedy steps, accept on Metropolis criterion
    against real TILOS proxy. The local relaxation is the key over plain SA —
    each move lands in a true local minimum of the surrounding basin.
    """
    rng = random.Random(seed)
    n_hard = benchmark.num_hard_macros
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_idx = np.where(movable)[0].tolist()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    scale = max(cw, ch)

    orig_positions = benchmark.macro_positions.clone()
    benchmark.macro_positions = positions.clone()
    try:
        incp = IncrementalProxy(benchmark)
    finally:
        benchmark.macro_positions = orig_positions

    cur_real = compute_proxy_cost(positions, benchmark, plc)["proxy_cost"]
    initial_real = cur_real
    best_real = cur_real
    best_pos = positions.clone()

    T = T_init_frac * cur_real

    t0 = time.time()
    accepted = 0
    rejected = 0

    for episode in range(n_episodes):
        if time.time() - t0 > time_budget:
            break
        if not movable_idx:
            break

        i = rng.choice(movable_idx)
        old_x = float(incp.positions[i, 0]); old_y = float(incp.positions[i, 1])
        hw = float(incp.sizes[i, 0] / 2); hh = float(incp.sizes[i, 1] / 2)

        # Snapshot
        snap = incp.snapshot()

        # Large jump
        jump_x, jump_y = _propose_move(rng, old_x, old_y, hw, hh, cw, ch,
                                        scale * perturb_frac)
        if incp.positions[i, 0] != jump_x or incp.positions[i, 1] != jump_y:
            incp.positions[i, 0] = jump_x; incp.positions[i, 1] = jump_y
            if incp.overlaps_any(i):
                incp.positions[i, 0] = old_x; incp.positions[i, 1] = old_y
                rejected += 1
                continue
            incp.positions[i, 0] = old_x; incp.positions[i, 1] = old_y
            incp.move_macro(i, jump_x, jump_y)

        # Local relaxation: K small greedy steps
        relax_step = scale * relax_step_frac
        cur_proxy = incp.proxy()
        for _ in range(relax_steps):
            cx = float(incp.positions[i, 0]); cy = float(incp.positions[i, 1])
            nx, ny = _propose_move(rng, cx, cy, hw, hh, cw, ch, relax_step)
            old_pos = (cx, cy)
            incp.positions[i, 0] = nx; incp.positions[i, 1] = ny
            if incp.overlaps_any(i):
                incp.positions[i, 0] = old_pos[0]; incp.positions[i, 1] = old_pos[1]
                continue
            incp.positions[i, 0] = old_pos[0]; incp.positions[i, 1] = old_pos[1]
            incp.move_macro(i, nx, ny)
            new_proxy = incp.proxy()
            if new_proxy < cur_proxy:
                cur_proxy = new_proxy
            else:
                incp.move_macro(i, cx, cy)

        # Validate against TILOS, Metropolis on real proxy
        new_positions = positions.clone()
        new_positions[:n_hard, 0] = torch.from_numpy(incp.positions[:n_hard, 0]).float()
        new_positions[:n_hard, 1] = torch.from_numpy(incp.positions[:n_hard, 1]).float()
        new_real = compute_proxy_cost(new_positions, benchmark, plc)["proxy_cost"]

        delta = new_real - cur_real
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            cur_real = new_real
            if new_real < best_real:
                best_real = new_real
                best_pos = new_positions.clone()
                positions = new_positions
            accepted += 1
        else:
            incp.restore(snap)
            rejected += 1

    if verbose:
        elapsed = time.time() - t0
        gain = (best_real - initial_real) / initial_real * 100
        print(f"  [BH] {accepted} acc / {rejected} rej in {elapsed:.1f}s, "
              f"start={initial_real:.4f} → best={best_real:.4f} ({gain:+.2f}%)")

    return best_pos


def parallel_tempering(positions: torch.Tensor,
                       benchmark: Benchmark,
                       plc,
                       n_chains: int = 4,
                       T_min_frac: float = 0.001,
                       T_max_frac: float = 0.05,
                       n_swap_rounds: int = 20,
                       sa_steps_per_round: int = 2000,
                       step_init_frac: float = 0.05,
                       step_end_frac: float = 0.0005,
                       time_budget: float = 600.0,
                       seed: int = 0,
                       verbose: bool = False) -> torch.Tensor:
    """
    Parallel-tempering / replica-exchange wrapper around single-macro SA.

    Run `n_chains` chains in round-robin, each at its own fixed temperature
    on a geometric ladder. Hot chains tunnel barriers, cold chains refine.
    Between rounds, propose Metropolis swaps of full configurations between
    adjacent chains using the canonical PT criterion:
        P(swap i,j) = min(1, exp((1/T_i - 1/T_j) * (E_i - E_j)))

    Best position is tracked on cold chain via real TILOS proxy (validated
    each round). Step size anneals geometrically across rounds (shared across
    all chains; their differing T's still produce different acceptance rates).

    `positions`: [num_macros, 2] post-legalize tensor.
    Returns best positions found across all chains.
    """
    rng = random.Random(seed)
    n_hard = benchmark.num_hard_macros
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    movable_idx = np.where(movable)[0].tolist()
    if not movable_idx:
        return positions
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    scale = max(cw, ch)

    # Build n_chains IncrementalProxy instances, each from the same positions.
    orig_positions = benchmark.macro_positions.clone()
    benchmark.macro_positions = positions.clone()
    try:
        chains = [IncrementalProxy(benchmark) for _ in range(n_chains)]
    finally:
        benchmark.macro_positions = orig_positions

    # T ladder geometric from T_min to T_max, anchored to initial proxy.
    initial_proxy = chains[0].proxy()
    T_min = T_min_frac * initial_proxy
    T_max = T_max_frac * initial_proxy
    if n_chains == 1:
        T_ladder = [T_min]
    else:
        ratio = (T_max / T_min) ** (1.0 / (n_chains - 1))
        T_ladder = [T_min * (ratio ** k) for k in range(n_chains)]

    # Tracking
    initial_real = compute_proxy_cost(positions, benchmark, plc)["proxy_cost"]
    best_real = initial_real
    best_positions = positions.clone()
    sa_accepts = [0] * n_chains
    sa_rejects = [0] * n_chains
    swap_attempts = 0
    swap_accepts = 0

    if verbose:
        print(f"  [PT] init real={initial_real:.4f} T_ladder={[f'{t:.4f}' for t in T_ladder]}")

    t0 = time.time()
    step0 = scale * step_init_frac
    step_end = scale * step_end_frac
    if n_swap_rounds > 1:
        step_decay_per_round = (step_end / step0) ** (1.0 / (n_swap_rounds - 1))
    else:
        step_decay_per_round = 1.0
    step = step0

    for round_idx in range(n_swap_rounds):
        if time.time() - t0 > time_budget:
            if verbose:
                print(f"  [PT] time budget hit at round {round_idx}")
            break

        # Advance each chain at its own T
        for c, incp in enumerate(chains):
            T = T_ladder[c]
            for sa_step in range(sa_steps_per_round):
                i = rng.choice(movable_idx)
                old_x = float(incp.positions[i, 0])
                old_y = float(incp.positions[i, 1])
                hw = float(incp.sizes[i, 0] / 2)
                hh = float(incp.sizes[i, 1] / 2)
                nx, ny = _propose_move(rng, old_x, old_y, hw, hh, cw, ch, step)
                # Quick overlap check
                incp.positions[i, 0] = nx
                incp.positions[i, 1] = ny
                if incp.overlaps_any(i):
                    incp.positions[i, 0] = old_x
                    incp.positions[i, 1] = old_y
                    sa_rejects[c] += 1
                    continue
                # Restore for proper move_macro semantics
                incp.positions[i, 0] = old_x
                incp.positions[i, 1] = old_y
                old_proxy = incp.proxy()
                incp.move_macro(i, nx, ny)
                new_proxy = incp.proxy()
                delta = new_proxy - old_proxy
                if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                    sa_accepts[c] += 1
                else:
                    incp.move_macro(i, old_x, old_y)
                    sa_rejects[c] += 1

        # Validate cold chain (chain 0) against TILOS proxy and update best.
        cold = chains[0]
        new_pos = positions.clone()
        new_pos[:n_hard, 0] = torch.from_numpy(cold.positions[:n_hard, 0]).float()
        new_pos[:n_hard, 1] = torch.from_numpy(cold.positions[:n_hard, 1]).float()
        real_now = compute_proxy_cost(new_pos, benchmark, plc)["proxy_cost"]
        if real_now < best_real:
            best_real = real_now
            best_positions = new_pos.clone()

        # Try adjacent swaps (alternate even/odd pairs each round).
        # Standard PT: P_accept = min(1, exp((β_i - β_j) * (E_i - E_j)))
        # where β = 1/T. With i hotter (lower β), j colder (higher β):
        #   exp((β_j - β_i) * (E_j - E_i))  reduces to standard form.
        pair_offset = round_idx % 2
        for c in range(pair_offset, n_chains - 1, 2):
            E_lo = chains[c].proxy()
            E_hi = chains[c + 1].proxy()
            beta_lo = 1.0 / max(T_ladder[c], 1e-12)
            beta_hi = 1.0 / max(T_ladder[c + 1], 1e-12)
            log_p = (beta_lo - beta_hi) * (E_lo - E_hi)
            swap_attempts += 1
            if log_p >= 0 or rng.random() < math.exp(log_p):
                # Swap full configurations: snapshot then restore on swapped target
                snap_lo = chains[c].snapshot()
                snap_hi = chains[c + 1].snapshot()
                chains[c].restore(snap_hi)
                chains[c + 1].restore(snap_lo)
                swap_accepts += 1

        if verbose:
            tot_acc = sum(sa_accepts)
            tot_rej = sum(sa_rejects)
            chain_proxies = [f"{c.proxy():.4f}" for c in chains]
            print(f"  [PT round {round_idx}] step={step:.1f} sa+/-={tot_acc}/{tot_rej} "
                  f"swap+/-={swap_accepts}/{swap_attempts} "
                  f"chains_proxy={chain_proxies} cold_real={real_now:.4f} "
                  f"best={best_real:.4f}")

        step *= step_decay_per_round

    if verbose:
        elapsed = time.time() - t0
        improv = (best_real - initial_real) / initial_real * 100
        print(f"  [PT] DONE in {elapsed:.1f}s | start={initial_real:.4f} "
              f"-> best={best_real:.4f} ({improv:+.2f}%) | swaps={swap_accepts}/{swap_attempts}")

    return best_positions


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
