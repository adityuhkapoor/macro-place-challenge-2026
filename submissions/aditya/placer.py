"""
Aditya v1 — Analytical Global Placement + Legalization + Swap Refinement

Pipeline:
  1. Gradient-based global placement on hard macros only.
       - Smooth HPWL via weighted-average (WA) wirelength model.
       - Density penalty via gaussian-bell macro footprints on a grid (ePlace-lite).
       - Adam optimizer with density-weight ramp schedule.
  2. Legalization via radial search (greedy minimum-displacement, reused from will_seed).
  3. Pairwise swap refinement on full HPWL.
  4. Soft macros kept at initial positions.

Usage:
    uv run evaluate submissions/aditya/placer.py -b ibm01
    uv run evaluate submissions/aditya/placer.py --all
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark


def _build_net_index(net_nodes: List[torch.Tensor], num_movable: int):
    """
    Pad nets into a [num_nets, max_pins] index tensor + mask.
    Out-of-net positions get index = num_movable (sentinel) so we can mask later.
    Only keeps nets with >= 2 pins.
    """
    keep = [net for net in net_nodes if net.numel() >= 2]
    if not keep:
        return None, None, None
    max_pins = max(net.numel() for net in keep)
    sentinel = num_movable
    idx = torch.full((len(keep), max_pins), sentinel, dtype=torch.long)
    mask = torch.zeros((len(keep), max_pins), dtype=torch.bool)
    for i, net in enumerate(keep):
        n = net.numel()
        idx[i, :n] = net
        mask[i, :n] = True
    return idx, mask, len(keep)


def _wa_wirelength(positions_with_sentinel: torch.Tensor,
                   net_idx: torch.Tensor, net_mask: torch.Tensor,
                   gamma: float) -> torch.Tensor:
    """
    Weighted-Average smooth HPWL (ePlace WA model). Numerically stabilized:
    - Subtract per-net max/min before exp (so largest exponent is 0)
    - Mask masked-pin positions out of pins entirely (replace with pin_max for
      e_pos branch, pin_min for e_neg branch) so they contribute exp(0)*0=0
      with no risk of underflow producing 0/0 NaNs.
    """
    pins = positions_with_sentinel[net_idx]  # [num_nets, max_pins, 2]
    mask_b = net_mask.unsqueeze(-1)  # [num_nets, max_pins, 1] bool
    mask_f = mask_b.float()

    very_neg = torch.tensor(-1e9, device=pins.device, dtype=pins.dtype)
    very_pos = torch.tensor(1e9, device=pins.device, dtype=pins.dtype)

    pins_for_max = torch.where(mask_b, pins, very_neg)
    pins_for_min = torch.where(mask_b, pins, very_pos)
    pin_max = pins_for_max.max(dim=1, keepdim=True).values  # [num_nets, 1, 2]
    pin_min = pins_for_min.min(dim=1, keepdim=True).values  # [num_nets, 1, 2]

    # For exponent computation, replace masked pin positions with pin_max
    # (so exp((pin_max - pin_max)/gamma) = exp(0) = 1; masked off) — but since
    # we multiply by mask we don't care what they contribute. The risk is
    # (pins - pin_max) being a huge negative number for sentinel pins at (0,0)
    # when pin_max is large; exp underflows to 0, fine. The real risk is in
    # the (pins * e_pos) numerator: pins=0 * 0 = 0 OK.
    pins_safe_max = torch.where(mask_b, pins, pin_max.expand_as(pins))
    pins_safe_min = torch.where(mask_b, pins, pin_min.expand_as(pins))

    e_pos = torch.exp((pins_safe_max - pin_max) / gamma) * mask_f
    e_neg = torch.exp(-(pins_safe_min - pin_min) / gamma) * mask_f
    eps = 1e-12
    wa_max = (pins_safe_max * e_pos).sum(dim=1) / (e_pos.sum(dim=1) + eps)
    wa_min = (pins_safe_min * e_neg).sum(dim=1) / (e_neg.sum(dim=1) + eps)
    hpwl = (wa_max - wa_min).sum(dim=1)  # [num_nets] (sum of x and y)
    return hpwl.sum()


def _density_penalty(pos: torch.Tensor, sizes: torch.Tensor,
                     canvas_w: float, canvas_h: float,
                     grid_x: int, grid_y: int) -> torch.Tensor:
    """
    Bell-shaped density: each macro contributes a gaussian footprint to bins.
    Returns squared overflow above target density (sum across bins).

    pos: [N, 2] hard macro centers
    sizes: [N, 2] hard macro widths/heights
    """
    bin_w = canvas_w / grid_x
    bin_h = canvas_h / grid_y
    # Bin centers
    bx = (torch.arange(grid_x, device=pos.device, dtype=pos.dtype) + 0.5) * bin_w
    by = (torch.arange(grid_y, device=pos.device, dtype=pos.dtype) + 0.5) * bin_h

    # Macro half-sizes
    hw = sizes[:, 0] / 2  # [N]
    hh = sizes[:, 1] / 2

    # Sigma chosen so footprint roughly matches macro size
    sigma_x = (hw + bin_w) * 0.5  # [N]
    sigma_y = (hh + bin_h) * 0.5  # [N]

    # X-axis bell: contribution from each macro to each x-bin
    # [N, grid_x]
    dx = bx.unsqueeze(0) - pos[:, 0:1]
    bell_x = torch.exp(-(dx * dx) / (2 * sigma_x.unsqueeze(1) ** 2))
    dy = by.unsqueeze(0) - pos[:, 1:2]
    bell_y = torch.exp(-(dy * dy) / (2 * sigma_y.unsqueeze(1) ** 2))

    # Normalize so each macro's total contribution = its area
    norm_x = bell_x.sum(dim=1, keepdim=True) + 1e-12
    norm_y = bell_y.sum(dim=1, keepdim=True) + 1e-12
    area = sizes[:, 0] * sizes[:, 1]  # [N]
    weight = area / (norm_x.squeeze(1) * norm_y.squeeze(1) + 1e-12)  # [N]

    # Aggregate density into [grid_y, grid_x]: (bell_y[N, gy] * weight) @ bell_x[N, gx]
    bell_y_w = bell_y * weight.unsqueeze(1)  # [N, grid_y]
    density = bell_y_w.t() @ bell_x  # [grid_y, grid_x]

    bin_area = bin_w * bin_h
    target = area.sum() / (canvas_w * canvas_h)  # macro area fraction
    target_per_bin = target * bin_area
    overflow = F.relu(density - target_per_bin)
    return (overflow * overflow).sum()


def _rudy_penalty(positions_with_sentinel: torch.Tensor,
                  net_idx: torch.Tensor, net_mask: torch.Tensor,
                  canvas_w: float, canvas_h: float,
                  grid_x: int, grid_y: int,
                  gamma: float) -> torch.Tensor:
    """
    Differentiable RUDY routing demand. For each net:
        W, H = smooth bbox dims (WA)
        demand = (W+H) / (W*H+eps)
    Spread demand over bins covered by the net's bbox (sigmoid coverage).
    Penalty = sum of squared bin RUDY (heavier penalty on hot bins).
    """
    pins = positions_with_sentinel[net_idx]  # [num_nets, max_pins, 2]
    mask = net_mask.unsqueeze(-1).float()

    very_neg = torch.tensor(-1e9, dtype=pins.dtype, device=pins.device)
    very_pos = torch.tensor(1e9, dtype=pins.dtype, device=pins.device)
    pins_for_max = torch.where(mask.bool(), pins, very_neg)
    pins_for_min = torch.where(mask.bool(), pins, very_pos)

    pin_max_seed = pins_for_max.max(dim=1, keepdim=True).values
    pin_min_seed = pins_for_min.min(dim=1, keepdim=True).values

    e_pos = torch.exp((pins - pin_max_seed) / gamma) * mask
    e_neg = torch.exp(-(pins - pin_min_seed) / gamma) * mask
    eps = 1e-9
    wa_max = (pins * e_pos).sum(dim=1) / (e_pos.sum(dim=1) + eps)
    wa_min = (pins * e_neg).sum(dim=1) / (e_neg.sum(dim=1) + eps)

    W = (wa_max[:, 0] - wa_min[:, 0]).clamp(min=eps)  # [num_nets]
    H = (wa_max[:, 1] - wa_min[:, 1]).clamp(min=eps)
    demand = (W + H) / (W * H + eps)  # [num_nets]
    # Cap demand to avoid singular contributions for very small bboxes
    demand = demand.clamp(max=10.0 / max(canvas_w, canvas_h) * 100.0)

    bin_w = canvas_w / grid_x
    bin_h = canvas_h / grid_y
    bx = (torch.arange(grid_x, dtype=pins.dtype, device=pins.device) + 0.5) * bin_w
    by = (torch.arange(grid_y, dtype=pins.dtype, device=pins.device) + 0.5) * bin_h

    T = bin_w * 0.5  # smoothing temperature
    cov_x = (torch.sigmoid((bx.unsqueeze(0) - wa_min[:, 0:1]) / T) -
             torch.sigmoid((bx.unsqueeze(0) - wa_max[:, 0:1]) / T))  # [num_nets, grid_x]
    cov_y = (torch.sigmoid((by.unsqueeze(0) - wa_min[:, 1:2]) / T) -
             torch.sigmoid((by.unsqueeze(0) - wa_max[:, 1:2]) / T))  # [num_nets, grid_y]

    weighted_cov_y = cov_y * demand.unsqueeze(1)  # [num_nets, grid_y]
    rudy = weighted_cov_y.t() @ cov_x  # [grid_y, grid_x]

    # Smooth top-k via squared-overflow penalty above mean
    target = rudy.mean().detach() * 1.5  # encourage uniformity
    overflow = F.relu(rudy - target)
    return (overflow * overflow).sum()


def _boundary_penalty(pos: torch.Tensor, sizes: torch.Tensor,
                      canvas_w: float, canvas_h: float) -> torch.Tensor:
    """Penalty for macros sticking outside canvas (smooth)."""
    hw = sizes[:, 0] / 2
    hh = sizes[:, 1] / 2
    left = F.relu(hw - pos[:, 0])
    right = F.relu(pos[:, 0] - (canvas_w - hw))
    bot = F.relu(hh - pos[:, 1])
    top = F.relu(pos[:, 1] - (canvas_h - hh))
    return (left * left + right * right + bot * bot + top * top).sum()


def _legalize(pos: np.ndarray, movable: np.ndarray, sizes: np.ndarray,
              canvas_w: float, canvas_h: float) -> np.ndarray:
    """
    Greedy minimum-displacement legalization.
    Place macros in order of decreasing area; for each, search outward in a grid
    until a non-overlapping spot is found. Reused approach from will_seed.
    """
    n = len(pos)
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

    order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()

    # Fixed macros: use original positions, mark placed
    for idx in range(n):
        legal[idx, 0] = np.clip(legal[idx, 0], half_w[idx], canvas_w - half_w[idx])
        legal[idx, 1] = np.clip(legal[idx, 1], half_h[idx], canvas_h - half_h[idx])
        if not movable[idx]:
            placed[idx] = True

    gap = 0.05
    for idx in order:
        if not movable[idx]:
            continue
        # Quick-accept current position
        if placed.any():
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            c = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap) & placed
            c[idx] = False
            if not c.any():
                placed[idx] = True
                continue

        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
        best_p = legal[idx].copy()
        best_d = float("inf")
        for r in range(1, 200):
            found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = np.clip(pos[idx, 0] + dxm * step, half_w[idx], canvas_w - half_w[idx])
                    cy = np.clip(pos[idx, 1] + dym * step, half_h[idx], canvas_h - half_h[idx])
                    if placed.any():
                        dx = np.abs(cx - legal[:, 0])
                        dy = np.abs(cy - legal[:, 1])
                        c = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap) & placed
                        c[idx] = False
                        if c.any():
                            continue
                    d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                    if d < best_d:
                        best_d = d
                        best_p = np.array([cx, cy])
                        found = True
            if found:
                break
        legal[idx] = best_p
        placed[idx] = True
    return legal


def _swap_refine(pos: np.ndarray, edges: np.ndarray, edge_weights: np.ndarray,
                 movable: np.ndarray, sizes: np.ndarray,
                 canvas_w: float, canvas_h: float, iters: int) -> np.ndarray:
    """
    Pairwise swap + small-shift SA refinement. Wirelength proxy on macro-pair edges.
    O(N) overlap check per move using broadcasting.
    """
    n = len(pos)
    if iters <= 0 or len(edges) == 0:
        return pos
    pos = pos.copy()
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

    movable_idx = np.where(movable)[0]
    if len(movable_idx) == 0:
        return pos

    # Adjacency by edges (for "swap a connected neighbor" moves)
    neighbors: List[List[int]] = [[] for _ in range(n)]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)

    def wl_cost():
        dx = np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0])
        dy = np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
        return float((edge_weights * (dx + dy)).sum())

    gap = 0.05

    def overlaps_any(idx):
        dx = np.abs(pos[idx, 0] - pos[:, 0])
        dy = np.abs(pos[idx, 1] - pos[:, 1])
        c = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap)
        c[idx] = False
        return bool(c.any())

    cur = wl_cost()
    best = cur
    best_pos = pos.copy()
    T_start = max(canvas_w, canvas_h) * 0.10
    T_end = max(canvas_w, canvas_h) * 0.001

    rng = random.Random(0)
    for step in range(iters):
        frac = step / iters
        T = T_start * (T_end / T_start) ** frac
        i = rng.choice(movable_idx.tolist())
        old_x, old_y = pos[i, 0], pos[i, 1]
        r = rng.random()

        if r < 0.55:
            # Gaussian shift
            sigma = T * (0.4 + 0.6 * (1 - frac))
            pos[i, 0] = np.clip(old_x + rng.gauss(0, sigma), half_w[i], canvas_w - half_w[i])
            pos[i, 1] = np.clip(old_y + rng.gauss(0, sigma), half_h[i], canvas_h - half_h[i])
            if overlaps_any(i):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                continue
        elif r < 0.85:
            # Swap with neighbor or random
            if neighbors[i] and rng.random() < 0.7:
                cands = [j for j in neighbors[i] if movable[j] and j != i]
                j = rng.choice(cands) if cands else rng.choice(movable_idx.tolist())
            else:
                j = rng.choice(movable_idx.tolist())
            if j == i:
                continue
            old_jx, old_jy = pos[j, 0], pos[j, 1]
            pos[i, 0] = np.clip(old_jx, half_w[i], canvas_w - half_w[i])
            pos[i, 1] = np.clip(old_jy, half_h[i], canvas_h - half_h[i])
            pos[j, 0] = np.clip(old_x, half_w[j], canvas_w - half_w[j])
            pos[j, 1] = np.clip(old_y, half_h[j], canvas_h - half_h[j])
            if overlaps_any(i) or overlaps_any(j):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                pos[j, 0] = old_jx; pos[j, 1] = old_jy
                continue
        else:
            # Move toward connected neighbor centroid
            if not neighbors[i]:
                continue
            cx = float(np.mean([pos[j, 0] for j in neighbors[i]]))
            cy = float(np.mean([pos[j, 1] for j in neighbors[i]]))
            alpha = rng.uniform(0.05, 0.3)
            pos[i, 0] = np.clip(old_x + alpha * (cx - old_x), half_w[i], canvas_w - half_w[i])
            pos[i, 1] = np.clip(old_y + alpha * (cy - old_y), half_h[i], canvas_h - half_h[i])
            if overlaps_any(i):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                continue

        new = wl_cost()
        delta = new - cur
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            cur = new
            if cur < best:
                best = cur
                best_pos = pos.copy()
        else:
            # revert
            pos[i, 0] = old_x; pos[i, 1] = old_y
            # j was modified only in swap branch; in that branch we already reverted on overlap
            # Non-overlap-rejected swap: must revert j too
            # (We re-run a quick check: if j modified, restore.)
            # Simplification: always set j back if it exists
            if r >= 0.55 and r < 0.85:
                pos[j, 0] = old_jx; pos[j, 1] = old_jy

    return best_pos


def _build_pair_edges_from_nets(net_nodes: List[torch.Tensor], num_hard: int):
    """
    Convert hyperedges to weighted pairwise edges (clique decomposition).
    Each net of k macros contributes C(k,2) pairs each with weight 1/(k-1).
    Only macros with index < num_hard.
    """
    edge_dict = {}
    for net in net_nodes:
        ids = [int(x) for x in net.tolist() if int(x) < num_hard]
        if len(ids) < 2:
            continue
        ids_sorted = sorted(set(ids))
        if len(ids_sorted) < 2:
            continue
        w = 1.0 / (len(ids_sorted) - 1)
        for i in range(len(ids_sorted)):
            for j in range(i + 1, len(ids_sorted)):
                pair = (ids_sorted[i], ids_sorted[j])
                edge_dict[pair] = edge_dict.get(pair, 0.0) + w
    if not edge_dict:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    edges = np.array(list(edge_dict.keys()), dtype=np.int64)
    weights = np.array([edge_dict[e] for e in edge_dict], dtype=np.float64)
    return edges, weights


class AdityaPlacer:
    """Analytical global placement + legalization + swap refinement."""

    def __init__(self,
                 seed: int = 42,
                 global_iters: int = 600,
                 swap_iters: int = 6000,
                 grid_size: int = 64,
                 lr_start: float = 0.05,
                 lambda_rudy: float = 0.0,
                 verbose: bool = False):
        self.seed = seed
        self.global_iters = global_iters
        self.swap_iters = swap_iters
        self.grid_size = grid_size
        self.lr_start = lr_start
        self.lambda_rudy = lambda_rudy
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        device = "cpu"  # CPU is plenty fast at this scale and avoids surprises

        n_hard = benchmark.num_hard_macros
        n_total = benchmark.num_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        scale = max(cw, ch)

        sizes_full = benchmark.macro_sizes.to(device)  # [N, 2]
        sizes_hard = sizes_full[:n_hard]
        movable = benchmark.get_movable_mask().to(device)
        movable_hard = movable[:n_hard]

        # ----- Build "all positions" tensor with hard (variable) + soft + ports (fixed) -----
        # net_nodes index space: [0, n_hard) hard, [n_hard, n_total) soft, [n_total, n_total+n_ports) ports
        n_ports = benchmark.port_positions.shape[0]
        n_all = n_total + n_ports

        soft_pos = benchmark.macro_positions[n_hard:n_total].to(device)  # fixed
        port_pos = benchmark.port_positions.to(device) if n_ports > 0 else torch.zeros(0, 2, device=device)

        # Initial hard positions (perturb to break symmetry slightly if all overlapping at center)
        init_hard = benchmark.macro_positions[:n_hard].to(device).clone()
        # Center movable hard macros around canvas mid for a clean optimization start
        if movable_hard.any():
            init_hard[movable_hard] = torch.tensor(
                [cw / 2, ch / 2], device=device, dtype=init_hard.dtype
            )
            # small random perturbation
            jitter = (torch.rand_like(init_hard[movable_hard]) - 0.5) * (scale * 0.1)
            init_hard[movable_hard] = init_hard[movable_hard] + jitter

        hard_var = torch.nn.Parameter(init_hard.clone())

        # Build padded net index tensor over the FULL universe of n_all + 1 (sentinel last)
        net_idx, net_mask, num_nets = _build_net_index(benchmark.net_nodes, n_all)
        if net_idx is not None:
            net_idx = net_idx.to(device)
            net_mask = net_mask.to(device)

        # Adam optimizer
        opt = torch.optim.Adam([hard_var], lr=self.lr_start * scale * 0.01)

        # Gamma for WA (smaller = closer to true HPWL but stiffer gradients)
        gamma = 0.01 * scale

        # Density-weight schedule (low → high)
        lambda_d_start = 0.0001
        lambda_d_end = 0.5

        t0 = time.time()
        if num_nets is None or num_nets == 0:
            if self.verbose:
                print("[aditya] no nets, skipping global placement")
        else:
            for step in range(self.global_iters):
                opt.zero_grad()
                # Build full position tensor: hard (variable) + soft (fixed) + ports (fixed) + sentinel
                hard_p = hard_var
                # Apply fixed-mask: fixed hard macros stay at their original spot
                if (~movable_hard).any():
                    hard_p = torch.where(movable_hard.unsqueeze(1), hard_p, init_hard)
                pieces = [hard_p, soft_pos]
                if n_ports > 0:
                    pieces.append(port_pos)
                pieces.append(torch.zeros(1, 2, device=device, dtype=hard_p.dtype))  # sentinel
                all_pos = torch.cat(pieces, dim=0)  # [n_all + 1, 2]

                wl = _wa_wirelength(all_pos, net_idx, net_mask, gamma)

                den = _density_penalty(hard_p, sizes_hard, cw, ch,
                                       self.grid_size, self.grid_size)
                bnd = _boundary_penalty(hard_p, sizes_hard, cw, ch)

                # Schedule density weight (cosine ramp)
                progress = step / max(1, self.global_iters - 1)
                lam = lambda_d_start + (lambda_d_end - lambda_d_start) * (1 - math.cos(math.pi * progress)) / 2

                # Optional RUDY congestion penalty
                if self.lambda_rudy > 0:
                    rudy = _rudy_penalty(all_pos, net_idx, net_mask,
                                         cw, ch, self.grid_size // 2, self.grid_size // 2,
                                         gamma)
                    lam_r = self.lambda_rudy * max(0.0, (progress - 0.2) / 0.8)
                else:
                    rudy = torch.tensor(0.0, device=device)
                    lam_r = 0.0

                loss = wl + lam * den + 1.0 * bnd + lam_r * rudy

                # NaN guard: if loss went bad, skip the step (keep current positions)
                if not torch.isfinite(loss):
                    if self.verbose and step % 50 == 0:
                        print(f"[aditya] step {step}: non-finite loss "
                              f"(wl={float(wl) if torch.isfinite(wl) else 'NaN'}, "
                              f"den={float(den) if torch.isfinite(den) else 'NaN'}, "
                              f"bnd={float(bnd) if torch.isfinite(bnd) else 'NaN'})")
                    opt.zero_grad()
                    continue

                loss.backward()
                # Zero gradients of fixed macros
                if (~movable_hard).any() and hard_var.grad is not None:
                    hard_var.grad[~movable_hard] = 0.0
                # Replace any non-finite gradients with zero (defensive)
                if hard_var.grad is not None:
                    bad = ~torch.isfinite(hard_var.grad)
                    if bad.any():
                        hard_var.grad[bad] = 0.0
                opt.step()

                if self.verbose and step % 100 == 0:
                    print(f"[aditya] step {step:4d}  wl={float(wl):.3f}  "
                          f"den={float(den):.3e}  bnd={float(bnd):.3e}  lam={lam:.3f}")

        if self.verbose:
            print(f"[aditya] global placement done in {time.time()-t0:.2f}s")

        # ----- Legalize -----
        global_pos = hard_var.detach().cpu().numpy().astype(np.float64)
        sizes_np = sizes_hard.cpu().numpy().astype(np.float64)
        movable_np = movable_hard.cpu().numpy()

        legal = _legalize(global_pos, movable_np, sizes_np, cw, ch)

        # ----- Swap refinement -----
        edges, edge_weights = _build_pair_edges_from_nets(benchmark.net_nodes, n_hard)
        if self.verbose:
            print(f"[aditya] {len(edges)} pair-edges; running {self.swap_iters} swap iters")
        refined = _swap_refine(legal, edges, edge_weights, movable_np, sizes_np,
                               cw, ch, self.swap_iters)

        # ----- Assemble full placement -----
        full = benchmark.macro_positions.clone()
        full[:n_hard] = torch.tensor(refined, dtype=torch.float32)
        return full
