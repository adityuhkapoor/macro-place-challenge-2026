"""
Aditya v3 — v1 + soft-macro co-optimization.

The only change from v1: optimize soft-macro positions alongside hard.
Soft macros are stdcell clusters; they affect HPWL, density, and congestion
even though they are allowed to overlap each other. Cheng et al. ISPD'23
attribute Cadence CMP's lead to concurrent macro+stdcell placement, and
SETUP.md explicitly warns that "moving hard macros without repositioning
soft macros will degrade wirelength and density."

Pipeline:
  1. Gradient-based global placement on (hard ∪ soft) macros.
       - Smooth HPWL via WA model.
       - Bell density penalty over hard+soft footprints (top-10% bin overflow).
       - Boundary penalty for both (must stay in canvas).
  2. Legalize hard macros only (soft allowed to overlap).
  3. Swap refinement on hard macros.

Validate against TILOS at end.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from placer import (  # noqa: E402
    _build_net_index,
    _wa_wirelength,
    _density_penalty,
    _boundary_penalty,
    _legalize,
    _swap_refine,
    _build_pair_edges_from_nets,
)


class AdityaPlacerV3:
    """Analytical (hard+soft co-optimization) + legalization + swap refinement."""

    def __init__(self,
                 seed: int = 42,
                 global_iters: int = 600,
                 swap_iters: int = 6000,
                 grid_size: int = 64,
                 lr_start: float = 0.05,
                 verbose: bool = False):
        self.seed = seed
        self.global_iters = global_iters
        self.swap_iters = swap_iters
        self.grid_size = grid_size
        self.lr_start = lr_start
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        device = "cpu"

        n_hard = benchmark.num_hard_macros
        n_total = benchmark.num_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        scale = max(cw, ch)

        sizes_full = benchmark.macro_sizes.to(device)
        movable_full = benchmark.get_movable_mask().to(device)

        n_ports = benchmark.port_positions.shape[0]
        n_all = n_total + n_ports

        port_pos = (benchmark.port_positions.to(device) if n_ports > 0
                    else torch.zeros(0, 2, device=device))

        # Initial positions
        init_pos = benchmark.macro_positions.to(device).clone()  # [n_total, 2]
        # Center hard macros around mid with jitter; leave soft at given positions
        movable_hard_only = movable_full & benchmark.get_hard_macro_mask().to(device)
        if movable_hard_only.any():
            init_pos[movable_hard_only] = torch.tensor(
                [cw / 2, ch / 2], device=device, dtype=init_pos.dtype
            )
            jitter = (torch.rand_like(init_pos[movable_hard_only]) - 0.5) * (scale * 0.1)
            init_pos[movable_hard_only] = init_pos[movable_hard_only] + jitter

        all_var = torch.nn.Parameter(init_pos.clone())  # [n_total, 2]

        net_idx, net_mask, num_nets = _build_net_index(benchmark.net_nodes, n_all)
        if net_idx is not None:
            net_idx = net_idx.to(device)
            net_mask = net_mask.to(device)

        opt = torch.optim.Adam([all_var], lr=self.lr_start * scale * 0.01)
        gamma = 0.01 * scale

        lambda_d_start = 0.0001
        lambda_d_end = 0.5

        t0 = time.time()
        if num_nets is None or num_nets == 0:
            pass
        else:
            for step in range(self.global_iters):
                opt.zero_grad()
                # Build full position tensor with fixed mask applied
                cur_pos = all_var
                if (~movable_full).any():
                    cur_pos = torch.where(movable_full.unsqueeze(1), cur_pos, init_pos)

                pieces = [cur_pos]
                if n_ports > 0:
                    pieces.append(port_pos)
                pieces.append(torch.zeros(1, 2, device=device, dtype=cur_pos.dtype))
                all_pos = torch.cat(pieces, dim=0)

                wl = _wa_wirelength(all_pos, net_idx, net_mask, gamma)

                # Density over BOTH hard and soft (they all consume area)
                den = _density_penalty(cur_pos, sizes_full, cw, ch,
                                       self.grid_size, self.grid_size)
                bnd = _boundary_penalty(cur_pos, sizes_full, cw, ch)

                progress = step / max(1, self.global_iters - 1)
                lam = lambda_d_start + (lambda_d_end - lambda_d_start) * \
                      (1 - math.cos(math.pi * progress)) / 2

                loss = wl + lam * den + 1.0 * bnd

                if not torch.isfinite(loss):
                    if self.verbose and step % 50 == 0:
                        print(f"[v3] step {step}: non-finite loss, skipping")
                    opt.zero_grad()
                    continue

                loss.backward()
                if (~movable_full).any() and all_var.grad is not None:
                    all_var.grad[~movable_full] = 0.0
                if all_var.grad is not None:
                    bad = ~torch.isfinite(all_var.grad)
                    if bad.any():
                        all_var.grad[bad] = 0.0
                opt.step()

                if self.verbose and step % 100 == 0:
                    print(f"[v3] step {step:4d}  wl={float(wl):.3f}  "
                          f"den={float(den):.3e}  bnd={float(bnd):.3e}  lam={lam:.3f}")

        if self.verbose:
            print(f"[v3] global placement done in {time.time()-t0:.2f}s")

        # ----- Legalize hard macros only -----
        global_pos = all_var.detach().cpu().numpy().astype(np.float64)
        hard_pos_np = global_pos[:n_hard].copy()
        sizes_hard_np = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
        movable_hard_np = (movable_full[:n_hard]).cpu().numpy()

        legal_hard = _legalize(hard_pos_np, movable_hard_np, sizes_hard_np, cw, ch)

        # ----- Swap refinement on hard -----
        edges, edge_weights = _build_pair_edges_from_nets(benchmark.net_nodes, n_hard)
        refined_hard = _swap_refine(legal_hard, edges, edge_weights,
                                    movable_hard_np, sizes_hard_np,
                                    cw, ch, self.swap_iters)

        # ----- Assemble: refined hard + optimized soft -----
        full_np = global_pos.copy()
        full_np[:n_hard] = refined_hard
        # Soft positions stay where global optimization placed them
        # (clip to canvas just in case)
        for i in range(n_hard, n_total):
            hw = sizes_full[i, 0].item() / 2
            hh = sizes_full[i, 1].item() / 2
            full_np[i, 0] = float(np.clip(full_np[i, 0], hw, cw - hw))
            full_np[i, 1] = float(np.clip(full_np[i, 1], hh, ch - hh))

        return torch.tensor(full_np, dtype=torch.float32)
