"""
Spectral / Laplacian-eigenmap initialization for macro placement.

Idea: build the netlist hypergraph adjacency, compute Laplacian L = D - A,
take eigenvectors 2 and 3 (skipping the trivial constant eigenvector), use
them as 2D coordinates for the macros. Affine-stretch to fit canvas.

Avoids the "Cheng-Kuh collapse" (where unconstrained quadratic placement
collapses all macros to the centroid) by:
  - Using eigenmaps with proper boundary conditions
  - Anchoring with port positions (acting as boundary nodes)

Falls back to original positions if numerical issues.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from macro_place.benchmark import Benchmark


def spectral_init(benchmark: Benchmark, port_anchor_w: float = 5.0,
                  jitter_frac: float = 0.02,
                  seed: int = 0) -> Optional[torch.Tensor]:
    """
    Returns an [num_hard_macros, 2] tensor of spectral-init positions.
    Returns None if it can't be computed (small graphs, numerical fail).
    Otherwise the caller should use these as the starting position for
    movable hard macros.
    """
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.num_macros
    n_ports = benchmark.port_positions.shape[0]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    if n_hard < 4:
        return None

    # Build clique-expansion adjacency over (hard ∪ soft ∪ ports).
    # Each net contributes 1/(degree-1) weight per edge to keep mass balanced.
    n_all = n_total + n_ports
    rows = []
    cols = []
    data = []
    for nodes_t in benchmark.net_nodes:
        nodes = nodes_t.numpy().tolist()
        deg = len(nodes)
        if deg < 2 or deg > 50:  # skip high-fanout broadcast nets
            continue
        w = 1.0 / max(deg - 1, 1)
        for i in nodes:
            for j in nodes:
                if i != j:
                    rows.append(i); cols.append(j); data.append(w)

    if not rows:
        return None

    A = sp.coo_matrix((data, (rows, cols)), shape=(n_all, n_all)).tocsr()
    A = (A + A.T) * 0.5  # symmetrize
    deg = np.asarray(A.sum(axis=1)).flatten()

    # Anchor port + non-movable nodes to their fixed positions via large
    # diagonal weights, with target = current x or y (for x and y separately).
    movable_mask = benchmark.get_movable_mask().numpy()
    soft_pos_np = benchmark.macro_positions[n_hard:n_total].numpy().astype(np.float64)
    port_pos_np = benchmark.port_positions.numpy().astype(np.float64) if n_ports > 0 else np.zeros((0, 2), dtype=np.float64)
    init_hard_np = benchmark.macro_positions[:n_hard].numpy().astype(np.float64)

    # Concatenate fixed positions: [hard, soft, ports]
    fixed_pos = np.concatenate([init_hard_np, soft_pos_np, port_pos_np], axis=0)

    # Solve quadratic with fixed-position constraints (Cheng-Kuh style):
    # min sum_ij A_ij ||x_i - x_j||^2 + sum_k anchor_w (x_k - fixed_x_k)^2
    # The first sum gives Laplacian L; second adds diagonal.
    # Movable rows: free; non-movable: pinned via large anchor.
    # We solve x and y separately.
    L = sp.diags(deg) - A
    anchor_diag = np.zeros(n_all)
    # Fixed: ports + non-movable hard + soft
    anchor_diag[n_hard + 0: n_hard + (n_total - n_hard)] = port_anchor_w  # soft (treated fixed)
    anchor_diag[n_hard + (n_total - n_hard):] = port_anchor_w  # ports
    # Non-movable hard
    for i in range(n_hard):
        if not movable_mask[i]:
            anchor_diag[i] = port_anchor_w

    M = L + sp.diags(anchor_diag)
    M = M.tocsc()
    bx = anchor_diag * fixed_pos[:, 0]
    by = anchor_diag * fixed_pos[:, 1]

    try:
        x = spla.spsolve(M, bx)
        y = spla.spsolve(M, by)
    except Exception:
        return None

    # Take only hard macro coords
    hard_x = x[:n_hard]
    hard_y = y[:n_hard]

    # Sanity: clip to canvas
    hw = benchmark.macro_sizes[:n_hard, 0].numpy() / 2
    hh = benchmark.macro_sizes[:n_hard, 1].numpy() / 2
    hard_x = np.clip(hard_x, hw, cw - hw)
    hard_y = np.clip(hard_y, hh, ch - hh)

    # For non-movable macros, keep their original positions
    for i in range(n_hard):
        if not movable_mask[i]:
            hard_x[i] = init_hard_np[i, 0]
            hard_y[i] = init_hard_np[i, 1]

    # Add small jitter to break symmetry
    rng = np.random.RandomState(seed)
    if jitter_frac > 0:
        jit_x = rng.uniform(-jitter_frac, jitter_frac, n_hard) * cw
        jit_y = rng.uniform(-jitter_frac, jitter_frac, n_hard) * ch
        for i in range(n_hard):
            if movable_mask[i]:
                hard_x[i] = np.clip(hard_x[i] + jit_x[i], hw[i], cw - hw[i])
                hard_y[i] = np.clip(hard_y[i] + jit_y[i], hh[i], ch - hh[i])

    return torch.from_numpy(np.stack([hard_x, hard_y], axis=1)).float()
