"""
Aditya v4 — Analytical macro placer.

Pipeline:
  1. Gradient-based global placement on hard-macro centers (soft macros fixed).
  2. Radial-search legalization.
  3. (Optional) pairwise-swap SA refinement on weighted pair edges. Empirically
     net-negative on TILOS proxy, default disabled (swap_iters=0).

Loss components:
  * Wirelength: WA (weighted-average) smooth max on pin positions, with optional
    LSE alternative. Pin offsets read from .plc.
      WA_max(x) = Σ x·e^(x/γ) / Σ e^(x/γ), HPWL ≈ WA_max - WA_min.
  * Density: multi-scale Gaussian-style bell deposition at scales (8, 16, 32,
    64, TILOS-grid). Bidirectional squared-diff against per-bin target +
    Coulomb (1/r) coupling at coarse scales (≤16) for long-range distribution.
  * Congestion: pin-density grid (32×32), box-smoothed, top-5% mean.

Net handling:
  * High-fanout filter: drop nets with degree > 30 (clock/scan/reset broadcast
    nets pull everything to centroid).
  * Soft-macro density floor precomputed and added to hard-macro density grid.

Soft-macro co-optimization left off pending an FFT-grade density model — bell
deposition collapsed in earlier experiments (see placer_v3 notes).
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from placer import _legalize, _swap_refine, _build_pair_edges_from_nets  # noqa: E402


# ----------------------- pin extraction (uses PlacementCost) ---------------

def _load_plc_for_benchmark(name: str):
    try:
        from macro_place.loader import load_benchmark_from_dir, load_benchmark
    except Exception:
        return None
    ibm_root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if ibm_root.exists():
        try:
            _, plc = load_benchmark_from_dir(str(ibm_root))
            return plc
        except Exception:
            return None
    ng45_map = {"ariane133_ng45": "ariane133", "ariane136_ng45": "ariane136",
                "nvdla_ng45": "nvdla", "mempool_tile_ng45": "mempool_tile"}
    d = ng45_map.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            try:
                _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
                return plc
            except Exception:
                return None
    return None


def _build_pin_arrays(benchmark: Benchmark, plc,
                      ignore_net_degree: int = 0,
                      cluster_pins_on_macro: bool = False) -> Optional[Dict]:
    """
    Build per-net flat pin arrays:
      pin_owner: [P] int — index into [hard, soft, port] universe (>=0)
                 with -1 meaning fixed-port (use port_xy)
      pin_param_idx: [P] int — index into the trainable hard params, -1 for fixed
      pin_offset_xy: [P, 2] float — offset from owner center (0 for ports)
      pin_fixed_xy: [P, 2] float — for fixed pins (ports / fixed-hard / soft)
      pin_net_id: [P] int — which net
      net_weights: [N_nets] float
    """
    try:
        N = benchmark.num_macros
        H = benchmark.num_hard_macros

        # plc-idx → bench-idx mapping for module owners
        plc_to_bench: Dict[int, int] = {}
        for i, p in enumerate(benchmark.hard_macro_indices):
            plc_to_bench[p] = i
        for i, p in enumerate(benchmark.soft_macro_indices):
            plc_to_bench[p] = H + i
        for i, p in enumerate(plc.port_indices):
            plc_to_bench[p] = N + i

        name_to_bench: Dict[str, int] = {}
        for plc_idx, bidx in plc_to_bench.items():
            name_to_bench[plc.modules_w_pins[plc_idx].get_name()] = bidx

        # Hard pin name → (owner bench idx, dx, dy)
        pin_name_to_info: Dict[str, Tuple[int, float, float]] = {}
        for plc_idx in plc.hard_macro_pin_indices:
            pin = plc.modules_w_pins[plc_idx]
            owner = pin.get_macro_name() if hasattr(pin, "get_macro_name") else None
            if owner is None or owner not in name_to_bench:
                continue
            pin_name_to_info[pin.get_name()] = (
                name_to_bench[owner],
                float(pin.x_offset), float(pin.y_offset),
            )

        port_pos = benchmark.port_positions.numpy().astype(np.float64) if benchmark.port_positions.numel() > 0 else np.zeros((0, 2), dtype=np.float64)

        pin_owner_l: List[int] = []
        pin_off_l: List[Tuple[float, float]] = []
        pin_fixed_l: List[Tuple[float, float]] = []
        pin_net_l: List[int] = []
        used_nets: List[int] = []
        net_weights_keep: List[float] = []

        next_net_id = 0
        cluster_count = [0]  # mutable closure for diagnostics
        for net_id, (driver, sinks) in enumerate(plc.nets.items()):
            net_pins: List[Tuple[int, float, float, float, float]] = []  # owner_bench_idx, dx, dy, fx, fy
            for pin_name in [driver] + sinks:
                # Macro pin: "MACRO/PIN" — parent is macro
                # Port: just "PORT_NAME"
                if "/" in pin_name:
                    info = pin_name_to_info.get(pin_name)
                    if info is None:
                        continue
                    owner_b, dx, dy = info
                    if owner_b < N:  # macro
                        net_pins.append((owner_b, dx, dy, 0.0, 0.0))
                    else:  # ??? shouldn't happen for macro-pins
                        pass
                else:
                    parent = pin_name
                    if parent in name_to_bench:
                        b = name_to_bench[parent]
                        if b >= N:
                            pi = b - N
                            if 0 <= pi < port_pos.shape[0]:
                                net_pins.append((-1, 0.0, 0.0,
                                                 float(port_pos[pi, 0]),
                                                 float(port_pos[pi, 1])))
            # Need ≥ 2 distinct owners for net to matter
            unique_owners = set((p[0], p[3], p[4]) for p in net_pins)
            if len(unique_owners) < 2:
                continue
            # Drop high-fanout nets (clock/scan/reset broadcast) — they pull
            # everything to centroid and dominate gradient. DREAMPlace default
            # threshold is 100; Hier-RTLMP folklore is 50.
            if ignore_net_degree > 0 and len(net_pins) > ignore_net_degree:
                continue
            # Pin clustering on macros: multiple pins on same macro inflate
            # net bbox even when macro is at optimal location. Aggregate to
            # median offset per (owner, net) to remove degenerate stretch.
            if cluster_pins_on_macro:
                from collections import defaultdict
                by_owner = defaultdict(list)
                for p in net_pins:
                    if p[0] >= 0:  # macro-owned
                        by_owner[p[0]].append(p)
                    else:  # port — keep distinct
                        by_owner[("port", p[3], p[4])].append(p)
                clustered = []
                got_clustered = False
                for key, pins in by_owner.items():
                    if isinstance(key, int) and len(pins) > 1:
                        got_clustered = True
                        # Median offset for this macro on this net
                        dxs = sorted(p[1] for p in pins)
                        dys = sorted(p[2] for p in pins)
                        mdx = dxs[len(dxs) // 2]
                        mdy = dys[len(dys) // 2]
                        clustered.append((pins[0][0], mdx, mdy, 0.0, 0.0))
                    else:
                        clustered.extend(pins)
                if got_clustered:
                    cluster_count[0] = cluster_count[0] + 1
                net_pins = clustered
            for owner_b, dx, dy, fx, fy in net_pins:
                pin_owner_l.append(owner_b)
                pin_off_l.append((dx, dy))
                pin_fixed_l.append((fx, fy))
                pin_net_l.append(next_net_id)
            used_nets.append(net_id)
            w = float(benchmark.net_weights[net_id].item()) if net_id < len(benchmark.net_weights) else 1.0
            net_weights_keep.append(w)
            next_net_id += 1

        if not pin_owner_l:
            return None

        return {
            "pin_owner": np.asarray(pin_owner_l, dtype=np.int64),
            "pin_offset": np.asarray(pin_off_l, dtype=np.float64),
            "pin_fixed": np.asarray(pin_fixed_l, dtype=np.float64),
            "pin_net": np.asarray(pin_net_l, dtype=np.int64),
            "num_nets": next_net_id,
            "net_weights": np.asarray(net_weights_keep, dtype=np.float64),
        }
    except Exception:
        return None


# ----------------------- LSE smooth wirelength -----------------------------

def _wa_wirelength(pin_xy: torch.Tensor, pin_net: torch.Tensor,
                   num_nets: int, net_weights: torch.Tensor,
                   gamma: float) -> torch.Tensor:
    """
    Weighted-average wirelength per net (DREAMPlace / AutoDMP standard):
        WA_max(x) = Σ x·e^(x/γ) / Σ e^(x/γ)
        WA_min(x) = Σ x·e^(-x/γ) / Σ e^(-x/γ)
        HPWL_net ≈ (WA_max(xs) - WA_min(xs)) + (WA_max(ys) - WA_min(ys))
    Smoother gradient than LSE — gradient flows to all pins, not just extremes.
    """
    device = pin_xy.device
    dtype = pin_xy.dtype
    very_neg = torch.full((num_nets, 2), -1e9, device=device, dtype=dtype)
    very_pos = torch.full((num_nets, 2), 1e9, device=device, dtype=dtype)
    pin_max = very_neg.scatter_reduce(0, pin_net.unsqueeze(1).expand_as(pin_xy),
                                       pin_xy, reduce="amax", include_self=True)
    pin_min = very_pos.scatter_reduce(0, pin_net.unsqueeze(1).expand_as(pin_xy),
                                       pin_xy, reduce="amin", include_self=True)

    # Stabilize: subtract per-net max for e^(x/γ), per-net min for e^(-x/γ)
    e_pos = torch.exp((pin_xy - pin_max[pin_net]) / gamma)        # [P, 2]
    e_neg = torch.exp(-(pin_xy - pin_min[pin_net]) / gamma)       # [P, 2]
    xe_pos = pin_xy * e_pos
    xe_neg = pin_xy * e_neg

    sum_e_pos = torch.zeros((num_nets, 2), device=device, dtype=dtype).scatter_add_(
        0, pin_net.unsqueeze(1).expand_as(e_pos), e_pos)
    sum_e_neg = torch.zeros((num_nets, 2), device=device, dtype=dtype).scatter_add_(
        0, pin_net.unsqueeze(1).expand_as(e_neg), e_neg)
    sum_xe_pos = torch.zeros((num_nets, 2), device=device, dtype=dtype).scatter_add_(
        0, pin_net.unsqueeze(1).expand_as(xe_pos), xe_pos)
    sum_xe_neg = torch.zeros((num_nets, 2), device=device, dtype=dtype).scatter_add_(
        0, pin_net.unsqueeze(1).expand_as(xe_neg), xe_neg)

    wa_max = sum_xe_pos / sum_e_pos.clamp_min(1e-30)
    wa_min = sum_xe_neg / sum_e_neg.clamp_min(1e-30)
    hpwl_per_net = (wa_max - wa_min).sum(dim=1)
    return (net_weights * hpwl_per_net).sum()


def _lse_wirelength(pin_xy: torch.Tensor, pin_net: torch.Tensor,
                    num_nets: int, net_weights: torch.Tensor,
                    gamma: float) -> torch.Tensor:
    """
    Log-sum-exp smooth max wirelength per net.
        smooth_max(x) ≈ gamma * log(sum exp(x/gamma))
        smooth_min(x) ≈ -gamma * log(sum exp(-x/gamma))
        HPWL_net ≈ (smooth_max(xs) - smooth_min(xs)) + (smooth_max(ys) - smooth_min(ys))

    Per-net stable: subtract per-net max/min before exp.
    """
    device = pin_xy.device
    dtype = pin_xy.dtype
    # Compute per-net hard max/min for stability
    very_neg = torch.full((num_nets, 2), -1e9, device=device, dtype=dtype)
    very_pos = torch.full((num_nets, 2), 1e9, device=device, dtype=dtype)
    pin_max = very_neg.scatter_reduce(0, pin_net.unsqueeze(1).expand_as(pin_xy),
                                       pin_xy, reduce="amax", include_self=True)
    pin_min = very_pos.scatter_reduce(0, pin_net.unsqueeze(1).expand_as(pin_xy),
                                       pin_xy, reduce="amin", include_self=True)

    # Now compute log-sum-exp smoothed extremes
    e_pos = torch.exp((pin_xy - pin_max[pin_net]) / gamma)
    e_neg = torch.exp(-(pin_xy - pin_min[pin_net]) / gamma)
    s_pos = torch.zeros((num_nets, 2), device=device, dtype=dtype).scatter_add_(
        0, pin_net.unsqueeze(1).expand_as(e_pos), e_pos)
    s_neg = torch.zeros((num_nets, 2), device=device, dtype=dtype).scatter_add_(
        0, pin_net.unsqueeze(1).expand_as(e_neg), e_neg)

    smooth_max = pin_max + gamma * torch.log(s_pos.clamp_min(1e-30))
    smooth_min = pin_min - gamma * torch.log(s_neg.clamp_min(1e-30))
    hpwl_per_net = (smooth_max - smooth_min).sum(dim=1)  # [num_nets]
    return (net_weights * hpwl_per_net).sum()


# ----------------------- multi-scale Gaussian density ---------------------

def _gaussian_kernel_2d(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    g = g / g.sum()
    k = g.unsqueeze(0) * g.unsqueeze(1)
    return k.unsqueeze(0).unsqueeze(0)


def _coulomb_kernel_2d(size: int, device, dtype) -> torch.Tensor:
    """1/r truncated kernel — heavier tails than Gaussian, long-range coupling.
    Conceptually distinct from FFT-Poisson (which is periodic / DCT-II)."""
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r = torch.sqrt(xx * xx + yy * yy)
    k = 1.0 / (r + 1.0)
    # Subtract mean so kernel doesn't add a uniform bias to density
    k = k - k.mean()
    k = k / k.abs().sum().clamp_min(1e-9)
    return k.unsqueeze(0).unsqueeze(0)


def _density_at_scale(positions: torch.Tensor, sizes: torch.Tensor,
                      canvas_w: float, canvas_h: float,
                      grid_x: int, grid_y: int) -> torch.Tensor:
    """
    Bell-shaped (triangular) deposition for differentiability.
    Returns [grid_y, grid_x] density grid.
    """
    bin_w = canvas_w / grid_x
    bin_h = canvas_h / grid_y
    bx = (torch.arange(grid_x, device=positions.device, dtype=positions.dtype) + 0.5) * bin_w
    by = (torch.arange(grid_y, device=positions.device, dtype=positions.dtype) + 0.5) * bin_h

    hw = sizes[:, 0] / 2 + bin_w / 2  # bell half-width (macro half + half-bin)
    hh = sizes[:, 1] / 2 + bin_h / 2

    dx = bx.unsqueeze(0) - positions[:, 0:1]  # [N, grid_x]
    dy = by.unsqueeze(0) - positions[:, 1:2]  # [N, grid_y]

    wx = F.relu(1.0 - dx.abs() / hw.unsqueeze(1))
    wy = F.relu(1.0 - dy.abs() / hh.unsqueeze(1))

    # Normalize per macro so each contributes its full area
    nx = wx.sum(dim=1, keepdim=True).clamp_min(1e-9)
    ny = wy.sum(dim=1, keepdim=True).clamp_min(1e-9)
    wxn = wx / nx
    wyn = wy / ny
    area = (sizes[:, 0] * sizes[:, 1]).unsqueeze(1)  # [N, 1]

    # Aggregate: (area * wyn).T [grid_y, N] @ wxn [N, grid_x]
    return (area * wyn).t() @ wxn


def _fft_poisson_density_loss(positions: torch.Tensor, sizes: torch.Tensor,
                               canvas_w: float, canvas_h: float,
                               fixed_density_floor: dict = None,
                               tilos_gxgy: tuple = None,
                               grid_x: int = 64, grid_y: int = 64,
                               poisson_w: float = 1.0,
                               match_loss_w: float = 1.0) -> torch.Tensor:
    """
    ePlace-style FFT-Poisson density loss.

    Build density grid via bell deposition, solve ∇²φ = -ρ on a doubly-mirrored
    2N×2N grid (image-charge trick → zero-Neumann BC on the original canvas).
    Loss = energy ⟨ρ, φ⟩ + bidirectional squared-diff vs target.

    The energy term is the canonical "electrostatic" potential — pulls density
    toward uniform distribution with correct long-range coupling, unlike
    multi-scale Gaussian which only captures short/medium range.
    """
    # Use TILOS grid if provided, else fixed grid
    if tilos_gxgy is not None:
        grid_x, grid_y = tilos_gxgy

    # Total expected mass for normalization
    total_macro_area = (sizes[:, 0] * sizes[:, 1]).sum()
    if fixed_density_floor is not None:
        total_macro_area = total_macro_area + fixed_density_floor.get("total_area", 0.0)
    canvas_area = canvas_w * canvas_h
    target_density = total_macro_area / canvas_area

    bin_w = canvas_w / grid_x
    bin_h = canvas_h / grid_y
    bin_area = bin_w * bin_h
    target_per_bin = target_density * bin_area

    rho = _density_at_scale(positions, sizes, canvas_w, canvas_h, grid_x, grid_y)
    if fixed_density_floor is not None:
        key = f"tilos_{grid_x}x{grid_y}" if tilos_gxgy is not None else grid_x
        if key in fixed_density_floor:
            rho = rho + fixed_density_floor[key]

    # Mean-subtract so total mass is 0 (needed for Poisson well-posedness on
    # zero-Neumann domain — DC mode has zero eigenvalue).
    rho_zm = rho - rho.mean()

    # Image-charge mirror: density on [0, L] extends to [0, 2L] with
    # ρ(2L - x) = ρ(x). The FFT on this 2N grid corresponds to DCT-II on N,
    # which gives ∇φ·n̂ = 0 (zero-Neumann) at original boundaries.
    rho_x_mirror = torch.cat([rho_zm, torch.flip(rho_zm, [1])], dim=1)
    rho_full = torch.cat([rho_x_mirror, torch.flip(rho_x_mirror, [0])], dim=0)

    Ny2, Nx2 = rho_full.shape
    rho_hat = torch.fft.fft2(rho_full)

    # Laplacian eigenvalues on 2N×2N periodic grid:
    # λ_kx = -2/dx² · (1 - cos(2πkx/Nx2)), similar for ky
    kx_idx = torch.arange(Nx2, device=rho.device, dtype=rho.dtype)
    ky_idx = torch.arange(Ny2, device=rho.device, dtype=rho.dtype)
    lam_x = -2.0 / (bin_w * bin_w) * (1.0 - torch.cos(2 * math.pi * kx_idx / Nx2))
    lam_y = -2.0 / (bin_h * bin_h) * (1.0 - torch.cos(2 * math.pi * ky_idx / Ny2))
    lam = lam_y.unsqueeze(1) + lam_x.unsqueeze(0)  # [Ny2, Nx2]
    # Avoid div by zero at DC mode (k=0); set to 1, will zero phi_hat[0,0]
    lam_safe = torch.where(lam.abs() < 1e-12,
                           torch.ones_like(lam), lam)

    # Solve ∇²φ = -ρ → φ̂ = -ρ̂ / λ
    phi_hat = -rho_hat / lam_safe
    phi_hat[0, 0] = 0.0
    phi_full = torch.fft.ifft2(phi_hat).real
    phi = phi_full[:grid_y, :grid_x]

    # Energy: <ρ, φ> · bin_area  (drives density to uniform)
    energy = (rho_zm * phi).sum() * bin_area

    # Bidirectional matching loss against target_per_bin (regularizer for the
    # actual TILOS top-10% scoring)
    diff = rho - target_per_bin
    match_loss = (diff * diff).sum()

    return poisson_w * energy + match_loss_w * match_loss


def _multiscale_gaussian_density_loss(positions: torch.Tensor, sizes: torch.Tensor,
                                      canvas_w: float, canvas_h: float,
                                      fixed_density_floor: dict = None,
                                      scales=(8, 16, 32, 64),
                                      tilos_gxgy: tuple = None,
                                      sigma_bins: float = 1.5,
                                      topk_frac: float = 0.10,
                                      asymmetric: bool = False,
                                      coulomb_w: float = 0.1) -> torch.Tensor:
    """
    Long-range density loss: at each scale, smooth density grid with Gaussian,
    take squared overflow above target, sum over scales.

    fixed_density_floor: optional dict {scale -> [grid, grid] tensor} of
    pre-computed density from fixed (soft + port) footprints. Added to hard
    density before computing overflow. This makes the optimizer see "where
    soft macros already are" and avoid stacking hard on top.
    """
    # Total area used by EVERYTHING (including fixed) for target normalization
    total_macro_area = (sizes[:, 0] * sizes[:, 1]).sum()
    if fixed_density_floor is not None:
        # Approximate: fixed contribution to total area is the floor's sum
        # at the finest scale
        total_macro_area = total_macro_area + fixed_density_floor.get("total_area", 0.0)
    canvas_area = canvas_w * canvas_h
    target_density = total_macro_area / canvas_area

    loss = positions.new_zeros(())
    scale_list = list(scales)
    if tilos_gxgy is not None:
        scale_list.append(tilos_gxgy)
    for g in scale_list:
        gx, gy = (g, g) if isinstance(g, int) else g
        bin_area = (canvas_w / gx) * (canvas_h / gy)
        target_per_bin = target_density * bin_area

        density = _density_at_scale(positions, sizes, canvas_w, canvas_h, gx, gy)
        if fixed_density_floor is not None:
            key = g if isinstance(g, int) else f"tilos_{gx}x{gy}"
            if key in fixed_density_floor:
                density = density + fixed_density_floor[key]

        diff = density - target_per_bin
        if asymmetric:
            # TILOS only penalizes top-10% bins — match by penalizing only
            # over-density. Allows under-filled regions without penalty.
            diff = F.relu(diff)
        loss = loss + (diff * diff).sum()

        # Coulomb (1/r) kernel: long-range coupling via direct convolution.
        # Apply only at coarse scales — at fine resolution (TILOS grid) the
        # kernel becomes large and the conv dominates runtime, while the
        # long-range signal is already captured at the coarse scales.
        if max(gx, gy) <= 16:
            ksize = min(2 * gx + 1, 2 * gy + 1)
            if ksize % 2 == 0:
                ksize += 1
            kernel = _coulomb_kernel_2d(ksize, density.device, density.dtype)
            potential = F.conv2d(density.unsqueeze(0).unsqueeze(0), kernel,
                                 padding=ksize // 2).squeeze(0).squeeze(0)
            loss = loss + coulomb_w * (density * potential).sum()
    return loss


# ----------------------- RUDY congestion ----------------------------------

def _rudy_congestion(pin_xy: torch.Tensor, pin_net: torch.Tensor,
                     num_nets: int, net_weights: torch.Tensor,
                     canvas_w: float, canvas_h: float,
                     grid_x: int, grid_y: int,
                     hroutes_per_micron: float, vroutes_per_micron: float,
                     gamma: float,
                     smooth_range: int = 2,
                     topk_frac: float = 0.05) -> torch.Tensor:
    """
    RUDY congestion matching TILOS PlacementCost.get_congestion_cost():
      - Separate H demand (x-span) and V demand (y-span).
      - Bell-deposit each via net-bbox carrier onto the bin grid.
      - Normalize by per-bin H/V routing capacity from .plc.
      - Smooth-range-2 box filter: V spreads horizontally (±2 cols),
        H spreads vertically (±2 rows).
      - Score = top-5% ABU of concatenated H+V grids (2× bin count).
    """
    device = pin_xy.device
    dtype = pin_xy.dtype

    very_neg = torch.full((num_nets, 2), -1e9, device=device, dtype=dtype)
    very_pos = torch.full((num_nets, 2), 1e9, device=device, dtype=dtype)
    pin_max = very_neg.scatter_reduce(0, pin_net.unsqueeze(1).expand_as(pin_xy),
                                       pin_xy, reduce="amax", include_self=True)
    pin_min = very_pos.scatter_reduce(0, pin_net.unsqueeze(1).expand_as(pin_xy),
                                       pin_xy, reduce="amin", include_self=True)

    e_pos = torch.exp((pin_xy - pin_max[pin_net]) / gamma)
    e_neg = torch.exp(-(pin_xy - pin_min[pin_net]) / gamma)
    s_pos = torch.zeros((num_nets, 2), device=device, dtype=dtype).scatter_add_(
        0, pin_net.unsqueeze(1).expand_as(e_pos), e_pos)
    s_neg = torch.zeros((num_nets, 2), device=device, dtype=dtype).scatter_add_(
        0, pin_net.unsqueeze(1).expand_as(e_neg), e_neg)
    bb_max = pin_max + gamma * torch.log(s_pos.clamp_min(1e-30))
    bb_min = pin_min - gamma * torch.log(s_neg.clamp_min(1e-30))

    bin_w = canvas_w / grid_x
    bin_h = canvas_h / grid_y
    bx = (torch.arange(grid_x, device=device, dtype=dtype) + 0.5) * bin_w
    by = (torch.arange(grid_y, device=device, dtype=dtype) + 0.5) * bin_h

    # x-span and y-span per net (used as demand magnitudes)
    x_span = (bb_max[:, 0] - bb_min[:, 0]).clamp_min(0.0)  # [num_nets]
    y_span = (bb_max[:, 1] - bb_min[:, 1]).clamp_min(0.0)
    h_demand = x_span * net_weights  # [num_nets]
    v_demand = y_span * net_weights

    # Bell deposition centered at bbox center with half-width = bbox/2 + bin/2
    cx = (bb_max[:, 0] + bb_min[:, 0]) * 0.5
    cy = (bb_max[:, 1] + bb_min[:, 1]) * 0.5
    hwx = (bb_max[:, 0] - bb_min[:, 0]) * 0.5 + bin_w * 0.5
    hhy = (bb_max[:, 1] - bb_min[:, 1]) * 0.5 + bin_h * 0.5

    dx = bx.unsqueeze(0) - cx.unsqueeze(1)   # [num_nets, grid_x]
    dy = by.unsqueeze(0) - cy.unsqueeze(1)   # [num_nets, grid_y]
    rwx = F.relu(1.0 - dx.abs() / hwx.unsqueeze(1))
    rwy = F.relu(1.0 - dy.abs() / hhy.unsqueeze(1))
    rwx_n = rwx / rwx.sum(dim=1, keepdim=True).clamp_min(1e-9)
    rwy_n = rwy / rwy.sum(dim=1, keepdim=True).clamp_min(1e-9)

    # Per-bin H and V demand grids
    h_grid = (h_demand.unsqueeze(1) * rwy_n).t() @ rwx_n  # [grid_y, grid_x]
    v_grid = (v_demand.unsqueeze(1) * rwy_n).t() @ rwx_n

    # Capacity normalization per the .plc model
    hcap = hroutes_per_micron * bin_h * bin_w
    vcap = vroutes_per_micron * bin_h * bin_w
    h_norm = h_grid / max(hcap, 1e-9)
    v_norm = v_grid / max(vcap, 1e-9)

    # Smooth: V spreads ±sr horizontally (cols), H spreads ±sr vertically (rows)
    sr = smooth_range
    if sr > 0:
        weight = 1.0 / (2 * sr + 1)
        v_smooth = torch.zeros_like(v_norm)
        h_smooth = torch.zeros_like(h_norm)
        for d in range(-sr, sr + 1):
            if d < 0:
                v_smooth[:, -d:] = v_smooth[:, -d:] + v_norm[:, :d] * weight
                h_smooth[-d:, :] = h_smooth[-d:, :] + h_norm[:d, :] * weight
            elif d > 0:
                v_smooth[:, :-d] = v_smooth[:, :-d] + v_norm[:, d:] * weight
                h_smooth[:-d, :] = h_smooth[:-d, :] + h_norm[d:, :] * weight
            else:
                v_smooth = v_smooth + v_norm * weight
                h_smooth = h_smooth + h_norm * weight
    else:
        v_smooth = v_norm
        h_smooth = h_norm

    # Top-5% ABU of concatenated H+V (2*gx*gy bins)
    combined = torch.cat([h_smooth.flatten(), v_smooth.flatten()])
    k = max(1, int(combined.numel() * topk_frac))
    top, _ = combined.topk(k)
    return top.mean()


# ----------------------- pin-density congestion ---------------------------

def _pin_density_congestion(pin_xy: torch.Tensor, pin_net: torch.Tensor,
                             net_weights: torch.Tensor,
                             canvas_w: float, canvas_h: float,
                             grid_x: int, grid_y: int,
                             smooth_range: int = 2,
                             topk_frac: float = 0.05) -> torch.Tensor:
    """
    Pin-density congestion: count weighted pins per bin (using bell deposition
    so it's differentiable in pin position). Smooth with box filter, take
    top-K mean.

    Different signal from RUDY: RUDY is wire-bbox-area-spread; pin density is
    where pin endpoints physically cluster.
    """
    device = pin_xy.device
    dtype = pin_xy.dtype
    bin_w = canvas_w / grid_x
    bin_h = canvas_h / grid_y
    bx = (torch.arange(grid_x, device=device, dtype=dtype) + 0.5) * bin_w
    by = (torch.arange(grid_y, device=device, dtype=dtype) + 0.5) * bin_h

    hw = bin_w  # bell half-width = bin_w (one-bin spread)
    hh = bin_h

    pin_w = net_weights[pin_net]  # weight per pin from its net
    dx = bx.unsqueeze(0) - pin_xy[:, 0:1]  # [P, grid_x]
    dy = by.unsqueeze(0) - pin_xy[:, 1:2]
    wx = F.relu(1.0 - dx.abs() / hw)
    wy = F.relu(1.0 - dy.abs() / hh)
    nx = wx.sum(dim=1, keepdim=True).clamp_min(1e-9)
    ny = wy.sum(dim=1, keepdim=True).clamp_min(1e-9)
    wxn = wx / nx
    wyn = wy / ny

    # Aggregate weighted pin density into [grid_y, grid_x]
    contrib = pin_w.unsqueeze(1)  # [P, 1]
    grid = (contrib * wyn).t() @ wxn  # [grid_y, grid_x]

    # Box smoothing range=smooth_range
    if smooth_range > 0:
        ksize = 2 * smooth_range + 1
        kernel = torch.ones(1, 1, ksize, ksize, device=device, dtype=dtype) / (ksize * ksize)
        grid = F.conv2d(grid.unsqueeze(0).unsqueeze(0), kernel,
                        padding=smooth_range).squeeze(0).squeeze(0)

    flat = grid.flatten()
    k = max(1, int(flat.numel() * topk_frac))
    top, _ = flat.topk(k)
    return top.mean()


# ----------------------- the placer ---------------------------------------

class AdityaPlacerV4:
    def __init__(self,
                 seed: int = 42,
                 global_iters: int = 800,
                 swap_iters: int = 0,
                 lr_frac: float = 0.003,
                 ov_start: float = 20.0,
                 ov_end: float = 2000.0,
                 den_w: float = 5.0,
                 cong_w: float = 0.05,
                 bd_w: float = 100.0,
                 anchor_k: int = 0,  # disabled — hypothesis A rejected
                 anchor_w: float = 5.0,
                 anchor_frac: float = 0.5,
                 init_mode: str = "given",  # "given" | "center" | "spectral"
                 spectral_anchor_w: float = 5.0,
                 spectral_jitter_frac: float = 0.02,
                 ignore_net_degree: int = 30,  # filter clock/scan/reset broadcast nets
                 cluster_pins_on_macro: bool = False,
                 asymmetric_density: bool = False,
                 coulomb_w: float = 0.1,
                 wl_mode: str = "wa",  # "lse" | "wa"
                 cong_mode: str = "pin_density",  # "pin_density" | "rudy"
                 density_mode: str = "multiscale",  # "multiscale" | "fft_poisson"
                 fft_poisson_w: float = 1.0,
                 fft_match_w: float = 1.0,
                 lr_schedule: str = "constant",  # "constant" | "cosine"
                 lr_end_frac: float = 0.1,  # cosine end = lr_frac * lr_end_frac
                 sgld_noise: float = 0.0,  # Langevin noise scale (0 = disabled)
                 sgld_start_frac: float = 0.5,  # inject noise from this fraction of iters
                 sgld_end_frac: float = 0.9,    # stop noise by this fraction
                 lns_episodes: int = 30,  # default LNS post-process: -0.4% on 3-bench
                 lns_subset_size: int = 12,
                 lns_sa_steps: int = 800,
                 lns_time_budget: float = 600.0,
                 verbose: bool = False):
        self.seed = seed
        self.global_iters = global_iters
        self.swap_iters = swap_iters
        self.lr_frac = lr_frac
        self.ov_start = ov_start
        self.ov_end = ov_end
        self.den_w = den_w
        self.cong_w = cong_w
        self.bd_w = bd_w
        self.anchor_k = anchor_k
        self.anchor_w = anchor_w
        self.anchor_frac = anchor_frac
        self.init_mode = init_mode
        self.spectral_anchor_w = spectral_anchor_w
        self.spectral_jitter_frac = spectral_jitter_frac
        self.ignore_net_degree = ignore_net_degree
        self.cluster_pins_on_macro = cluster_pins_on_macro
        self.asymmetric_density = asymmetric_density
        self.coulomb_w = coulomb_w
        self.wl_mode = wl_mode
        self.cong_mode = cong_mode
        self.density_mode = density_mode
        self.fft_poisson_w = fft_poisson_w
        self.fft_match_w = fft_match_w
        self.lr_schedule = lr_schedule
        self.lr_end_frac = lr_end_frac
        self.sgld_noise = sgld_noise
        self.sgld_start_frac = sgld_start_frac
        self.sgld_end_frac = sgld_end_frac
        self.lns_episodes = lns_episodes
        self.lns_subset_size = lns_subset_size
        self.lns_sa_steps = lns_sa_steps
        self.lns_time_budget = lns_time_budget
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        device = "cpu"

        n_hard = benchmark.num_hard_macros
        n_total = benchmark.num_macros
        n_ports = benchmark.port_positions.shape[0]
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        scale = max(cw, ch)

        sizes_full = benchmark.macro_sizes.to(device).float()
        sizes_hard = sizes_full[:n_hard]
        movable_hard = (benchmark.get_movable_mask()[:n_hard]).to(device)

        # Use TILOS's exact grid for density and congestion — matching the
        # scoring grid is the single biggest score lever (see analysis).
        tilos_gr = max(8, int(benchmark.grid_rows))
        tilos_gc = max(8, int(benchmark.grid_cols))

        # Init: "given" uses benchmark positions as-is (preserves basin from
        # initial.plc). "center" puts movable hard at canvas mid + jitter.
        init_hard = benchmark.macro_positions[:n_hard].to(device).float().clone()
        if self.init_mode == "center" and movable_hard.any():
            init_hard[movable_hard] = torch.tensor([cw / 2, ch / 2], device=device)
            jitter = (torch.rand_like(init_hard[movable_hard]) - 0.5) * (scale * 0.1)
            init_hard[movable_hard] = init_hard[movable_hard] + jitter
        elif self.init_mode == "spectral" and movable_hard.any():
            from spectral_init import spectral_init
            sp_pos = spectral_init(
                benchmark,
                port_anchor_w=self.spectral_anchor_w,
                jitter_frac=self.spectral_jitter_frac,
                seed=self.seed,
            )
            if sp_pos is not None:
                # Only overwrite movable; non-movable stays at given positions.
                mh = movable_hard
                init_hard[mh] = sp_pos.to(device)[mh]
            else:
                if self.verbose:
                    print("  [spectral_init returned None — falling back to given]")

        hard_var = torch.nn.Parameter(init_hard.clone())

        # Soft + ports stay fixed (point-macro for soft; learning bell density too
        # would risk the v3 collapse seen earlier).
        soft_pos = benchmark.macro_positions[n_hard:n_total].to(device).float()
        port_pos = benchmark.port_positions.to(device).float() if n_ports > 0 else torch.zeros(0, 2, device=device)

        # Build pin arrays (with offsets)
        plc = _load_plc_for_benchmark(benchmark.name)
        pin_data = _build_pin_arrays(
            benchmark, plc,
            ignore_net_degree=self.ignore_net_degree,
            cluster_pins_on_macro=self.cluster_pins_on_macro,
        ) if plc is not None else None

        if pin_data is None:
            # Fallback: point-macro pins (no offsets) — degrades quality
            if self.verbose:
                print("[v4] WARN: no pin offsets, falling back to point-macro HPWL")
            return self._fallback_no_pins(benchmark, hard_var, sizes_hard,
                                          movable_hard, init_hard,
                                          soft_pos, port_pos,
                                          cw, ch, n_hard, n_total, n_ports)

        # Convert to torch
        pin_owner = torch.from_numpy(pin_data["pin_owner"]).to(device)
        pin_offset = torch.from_numpy(pin_data["pin_offset"]).to(device).float()
        pin_fixed = torch.from_numpy(pin_data["pin_fixed"]).to(device).float()
        pin_net = torch.from_numpy(pin_data["pin_net"]).to(device)
        num_nets = pin_data["num_nets"]
        net_weights = torch.from_numpy(pin_data["net_weights"]).to(device).float()

        # Categorize pins
        is_hard = (pin_owner >= 0) & (pin_owner < n_hard)
        is_soft = (pin_owner >= n_hard) & (pin_owner < n_total)
        is_port = (pin_owner == -1)

        owner_safe = pin_owner.clamp(min=0)

        # ----- Build port-centroid anchors for top-K most-connected hard macros -----
        anchor_idx = torch.empty(0, dtype=torch.long, device=device)
        anchor_target = torch.empty(0, 2, device=device)
        if self.anchor_k > 0 and n_ports > 0:
            # For each net, find which hard macros it touches and which ports
            np_pin_owner = pin_data["pin_owner"]
            np_pin_fixed = pin_data["pin_fixed"]
            np_pin_net = pin_data["pin_net"]
            num_nets_local = pin_data["num_nets"]
            net_w_np = pin_data["net_weights"]

            hard_net_count = np.zeros(n_hard, dtype=np.int64)
            hard_port_sum = np.zeros((n_hard, 2), dtype=np.float64)
            hard_port_w = np.zeros(n_hard, dtype=np.float64)

            # Group pins by net for fast lookup
            for net_id in range(num_nets_local):
                net_mask = np_pin_net == net_id
                pins = np.where(net_mask)[0]
                hard_owners = []
                port_xys = []
                for pi in pins:
                    o = np_pin_owner[pi]
                    if 0 <= o < n_hard:
                        hard_owners.append(o)
                    elif o == -1:
                        port_xys.append(np_pin_fixed[pi])
                if not hard_owners:
                    continue
                w = float(net_w_np[net_id])
                # Each hard owner gets one port-centroid contribution per net
                if port_xys:
                    centroid = np.mean(port_xys, axis=0)
                    for h in set(hard_owners):
                        hard_port_sum[h] += centroid * w
                        hard_port_w[h] += w
                for h in set(hard_owners):
                    hard_net_count[h] += 1

            # Top-K by net degree (only macros with port-connected nets)
            connected = hard_port_w > 0
            ranked = np.argsort(-hard_net_count * connected)[:self.anchor_k]
            ranked = ranked[connected[ranked]]
            ranked = ranked[(movable_hard.cpu().numpy())[ranked]]

            if len(ranked) > 0:
                anchor_idx = torch.from_numpy(ranked).to(device).long()
                centroids = hard_port_sum[ranked] / hard_port_w[ranked, None].clip(min=1e-9)
                anchor_target = torch.from_numpy(centroids).to(device).float()
                if self.verbose:
                    print(f"[v4] anchor: {len(ranked)} macros to port centroids")

        # ----- Precompute fixed (soft) density floor at all scales -----
        # This lets the hard-macro density gradient see "where soft is" and
        # avoid stacking on top of soft-dense regions.
        # Scales include TILOS's grid (variable per benchmark).
        density_scales = (8, 16, 32, 64, (tilos_gc, tilos_gr))
        fixed_density_floor = None
        if soft_pos.shape[0] > 0:
            sizes_soft = sizes_full[n_hard:n_total]
            fixed_density_floor = {}
            for g in density_scales:
                gx, gy = (g, g) if isinstance(g, int) else g
                key = g if isinstance(g, int) else f"tilos_{gx}x{gy}"
                fixed_density_floor[key] = _density_at_scale(
                    soft_pos, sizes_soft, cw, ch, gx, gy
                ).detach()
            fixed_density_floor["total_area"] = float(
                (sizes_soft[:, 0] * sizes_soft[:, 1]).sum().item()
            )

        gamma = 0.01 * scale  # reverted from H
        lr_init = self.lr_frac * scale
        opt = torch.optim.Adam([hard_var], lr=lr_init)

        log_ov_start = math.log(self.ov_start)
        log_ov_end = math.log(self.ov_end)
        # density weight ramp
        log_den_start = math.log(0.001)
        log_den_end = math.log(self.den_w)

        t0 = time.time()
        for step in range(self.global_iters):
            # Cosine LR schedule (optional)
            if self.lr_schedule == "cosine":
                t_lr = step / max(self.global_iters - 1, 1)
                cosine = 0.5 * (1 + math.cos(math.pi * t_lr))
                lr_now = lr_init * (self.lr_end_frac + (1 - self.lr_end_frac) * cosine)
                for g in opt.param_groups:
                    g["lr"] = lr_now
            opt.zero_grad()
            cur_hard = hard_var
            if (~movable_hard).any():
                cur_hard = torch.where(movable_hard.unsqueeze(1), cur_hard, init_hard)

            # Compute pin positions
            # hard pins: cur_hard[owner] + offset
            # soft pins: soft_pos[owner-n_hard] + offset
            # port pins: pin_fixed
            pin_pos = torch.zeros_like(pin_offset)  # [P, 2]
            # hard
            hard_pos = cur_hard[owner_safe.clamp(max=n_hard - 1)]
            soft_idx = (owner_safe - n_hard).clamp(min=0, max=max(n_total - n_hard - 1, 0))
            soft_owner_pos = soft_pos[soft_idx] if soft_pos.shape[0] > 0 else torch.zeros_like(hard_pos)
            owner_pos = torch.where(is_hard.unsqueeze(1), hard_pos, soft_owner_pos)
            owner_pos = torch.where(is_port.unsqueeze(1), pin_fixed, owner_pos + pin_offset)

            # WL via LSE
            t = step / max(self.global_iters - 1, 1)
            wl_fn = _wa_wirelength if self.wl_mode == "wa" else _lse_wirelength
            wl = wl_fn(owner_pos, pin_net, num_nets, net_weights, gamma)
            wl_norm = wl / ((cw + ch) * net_weights.sum().clamp_min(1.0))

            # Multi-scale Gaussian density with fixed soft-macro floor
            # + TILOS-grid scale for direct alignment with scoring resolution
            if self.density_mode == "fft_poisson":
                den = _fft_poisson_density_loss(
                    cur_hard, sizes_hard, cw, ch,
                    fixed_density_floor=fixed_density_floor,
                    tilos_gxgy=(tilos_gc, tilos_gr),
                    poisson_w=self.fft_poisson_w,
                    match_loss_w=self.fft_match_w,
                )
            else:
                den = _multiscale_gaussian_density_loss(
                    cur_hard, sizes_hard, cw, ch,
                    fixed_density_floor=fixed_density_floor,
                    tilos_gxgy=(tilos_gc, tilos_gr),
                    asymmetric=self.asymmetric_density,
                    coulomb_w=self.coulomb_w,
                )

            # Congestion proxy
            if self.cong_mode == "rudy":
                cong = _rudy_congestion(
                    owner_pos, pin_net, num_nets, net_weights,
                    cw, ch, grid_x=tilos_gc, grid_y=tilos_gr,
                    hroutes_per_micron=float(benchmark.hroutes_per_micron),
                    vroutes_per_micron=float(benchmark.vroutes_per_micron),
                    gamma=gamma,
                )
            else:
                cong = _pin_density_congestion(owner_pos, pin_net, net_weights,
                                                cw, ch, grid_x=32, grid_y=32)

            # Boundary penalty
            hw = sizes_hard[:, 0] / 2
            hh = sizes_hard[:, 1] / 2
            bx_lo = F.relu(hw - cur_hard[:, 0])
            bx_hi = F.relu(cur_hard[:, 0] - (cw - hw))
            by_lo = F.relu(hh - cur_hard[:, 1])
            by_hi = F.relu(cur_hard[:, 1] - (ch - hh))
            bd = (bx_lo * bx_lo + bx_hi * bx_hi + by_lo * by_lo + by_hi * by_hi).sum() / scale

            den_w = math.exp(log_den_start + t * (log_den_end - log_den_start))
            cong_w_t = self.cong_w * max(0.0, (t - 0.2) / 0.8)

            # Anchor: linearly-decaying quadratic pull toward port centroids
            # for top-K most-connected hard macros. Active only for first
            # `anchor_frac` of iters.
            if anchor_idx.numel() > 0 and t < self.anchor_frac:
                anchor_decay = max(0.0, 1.0 - t / self.anchor_frac)
                anchor_diff = cur_hard[anchor_idx] - anchor_target
                anchor_pen = (anchor_diff * anchor_diff).sum() / (scale * scale)
                anchor_term = self.anchor_w * anchor_decay * anchor_pen
            else:
                anchor_term = positions_zero = cur_hard.new_zeros(())

            loss = (wl_norm + den_w * den / (scale * scale) + self.bd_w * bd
                    + cong_w_t * cong + anchor_term)

            if not torch.isfinite(loss):
                opt.zero_grad()
                continue

            loss.backward()
            if hard_var.grad is not None:
                if (~movable_hard).any():
                    hard_var.grad[~movable_hard] = 0.0
                bad = ~torch.isfinite(hard_var.grad)
                if bad.any():
                    hard_var.grad[bad] = 0.0
                # SGLD: add Langevin noise during escape phase
                # x_{t+1} = x_t - lr·∇f + √(2·lr·T)·N(0,I)
                # Cosine-shaped envelope between sgld_start_frac and sgld_end_frac
                if self.sgld_noise > 0.0 and self.sgld_start_frac <= t <= self.sgld_end_frac:
                    span = self.sgld_end_frac - self.sgld_start_frac
                    if span > 1e-9:
                        u = (t - self.sgld_start_frac) / span  # 0 → 1
                        envelope = 0.5 * (1.0 + math.cos(math.pi * u))  # 1 → 0
                    else:
                        envelope = 1.0
                    noise_scale = self.sgld_noise * scale * envelope
                    noise = torch.randn_like(hard_var.grad) * noise_scale
                    if (~movable_hard).any():
                        noise[~movable_hard] = 0.0
                    hard_var.grad = hard_var.grad + noise
            opt.step()

            if self.verbose and step % 100 == 0:
                print(f"[v4] step {step:4d}  wl={float(wl_norm):.4f}  "
                      f"den={float(den):.3e} (w={den_w:.3f})  "
                      f"cong={float(cong):.4f} (w={cong_w_t:.3f})  "
                      f"bd={float(bd):.3e}")

        if self.verbose:
            print(f"[v4] global done in {time.time()-t0:.2f}s")

        # ---- Legalize hard macros ----
        global_pos = hard_var.detach().cpu().numpy().astype(np.float64)
        sizes_np = sizes_hard.cpu().numpy().astype(np.float64)
        movable_np = movable_hard.cpu().numpy()
        legal = _legalize(global_pos, movable_np, sizes_np, cw, ch)

        # ---- Swap refinement ----
        edges, edge_weights = _build_pair_edges_from_nets(benchmark.net_nodes, n_hard)
        refined = _swap_refine(legal, edges, edge_weights, movable_np, sizes_np,
                               cw, ch, self.swap_iters)

        # ---- Assemble ----
        full = benchmark.macro_positions.clone()
        full[:n_hard] = torch.tensor(refined, dtype=torch.float32)

        # ---- LNS post-process (optional) ----
        if self.lns_episodes > 0:
            from lns_refine import lns_refine
            full = lns_refine(
                full, benchmark, plc,
                time_budget=self.lns_time_budget,
                n_episodes=self.lns_episodes,
                subset_size=self.lns_subset_size,
                sa_steps=self.lns_sa_steps,
                sa_mode="greedy",
                step_init_frac=0.02, step_end_frac=0.0002,
                seed=self.seed,
                verbose=self.verbose,
            )
        return full

    def _fallback_no_pins(self, benchmark, hard_var, sizes_hard, movable_hard,
                          init_hard, soft_pos, port_pos, cw, ch, n_hard, n_total, n_ports):
        # Fall through to v1's implementation if pins aren't available
        from placer import AdityaPlacer
        return AdityaPlacer(seed=self.seed).place(benchmark)
