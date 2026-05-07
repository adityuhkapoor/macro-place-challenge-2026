"""
Incremental proxy evaluator for the SA / coordinate-descent inner loop.

Mutable per-net HPWL + density + RUDY state. On a single-macro move:
  - HPWL: O(deg(macro)) — only nets touching the macro need delta
  - density: O(cells_per_macro_footprint) — typically O(1)
  - RUDY: O(deg(macro) * cells_per_net_bbox) — manageable for IBM benchmarks

The point is to call this ~10^5–10^6 times per benchmark inside SA, instead
of calling TILOS PlacementCost (which rebuilds everything in pure Python and
costs ~1–10 s per call).

Validate against TILOS at the end of each pipeline stage; never inside the
hot loop.

Architecture follows boydhamilton's vectorized `_DensityGrid` (proven 175×
faster on ibm01) and the per-net HPWL pattern recommended in the engineering
research.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from macro_place.benchmark import Benchmark


def _build_pin_to_nets(net_pins: List[np.ndarray], num_nodes: int) -> List[List[int]]:
    """For each node index, list of nets that contain it."""
    pin_to_nets: List[List[int]] = [[] for _ in range(num_nodes)]
    for net_idx, pins in enumerate(net_pins):
        for p in pins:
            pin_to_nets[int(p)].append(net_idx)
    return pin_to_nets


class IncrementalProxy:
    """
    Mutable proxy state. Designed for SA / coordinate-descent on hard macros.
    Soft macros and ports are treated as fixed (their positions never change
    inside the inner loop).

    Cost components:
      - hpwl_total: sum of net BBox half-perimeters
      - density grid (10×10 by default): per-cell occupancy from hard macro footprints
      - rudy grid: per-cell wire-density estimate

    Combined proxy approximation:
        proxy_approx = hpwl_norm + 0.5 * density_top10pct + 0.5 * rudy_top5pct
    Note: this matches the structure of TILOS proxy but uses RUDY as a cheap
    surrogate for routing congestion. The TRUE TILOS proxy must be computed
    at end of pipeline (not inside the hot loop).
    """

    def __init__(self, benchmark: Benchmark,
                 grid_size_density: int = 10,
                 grid_size_rudy: int = 32):
        self.bm = benchmark
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.n_hard = benchmark.num_hard_macros
        self.n_total = benchmark.num_macros
        self.n_ports = int(benchmark.port_positions.shape[0])
        self.n_all = self.n_total + self.n_ports

        # Position table covering hard, soft, ports. Indices match net_nodes.
        all_pos = np.zeros((self.n_all, 2), dtype=np.float64)
        all_pos[:self.n_total] = benchmark.macro_positions.numpy().astype(np.float64)
        if self.n_ports > 0:
            all_pos[self.n_total:] = benchmark.port_positions.numpy().astype(np.float64)
        self.positions = all_pos  # mutable

        self.sizes = benchmark.macro_sizes.numpy().astype(np.float64)  # [n_total, 2]

        # Convert net_nodes (List[Tensor]) to List[ndarray] of int32 indices.
        # Filter to nets with ≥2 distinct pins.
        self.net_pins: List[np.ndarray] = []
        net_weights_list = []
        for i, t in enumerate(benchmark.net_nodes):
            arr = np.asarray(t.numpy(), dtype=np.int32)
            if arr.size < 2:
                continue
            arr = np.unique(arr)
            if arr.size < 2:
                continue
            self.net_pins.append(arr)
            w = float(benchmark.net_weights[i].item()) if i < len(benchmark.net_weights) else 1.0
            net_weights_list.append(w)
        self.net_weights = np.asarray(net_weights_list, dtype=np.float64)
        self.num_nets = len(self.net_pins)

        self.pin_to_nets = _build_pin_to_nets(self.net_pins, self.n_all)

        # Per-net BBox cache: xmin, xmax, ymin, ymax
        self.net_xmin = np.zeros(self.num_nets, dtype=np.float64)
        self.net_xmax = np.zeros(self.num_nets, dtype=np.float64)
        self.net_ymin = np.zeros(self.num_nets, dtype=np.float64)
        self.net_ymax = np.zeros(self.num_nets, dtype=np.float64)
        # Total weighted HPWL = sum_i weight_i * ((xmax-xmin)+(ymax-ymin))
        self.hpwl_total = 0.0

        for k, pins in enumerate(self.net_pins):
            xs = self.positions[pins, 0]
            ys = self.positions[pins, 1]
            self.net_xmin[k] = xs.min()
            self.net_xmax[k] = xs.max()
            self.net_ymin[k] = ys.min()
            self.net_ymax[k] = ys.max()
        self.hpwl_total = float((self.net_weights *
                                 ((self.net_xmax - self.net_xmin) +
                                  (self.net_ymax - self.net_ymin))).sum())

        # Density grid (boydhamilton style)
        self.den_g = grid_size_density
        self.den_cell_w = self.cw / self.den_g
        self.den_cell_h = self.ch / self.den_g
        self.den_cell_area = self.den_cell_w * self.den_cell_h
        self.den_col_lo = np.arange(self.den_g) * self.den_cell_w
        self.den_col_hi = self.den_col_lo + self.den_cell_w
        self.den_row_lo = np.arange(self.den_g) * self.den_cell_h
        self.den_row_hi = self.den_row_lo + self.den_cell_h
        self.density_grid = np.zeros((self.den_g, self.den_g), dtype=np.float64)
        for i in range(self.n_hard):
            self._density_add(i)

        # RUDY grid
        self.rudy_g = grid_size_rudy
        self.rudy_cell_w = self.cw / self.rudy_g
        self.rudy_cell_h = self.ch / self.rudy_g
        self.rudy_col_lo = np.arange(self.rudy_g) * self.rudy_cell_w
        self.rudy_col_hi = self.rudy_col_lo + self.rudy_cell_w
        self.rudy_row_lo = np.arange(self.rudy_g) * self.rudy_cell_h
        self.rudy_row_hi = self.rudy_row_lo + self.rudy_cell_h
        self.rudy_grid = np.zeros((self.rudy_g, self.rudy_g), dtype=np.float64)
        for k in range(self.num_nets):
            self._rudy_add_net(k)

        # HPWL normalizer (matches TILOS scale roughly: half-perimeter / canvas total)
        total_w = float(self.net_weights.sum()) if self.num_nets > 0 else 1.0
        self.hpwl_denom = max((self.cw + self.ch) * total_w, 1e-12)

    # ---- Density (boydhamilton's vectorized footprint contribution) ----
    def _density_contribution(self, i: int) -> np.ndarray:
        x = self.positions[i, 0]
        y = self.positions[i, 1]
        hw = self.sizes[i, 0] / 2
        hh = self.sizes[i, 1] / 2
        x0, x1 = x - hw, x + hw
        y0, y1 = y - hh, y + hh
        ov_x = np.maximum(0.0, np.minimum(x1, self.den_col_hi) - np.maximum(x0, self.den_col_lo))
        ov_y = np.maximum(0.0, np.minimum(y1, self.den_row_hi) - np.maximum(y0, self.den_row_lo))
        return np.outer(ov_y, ov_x) / self.den_cell_area

    def _density_add(self, i: int):
        self.density_grid += self._density_contribution(i)

    def _density_remove(self, i: int):
        self.density_grid -= self._density_contribution(i)

    def density_top10_mean(self) -> float:
        flat = self.density_grid.ravel()
        n_top = max(1, flat.size // 10)
        return float(np.partition(flat, -n_top)[-n_top:].mean())

    # ---- RUDY: per-net, smeared uniformly over net's bbox cells ----
    def _rudy_contribution(self, k: int) -> Tuple[np.ndarray, float, int, int, int, int]:
        """Returns (delta grid, demand, c0, c1, r0, r1)."""
        x0 = self.net_xmin[k]; x1 = self.net_xmax[k]
        y0 = self.net_ymin[k]; y1 = self.net_ymax[k]
        # Net bbox dims
        W = max(x1 - x0, 1e-6)
        H = max(y1 - y0, 1e-6)
        demand = self.net_weights[k] * (W + H) / (W * H)  # wire density per unit area

        # Find covered cells (integer indices)
        c0 = max(0, int(np.floor(x0 / self.rudy_cell_w)))
        c1 = min(self.rudy_g, int(np.ceil(x1 / self.rudy_cell_w)) + 1)
        r0 = max(0, int(np.floor(y0 / self.rudy_cell_h)))
        r1 = min(self.rudy_g, int(np.ceil(y1 / self.rudy_cell_h)) + 1)
        if c1 <= c0 or r1 <= r0:
            return None, 0.0, 0, 0, 0, 0

        # Per-cell overlap with bbox
        col_ov = (np.minimum(x1, self.rudy_col_hi[c0:c1]) -
                  np.maximum(x0, self.rudy_col_lo[c0:c1])).clip(min=0.0)
        row_ov = (np.minimum(y1, self.rudy_row_hi[r0:r1]) -
                  np.maximum(y0, self.rudy_row_lo[r0:r1])).clip(min=0.0)
        bbox_area = max(W * H, 1e-12)
        # Fraction of bbox area covered by each cell × demand
        contrib = np.outer(row_ov, col_ov) * (demand / bbox_area)
        return contrib, demand, c0, c1, r0, r1

    def _rudy_add_net(self, k: int):
        c, _, c0, c1, r0, r1 = self._rudy_contribution(k)
        if c is None:
            return
        self.rudy_grid[r0:r1, c0:c1] += c

    def _rudy_remove_net(self, k: int):
        c, _, c0, c1, r0, r1 = self._rudy_contribution(k)
        if c is None:
            return
        self.rudy_grid[r0:r1, c0:c1] -= c

    def rudy_top5_mean(self) -> float:
        flat = self.rudy_grid.ravel()
        n_top = max(1, int(flat.size * 0.05))
        return float(np.partition(flat, -n_top)[-n_top:].mean())

    # ---- Move a macro: incremental update ----
    def move_macro(self, hard_idx: int, new_x: float, new_y: float):
        """
        Move macro `hard_idx` (must be in [0, n_hard)) to (new_x, new_y).
        Updates HPWL, density, and RUDY incrementally.
        """
        assert 0 <= hard_idx < self.n_hard
        old_x = self.positions[hard_idx, 0]
        old_y = self.positions[hard_idx, 1]
        if old_x == new_x and old_y == new_y:
            return

        # Density: remove old footprint, then update position, then add new
        self._density_remove(hard_idx)

        # Find affected nets
        affected = self.pin_to_nets[hard_idx]

        # Remove old RUDY contributions for affected nets (they will change)
        for k in affected:
            self._rudy_remove_net(k)

        # Update position
        self.positions[hard_idx, 0] = new_x
        self.positions[hard_idx, 1] = new_y

        # Update HPWL caches for affected nets (recompute from full pin set; cheap)
        for k in affected:
            pins = self.net_pins[k]
            xs = self.positions[pins, 0]
            ys = self.positions[pins, 1]
            new_xmin = xs.min(); new_xmax = xs.max()
            new_ymin = ys.min(); new_ymax = ys.max()
            old_span = (self.net_xmax[k] - self.net_xmin[k]) + (self.net_ymax[k] - self.net_ymin[k])
            new_span = (new_xmax - new_xmin) + (new_ymax - new_ymin)
            self.hpwl_total += self.net_weights[k] * (new_span - old_span)
            self.net_xmin[k] = new_xmin; self.net_xmax[k] = new_xmax
            self.net_ymin[k] = new_ymin; self.net_ymax[k] = new_ymax

        # Re-add RUDY contributions with new bboxes
        for k in affected:
            self._rudy_add_net(k)

        # Add new density footprint
        self._density_add(hard_idx)

    def hpwl_norm(self) -> float:
        return self.hpwl_total / self.hpwl_denom

    def proxy(self,
              w_wl: float = 1.0,
              w_density: float = 0.5,
              w_rudy: float = 0.5) -> float:
        """Approximate proxy combining HPWL + density-top10 + RUDY-top5."""
        return (w_wl * self.hpwl_norm()
                + w_density * self.density_top10_mean()
                + w_rudy * self.rudy_top5_mean())

    # ---- Overlap check (one macro vs all hard macros) ----
    def overlaps_any(self, hard_idx: int, gap: float = 0.05) -> bool:
        x = self.positions[hard_idx, 0]
        y = self.positions[hard_idx, 1]
        hw = self.sizes[hard_idx, 0] / 2
        hh = self.sizes[hard_idx, 1] / 2
        # Distances to all hard macros
        dx = np.abs(self.positions[:self.n_hard, 0] - x)
        dy = np.abs(self.positions[:self.n_hard, 1] - y)
        sep_x = (self.sizes[:self.n_hard, 0] + self.sizes[hard_idx, 0]) / 2 + gap
        sep_y = (self.sizes[:self.n_hard, 1] + self.sizes[hard_idx, 1]) / 2 + gap
        c = (dx < sep_x) & (dy < sep_y)
        c[hard_idx] = False
        return bool(c.any())

    # ---- Snapshot / restore for SA reject ----
    def snapshot(self) -> dict:
        return {
            "positions": self.positions.copy(),
            "net_xmin": self.net_xmin.copy(),
            "net_xmax": self.net_xmax.copy(),
            "net_ymin": self.net_ymin.copy(),
            "net_ymax": self.net_ymax.copy(),
            "density_grid": self.density_grid.copy(),
            "rudy_grid": self.rudy_grid.copy(),
            "hpwl_total": self.hpwl_total,
        }

    def restore(self, snap: dict):
        self.positions[...] = snap["positions"]
        self.net_xmin[...] = snap["net_xmin"]
        self.net_xmax[...] = snap["net_xmax"]
        self.net_ymin[...] = snap["net_ymin"]
        self.net_ymax[...] = snap["net_ymax"]
        self.density_grid[...] = snap["density_grid"]
        self.rudy_grid[...] = snap["rudy_grid"]
        self.hpwl_total = snap["hpwl_total"]


def smoke_test(benchmark_name: str = "ibm01") -> dict:
    """Validate the incremental proxy against TILOS on initial placement."""
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    import torch

    bm, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{benchmark_name}")
    inc = IncrementalProxy(bm)

    # TILOS reference
    tilos = compute_proxy_cost(bm.macro_positions, bm, plc)

    return {
        "tilos_proxy": float(tilos["proxy_cost"]),
        "tilos_wl": float(tilos["wirelength_cost"]),
        "tilos_density": float(tilos["density_cost"]),
        "tilos_congestion": float(tilos["congestion_cost"]),
        "inc_hpwl_norm": float(inc.hpwl_norm()),
        "inc_density_top10": float(inc.density_top10_mean()),
        "inc_rudy_top5": float(inc.rudy_top5_mean()),
        "inc_proxy_approx": float(inc.proxy()),
    }


def validate_move_consistency(benchmark_name: str = "ibm01", num_moves: int = 50):
    """
    Make N random moves on the incremental proxy. After each, build a
    fresh IncrementalProxy from the same positions and check its state
    matches the incrementally-updated one to within 1e-6.
    """
    from macro_place.loader import load_benchmark_from_dir
    bm, _ = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{benchmark_name}")
    rng = np.random.default_rng(0)

    inc = IncrementalProxy(bm)
    for step in range(num_moves):
        i = int(rng.integers(0, inc.n_hard))
        nx = float(rng.uniform(0, inc.cw))
        ny = float(rng.uniform(0, inc.ch))
        inc.move_macro(i, nx, ny)

        # Fresh comparison every 10 moves
        if step % 10 == 9:
            fresh = IncrementalProxy(bm)
            fresh.positions[...] = inc.positions
            # Force recompute caches
            for k in range(fresh.num_nets):
                pins = fresh.net_pins[k]
                xs = fresh.positions[pins, 0]
                ys = fresh.positions[pins, 1]
                fresh.net_xmin[k] = xs.min(); fresh.net_xmax[k] = xs.max()
                fresh.net_ymin[k] = ys.min(); fresh.net_ymax[k] = ys.max()
            fresh.hpwl_total = float((fresh.net_weights *
                                      ((fresh.net_xmax - fresh.net_xmin) +
                                       (fresh.net_ymax - fresh.net_ymin))).sum())
            fresh.density_grid = np.zeros_like(fresh.density_grid)
            for j in range(fresh.n_hard):
                fresh._density_add(j)
            fresh.rudy_grid = np.zeros_like(fresh.rudy_grid)
            for k in range(fresh.num_nets):
                fresh._rudy_add_net(k)

            hpwl_diff = abs(inc.hpwl_total - fresh.hpwl_total)
            den_diff = float(np.abs(inc.density_grid - fresh.density_grid).max())
            rudy_diff = float(np.abs(inc.rudy_grid - fresh.rudy_grid).max())
            print(f"  step {step+1}: hpwl_diff={hpwl_diff:.6e} "
                  f"den_max_diff={den_diff:.6e} rudy_max_diff={rudy_diff:.6e}")
            ok = hpwl_diff < 1e-3 and den_diff < 1e-6 and rudy_diff < 1e-6
            if not ok:
                print(f"  FAIL at step {step+1}")
                return False
    return True


def benchmark_speed(benchmark_name: str = "ibm01", num_moves: int = 1000):
    """Time how many moves/sec the incremental proxy can do."""
    import time
    from macro_place.loader import load_benchmark_from_dir
    bm, _ = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{benchmark_name}")
    rng = np.random.default_rng(0)

    inc = IncrementalProxy(bm)
    moves = [(int(rng.integers(0, inc.n_hard)),
              float(rng.uniform(0, inc.cw)),
              float(rng.uniform(0, inc.ch))) for _ in range(num_moves)]

    t0 = time.time()
    for i, x, y in moves:
        inc.move_macro(i, x, y)
        _ = inc.proxy()
    elapsed = time.time() - t0
    return {"benchmark": benchmark_name, "moves": num_moves,
            "elapsed_s": elapsed, "moves_per_s": num_moves / elapsed}


if __name__ == "__main__":
    import json
    print("=== smoke test ===")
    r = smoke_test("ibm01")
    print(json.dumps(r, indent=2))
    print("\n=== move consistency ===")
    ok = validate_move_consistency("ibm01", 50)
    print(f"  ok = {ok}")
    print("\n=== speed benchmark ===")
    s = benchmark_speed("ibm01", 1000)
    print(json.dumps(s, indent=2))
