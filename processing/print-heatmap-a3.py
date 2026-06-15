#!/usr/bin/env python3
"""
Generate a real-size tiled A3 PDF from heatmap CSV data.

The script reconstructs the numeric power grid from heatmap.csv and renders
each page at the requested DPI using interpolation (IDW by default).
An additional A3 page with the heatmap colorbar in micro-watts is appended
by default.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter


MM_PER_INCH = 25.4
A3_PORTRAIT_MM = (297.0, 420.0)
A3_LANDSCAPE_MM = (420.0, 297.0)
EPS = 1e-12
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_FOLDER = os.path.join(PROJECT_ROOT, "data", "RECI-merged")


@dataclass(frozen=True)
class Layout:
    orientation: str
    page_w_mm: float
    page_h_mm: float
    print_w_mm: float
    print_h_mm: float
    x_starts_mm: list[float]
    y_starts_mm: list[float]
    overlap_mm: float
    bleed_mm: float

    @property
    def cols(self) -> int:
        return len(self.x_starts_mm)

    @property
    def rows(self) -> int:
        return len(self.y_starts_mm)

    @property
    def pages(self) -> int:
        return self.cols * self.rows


@dataclass(frozen=True)
class TilePlacement:
    page_num: int
    row: int
    col: int
    x_start_mm: float
    y_start_mm: float
    tile_w_mm: float
    tile_h_mm: float
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a heatmap as a real-size, tiled A3 PDF for floor printing."
    )
    parser.add_argument("--folder", default=DEFAULT_FOLDER, help="Folder containing heatmap exports.")
    parser.add_argument(
        "--heatmap-csv",
        default="heatmap.csv",
        help="Path to heatmap CSV (x,y,z). Relative paths are resolved from --folder.",
    )
    parser.add_argument(
        "--x-edges",
        default="x_edges.csv",
        help="Path to x_edges CSV. Relative paths are resolved from --folder.",
    )
    parser.add_argument(
        "--y-edges",
        default="y_edges.csv",
        help="Path to y_edges CSV. Relative paths are resolved from --folder.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PDF path. Default: <folder>/heatmap_A3_tiled_idw2_150dpi.pdf",
    )
    parser.add_argument("--render-dpi", type=float, default=150.0, help="Tile render DPI (default: 150).")
    parser.add_argument(
        "--interp",
        choices=["idw", "linear", "nearest"],
        default="idw",
        help="Interpolation method (default: idw).",
    )
    parser.add_argument("--idw-power", type=float, default=2.0, help="IDW exponent p for 1/(d^p+eps).")
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=6.0,
        help="Outer white margin in mm for cutting/handling (default: 6.0).",
    )
    parser.add_argument(
        "--bleed-mm",
        type=float,
        default=3.0,
        help="Bleed around each tile in mm (default: 3.0).",
    )
    parser.add_argument("--overlap-mm", type=float, default=10.0, help="Overlap between adjacent tiles in mm.")
    parser.add_argument(
        "--orientation",
        choices=["auto", "portrait", "landscape"],
        default="auto",
        help="A3 orientation selection mode.",
    )
    parser.add_argument("--cmap", default="inferno", help="Matplotlib colormap name.")
    parser.add_argument("--vmin", type=float, default=None, help="Color scale minimum.")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale maximum.")
    parser.add_argument(
        "--colorbar-dpi",
        type=float,
        default=600.0,
        help="DPI used when rendering the dedicated colorbar page (default: 600).",
    )
    parser.add_argument(
        "--colorbar-tick-count",
        type=int,
        default=13,
        help="Number of ticks on the dedicated colorbar page (default: 13).",
    )
    parser.add_argument(
        "--draw-borders",
        action="store_true",
        help="Draw tile border + registration marks (disabled by default for borderless printing).",
    )
    parser.add_argument(
        "--no-trim-marks",
        dest="trim_marks",
        action="store_false",
        help="Disable trim marks at tile cut lines.",
    )
    parser.set_defaults(trim_marks=True)
    parser.add_argument(
        "--no-colorbar-page",
        dest="colorbar_page",
        action="store_false",
        help="Do not append a dedicated A3 colorbar page.",
    )
    parser.set_defaults(colorbar_page=True)
    parser.add_argument(
        "--no-layout-page",
        dest="layout_page",
        action="store_false",
        help="Do not append a dedicated page-layout legend page.",
    )
    parser.set_defaults(layout_page=True)
    parser.add_argument("--dry-run", action="store_true", help="Print layout summary without generating a PDF.")
    return parser.parse_args()


def resolve_path(base_folder: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.join(base_folder, value)


def load_edges(path: str, axis_name: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{axis_name} edges file not found: {path}")
    data = np.loadtxt(path, delimiter=",", dtype=float)
    data = np.atleast_1d(data).astype(float)
    if data.size < 2:
        raise ValueError(f"{axis_name} edges must contain at least 2 values: {path}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{axis_name} edges contain non-finite values: {path}")
    if not np.all(np.diff(data) > 0):
        raise ValueError(f"{axis_name} edges must be strictly increasing: {path}")
    return data


def load_heatmap_grid(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Heatmap CSV file not found: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        raise ValueError(f"Heatmap CSV is empty: {path}")
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)

    required = {"x", "y", "z"}
    names = set(data.dtype.names or [])
    if not required.issubset(names):
        raise ValueError(f"Heatmap CSV must contain columns x,y,z: {path}")

    xs = np.asarray(data["x"], dtype=float)
    ys = np.asarray(data["y"], dtype=float)
    zs = np.asarray(data["z"], dtype=float)
    finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    if not finite.any():
        raise ValueError(f"Heatmap CSV has no finite rows: {path}")
    xs, ys, zs = xs[finite], ys[finite], zs[finite]

    x_unique = np.unique(xs)
    y_unique = np.unique(ys)
    if x_unique.size < 2 or y_unique.size < 2:
        raise ValueError(f"Heatmap CSV must contain at least a 2x2 grid: {path}")

    nx, ny = x_unique.size, y_unique.size
    xi = np.searchsorted(x_unique, xs)
    yi = np.searchsorted(y_unique, ys)
    flat = xi * ny + yi

    sums = np.bincount(flat, weights=zs, minlength=nx * ny)
    counts = np.bincount(flat, minlength=nx * ny)
    grid = np.full(nx * ny, np.nan, dtype=float)
    mask = counts > 0
    grid[mask] = sums[mask] / counts[mask]
    grid = grid.reshape(nx, ny)

    if not mask.all():
        missing = int((~mask).sum())
        raise ValueError(
            "Heatmap CSV does not form a full rectangular grid "
            f"(missing {missing} cells). Regenerate with filled heatmap export."
        )

    return x_unique, y_unique, grid


def compute_starts(total_mm: float, window_mm: float, step_mm: float) -> list[float]:
    if total_mm <= window_mm + 1e-9:
        return [0.0]

    starts = [0.0]
    while starts[-1] + window_mm < total_mm - 1e-9:
        nxt = starts[-1] + step_mm
        if nxt + window_mm >= total_mm - 1e-9:
            break
        starts.append(nxt)

    last = max(total_mm - window_mm, 0.0)
    if last - starts[-1] > 1e-9:
        starts.append(last)

    out: list[float] = []
    for s in starts:
        if not out or abs(s - out[-1]) > 1e-6:
            out.append(float(s))
    return out


def build_layout(
    map_w_mm: float,
    map_h_mm: float,
    page_mm: Tuple[float, float],
    margin_mm: float,
    bleed_mm: float,
    overlap_mm: float,
    orientation_name: str,
) -> Layout:
    page_w_mm, page_h_mm = page_mm
    usable_w_mm = page_w_mm - 2.0 * margin_mm
    usable_h_mm = page_h_mm - 2.0 * margin_mm
    if usable_w_mm <= 0 or usable_h_mm <= 0:
        raise ValueError(
            f"Invalid margin ({margin_mm} mm): printable area is non-positive "
            f"for A3 {orientation_name} ({page_w_mm} x {page_h_mm} mm)."
        )
    if bleed_mm < 0:
        raise ValueError("bleed-mm must be >= 0.")
    print_w_mm = usable_w_mm - 2.0 * bleed_mm
    print_h_mm = usable_h_mm - 2.0 * bleed_mm
    if print_w_mm <= 0 or print_h_mm <= 0:
        raise ValueError(
            f"Invalid bleed ({bleed_mm} mm): trim area is non-positive "
            f"after margins for A3 {orientation_name}."
        )
    if overlap_mm < 0:
        raise ValueError("overlap-mm must be >= 0.")

    step_x_mm = print_w_mm - overlap_mm
    step_y_mm = print_h_mm - overlap_mm
    if step_x_mm <= 0 or step_y_mm <= 0:
        raise ValueError(
            f"Invalid overlap ({overlap_mm} mm): stride must stay positive "
            f"(x stride={step_x_mm:.2f}, y stride={step_y_mm:.2f})."
        )

    x_starts_mm = compute_starts(map_w_mm, print_w_mm, step_x_mm)
    y_starts_mm = compute_starts(map_h_mm, print_h_mm, step_y_mm)
    return Layout(
        orientation=orientation_name,
        page_w_mm=page_w_mm,
        page_h_mm=page_h_mm,
        print_w_mm=print_w_mm,
        print_h_mm=print_h_mm,
        x_starts_mm=x_starts_mm,
        y_starts_mm=y_starts_mm,
        overlap_mm=overlap_mm,
        bleed_mm=bleed_mm,
    )


def choose_layout(
    map_w_mm: float,
    map_h_mm: float,
    margin_mm: float,
    bleed_mm: float,
    overlap_mm: float,
    orientation: str,
) -> Layout:
    portrait = build_layout(
        map_w_mm,
        map_h_mm,
        A3_PORTRAIT_MM,
        margin_mm,
        bleed_mm,
        overlap_mm,
        orientation_name="portrait",
    )
    landscape = build_layout(
        map_w_mm,
        map_h_mm,
        A3_LANDSCAPE_MM,
        margin_mm,
        bleed_mm,
        overlap_mm,
        orientation_name="landscape",
    )

    if orientation == "portrait":
        return portrait
    if orientation == "landscape":
        return landscape

    if portrait.pages != landscape.pages:
        return portrait if portrait.pages < landscape.pages else landscape

    portrait_area = portrait.print_w_mm * portrait.print_h_mm
    landscape_area = landscape.print_w_mm * landscape.print_h_mm
    if not math.isclose(portrait_area, landscape_area):
        return portrait if portrait_area > landscape_area else landscape
    return landscape


def nearest_indices(coords: np.ndarray, centers: np.ndarray) -> np.ndarray:
    mids = 0.5 * (centers[:-1] + centers[1:])
    idx = np.searchsorted(mids, coords, side="left")
    return np.clip(idx, 0, centers.size - 1)


def bounds_and_fraction(coords: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    right = np.searchsorted(centers, coords, side="right")
    left = right - 1
    left = np.clip(left, 0, centers.size - 1)
    right = np.clip(right, 0, centers.size - 1)

    x0 = centers[left]
    x1 = centers[right]
    denom = x1 - x0
    frac = np.zeros_like(coords, dtype=float)
    varying = denom > 0
    frac[varying] = (coords[varying] - x0[varying]) / denom[varying]
    return left, right, frac


def interpolate_chunk_nearest(
    z_grid: np.ndarray,
    ix_near: np.ndarray,
    iy_near: np.ndarray,
) -> np.ndarray:
    return z_grid[ix_near[None, :], iy_near[:, None]]


def interpolate_chunk_linear(
    z_grid: np.ndarray,
    ix0: np.ndarray,
    ix1: np.ndarray,
    tx: np.ndarray,
    iy0: np.ndarray,
    iy1: np.ndarray,
    ty: np.ndarray,
) -> np.ndarray:
    z00 = z_grid[ix0[None, :], iy0[:, None]]
    z10 = z_grid[ix1[None, :], iy0[:, None]]
    z01 = z_grid[ix0[None, :], iy1[:, None]]
    z11 = z_grid[ix1[None, :], iy1[:, None]]
    tx2 = tx[None, :]
    ty2 = ty[:, None]
    return (
        (1.0 - tx2) * (1.0 - ty2) * z00
        + tx2 * (1.0 - ty2) * z10
        + (1.0 - tx2) * ty2 * z01
        + tx2 * ty2 * z11
    )


def interpolate_chunk_idw(
    x_query: np.ndarray,
    y_query: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    z_grid: np.ndarray,
    ix0: np.ndarray,
    ix1: np.ndarray,
    iy0: np.ndarray,
    iy1: np.ndarray,
    p: float,
) -> np.ndarray:
    if p <= 0:
        raise ValueError("idw-power must be > 0.")

    z00 = z_grid[ix0[None, :], iy0[:, None]]
    z10 = z_grid[ix1[None, :], iy0[:, None]]
    z01 = z_grid[ix0[None, :], iy1[:, None]]
    z11 = z_grid[ix1[None, :], iy1[:, None]]

    dx0 = np.abs(x_query - x_centers[ix0])
    dx1 = np.abs(x_query - x_centers[ix1])
    dy0 = np.abs(y_query - y_centers[iy0])
    dy1 = np.abs(y_query - y_centers[iy1])

    d00 = np.hypot(dx0[None, :], dy0[:, None])
    d10 = np.hypot(dx1[None, :], dy0[:, None])
    d01 = np.hypot(dx0[None, :], dy1[:, None])
    d11 = np.hypot(dx1[None, :], dy1[:, None])

    w00 = 1.0 / (np.power(d00, p) + EPS)
    w10 = 1.0 / (np.power(d10, p) + EPS)
    w01 = 1.0 / (np.power(d01, p) + EPS)
    w11 = 1.0 / (np.power(d11, p) + EPS)

    numer = w00 * z00 + w10 * z10 + w01 * z01 + w11 * z11
    denom = w00 + w10 + w01 + w11
    out = numer / denom

    # Preserve exact values at exact sample centers (distance == 0).
    z_exact = np.full_like(out, np.nan, dtype=float)
    zero00 = d00 == 0.0
    zero10 = d10 == 0.0
    zero01 = d01 == 0.0
    zero11 = d11 == 0.0
    any_zero = zero00 | zero10 | zero01 | zero11
    if np.any(any_zero):
        z_exact = np.where(zero00, z00, z_exact)
        z_exact = np.where(np.isnan(z_exact) & zero10, z10, z_exact)
        z_exact = np.where(np.isnan(z_exact) & zero01, z01, z_exact)
        z_exact = np.where(np.isnan(z_exact) & zero11, z11, z_exact)
        out = np.where(any_zero, z_exact, out)
    return out


def render_tile_power(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    z_grid: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    px_w: int,
    px_h: int,
    method: str,
    idw_power: float,
    chunk_rows: int = 256,
) -> np.ndarray:
    if px_w <= 0 or px_h <= 0:
        raise ValueError(f"Tile pixel size must be positive, got {px_w}x{px_h}.")

    x_step = (x_max - x_min) / px_w
    y_step = (y_max - y_min) / px_h
    xq = x_min + (np.arange(px_w, dtype=float) + 0.5) * x_step
    yq = y_min + (np.arange(px_h, dtype=float) + 0.5) * y_step

    xq = np.clip(xq, x_centers[0], x_centers[-1])
    yq = np.clip(yq, y_centers[0], y_centers[-1])

    out = np.empty((px_h, px_w), dtype=float)

    if method == "nearest":
        ix_near = nearest_indices(xq, x_centers)
        for y0 in range(0, px_h, chunk_rows):
            y1 = min(y0 + chunk_rows, px_h)
            iy_near = nearest_indices(yq[y0:y1], y_centers)
            out[y0:y1, :] = interpolate_chunk_nearest(z_grid, ix_near, iy_near)
        return out

    ix0, ix1, tx = bounds_and_fraction(xq, x_centers)
    for y0 in range(0, px_h, chunk_rows):
        y1 = min(y0 + chunk_rows, px_h)
        yc = yq[y0:y1]
        iy0, iy1, ty = bounds_and_fraction(yc, y_centers)
        if method == "linear":
            out[y0:y1, :] = interpolate_chunk_linear(z_grid, ix0, ix1, tx, iy0, iy1, ty)
        elif method == "idw":
            out[y0:y1, :] = interpolate_chunk_idw(
                xq,
                yc,
                x_centers,
                y_centers,
                z_grid,
                ix0,
                ix1,
                iy0,
                iy1,
                p=idw_power,
            )
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")

    return out


def draw_registration_marks(
    ax: plt.Axes,
    left: float,
    bottom: float,
    width: float,
    height: float,
    mark_len_mm: float = 5.0,
    lw: float = 0.8,
) -> None:
    right = left + width
    top = bottom + height

    # Bottom-left
    ax.plot([left - mark_len_mm, left], [bottom, bottom], color="black", linewidth=lw)
    ax.plot([left, left], [bottom - mark_len_mm, bottom], color="black", linewidth=lw)
    # Bottom-right
    ax.plot([right, right + mark_len_mm], [bottom, bottom], color="black", linewidth=lw)
    ax.plot([right, right], [bottom - mark_len_mm, bottom], color="black", linewidth=lw)
    # Top-left
    ax.plot([left - mark_len_mm, left], [top, top], color="black", linewidth=lw)
    ax.plot([left, left], [top, top + mark_len_mm], color="black", linewidth=lw)
    # Top-right
    ax.plot([right, right + mark_len_mm], [top, top], color="black", linewidth=lw)
    ax.plot([right, right], [top, top + mark_len_mm], color="black", linewidth=lw)


def print_summary(
    map_w_mm: float,
    map_h_mm: float,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    z_grid: np.ndarray,
    layout: Layout,
    args: argparse.Namespace,
    output_path: str,
) -> None:
    finite = np.isfinite(z_grid)
    zmin = float(np.nanmin(z_grid[finite]))
    zmax = float(np.nanmax(z_grid[finite]))
    vmin = args.vmin if args.vmin is not None else zmin
    vmax = args.vmax if args.vmax is not None else zmax

    print("Heatmap floor-print configuration:")
    print(f"- folder: {os.path.abspath(args.folder)}")
    print(f"- heatmap csv: {os.path.abspath(resolve_path(args.folder, args.heatmap_csv))}")
    print(f"- x/y edges: {os.path.abspath(resolve_path(args.folder, args.x_edges))}, {os.path.abspath(resolve_path(args.folder, args.y_edges))}")
    print(f"- source grid: {x_centers.size} x {y_centers.size} centers")
    print(
        f"- map extent: x [{x_edges[0]:.6f}, {x_edges[-1]:.6f}] m, "
        f"y [{y_edges[0]:.6f}, {y_edges[-1]:.6f}] m"
    )
    print(f"- map physical size: {map_w_mm:.1f} mm x {map_h_mm:.1f} mm ({map_w_mm/1000:.3f} m x {map_h_mm/1000:.3f} m)")
    print(
        f"- orientation: {layout.orientation} (A3 {layout.page_w_mm:.0f} x {layout.page_h_mm:.0f} mm), "
        f"trim coverage: {layout.print_w_mm:.1f} x {layout.print_h_mm:.1f} mm"
    )
    print(f"- cut margin: {args.margin_mm:.1f} mm")
    print(f"- bleed: {layout.bleed_mm:.1f} mm")
    print(f"- overlap: {layout.overlap_mm:.1f} mm")
    print(f"- tile grid: {layout.cols} cols x {layout.rows} rows = {layout.pages} pages")
    print(f"- draw borders/marks: {args.draw_borders}")
    print(f"- trim marks: {args.trim_marks}")
    print(f"- separate colorbar page: {args.colorbar_page}")
    print(f"- separate layout page: {args.layout_page}")
    total_pages = layout.pages + (1 if args.colorbar_page else 0) + (1 if args.layout_page else 0)
    print(f"- total PDF pages: {total_pages}")
    print(f"- render dpi: {args.render_dpi:.1f}")
    print(f"- interpolation: {args.interp} (idw-power={args.idw_power})")
    print(f"- cmap: {args.cmap}, value range: [{vmin:.6g}, {vmax:.6g}]")
    print(f"- colorbar dpi: {args.colorbar_dpi:.1f}, ticks: {args.colorbar_tick_count}")
    print(f"- output: {os.path.abspath(output_path)}")
    if args.dry_run:
        print("- dry-run: true (no PDF written)")


def build_tile_placements(
    layout: Layout,
    x0_m: float,
    y0_m: float,
    map_w_mm: float,
    map_h_mm: float,
) -> list[TilePlacement]:
    placements: list[TilePlacement] = []
    x_starts = layout.x_starts_mm
    y_starts_top_to_bottom = list(reversed(layout.y_starts_mm))

    page_num = 1
    for row_idx, y_start_mm in enumerate(y_starts_top_to_bottom, start=1):
        for col_idx, x_start_mm in enumerate(x_starts, start=1):
            tile_w_mm = min(layout.print_w_mm, map_w_mm - x_start_mm)
            tile_h_mm = min(layout.print_h_mm, map_h_mm - y_start_mm)
            x_min = x0_m + x_start_mm / 1000.0
            x_max = x_min + tile_w_mm / 1000.0
            y_min = y0_m + y_start_mm / 1000.0
            y_max = y_min + tile_h_mm / 1000.0
            placements.append(
                TilePlacement(
                    page_num=page_num,
                    row=row_idx,
                    col=col_idx,
                    x_start_mm=x_start_mm,
                    y_start_mm=y_start_mm,
                    tile_w_mm=tile_w_mm,
                    tile_h_mm=tile_h_mm,
                    x_min_m=x_min,
                    x_max_m=x_max,
                    y_min_m=y_min,
                    y_max_m=y_max,
                )
            )
            page_num += 1
    return placements


def print_pdf_layout_mapping(
    placements: list[TilePlacement],
    args: argparse.Namespace,
) -> None:
    print("\nPDF layout mapping:")
    if not placements:
        print("- no tile pages")
        return

    for p in placements:
        print(
            f"- page {p.page_num:>2}: R{p.row}C{p.col} | "
            f"x [{p.x_min_m:.3f}, {p.x_max_m:.3f}] m | "
            f"y [{p.y_min_m:.3f}, {p.y_max_m:.3f}] m"
        )

    print("- grid view (top row printed first):")
    max_page_digits = len(str(max(p.page_num for p in placements)))
    for row in sorted({p.row for p in placements}):
        row_items = [p for p in placements if p.row == row]
        row_items.sort(key=lambda x: x.col)
        labels = [f"p{p.page_num:0{max_page_digits}d}(R{p.row}C{p.col})" for p in row_items]
        print(f"  row {row}: " + " | ".join(labels))

    next_page = len(placements) + 1
    if args.colorbar_page:
        print(f"- page {next_page:>2}: colorbar [uW]")
        next_page += 1
    if args.layout_page:
        print(f"- page {next_page:>2}: layout legend")


def validate_value_range(z_grid: np.ndarray, vmin: float | None, vmax: float | None) -> Tuple[float, float]:
    finite = z_grid[np.isfinite(z_grid)]
    if finite.size == 0:
        raise ValueError("Source heatmap grid has no finite values.")
    vmin_use = float(np.nanmin(finite)) if vmin is None else float(vmin)
    vmax_use = float(np.nanmax(finite)) if vmax is None else float(vmax)
    if not np.isfinite(vmin_use) or not np.isfinite(vmax_use):
        raise ValueError("vmin/vmax must be finite.")
    if vmin_use >= vmax_use:
        raise ValueError(f"vmin must be smaller than vmax (got {vmin_use} >= {vmax_use}).")
    return vmin_use, vmax_use


def write_tiled_pdf(
    output_path: str,
    layout: Layout,
    x0_m: float,
    y0_m: float,
    map_w_mm: float,
    map_h_mm: float,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    z_grid: np.ndarray,
    args: argparse.Namespace,
    vmin: float,
    vmax: float,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(args.cmap)
    placements = build_tile_placements(
        layout=layout,
        x0_m=x0_m,
        y0_m=y0_m,
        map_w_mm=map_w_mm,
        map_h_mm=map_h_mm,
    )

    with PdfPages(output_path) as pdf:
        for p in placements:
            map_x_max_m = x0_m + map_w_mm / 1000.0
            map_y_max_m = y0_m + map_h_mm / 1000.0
            bleed_left_mm = min(layout.bleed_mm, max(0.0, (p.x_min_m - x0_m) * 1000.0))
            bleed_right_mm = min(layout.bleed_mm, max(0.0, (map_x_max_m - p.x_max_m) * 1000.0))
            bleed_bottom_mm = min(layout.bleed_mm, max(0.0, (p.y_min_m - y0_m) * 1000.0))
            bleed_top_mm = min(layout.bleed_mm, max(0.0, (map_y_max_m - p.y_max_m) * 1000.0))

            x_min_bleed_m = p.x_min_m - bleed_left_mm / 1000.0
            x_max_bleed_m = p.x_max_m + bleed_right_mm / 1000.0
            y_min_bleed_m = p.y_min_m - bleed_bottom_mm / 1000.0
            y_max_bleed_m = p.y_max_m + bleed_top_mm / 1000.0

            bleed_w_mm = p.tile_w_mm + bleed_left_mm + bleed_right_mm
            bleed_h_mm = p.tile_h_mm + bleed_bottom_mm + bleed_top_mm

            px_w = max(2, int(round(bleed_w_mm / MM_PER_INCH * args.render_dpi)))
            px_h = max(2, int(round(bleed_h_mm / MM_PER_INCH * args.render_dpi)))
            tile_power = render_tile_power(
                x_centers=x_centers,
                y_centers=y_centers,
                z_grid=z_grid,
                x_min=x_min_bleed_m,
                x_max=x_max_bleed_m,
                y_min=y_min_bleed_m,
                y_max=y_max_bleed_m,
                px_w=px_w,
                px_h=px_h,
                method=args.interp,
                idw_power=args.idw_power,
                chunk_rows=256,
            )

            fig = plt.figure(
                figsize=(layout.page_w_mm / MM_PER_INCH, layout.page_h_mm / MM_PER_INCH)
            )
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.set_xlim(0.0, layout.page_w_mm)
            ax.set_ylim(0.0, layout.page_h_mm)
            ax.axis("off")

            trim_left_mm = args.margin_mm + layout.bleed_mm
            trim_bottom_mm = args.margin_mm + layout.bleed_mm
            image_left_mm = trim_left_mm - bleed_left_mm
            image_bottom_mm = trim_bottom_mm - bleed_bottom_mm
            ax.imshow(
                tile_power,
                origin="lower",
                extent=[
                    image_left_mm,
                    image_left_mm + bleed_w_mm,
                    image_bottom_mm,
                    image_bottom_mm + bleed_h_mm,
                ],
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="equal",
            )

            if args.draw_borders:
                ax.add_patch(
                    plt.Rectangle(
                        (trim_left_mm, trim_bottom_mm),
                        p.tile_w_mm,
                        p.tile_h_mm,
                        fill=False,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                )
            if args.trim_marks and layout.bleed_mm > 0:
                mark_len_mm = min(4.0, layout.bleed_mm - 0.2)
                if mark_len_mm > 0.2:
                    draw_registration_marks(
                        ax,
                        trim_left_mm,
                        trim_bottom_mm,
                        p.tile_w_mm,
                        p.tile_h_mm,
                        mark_len_mm=mark_len_mm,
                        lw=0.7,
                    )

            page_label = f"R{p.row}C{p.col}"
            span = (
                f"page {p.page_num} | "
                f"x [{p.x_min_m:.3f}, {p.x_max_m:.3f}] m | "
                f"y [{p.y_min_m:.3f}, {p.y_max_m:.3f}] m | "
                f"{p.tile_w_mm:.0f}x{p.tile_h_mm:.0f} mm"
            )
            top_text_y = layout.page_h_mm - max(args.margin_mm * 0.45, 4.0)
            bottom_text_y = max(args.margin_mm * 0.35, 3.0)
            ax.text(
                trim_left_mm,
                top_text_y,
                f"{page_label}  {span}",
                fontsize=7,
                ha="left",
                va="center",
                color="black",
            )
            ax.text(
                trim_left_mm,
                bottom_text_y,
                "Print at 100% / Actual size. Disable fit-to-page.",
                fontsize=7,
                ha="left",
                va="center",
                color="black",
            )

            # 100 mm calibration segment for print-scale verification.
            calib_mm = min(100.0, layout.page_w_mm - (trim_left_mm + 30.0))
            if calib_mm >= 30.0:
                x1 = layout.page_w_mm - trim_left_mm - calib_mm
                x2 = layout.page_w_mm - trim_left_mm
                yb = bottom_text_y
                ax.plot([x1, x2], [yb, yb], color="black", linewidth=1.0)
                ax.plot([x1, x1], [yb - 1.0, yb + 1.0], color="black", linewidth=1.0)
                ax.plot([x2, x2], [yb - 1.0, yb + 1.0], color="black", linewidth=1.0)
                ax.text((x1 + x2) / 2.0, yb + 1.6, f"{int(round(calib_mm))} mm", fontsize=6, ha="center", va="bottom")

            pdf.savefig(fig)
            plt.close(fig)

        if args.colorbar_page:
            fig = plt.figure(
                figsize=(layout.page_w_mm / MM_PER_INCH, layout.page_h_mm / MM_PER_INCH),
                dpi=args.colorbar_dpi,
            )
            bg = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            bg.axis("off")

            is_landscape = layout.page_w_mm >= layout.page_h_mm
            if is_landscape:
                cax = fig.add_axes([0.12, 0.43, 0.76, 0.16])
                cbar_orientation = "horizontal"
            else:
                cax = fig.add_axes([0.44, 0.14, 0.12, 0.72])
                cbar_orientation = "vertical"

            scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            scalar_map.set_array([])
            cbar = fig.colorbar(scalar_map, cax=cax, orientation=cbar_orientation)
            ticks = np.linspace(vmin, vmax, args.colorbar_tick_count)
            cbar.set_ticks(ticks)
            cbar.formatter = FuncFormatter(lambda x, _: f"{x:.3g}")
            cbar.update_ticks()
            cbar.set_label("Power [uW]", fontsize=16)
            cbar.ax.tick_params(labelsize=12)

            bg.text(
                0.5,
                0.93,
                "Heatmap Colorbar",
                ha="center",
                va="center",
                fontsize=22,
                transform=bg.transAxes,
            )
            bg.text(
                0.5,
                0.87,
                f"Range: {vmin:.6g} to {vmax:.6g} uW",
                ha="center",
                va="center",
                fontsize=12,
                transform=bg.transAxes,
            )
            bg.text(
                0.5,
                0.05,
                "Print at 100% / Actual size. Disable fit-to-page.",
                ha="center",
                va="center",
                fontsize=10,
                transform=bg.transAxes,
            )

            pdf.savefig(fig, dpi=args.colorbar_dpi)
            plt.close(fig)

        if args.layout_page:
            fig = plt.figure(
                figsize=(layout.page_w_mm / MM_PER_INCH, layout.page_h_mm / MM_PER_INCH)
            )
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.set_xlim(0.0, layout.page_w_mm)
            ax.set_ylim(0.0, layout.page_h_mm)
            ax.axis("off")

            grid_left = 0.08 * layout.page_w_mm
            grid_right = 0.92 * layout.page_w_mm
            grid_bottom = 0.14 * layout.page_h_mm
            grid_top = 0.82 * layout.page_h_mm
            grid_w = grid_right - grid_left
            grid_h = grid_top - grid_bottom
            cell_w = grid_w / layout.cols
            cell_h = grid_h / layout.rows

            ax.text(
                layout.page_w_mm / 2.0,
                0.93 * layout.page_h_mm,
                "Tile Layout Legend",
                ha="center",
                va="center",
                fontsize=20,
            )
            ax.text(
                layout.page_w_mm / 2.0,
                0.88 * layout.page_h_mm,
                "Use this page to place printed sheets on the floor.",
                ha="center",
                va="center",
                fontsize=11,
            )

            for p in placements:
                cell_left = grid_left + (p.col - 1) * cell_w
                cell_bottom = grid_top - p.row * cell_h
                ax.add_patch(
                    plt.Rectangle(
                        (cell_left, cell_bottom),
                        cell_w,
                        cell_h,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.0,
                    )
                )
                ax.text(
                    cell_left + cell_w / 2.0,
                    cell_bottom + 0.62 * cell_h,
                    f"R{p.row}C{p.col}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.text(
                    cell_left + cell_w / 2.0,
                    cell_bottom + 0.40 * cell_h,
                    f"PDF page {p.page_num}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
                ax.text(
                    cell_left + cell_w / 2.0,
                    cell_bottom + 0.22 * cell_h,
                    f"x {p.x_min_m:.2f}-{p.x_max_m:.2f} m",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
                ax.text(
                    cell_left + cell_w / 2.0,
                    cell_bottom + 0.10 * cell_h,
                    f"y {p.y_min_m:.2f}-{p.y_max_m:.2f} m",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

            ax.text(
                grid_left,
                grid_top + 0.02 * layout.page_h_mm,
                "TOP (higher y)",
                ha="left",
                va="bottom",
                fontsize=9,
            )
            ax.text(
                grid_left,
                grid_bottom - 0.035 * layout.page_h_mm,
                "BOTTOM (lower y)",
                ha="left",
                va="top",
                fontsize=9,
            )
            ax.text(
                grid_left - 0.01 * layout.page_w_mm,
                grid_bottom - 0.06 * layout.page_h_mm,
                "LEFT (lower x)",
                ha="left",
                va="top",
                fontsize=9,
            )
            ax.text(
                grid_right,
                grid_bottom - 0.06 * layout.page_h_mm,
                "RIGHT (higher x)",
                ha="right",
                va="top",
                fontsize=9,
            )

            next_page = len(placements) + 1
            legend_items = []
            if args.colorbar_page:
                legend_items.append(f"Colorbar: PDF page {next_page}")
                next_page += 1
            legend_items.append(f"This layout legend: PDF page {next_page}")
            ax.text(
                layout.page_w_mm / 2.0,
                0.06 * layout.page_h_mm,
                " | ".join(legend_items),
                ha="center",
                va="center",
                fontsize=10,
            )

            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.render_dpi <= 0:
        raise ValueError("render-dpi must be > 0.")
    if args.colorbar_dpi <= 0:
        raise ValueError("colorbar-dpi must be > 0.")
    if args.colorbar_tick_count < 2:
        raise ValueError("colorbar-tick-count must be >= 2.")
    if args.idw_power <= 0:
        raise ValueError("idw-power must be > 0.")
    if args.bleed_mm < 0:
        raise ValueError("bleed-mm must be >= 0.")
    if args.margin_mm < 0:
        raise ValueError("margin-mm must be >= 0.")

    folder = args.folder
    if not os.path.isabs(folder):
        folder_cwd = os.path.abspath(folder)
        folder_repo = os.path.abspath(os.path.join(PROJECT_ROOT, folder))
        folder = folder_cwd if os.path.isdir(folder_cwd) else folder_repo
    args.folder = folder
    heatmap_csv_path = resolve_path(folder, args.heatmap_csv)
    x_edges_path = resolve_path(folder, args.x_edges)
    y_edges_path = resolve_path(folder, args.y_edges)
    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)
    else:
        output_path = os.path.abspath(
            os.path.join(folder, "heatmap_A3_tiled_idw2_150dpi.pdf")
        )

    x_edges = load_edges(x_edges_path, "x")
    y_edges = load_edges(y_edges_path, "y")
    x_centers, y_centers, z_grid = load_heatmap_grid(heatmap_csv_path)

    map_w_mm = float((x_edges[-1] - x_edges[0]) * 1000.0)
    map_h_mm = float((y_edges[-1] - y_edges[0]) * 1000.0)
    if map_w_mm <= 0 or map_h_mm <= 0:
        raise ValueError(
            f"Invalid physical map size from edges: {map_w_mm:.3f} x {map_h_mm:.3f} mm"
        )

    layout = choose_layout(
        map_w_mm=map_w_mm,
        map_h_mm=map_h_mm,
        margin_mm=args.margin_mm,
        bleed_mm=args.bleed_mm,
        overlap_mm=args.overlap_mm,
        orientation=args.orientation,
    )
    placements = build_tile_placements(
        layout=layout,
        x0_m=float(x_edges[0]),
        y0_m=float(y_edges[0]),
        map_w_mm=map_w_mm,
        map_h_mm=map_h_mm,
    )

    vmin, vmax = validate_value_range(z_grid, args.vmin, args.vmax)
    print_summary(
        map_w_mm=map_w_mm,
        map_h_mm=map_h_mm,
        x_edges=x_edges,
        y_edges=y_edges,
        x_centers=x_centers,
        y_centers=y_centers,
        z_grid=z_grid,
        layout=layout,
        args=args,
        output_path=output_path,
    )
    print_pdf_layout_mapping(placements, args)

    if args.dry_run:
        return

    write_tiled_pdf(
        output_path=output_path,
        layout=layout,
        x0_m=float(x_edges[0]),
        y0_m=float(y_edges[0]),
        map_w_mm=map_w_mm,
        map_h_mm=map_h_mm,
        x_centers=x_centers,
        y_centers=y_centers,
        z_grid=z_grid,
        args=args,
        vmin=vmin,
        vmax=vmax,
    )
    print(f"Wrote tiled A3 PDF to {output_path}")


if __name__ == "__main__":
    main()
