"""
Aggregate all position/value pairs inside each subfolder of ../data
and plot a heatmap of mean power for the concatenated samples.
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
try:
    from scipy.ndimage import distance_transform_edt
    HAS_DT = True
except Exception:
    HAS_DT = False


class scope_data(object):
    def __init__(self, pwr_pw):
        self.pwr_pw = pwr_pw


WAVELENGTH = 3e8 / 920e6  # meters

GRID_RES = 0.1 * WAVELENGTH  # meters (default; overridden by --grid-res-lambda)
SMALL_POWER_UW = 1e-8  # threshold for reporting tiny measurements (micro-watts)


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
SETTINGS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiment-settings.yaml"))
CMAP = "inferno"

# Ensure pickle can resolve project modules referenced in saved arrays
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_target_from_settings(settings_path=SETTINGS_PATH):
    """Return target_location from experiment-settings.yaml as [x, y, z?]."""
    if not os.path.exists(settings_path):
        return None
    try:
        with open(settings_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh) or {}
        target = settings.get("experiment_config", {}).get("target_location")
        if target is None:
            return None
        if isinstance(target, str):
            parts = [p.strip() for p in target.split(",") if p.strip()]
        elif isinstance(target, (list, tuple)):
            parts = list(target)
        else:
            return None
        vals = [float(p) for p in parts]
        return vals if len(vals) >= 2 else None
    except Exception as exc:
        print(f"Failed to load target_location from {settings_path}: {exc}", file=sys.stderr)
        return None


def target_rect_from_xyz(target_xyz, rect_size=0.2 * WAVELENGTH):
    """Rectangle of fixed size (default 0.5 lambda) centered on target x/y."""
    if not target_xyz or len(target_xyz) < 2:
        return None
    tx, ty = target_xyz[0], target_xyz[1]
    half = rect_size / 2
    return (tx - half, ty - half, rect_size, rect_size)


def load_folder(folder_path):
    """Load and concatenate all *_positions.npy and *_values.npy pairs in a folder."""
    positions_parts = []
    values_parts = []

    for name in sorted(os.listdir(folder_path)):
        if not name.endswith("_positions.npy"):
            continue
        base = name[: -len("_positions.npy")]
        pos_path = os.path.join(folder_path, name)
        val_path = os.path.join(folder_path, f"{base}_values.npy")
        if not os.path.exists(val_path):
            print(f"Skipping {base}: missing values file")
            continue
        pos_arr = np.load(pos_path, allow_pickle=True)
        val_arr = np.load(val_path, allow_pickle=True)
        if len(pos_arr) != len(val_arr):
            min_len = min(len(pos_arr), len(val_arr))
            print(
                f"\033[91mWarning: {base} positions ({len(pos_arr)}) != values ({len(val_arr)}); "
                f"truncating both to {min_len}\033[0m"
            )
            pos_arr = pos_arr[:min_len]
            val_arr = val_arr[:min_len]
        positions_parts.append(pos_arr)
        values_parts.append(val_arr)

    if not positions_parts:
        raise ValueError(f"No position/value pairs found in {folder_path}")

    positions = np.concatenate(positions_parts)
    values = np.concatenate(values_parts)
    print(f"{os.path.basename(folder_path)}: merged {len(positions_parts)} pairs, {len(positions)} samples")
    return positions, values


def filter_small_values(folder_path, positions, values, vs, threshold=SMALL_POWER_UW):
    """
    Log and drop zero or near-zero power samples (threshold in uW).
    Returns filtered positions, values, and vs arrays.
    """
    zeros = vs == 0.0
    small = (vs > 0.0) & (vs < threshold)
    drop_mask = ~(zeros | small)

    reports = []
    if zeros.any():
        reports.append(f"{zeros.sum()} zeros")
    if small.any():
        reports.append(f"{small.sum()} below {threshold:.1e} uW (min {vs[small].min():.2e})")

    dropped = len(vs) - int(drop_mask.sum())
    report = ", ".join(reports) if reports else ""
    if dropped:
        return positions[drop_mask], values[drop_mask], vs[drop_mask], dropped, report

    return positions, values, vs, 0, report


def drop_consecutive_equal_values(positions, values):
    """
    Remove runs of consecutive measurements that have identical power.
    The same indices are removed from positions to keep arrays aligned.
    """
    if len(positions) != len(values):
        min_len = min(len(positions), len(values))
        print(
            f"Warning: length mismatch positions={len(positions)} values={len(values)}; truncating to {min_len}"
        )
        positions = positions[:min_len]
        values = values[:min_len]

    keep_idx = [0]
    last_power = values[0].pwr_pw
    for idx in range(1, len(values)):
        if values[idx].pwr_pw != last_power:
            keep_idx.append(idx)
            last_power = values[idx].pwr_pw

    if len(keep_idx) == len(values):
        return positions, values, 0

    dropped = len(values) - len(keep_idx)
    return positions[keep_idx], values[keep_idx], dropped


def drop_nonincreasing_timestamps(positions, values):
    """
    Drop samples whose position timestamp does not increase vs. the previous one.
    Assumes any duplicates/non-increasing timestamps occur consecutively.
    """
    if len(positions) != len(values):
        min_len = min(len(positions), len(values))
        print(
            f"Warning: length mismatch positions={len(positions)} values={len(values)}; truncating to {min_len}"
        )
        positions = positions[:min_len]
        values = values[:min_len]

    if len(positions) <= 1:
        return positions, values

    keep_idx = [0]
    equal_count = 0
    decrease_count = 0
    last_t = getattr(positions[0], "t", None)
    for idx in range(1, len(positions)):
        curr_t = getattr(positions[idx], "t", None)
        if last_t is None or curr_t is None:
            keep_idx.append(idx)
            last_t = curr_t
            continue
        if curr_t > last_t:
            keep_idx.append(idx)
            last_t = curr_t
        elif curr_t == last_t:
            equal_count += 1
        else:
            decrease_count += 1

    if len(keep_idx) == len(positions):
        return positions, values, 0, {"equal": 0, "decrease": 0}

    dropped = len(positions) - len(keep_idx)
    return positions[keep_idx], values[keep_idx], dropped, {
        "equal": equal_count,
        "decrease": decrease_count,
    }


def heatmap_delta_db(curr_heatmap, base_heatmap):
    """
    Compute delta in dB: 10*log10(curr) - 10*log10(base).
    Cells with non-positive or NaN values in either map become NaN.
    """
    diff = np.full_like(curr_heatmap, np.nan, dtype=float)
    valid = (
        np.isfinite(curr_heatmap)
        & np.isfinite(base_heatmap)
        & (curr_heatmap > 0)
        & (base_heatmap > 0)
    )
    if not np.any(valid):
        return diff

    curr_db = np.zeros_like(curr_heatmap, dtype=float)
    base_db = np.zeros_like(base_heatmap, dtype=float)
    curr_db[valid] = 10 * np.log10(curr_heatmap[valid])
    base_db[valid] = 10 * np.log10(base_heatmap[valid])
    diff[valid] = curr_db[valid] - base_db[valid]
    return diff


def _target_mask(x_edges, y_edges, target_rect):
    if not target_rect:
        return None
    x0, y0, w, h = target_rect
    x1, y1 = x0 + w, y0 + h
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    mask_x = (xc >= x0) & (xc <= x1)
    mask_y = (yc >= y0) & (yc <= y1)
    if not mask_x.any() or not mask_y.any():
        return None
    return np.outer(mask_x, mask_y)


def gain_stats(curr, base, x_edges, y_edges, target_rect=None):
    """Return avg/max gain (linear and dB) vs baseline; optionally within target rect."""
    mask = (
        np.isfinite(curr)
        & np.isfinite(base)
        & (curr > 0)
        & (base > 0)
    )
    def _stats(mask_local):
        if mask_local is None or not np.any(mask_local):
            return None
        ratio = curr[mask_local] / base[mask_local]
        avg_lin = float(np.mean(ratio))
        max_lin = float(np.max(ratio))
        return {
            "avg_lin": avg_lin,
            "max_lin": max_lin,
            "avg_db": 10 * np.log10(avg_lin),
            "max_db": 10 * np.log10(max_lin),
        }

    global_stats = _stats(mask)
    target_mask = _target_mask(x_edges, y_edges, target_rect)
    target_stats = _stats(mask & target_mask) if target_mask is not None else None
    return global_stats, target_stats


def compute_heatmap(xs, ys, vs, grid_res, agg="median", x_edges=None, y_edges=None):
    """Bin values onto a 2D grid and compute an aggregate power per cell."""
    if x_edges is None or y_edges is None:
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        x_edges = np.arange(min_x, max_x + grid_res, grid_res)
        y_edges = np.arange(min_y, max_y + grid_res, grid_res)

    heatmap = np.full((len(x_edges) - 1, len(y_edges) - 1), np.nan, dtype=float)
    if agg not in {"mean", "median", "max", "min"}:
        raise ValueError("agg must be one of: mean, median, max, min")
    cell_values = defaultdict(list)
    counts = np.zeros_like(heatmap, dtype=int)

    xi = np.digitize(xs, x_edges) - 1
    yi = np.digitize(ys, y_edges) - 1

    for i_x, i_y, v in zip(xi, yi, vs):
        if 0 <= i_x < heatmap.shape[0] and 0 <= i_y < heatmap.shape[1]:
            cell_values[(i_x, i_y)].append(v)
            counts[i_x, i_y] += 1

    agg_funcs = {
        "mean": np.mean,
        "median": np.median,
        "max": np.max,
        "min": np.min,
    }
    func = agg_funcs[agg]
    for (i_x, i_y), values in cell_values.items():
        if values:
            heatmap[i_x, i_y] = float(func(values))
    return heatmap, counts, x_edges, y_edges, xi, yi


def fill_empty_cells_nearest(heatmap):
    """Fill NaN cells with nearest non-NaN value using distance transform."""
    if not HAS_DT:
        print("Warning: scipy.ndimage.distance_transform_edt not available; cannot fill empty cells.")
        return heatmap
    mask = np.isnan(heatmap)
    if not mask.any():
        return heatmap
    filled = heatmap.copy()
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    filled[mask] = heatmap[tuple(idx[:, mask])]
    return filled


def plot_heatmap(
    folder,
    heatmap,
    counts,
    x_edges,
    y_edges,
    recent_cells=None,
    target_rect=None,
    agg="mean",
    show=True,
    save_bitmap=False,
    png_name="heatmap.png",
    bitmap_name="heatmap_bitmap.png",
):
    """Render heatmaps with axes in meters (linear uW and dBm)."""

    def _draw(ax, add_axes=True):
        img = ax.imshow(
            heatmap.T,
            origin="lower",
            cmap=CMAP,
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        )
        ax.set_aspect("equal", adjustable="box")
        if add_axes:
            agg_label = "Median" if agg == "median" else "Mean"
            ax.set_title(f"{os.path.basename(folder)} | {agg_label.lower()} power per cell [uW]")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            cbar = plt.colorbar(img, ax=ax)
            cbar.ax.set_ylabel(f"{agg_label} power per cell [uW]")
        if recent_cells:
            for idx, (i_x, i_y) in enumerate(recent_cells):
                if 0 <= i_x < len(x_edges) - 1 and 0 <= i_y < len(y_edges) - 1:
                    edgecolor = "lime" if idx == len(recent_cells) - 1 else "red"
                    ax.add_patch(
                        plt.Rectangle(
                            (x_edges[i_x], y_edges[i_y]),
                            x_edges[i_x + 1] - x_edges[i_x],
                            y_edges[i_y + 1] - y_edges[i_y],
                            fill=False,
                            edgecolor=edgecolor,
                            linewidth=2,
                        )
                    )
        if target_rect and add_axes:
            x0, y0, w, h = target_rect
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0),
                    w,
                    h,
                    fill=False,
                    edgecolor="green",
                    linewidth=2,
                )
            )
        if not add_axes:
            ax.axis("off")
        return img

    fig, ax = plt.subplots()
    _draw(ax, add_axes=True)
    fig.tight_layout()
    plt.savefig(os.path.join(folder, png_name))
    if show:
        plt.show()
    else:
        plt.close(fig)

    if save_bitmap:
        fig2, ax2 = plt.subplots()
        _draw(ax2, add_axes=False)
        fig2.tight_layout(pad=0)
        plt.savefig(os.path.join(folder, bitmap_name), bbox_inches="tight", pad_inches=0)
        plt.close(fig2)

    # Counts heatmap
    fig_counts, ax_counts = plt.subplots()
    img_counts = ax_counts.imshow(
        counts.T,
        origin="lower",
        cmap="viridis",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    ax_counts.set_aspect("equal", adjustable="box")
    ax_counts.set_title(f"{os.path.basename(folder)} | samples per cell")
    ax_counts.set_xlabel("x [m]")
    ax_counts.set_ylabel("y [m]")
    cbar_counts = fig_counts.colorbar(img_counts, ax=ax_counts)
    cbar_counts.ax.set_ylabel("Samples per cell")
    if target_rect and show:
        x0, y0, w, h = target_rect
        ax_counts.add_patch(
            plt.Rectangle(
                (x0, y0),
                w,
                h,
                fill=False,
                edgecolor="green",
                linewidth=2,
            )
        )
    fig_counts.tight_layout()
    plt.savefig(os.path.join(folder, "heatmap_counts.png"))
    if save_bitmap:
        fig_counts_bitmap, ax_counts_bitmap = plt.subplots()
        img_counts_bitmap = ax_counts_bitmap.imshow(
            counts.T,
            origin="lower",
            cmap="viridis",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        )
        ax_counts_bitmap.set_aspect("equal", adjustable="box")
        ax_counts_bitmap.axis("off")
        fig_counts_bitmap.tight_layout(pad=0)
        plt.savefig(
            os.path.join(folder, "heatmap_counts_bitmap.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig_counts_bitmap)
    if show:
        plt.show()
    else:
        plt.close(fig_counts)

    # dBm plot (vmin fixed to -30 dBm)
    heatmap_dbm = 10 * np.log10(np.clip(heatmap * 1e-6, 1e-15, None) / 1e-3)  # uW->W then to dBm
    fig_dbm, ax_dbm = plt.subplots()
    img_dbm = ax_dbm.imshow(
        heatmap_dbm.T,
        origin="lower",
        cmap=CMAP,
        # vmin=-80,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    ax_dbm.set_aspect("equal", adjustable="box")
    ax_dbm.set_title(f"{os.path.basename(folder)} | power per cell [dBm]")
    ax_dbm.set_xlabel("x [m]")
    ax_dbm.set_ylabel("y [m]")
    cbar_dbm = fig_dbm.colorbar(img_dbm, ax=ax_dbm)
    cbar_dbm.ax.set_ylabel("Power per cell [dBm]")
    if target_rect:
        x0, y0, w, h = target_rect
        ax_dbm.add_patch(
            plt.Rectangle(
                (x0, y0),
                w,
                h,
                fill=False,
                edgecolor="green",
                linewidth=2,
            )
        )
    fig_dbm.tight_layout()
    plt.savefig(os.path.join(folder, png_name.replace(".png", "_dBm.png")))
    if show:
        plt.show()
    else:
        plt.close(fig_dbm)


def plot_heatmap_3d(
    folder,
    heatmap,
    x_edges,
    y_edges,
    agg="mean",
    show=True,
    png_name="heatmap_3d.png",
    z_label="Power per cell [uW]",
    title=None,
    save_pdf=True,
):
    """Render a 3D surface plot of the heatmap and save to disk."""
    finite_vals = heatmap[np.isfinite(heatmap)]
    if finite_vals.size == 0:
        print(f"{os.path.basename(folder)}: no finite values for 3D plot; skipping.")
        return

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    grid_z = np.ma.array(heatmap, mask=~np.isfinite(heatmap))

    agg_label = "Median" if agg == "median" else "Mean"
    plot_title = title or f"{os.path.basename(folder)} | {agg_label.lower()} power per cell (3D)"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Transparent background for easy compositing
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    surf = ax.plot_surface(
        grid_x,
        grid_y,
        grid_z,
        cmap=CMAP,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel(z_label)
    ax.set_title(plot_title)
    ax.grid(False)
    ax.view_init(elev=30, azim=225)
    colorbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, aspect=20)
    colorbar.ax.set_facecolor("none")
    fig.tight_layout()
    png_path = os.path.join(folder, png_name)
    plt.savefig(png_path, transparent=True)
    if save_pdf:
        base, _ = os.path.splitext(png_name)
        pdf_path = os.path.join(folder, f"{base}.pdf")
        plt.savefig(pdf_path, transparent=True)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _cell_center(i_x, i_y, x_edges, y_edges):
    """Return center coordinate for a heatmap cell index."""
    cx = (x_edges[i_x] + x_edges[i_x + 1]) / 2.0
    cy = (y_edges[i_y] + y_edges[i_y + 1]) / 2.0
    return cx, cy


def write_folder_log(
    folder,
    heatmap,
    counts,
    x_edges,
    y_edges,
    target_xyz,
    agg,
    first_cell=None,
    first_pos=None,
    baseline_heatmap=None,
    baseline_x_edges=None,
    baseline_y_edges=None,
    baseline_name=None,
):
    """Write a per-folder summary stats log."""
    log_path = os.path.join(folder, "heatmap.txt")
    if np.isfinite(heatmap).any():
        max_idx = np.nanargmax(heatmap)
        i_x, i_y = np.unravel_index(max_idx, heatmap.shape)
        max_val = float(heatmap[i_x, i_y])
        max_x, max_y = _cell_center(i_x, i_y, x_edges, y_edges)
        max_count = int(counts[i_x, i_y])
    else:
        i_x = i_y = None
        max_val = float("nan")
        max_x = max_y = float("nan")
        max_count = 0

    total_samples = int(counts.sum())
    nonzero_counts = counts[counts > 0]
    min_count = int(nonzero_counts.min()) if nonzero_counts.size else 0
    max_count = int(nonzero_counts.max()) if nonzero_counts.size else 0
    num_cells = int(counts.size)
    num_nonzero_cells = int(nonzero_counts.size)

    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"folder: {os.path.basename(folder)}\n")
        fh.write(f"aggregation: {agg}\n")
        fh.write(f"grid_res_m: {GRID_RES}\n")
        if target_xyz and len(target_xyz) >= 2:
            z_val = target_xyz[2] if len(target_xyz) > 2 else "n/a"
            fh.write(f"target_location: {target_xyz[0]:.6f}, {target_xyz[1]:.6f}, {z_val}\n")
        else:
            fh.write("target_location: n/a\n")
        fh.write(f"max_power_uW: {max_val:.6f}\n")
        fh.write(f"max_cell_center_m: {max_x:.6f}, {max_y:.6f}\n")
        if i_x is not None and i_y is not None:
            fh.write(f"max_cell_index: {i_x}, {i_y}\n")
            fh.write(f"max_cell_count: {max_count}\n")
        if first_cell is not None:
            fh.write(f"first_cell_index: {first_cell[0]}, {first_cell[1]}\n")
        if first_pos is not None:
            fh.write(f"first_position: {first_pos[0]:.6f}, {first_pos[1]:.6f}\n")
        if baseline_name:
            fh.write(f"baseline_folder_used: {baseline_name}\n")
        fh.write(f"total_samples: {total_samples}\n")
        fh.write(f"sum_counts: {total_samples}\n")
        fh.write(f"cells_total: {num_cells}\n")
        fh.write(f"cells_nonzero: {num_nonzero_cells}\n")
        fh.write(f"min_cell_count: {min_count}\n")
        fh.write(f"max_cell_count_all: {max_count}\n")
        if target_xyz and len(target_xyz) >= 2:
            tx, ty = target_xyz[0], target_xyz[1]
            ti_x = np.digitize(tx, x_edges) - 1
            ti_y = np.digitize(ty, y_edges) - 1
            if 0 <= ti_x < heatmap.shape[0] and 0 <= ti_y < heatmap.shape[1]:
                tgt_power = heatmap[ti_x, ti_y]
                tgt_count = int(counts[ti_x, ti_y])
                fh.write(f"target_cell_index: {ti_x}, {ti_y}\n")
                fh.write(f"target_cell_center_m: {_cell_center(ti_x, ti_y, x_edges, y_edges)[0]:.6f}, {_cell_center(ti_x, ti_y, x_edges, y_edges)[1]:.6f}\n")
                fh.write(f"target_power_uW: {float(tgt_power):.6f}\n")
                fh.write(f"target_cell_count: {tgt_count}\n")
                if (
                    baseline_heatmap is not None
                    and baseline_x_edges is not None
                    and baseline_y_edges is not None
                ):
                    b_ix = np.digitize(tx, baseline_x_edges) - 1
                    b_iy = np.digitize(ty, baseline_y_edges) - 1
                    if 0 <= b_ix < baseline_heatmap.shape[0] and 0 <= b_iy < baseline_heatmap.shape[1]:
                        base_power = baseline_heatmap[b_ix, b_iy]
                        if np.isfinite(base_power) and base_power > 0 and np.isfinite(tgt_power) and tgt_power > 0:
                            gain_lin = float(tgt_power / base_power)
                            gain_db = float(10 * np.log10(gain_lin))
                            fh.write(f"baseline_folder: {baseline_name or 'n/a'}\n")
                            fh.write(f"baseline_target_power_uW: {float(base_power):.6f}\n")
                            fh.write(f"target_gain_linear: {gain_lin:.6f}\n")
                            fh.write(f"target_gain_db: {gain_db:.2f}\n")
            else:
                fh.write("target_power_uW: n/a\n")



def plot_diff_heatmap(
    folder,
    baseline_name,
    diff_map,
    x_edges,
    y_edges,
    target_rect=None,
    show=True,
    save_bitmap=False,
    png_name=None,
    bitmap_name=None,
    title_override=None,
):
    """Plot the difference vs baseline in dB (folder - baseline) on aligned grid."""
    vmax_db = 10 * np.log10(42) # we expect a sqrt(42) ~16x power gain at most
    vmin_db = -vmax_db
    png_out = png_name or f"heatmap_vs_{baseline_name}_dB.png"
    bitmap_out = bitmap_name or f"heatmap_vs_{baseline_name}_dB_bitmap.png"

    def _draw(ax, add_axes=True):
        img = ax.imshow(
            diff_map.T,
            origin="lower",
            cmap=CMAP,
            vmin=vmin_db,
            vmax=vmax_db,
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        )
        ax.set_aspect("equal", adjustable="box")
        if add_axes:
            title = title_override or f"{os.path.basename(folder)} - {baseline_name} | delta power per cell [dB]"
            ax.set_title(title)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            cbar = plt.colorbar(img, ax=ax)
            cbar.ax.set_ylabel("Delta vs baseline [dB]")
        if target_rect and add_axes:
            x0, y0, w, h = target_rect
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0),
                    w,
                    h,
                    fill=False,
                    edgecolor="green",
                    linewidth=2,
                )
            )
        if not add_axes:
            ax.axis("off")
        return img

    fig, ax = plt.subplots()
    _draw(ax, add_axes=True)
    fig.tight_layout()
    plt.savefig(os.path.join(folder, png_out))
    if show:
        plt.show()
    else:
        plt.close(fig)

    if save_bitmap:
        fig2, ax2 = plt.subplots()
        _draw(ax2, add_axes=False)
        fig2.tight_layout(pad=0)
        plt.savefig(os.path.join(folder, bitmap_out), bbox_inches="tight", pad_inches=0)
        plt.close(fig2)


def export_heatmap_csv(folder, heatmap, x_edges, y_edges, suffix=""):
    """Export heatmap grid to long-form CSV (x,y,z) plus edge vectors."""
    suffix_str = f"_{suffix}" if suffix else ""
    grid_path = os.path.join(folder, f"heatmap{suffix_str}.csv")
    x_path = os.path.join(folder, f"x_edges{suffix_str}.csv")
    y_path = os.path.join(folder, f"y_edges{suffix_str}.csv")

    # Centers for each cell
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    rows = []
    for ix, xc in enumerate(x_centers):
        for iy, yc in enumerate(y_centers):
            z = heatmap[ix, iy]
            if np.isfinite(z):
                rows.append((xc, yc, z))

    with open(grid_path, "w", encoding="utf-8") as fh:
        fh.write("x,y,z\n")
        for x, y, z in rows:
            fh.write(f"{x:.6g},{y:.6g},{z:.6g}\n")

    np.savetxt(x_path, x_edges, delimiter=",", fmt="%.6g")
    np.savetxt(y_path, y_edges, delimiter=",", fmt="%.6g")
    print(f"Exported heatmap CSVs: {grid_path}, {x_path}, {y_path}")


def export_heatmap_tex(folder, x_edges, y_edges, heatmap, suffix="", title="", target_rect=None, vmin=None, vmax=None):
    """Write a small PGFPlots snippet that embeds the rendered bitmap with axes/ticks/colorbar drawn in TikZ."""
    suffix_str = f"_{suffix}" if suffix else ""
    tex_path = os.path.join(folder, f"heatmap{suffix_str}.tex")
    png_name = f"heatmap{suffix_str}_bitmap.png"
    png_rel_path = f"figures/{os.path.basename(folder)}/{png_name}"
    xmin, xmax = x_edges[0], x_edges[-1]
    ymin, ymax = y_edges[0], y_edges[-1]
    title_tex = title.replace("_", "\\_") if title else ""
    finite_vals = heatmap[np.isfinite(heatmap)]
    vmin_val = float(finite_vals.min()) if finite_vals.size else 0.0
    vmax_val = float(finite_vals.max()) if finite_vals.size else 1.0
    vmin_use = vmin if vmin is not None else vmin_val
    vmax_use = vmax if vmax is not None else vmax_val

    lines = [
        f"% PGFPlots includegraphics example for {png_rel_path}",
        "\\begin{tikzpicture}",
        "  \\begin{axis}[",
        "    axis equal image,",
        f"    xmin={xmin:.6g}, xmax={xmax:.6g},",
        f"    ymin={ymin:.6g}, ymax={ymax:.6g},",
        "    axis lines=box,",
        "    xlabel={x [m]},",
        "    ylabel={y [m]},",
        "    colorbar,",
        "    colormap name=inferno,",
        f"    point meta min={vmin_use:.6g},",
        f"    point meta max={vmax_use:.6g},",
        "    enlargelimits=false,",
        f"    title={{{title_tex}}}",
        "  ]",
        f"    \\addplot graphics [includegraphics cmd=\\pgfimage, xmin={xmin:.6g}, xmax={xmax:.6g}, ymin={ymin:.6g}, ymax={ymax:.6g}] {{{png_rel_path}}};",
    ]

    if target_rect:
        x0, y0, w, h = target_rect
        cx, cy = x0 + w / 2.0, y0 + h / 2.0
        lines.extend(
            [
                "    % Target circle (radius = wavelength/2)",
                f"    \\draw[red,dashed,thick] ({cx:.6g},{cy:.6g}) circle [radius={WAVELENGTH/2:.6g}];",
            ]
        )

    lines.extend(
        [
            "  \\end{axis}",
            "\\end{tikzpicture}",
            "",
        ]
    )

    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Exported LaTeX example: {tex_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate positions/values per data subfolder and plot a heatmap."
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Plot heatmaps for all subfolders (default plots only the most recent).",
    )
    parser.add_argument(
        "--plot-movement",
        action="store_true",
        help="Overlay rectangles for the last 5 visited grid cells.",
    )
    parser.add_argument(
        "--drop-consecutive-equal",
        action="store_true",
        help="Filter out consecutive samples with identical power values.",
    )
    parser.add_argument(
        "--no-drop-duplicate-timestamps",
        dest="drop_duplicate_timestamps",
        action="store_false",
        help="Disable filtering of samples where position timestamp does not increase (consecutive duplicates).",
    )
    parser.set_defaults(drop_duplicate_timestamps=True)
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Save plots to disk without displaying them.",
    )
    parser.add_argument(
        "--agg",
        choices=["mean", "median", "max", "min"],
        default="mean",
        help="Aggregation used for heatmap cells (default: mean).",
    )
    parser.add_argument(
        "--baseline-folder",
        default="RANDOM-1",
        help="Folder name to use as baseline for delta heatmap (default: RANDOM). Set empty to disable.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export heatmap grids and edges to CSV (for LaTeX/pgfplots).",
    )
    parser.add_argument(
        "--no-3d",
        dest="export_3d",
        action="store_false",
        help="Disable exporting 3D surface plots.",
    )
    parser.add_argument(
        "--fill-empty",
        action="store_true",
        help="Interpolate/fill empty cells with nearest non-empty value.",
    )
    parser.add_argument(
        "--grid-res-lambda",
        type=float,
        help="Grid resolution as a fraction of wavelength (e.g., 0.08 for 0.08*lambda). Overrides default.",
    )
    parser.set_defaults(export_3d=True)
    return parser.parse_args()


def print_run_summary(args, target_rect, baseline_folder_name, baseline_agg, grid_res):
    """Print selected options so it's clear how folders will be processed."""
    rect_desc = (
        f"{target_rect[2]:.3f}x{target_rect[3]:.3f} m at ({target_rect[0]:.3f}, {target_rect[1]:.3f})"
        if target_rect
        else "none"
    )
    baseline_desc = baseline_folder_name or "disabled"
    print(
        "\nRun configuration:\n"
        f"- data dir: {DATA_DIR}\n"
        f"- plot_all: {args.plot_all}\n"
        f"- save_only: {args.save_only}\n"
        f"- plot_movement: {args.plot_movement}\n"
        f"- agg (main): {args.agg}\n"
        f"- drop_duplicate_timestamps: {args.drop_duplicate_timestamps}\n"
        f"- drop_consecutive_equal: {args.drop_consecutive_equal}\n"
        f"- small-value filter threshold (uW): {SMALL_POWER_UW}\n"
        f"- target rectangle: {rect_desc}\n"
        f"- baseline folder: {baseline_desc} (agg={baseline_agg})\n"
        f"- fill_empty: {args.fill_empty}\n"
        f"- export_3d: {args.export_3d}\n"
        f"- grid_res: {grid_res:.4f} m\n"
    )


def print_drop_summary(
    folder_name,
    start_count,
    end_count,
    drop_ts,
    drop_dups,
    drop_small,
    small_report,
    ts_info=None,
):
    """Print a concise drop summary for a folder."""
    dropped_total = start_count - end_count
    parts = []
    if drop_ts:
        parts.append(f"ts={drop_ts}")
        if ts_info:
            if ts_info.get("equal"):
                parts.append(f"ts_equal={ts_info['equal']}")
            if ts_info.get("decrease"):
                parts.append(f"ts_decrease={ts_info['decrease']}")
    if drop_dups:
        parts.append(f"dup={drop_dups}")
    if drop_small:
        parts.append(f"small={drop_small}")
    if small_report:
        parts.append(small_report)
    detail = ", ".join(parts) if parts else "none"
    print(
        f"[{folder_name}] samples {start_count} -> {end_count} (dropped {dropped_total}; {detail})"
    )


def main():
    args = parse_args()
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    grid_res = GRID_RES
    if args.grid_res_lambda:
        grid_res = float(args.grid_res_lambda) * WAVELENGTH

    target_vals = load_target_from_settings()
    target_rect = target_rect_from_xyz(target_vals)

    # Sort subfolders by modification time (newest first) to view recent runs first
    folder_entries = []
    for name in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, name)
        if os.path.isdir(folder_path):
            folder_entries.append((os.path.getmtime(folder_path), name))

    if not folder_entries:
        raise ValueError(f"No subfolders found in {DATA_DIR}")

    # Precompute default baseline heatmap if requested
    baseline_heatmap = baseline_x_edges = baseline_y_edges = None
    baseline_agg = "mean"
    baseline_folder_name = args.baseline_folder.strip() if args.baseline_folder else ""

    def _build_baseline(baseline_name: str):
        if not baseline_name:
            return None, None, None
        baseline_path = os.path.join(DATA_DIR, baseline_name)
        if not os.path.isdir(baseline_path):
            print(f"Baseline folder not found: {baseline_path}")
            return None, None, None
        try:
            base_positions, base_values = load_folder(baseline_path)
            if args.drop_consecutive_equal:
                base_positions, base_values, _ = drop_consecutive_equal_values(
                    base_positions, base_values
                )
            base_vs = np.array([v.pwr_pw / 1e6 for v in base_values], dtype=float)
            base_positions, base_values, base_vs, _, _ = filter_small_values(
                baseline_path, base_positions, base_values, base_vs
            )
            base_xs = np.array([p.x for p in base_positions], dtype=float)
            base_ys = np.array([p.y for p in base_positions], dtype=float)
            base_heatmap, _, base_x_edges, base_y_edges, _, _ = compute_heatmap(
                base_xs, base_ys, base_vs, GRID_RES, agg=baseline_agg
            )
            return base_heatmap, base_x_edges, base_y_edges
        except Exception as exc:
            print(f"Failed to build baseline from {baseline_path}: {exc}")
            return None, None, None

    if baseline_folder_name:
        baseline_heatmap, baseline_x_edges, baseline_y_edges = _build_baseline(
            baseline_folder_name
        )
        if baseline_heatmap is None:
            baseline_folder_name = ""

    print_run_summary(args, target_rect, baseline_folder_name, baseline_agg, grid_res)

    folder_entries.sort(key=lambda x: x[0], reverse=True)
    for _, folder_name in folder_entries:
        folder_path = os.path.join(DATA_DIR, folder_name)
        try:
            positions, values = load_folder(folder_path)
        except ValueError as e:
            print(e)
            continue

        baseline_override_name = baseline_folder_name
        baseline_override_path = os.path.join(folder_path, "baseline.txt")
        if os.path.isfile(baseline_override_path):
            try:
                with open(baseline_override_path, "r", encoding="utf-8") as fh:
                    baseline_override_name = fh.read().strip()
            except Exception as exc:
                print(f"Failed to read baseline override in {baseline_override_path}: {exc}")
                baseline_override_name = baseline_folder_name
        if not baseline_override_name:
            baseline_override_name = ""

        start_count = len(values)
        drop_ts = drop_dups = drop_small = 0
        ts_info = None
        small_report = ""

        if args.drop_duplicate_timestamps:
            positions, values, drop_ts, ts_info = drop_nonincreasing_timestamps(positions, values)

        if args.drop_consecutive_equal:
            positions, values, drop_dups = drop_consecutive_equal_values(positions, values)

        vs = np.array([v.pwr_pw / 1e6 for v in values], dtype=float)  # uW

        positions, values, vs, drop_small, small_report = filter_small_values(
            folder_path, positions, values, vs
        )
        print_drop_summary(
            os.path.basename(folder_path),
            start_count,
            len(values),
            drop_ts,
            drop_dups,
            drop_small,
            small_report,
            ts_info,
        )

        xs = np.array([p.x for p in positions], dtype=float)
        ys = np.array([p.y for p in positions], dtype=float)

        if len(xs) and len(ys) and os.path.basename(folder_path).startswith("RECI"):
            first_target = [float(xs[0]), float(ys[0])]
            if target_vals and len(target_vals) > 2:
                first_target.append(target_vals[2])
            target_vals = first_target
            target_rect = target_rect_from_xyz(target_vals)

        heatmap, counts, x_edges, y_edges, xi, yi = compute_heatmap(
            xs, ys, vs, grid_res, agg=args.agg
        )
        if args.fill_empty:
            heatmap = fill_empty_cells_nearest(heatmap)

        if args.export_csv:
            export_heatmap_csv(folder_path, heatmap, x_edges, y_edges)
            export_heatmap_tex(
                folder_path,
                x_edges,
                y_edges,
                heatmap,
                title=f"{os.path.basename(folder_path)} | {args.agg} power [uW]",
                target_rect=target_rect,
            )

        # If baseline available, compute aligned heatmap and plot delta
        if baseline_override_name:
            if baseline_override_name == baseline_folder_name and baseline_heatmap is not None:
                curr_baseline_heatmap = baseline_heatmap
                curr_baseline_x_edges = baseline_x_edges
                curr_baseline_y_edges = baseline_y_edges
            else:
                curr_baseline_heatmap, curr_baseline_x_edges, curr_baseline_y_edges = _build_baseline(
                    baseline_override_name
                )
        else:
            curr_baseline_heatmap = curr_baseline_x_edges = curr_baseline_y_edges = None

        if (
            curr_baseline_heatmap is not None
            and curr_baseline_x_edges is not None
            and curr_baseline_y_edges is not None
        ):
            aligned_heatmap, _, _, _, _, _ = compute_heatmap(
                xs,
                ys,
                vs,
                grid_res,
                agg=baseline_agg,
                x_edges=curr_baseline_x_edges,
                y_edges=curr_baseline_y_edges,
            )
            if args.fill_empty:
                aligned_heatmap = fill_empty_cells_nearest(aligned_heatmap)
                curr_baseline_heatmap = fill_empty_cells_nearest(curr_baseline_heatmap)
            diff_map = heatmap_delta_db(aligned_heatmap, curr_baseline_heatmap)
            global_gain, target_gain = gain_stats(
                aligned_heatmap,
                curr_baseline_heatmap,
                curr_baseline_x_edges,
                curr_baseline_y_edges,
                target_rect,
            )
            gain_title = None
            if global_gain:
                gain_title = f"avg {global_gain['avg_db']:.1f}dB / max {global_gain['max_db']:.1f}dB"
                print(
                    f"Gain vs {baseline_override_name}: avg {global_gain['avg_db']:.2f} dB ({global_gain['avg_lin']:.2f}x), "
                    f"max {global_gain['max_db']:.2f} dB ({global_gain['max_lin']:.2f}x)"
                )
            if target_gain:
                target_str = f"target avg {target_gain['avg_db']:.1f}dB / max {target_gain['max_db']:.1f}dB"
                gain_title = f"{gain_title} | {target_str}" if gain_title else target_str
                print(
                    f"Target gain vs {baseline_override_name}: avg {target_gain['avg_db']:.2f} dB ({target_gain['avg_lin']:.2f}x), "
                    f"max {target_gain['max_db']:.2f} dB ({target_gain['max_lin']:.2f}x)"
                )
            plot_diff_heatmap(
                folder_path,
                baseline_override_name,
                diff_map,
                curr_baseline_x_edges,
                curr_baseline_y_edges,
                target_rect=target_rect,
                show=not args.save_only,
                save_bitmap=args.export_csv,
                png_name=f"heatmap_vs_{baseline_override_name}_dB.png",
                bitmap_name=f"heatmap_vs_{baseline_override_name}_dB_bitmap.png",
                title_override=gain_title,
            )
            if args.export_csv:
                suffix = f"vs_{baseline_override_name}_dB"
                export_heatmap_csv(
                    folder_path, diff_map, curr_baseline_x_edges, curr_baseline_y_edges, suffix=suffix
                )
                export_heatmap_tex(
                    folder_path,
                    curr_baseline_x_edges,
                    curr_baseline_y_edges,
                    diff_map,
                    suffix=suffix,
                    title=f"{os.path.basename(folder_path)} - {baseline_override_name} [dB]{' | ' + gain_title if gain_title else ''}",
                    target_rect=target_rect,
                    vmin=-10 * np.log10(42)-6,
                    vmax=10 * np.log10(42)+6,
                )

        recent_cells = None
        if args.plot_movement:
            # Take the last 5 distinct cells (most recent first), not just the last 5 samples.
            recent_cells = []
            seen = set()
            for cell in reversed(list(zip(xi, yi))):
                if cell in seen:
                    continue
                seen.add(cell)
                recent_cells.append(cell)
                if len(recent_cells) == 5:
                    break
            recent_cells.reverse()
        first_cell = (int(xi[0]), int(yi[0])) if len(xi) else None
        first_pos = (float(xs[0]), float(ys[0])) if len(xs) else None
        write_folder_log(
            folder_path,
            heatmap,
            counts,
            x_edges,
            y_edges,
            target_vals,
            args.agg,
            first_cell=first_cell,
            first_pos=first_pos,
            baseline_heatmap=curr_baseline_heatmap,
            baseline_x_edges=curr_baseline_x_edges,
            baseline_y_edges=curr_baseline_y_edges,
            baseline_name=baseline_override_name,
        )
        plot_heatmap(
            folder_path,
            heatmap,
            counts,
            x_edges,
            y_edges,
            recent_cells,
            target_rect,
            agg=args.agg,
            show=not args.save_only,
            save_bitmap=args.export_csv,
            png_name="heatmap.png",
            bitmap_name="heatmap_bitmap.png",
        )
        if args.export_3d:
            plot_heatmap_3d(
                folder_path,
                heatmap,
                x_edges,
                y_edges,
                agg=args.agg,
                show=not args.save_only,
                png_name="heatmap_3d.png",
                z_label="Power per cell [uW]",
                title=f"{os.path.basename(folder_path)} | {args.agg} power per cell [uW] (3D)",
            )
            heatmap_dbm = 10 * np.log10(np.clip(heatmap * 1e-6, 1e-15, None) / 1e-3)
            plot_heatmap_3d(
                folder_path,
                heatmap_dbm,
                x_edges,
                y_edges,
                agg=args.agg,
                show=not args.save_only,
                png_name="heatmap_3d_dBm.png",
                z_label="Power per cell [dBm]",
                title=f"{os.path.basename(folder_path)} | {args.agg} power per cell [dBm] (3D)",
            )
        if not args.plot_all:
            break


if __name__ == "__main__":
    main()
