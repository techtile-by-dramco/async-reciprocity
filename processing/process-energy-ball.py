#!/usr/bin/env python3
"""
Summarize energy-ball experiment results.

Reads the YAML produced by server/record/server-energy-ball.py and extracts
the max power per iteration. Outputs tables/plots and builds an xarray Dataset.
"""

import argparse
import glob
import os
import re
import sys
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

wavelength = 3e8 / 920e6  # meters
GRID_RES = 0.1 * wavelength  # meters
POS_CMAP = "inferno"


def wrap_phase(m):
    """Wrap phases to [-pi, pi]."""
    return np.angle(np.exp(1j * m))

# Ensure project root on sys.path for pickled objects (e.g., lib.* classes)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def find_latest_exp(path_glob: str) -> str | None:
    files = glob.glob(path_glob, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_measurements(path: str) -> list[dict]:
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    return data.get("measurments", []) if data else []


def iter_max_power(measurements: list[dict]) -> Iterable[Tuple[int, int, float]]:
    for meas in measurements:
        meas_id = meas.get("meas_id", -1)
        for it in meas.get("iterations", []):
            power = it.get("max_power_pw")
            if power is None:
                power = it.get("power_pw", 0.0)
            yield meas_id, it.get("iter", -1), power


def extract_iterations(measurements: list[dict]) -> list[dict]:
    """Return a flat list of iteration records with power and client phases."""
    rows = []
    for meas in measurements:
        meas_id = meas.get("meas_id", -1)
        for it in meas.get("iterations", []):
            power = it.get("max_power_pw")
            if power is None:
                power = it.get("power_pw", 0.0)
            rows.append(
                {
                    "meas_id": meas_id,
                    "iter": it.get("iter", -1),
                    "power_pw": power,
                    "clients": it.get("clients", []),
                }
            )
    return rows


def build_dataset(iterations: list[dict]) -> xr.Dataset:
    """Convert iterations to xarray with dims (iteration, host) and power/phase."""
    iter_vals = [row.get("iter", -1) for row in iterations]
    hosts = sorted(
        {
            client.get("host")
            for row in iterations
            for client in row.get("clients", [])
            if client.get("host")
        }
    )
    host_index = {h: i for i, h in enumerate(hosts)}
    num_iters = len(iter_vals)
    num_hosts = max(1, len(hosts))

    power_arr = np.full((num_iters, num_hosts), np.nan)
    phase_arr = np.full((num_iters, num_hosts), np.nan)

    for i, row in enumerate(iterations):
        power_uw = row.get("power_pw", 0.0) / 1e6
        power_arr[i, :] = power_uw  # broadcast across hosts
        for client in row.get("clients", []):
            host = client.get("host")
            if host in host_index:
                phase_arr[i, host_index[host]] = client.get("applied_phase", np.nan)

    ds = xr.Dataset(
        {
            "power_uW": (("iteration", "host"), power_arr[:, : len(hosts) or 1]),
            "applied_phase_rad": (("iteration", "host"), phase_arr[:, : len(hosts) or 1]),
        },
        coords={"iteration": iter_vals, "host": hosts or ["(none)"]},
    )
    return ds


def load_position_value_pairs(folder_path: str):
    """Load concatenated *_positions.npy and *_values.npy pairs from a folder."""
    positions_parts = []
    values_parts = []
    for name in sorted(os.listdir(folder_path)):
        if not name.endswith("_positions.npy"):
            continue
        base = name[: -len("_positions.npy")]
        pos_path = os.path.join(folder_path, name)
        val_path = os.path.join(folder_path, f"{base}_values.npy")
        if not os.path.exists(val_path):
            continue
        positions_parts.append(np.load(pos_path, allow_pickle=True))
        values_parts.append(np.load(val_path, allow_pickle=True))

    if not positions_parts:
        return None, None

    positions = np.concatenate(positions_parts)
    values = np.concatenate(values_parts)
    return positions, values


def compute_heatmap(xs, ys, vs, grid_res):
    """Bin values onto a 2D grid and compute mean per cell."""
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    x_edges = np.arange(min_x, max_x + grid_res, grid_res)
    y_edges = np.arange(min_y, max_y + grid_res, grid_res)

    heatmap = np.full((len(x_edges) - 1, len(y_edges) - 1), np.nan, dtype=float)
    sums = np.zeros_like(heatmap, dtype=float)
    counts = np.zeros_like(heatmap, dtype=int)

    xi = np.digitize(xs, x_edges) - 1
    yi = np.digitize(ys, y_edges) - 1

    for i_x, i_y, v in zip(xi, yi, vs):
        if 0 <= i_x < heatmap.shape[0] and 0 <= i_y < heatmap.shape[1]:
            sums[i_x, i_y] += v
            counts[i_x, i_y] += 1

    mask = counts > 0
    heatmap[mask] = sums[mask] / counts[mask]  # mean per cell
    return heatmap, counts, x_edges, y_edges, xi, yi


def plot_position_heatmap(folder_label, heatmap, counts, x_edges, y_edges, target=None):
    """Render a position/value heatmap in meters."""
    fig, ax = plt.subplots()
    img = ax.imshow(
        heatmap.T,
        origin="lower",
        cmap=POS_CMAP,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    ax.set_title(f"{folder_label} | mean power per cell [uW]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.set_ylabel("Mean power per cell [uW]")
    if target:
        tx, ty, w, h = target
        ax.add_patch(
            plt.Rectangle(
                (tx, ty),
                w,
                h,
                fill=False,
                edgecolor="lime",
                linewidth=2,
            )
        )
    fig.tight_layout()


def save_all_figs(save_dir: str, prefix: str):
    """Persist all open matplotlib figures as numbered PNGs with title in filename."""
    os.makedirs(save_dir, exist_ok=True)
    for idx, num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(num)
        title = None
        if fig._suptitle is not None:  # type: ignore[attr-defined]
            title = fig._suptitle.get_text()
        elif fig.axes:
            title = fig.axes[0].get_title()
        if title:
            title_clean = re.sub(r"\s+", "_", title.strip())
            title_clean = title_clean.replace(os.sep, "_")
            # Remove characters that are invalid on Windows filesystems
            title_clean = re.sub(r'[<>:"/\\\\|?*]+', "_", title_clean)
            fname = f"{prefix}-{idx}-{title_clean}.png"
        else:
            fname = f"{prefix}-{idx}.png"
        out_path = os.path.join(save_dir, fname)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {out_path}")


def load_target_from_settings() -> list[float] | None:
    """Return target_location from experiment-settings.yaml as [x, y, z?]."""
    settings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiment-settings.yaml"))
    if not os.path.exists(settings_path):
        return None
    try:
        with open(settings_path, "r") as fh:
            settings = yaml.safe_load(fh) or {}
        target = settings.get("experiment_config", {}).get("target_location")
        if target is None:
            return None
        # Allow comma-separated string or list/tuple
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


def main():
    parser = argparse.ArgumentParser(
        description="Process energy-ball YAML and list max power per iteration."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to exp-*.yml (defaults to latest in server/record/data)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render plots (disabled by default)",
    )
    parser.add_argument(
        "--save-dir",
        help="Directory to save generated plots as PNGs (defaults to folder containing the input YAML)",
    )
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Target xyz to highlight on the position heatmap (z ignored). Defaults to experiment-settings.yaml experiment_config.target_location.",
    )
    args = parser.parse_args()

    default_glob = os.path.join(
        os.path.dirname(__file__), "..", "data", "**", "exp-*.yml"
    )
    input_path = args.input or find_latest_exp(default_glob)
    if not input_path:
        print(f"No input provided and none found via {default_glob}", file=sys.stderr)
        sys.exit(1)

    measurements = load_measurements(input_path)
    if not measurements:
        print(f"No measurements found in {input_path}", file=sys.stderr)
        sys.exit(1)

    iterations = extract_iterations(measurements)
    if not iterations:
        print(f"No iterations found in {input_path}", file=sys.stderr)
        sys.exit(1)

    ds = build_dataset(iterations)
    rows_micro_w = [
        (row["meas_id"], row["iter"], row["power_pw"] / 1e6) for row in iterations
    ]
    print(f"Loaded {len(rows_micro_w)} iterations from {input_path}")
    print("meas_id\titer\tpower_uW")
    for meas_id, iter_id, power_uw in rows_micro_w:
        print(f"{meas_id}\t{iter_id}\t{power_uw}")

    # Plot max power per iteration (across all measurements)
    iters = ds["iteration"].to_numpy()
    power_matrix = ds["power_uW"].to_numpy()
    powers = np.nanmean(power_matrix, axis=1) if power_matrix.size else []
    max_idx = int(np.nanargmax(powers)) if len(powers) else 0
    max_iter_val = iters[max_idx] if len(iters) else None
    host_labels = ds["host"].to_numpy()

    # Store best phases (deg) at max-power iteration.
    if power_matrix.size and len(host_labels) and max_iter_val is not None:
        phase_matrix = ds["applied_phase_rad"].to_numpy()
        if max_idx < phase_matrix.shape[0]:
            phases_deg = np.rad2deg(wrap_phase(phase_matrix[max_idx]))
            best_phases = {}
            for host, deg in zip(host_labels, phases_deg):
                if np.isnan(deg):
                    continue
                best_phases[str(host)] = float(deg)
            out_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "client", "tx-phases-energy-ball.yml")
            )
            with open(out_path, "w") as fh:
                yaml.safe_dump(best_phases, fh)
            print(f"Wrote max-power phases (deg) for iter {max_iter_val} to {out_path}")

    do_plot = args.plot

    if do_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(iters, powers, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Power (uW)")
        plt.title(f"Energy-ball max power per iteration\n{os.path.basename(input_path)}")
        plt.grid(True)
        plt.tight_layout()

        # Heatmap of power per iteration and host; y-axis labels show power in uW.
        if power_matrix.size:
            plt.figure(figsize=(max(4, len(ds["host"]) * 0.5), max(3, len(powers) * 0.25)))
            im_power = plt.imshow(power_matrix, aspect="auto", cmap="plasma")
            plt.colorbar(im_power, label="Power (uW)")
            plt.xticks(range(len(ds["host"])), ds["host"].to_numpy(), rotation=45)
            plt.yticks(range(len(iters)), [f"iter {it}: {p:.2f} uW" for it, p in zip(iters, powers)])
            plt.xlabel("Host")
            plt.title("Power per iteration (heatmap)")
            plt.tight_layout()

        # Phase heatmaps (raw + normalized), wrapped to [-pi, pi].
        phase_matrix = ds["applied_phase_rad"].to_numpy()
        host_labels = ds["host"].to_numpy()
        iter_labels = iters
        if phase_matrix.size:
            wrap = lambda m: np.angle(np.exp(1j * m))
            pm = wrap(phase_matrix)

            plt.figure(figsize=(max(6, len(iter_labels) * 0.4), max(6, len(host_labels) * 0.2)))
            im_phase = plt.imshow(pm, aspect="auto", cmap="twilight_shifted", interpolation="nearest")
            plt.colorbar(im_phase, label="Phase (rad)")
            plt.xticks(range(len(host_labels)), host_labels, rotation=45)
            plt.yticks(range(len(iter_labels)), iter_labels)
            plt.xlabel("Host")
            plt.ylabel("Iteration")
            plt.title("Applied phases per client")
            plt.tight_layout()

            # Normalized against first iteration (row 0).
            norm_iter0 = wrap(phase_matrix - phase_matrix[:1, :])
            plt.figure(figsize=(max(6, len(iter_labels) * 0.4), max(6, len(host_labels) * 0.2)))
            im_phase_norm_iter = plt.imshow(norm_iter0, aspect="auto", cmap="twilight_shifted", interpolation="nearest")
            plt.colorbar(im_phase_norm_iter, label="Phase delta vs iter0 (rad)")
            plt.xticks(range(len(host_labels)), host_labels, rotation=45)
            plt.yticks(range(len(iter_labels)), iter_labels)
            plt.xlabel("Host")
            plt.ylabel("Iteration")
            plt.title("Applied phases (normalized to iteration 0)")
            plt.tight_layout()

            # Normalized against first host (column 0).
            norm_host0 = wrap(phase_matrix - phase_matrix[:, :1])
            plt.figure(figsize=(max(6, len(iter_labels) * 0.4), max(6, len(host_labels) * 0.2)))
            im_phase_norm_host = plt.imshow(norm_host0, aspect="auto", cmap="twilight_shifted", interpolation="nearest")
            plt.colorbar(im_phase_norm_host, label="Phase delta vs host0 (rad)")
            plt.xticks(range(len(host_labels)), host_labels, rotation=45)
            plt.yticks(range(len(iter_labels)), iter_labels)
            plt.xlabel("Host")
            plt.ylabel("Iteration")
            plt.title("Applied phases (normalized to host 0)")
            plt.tight_layout()

            # Compare first iteration vs max-power iteration (phase rows + delta).
            if max_iter_val is not None:
                compare_rows = wrap(np.vstack([phase_matrix[0], phase_matrix[max_idx]]))
                plt.figure(
                    figsize=(max(6, len(host_labels) * 0.4), 3)
                )
                im_compare = plt.imshow(compare_rows, aspect="auto", cmap="twilight_shifted", interpolation="nearest")
                plt.colorbar(im_compare, label="Phase (rad)")
                plt.xticks(range(len(host_labels)), host_labels, rotation=45)
                plt.yticks([0, 1], [f"iter {iter_labels[0]}", f"iter {max_iter_val} (max power)"])
                plt.xlabel("Host")
                plt.ylabel("Iteration")
                plt.title("Applied phases: iter0 vs max-power iteration")
                plt.tight_layout()

                delta_rows = wrap(phase_matrix[max_idx:max_idx+1, :] - phase_matrix[:1, :])
                plt.figure(
                    figsize=(max(6, len(host_labels) * 0.4), 3)
                )
                plt.scatter(range(len(host_labels)), np.rad2deg(delta_rows[0]), c="blue", s=50)
                plt.xticks(range(len(host_labels)), host_labels, rotation=45)
                plt.xlabel("Host")
                plt.ylabel("Delta row (deg)")
                plt.title("Phase delta: max-power iteration vs iter0")
                plt.tight_layout()

        # Position/value heatmap for the folder containing the input file.
        folder_for_positions = os.path.abspath(os.path.dirname(input_path))
        try:
            positions, values = load_position_value_pairs(folder_for_positions)
        except Exception as exc:
            print(f"Skipping position heatmap due to error: {exc}", file=sys.stderr)
            positions = values = None

        if positions is not None and values is not None:
            xs = np.array([p.x for p in positions], dtype=float)
            ys = np.array([p.y for p in positions], dtype=float)
            vs = np.array([v.pwr_pw / 1e6 for v in values], dtype=float)  # uW

            heatmap, counts, x_edges, y_edges, xi, yi = compute_heatmap(xs, ys, vs, GRID_RES)
            print(
                f"Position heatmap for {folder_for_positions}: {len(xs)} samples, grid {heatmap.shape[0]}x{heatmap.shape[1]}"
            )

            target_vals = args.target or load_target_from_settings() or [3.181, 1.774, 0.266]
            tx, ty = target_vals[0], target_vals[1]
            target_rect = (tx - GRID_RES / 2, ty - GRID_RES / 2, GRID_RES, GRID_RES)
            plot_position_heatmap(
                os.path.basename(folder_for_positions), heatmap, counts, x_edges, y_edges, target_rect
            )
        else:
            print(f"No *_positions.npy/_values.npy pairs found in {folder_for_positions}", file=sys.stderr)

        # Save plots to disk before showing them.
        save_dir = args.save_dir or os.path.abspath(os.path.dirname(input_path))
        if plt.get_fignums():
            save_all_figs(save_dir, os.path.splitext(os.path.basename(input_path))[0])

       
        # If plotting is disabled but save-dir is provided, still write any figures created elsewhere.
        if plt.get_fignums():
            save_dir = args.save_dir or os.path.abspath(os.path.dirname(input_path))
            save_all_figs(save_dir, os.path.splitext(os.path.basename(input_path))[0])

        plt.show()


if __name__ == "__main__":
    main()
