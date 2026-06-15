from TechtilePlotter.TechtilePlotter import TechtilePlotter
import numpy as np
from Positioner import PositionerValues
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom
import os
import sys

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

DATA_DIR = "../data"
FOLDER = "sionna1SDR"

cmap = "inferno"

wavelen = 3e8 / 920e6
zoom_val = 1
PLOT_LAST_VAL = 10  # number of most recent samples to highlight
GRID_RESOLUTION = 0.22 * wavelen  # in meters
LABEL_IN_WAVELENGTHS = False  # False plots labels in absolute meters
PLOT_DB = False

# Grid cell aggregation functions
STAT_FUNCS = {
    "median": np.median,
    # "mean": np.mean,
    # "max": np.max,
    # "min": np.min,
}

# -------------------------------------------------
# Directory and file names
# -------------------------------------------------

server_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Server dir: {server_dir}")

PROJECT_ROOT = os.path.dirname(server_dir)
sys.path.insert(0, PROJECT_ROOT)

from lib.yaml_utils import read_yaml_file
from lib.ep import RFEP

DATA_DIR = os.path.abspath(os.path.join(server_dir, "../data"))


# -------------------------------------------------
def load_and_merge_folder(folder):
    """Load and merge all *_positions.npy and *_values.npy pairs inside the folder."""
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    positions_parts = []
    values_parts = []

    for name in sorted(os.listdir(folder_path)):
        if not name.endswith("_positions.npy"):
            continue
        base = name.replace("_positions.npy", "")
        pos_path = os.path.join(folder_path, name)
        val_path = os.path.join(folder_path, f"{base}_values.npy")
        if not os.path.exists(val_path):
            print(f"Skipping {base}: missing values file")
            continue

        positions_parts.append(np.load(pos_path, allow_pickle=True))
        values_parts.append(np.load(val_path, allow_pickle=True))

    if not positions_parts:
        raise ValueError(f"No valid position/value pairs found in {folder_path}")

    merged_positions = np.concatenate(positions_parts)
    merged_values = np.concatenate(values_parts)
    merged_count = len(positions_parts)
    print(f"Merged {merged_count} file pairs from {folder_path}: {len(merged_positions)} samples total")
    return merged_positions, merged_values


# -------------------------------------------------
def compute_heatmap(values, grid_pos_ids, reducer):
    """
    values: 1D list/array with scalar values (same indexing as positions/o_values)
    grid_pos_ids: [Nx][Ny] list of lists containing indices into values
    reducer: np.mean / np.median / np.max / np.min
    """
    heatmap = np.full((len(grid_pos_ids), len(grid_pos_ids[0])), np.nan, dtype=float)
    x_bf = None
    y_bf = None
    last_positions = []
    last_indices = list(range(max(len(values) - PLOT_LAST_VAL, 0), len(values)))
    last_index_set = set(last_indices)
    idx_to_cell = {}

    for i_x, grid_along_y in enumerate(grid_pos_ids):
        for i_y, grid_along_xy_ids in enumerate(grid_along_y):
            if len(grid_along_xy_ids) == 0:
                continue

            heatmap[i_x, i_y] = reducer([values[_id] for _id in grid_along_xy_ids])

            # Keep your marker logic: cell containing sample id 0
            if 0 in grid_along_xy_ids:
                x_bf = i_x
                y_bf = i_y
            # New marker: cells containing the most recent samples
            for idx in grid_along_xy_ids:
                idx_int = int(idx)
                if idx_int in last_index_set and idx_int not in idx_to_cell:
                    idx_to_cell[idx_int] = (i_x, i_y)
    # Preserve recency order: oldest of the slice first, most recent last
    for idx in last_indices:
        if idx in idx_to_cell:
            i_x, i_y = idx_to_cell[idx]
            last_positions.append((i_x, i_y, idx))
    return heatmap, x_bf, y_bf, last_positions


def plot_heatmaps_for_stat(heatmap, xi, yi, x_bf, y_bf, last_positions, stat_name, cmap, zoom_val, wavelen):
    # Tick labels can be wavelengths or raw positions
    if LABEL_IN_WAVELENGTHS:
        xtick_labels = [f"{(x - xi[0]) / wavelen:.2f}" for x in xi][::4]
        ytick_labels = [f"{(y - yi[0]) / wavelen:.2f}" for y in yi][::4]
        axis_label = "distance in wavelengths"
    else:
        xtick_labels = [f"{x:.2f}" for x in xi][::4]
        ytick_labels = [f"{y:.2f}" for y in yi][::4]
        axis_label = "position [m]"

    xtick_pos = zoom_val * np.arange(len(xi))[::4]
    ytick_pos = zoom_val * np.arange(len(yi))[::4]

    # UE marker (only if available)
    ue_position = None
    if (x_bf is not None) and (y_bf is not None):
        ue_position = Rectangle(
            ((y_bf - 0.5) * zoom_val, (x_bf - 0.5) * zoom_val),
            1,
            1,
            fill=False,
            edgecolor="red",
            lw=3,
        )
    last_patches = []
    
    if last_positions:
        print(f"Highlighting last {len(last_positions)} positions:")
        # Red outlines, more recent = more opaque
        alphas = np.linspace(0.3, 0.9, len(last_positions))
        colors = [(1.0, 0.0, 0.0, alpha) for alpha in alphas]
        for (x_last, y_last, _idx), color in zip(last_positions, colors):
            last_patches.append(
                Rectangle(
                    ((y_last - 0.5) * zoom_val, (x_last - 0.5) * zoom_val),
                    1,
                    1,
                    fill=False,
                    edgecolor=color,
                    lw=3,
                )
            )

    upsampled_heatmap = zoom(heatmap, zoom=zoom_val, order=1)

    # -----------------------
    # Linear plot (uW)
    # -----------------------
    fig, ax = plt.subplots()
    ax.set_title(f"{stat_name} | uW")
    img_lin = ax.imshow(upsampled_heatmap, cmap=cmap, origin="lower")

    ax.set_xticks(xtick_pos, labels=xtick_labels)
    ax.set_yticks(ytick_pos, labels=ytick_labels)

    if ue_position is not None:
        ax.add_patch(ue_position)
    for patch in last_patches:
        ax.add_patch(patch)

    cbar = fig.colorbar(img_lin)
    cbar.ax.set_ylabel(f"{stat_name} power [uW]")

    ax.set_xlabel(axis_label)
    ax.set_ylabel(axis_label)
    fig.tight_layout()
    plt.show()

    # -----------------------
    # Log plot (dB)
    # NOTE: Keeps your original scaling idea; label is generic dB.
    # -----------------------
    if PLOT_DB:
        fig, ax = plt.subplots()
        ax.set_title(f"{stat_name} | dB")

        up = upsampled_heatmap.copy()
        up[up <= 0] = np.nan  # protect log

        img_db = ax.imshow(
            10 * np.log10(up / 10e3),  # keep your original scaling
            vmin=-60,
            vmax=None,
            cmap=cmap,
            origin="lower",
        )

        ax.set_xticks(xtick_pos, labels=xtick_labels)
        ax.set_yticks(ytick_pos, labels=ytick_labels)

        cbar = fig.colorbar(img_db)
        cbar.ax.set_ylabel(f"{stat_name} power [dB]")

        ax.set_xlabel(axis_label)
        ax.set_ylabel(axis_label)
        fig.tight_layout()
        plt.show()


# -------------------------------------------------
# MAIN
# -------------------------------------------------

positions, o_values = load_and_merge_folder(FOLDER)

print(f"Processing {len(positions)} samples for folder {FOLDER}")

print("CHANGE OF X POSITION DUE TO QTM position not same as antenna position")
y_positions = [p.y + 0.1 for p in positions]

temp_pos_list = PositionerValues(positions)
positions_list = PositionerValues.from_xyz(
    temp_pos_list.get_x_positions(), y_positions, temp_pos_list.get_z_positions()
)

# power in uW
values = np.array([v.pwr_pw / 1e6 for v in o_values], dtype=float)
print(f"MAX POWER (raw samples): {np.max(values):.2f} uW")


grid_pos_ids, xi, yi = positions_list.group_in_grids(
    GRID_RESOLUTION #, min_x=2.0, max_x=4.2, min_y=1.0, max_y=2.5
)

for stat_name, reducer in STAT_FUNCS.items():
    heatmap, x_bf, y_bf, last_positions = compute_heatmap(values, grid_pos_ids, reducer)
    print(x_bf, y_bf)
    print(f"{stat_name}: max(cell)={np.nanmax(heatmap):.2f} uW")

    plot_heatmaps_for_stat(
        heatmap=heatmap,
        xi=xi,
        yi=yi,
        x_bf=x_bf,
        y_bf=y_bf,
        last_positions=last_positions,
        stat_name=stat_name,
        cmap=cmap,
        zoom_val=zoom_val,
        wavelen=wavelen,
    )
