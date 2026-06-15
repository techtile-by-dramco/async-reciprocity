"""
Interactive Plotly heatmap that refreshes every 2 seconds.
Loads the most recent folder in ../data by default, or a specific folder
if provided as a command-line argument.
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

# Reuse settings from the matplotlib version
from plot_all_folders_heatmap import (
    CMAP,
    DATA_DIR,
    GRID_RES,
    compute_heatmap,
    load_folder,
    wavelength,
)


def resolve_folder_name(target: Optional[str]) -> Tuple[str, str]:
    """Return (folder_name, folder_path) for the requested or newest folder."""
    if target:
        folder_path = target
        if not os.path.isabs(folder_path):
            folder_path = os.path.join(DATA_DIR, folder_path)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        return os.path.basename(folder_path), folder_path

    folder_entries = []
    for name in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, name)
        if os.path.isdir(folder_path):
            folder_entries.append((os.path.getmtime(folder_path), name))

    if not folder_entries:
        raise ValueError(f"No subfolders found in {DATA_DIR}")

    folder_entries.sort(key=lambda x: x[0], reverse=True)
    newest_name = folder_entries[0][1]
    return newest_name, os.path.join(DATA_DIR, newest_name)


def compute_plot_data(folder_path: str):
    """Load folder data and prepare values for Plotly."""
    positions, values = load_folder(folder_path)
    xs = np.array([p.x for p in positions], dtype=float)
    ys = np.array([p.y for p in positions], dtype=float)
    vs = np.array([v.pwr_pw / 1e6 for v in values], dtype=float)  # uW

    heatmap, _, x_edges, y_edges, _, _ = compute_heatmap(xs, ys, vs, GRID_RES)
    z = heatmap.T
    z = np.where(np.isnan(z), None, z)  # Plotly prefers None over NaN

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    return x_centers, y_centers, z, x_edges, y_edges


def _target_cell(target_xy, x_edges, y_edges):
    """Return (i_x, i_y) for target point or None if outside grid."""
    if target_xy is None:
        return None
    x, y = target_xy
    i_x = np.digitize([x], x_edges) - 1
    i_y = np.digitize([y], y_edges) - 1
    ix, iy = int(i_x[0]), int(i_y[0])
    if 0 <= ix < len(x_edges) - 1 and 0 <= iy < len(y_edges) - 1:
        return ix, iy
    return None


def _target_rect(target_xy, x_edges, y_edges):
    """Rectangle covering ~0.1 wavelength, snapped to whole grid cells."""
    cell = _target_cell(target_xy, x_edges, y_edges)
    if cell is None:
        return None
    ix, iy = cell
    target_size = 0.1 * wavelength
    cells_span = max(1, int(round(target_size / GRID_RES)))

    start_ix = max(0, ix - (cells_span - 1) // 2)
    end_ix = min(len(x_edges) - 2, start_ix + cells_span - 1)
    start_ix = max(0, end_ix - cells_span + 1)

    start_iy = max(0, iy - (cells_span - 1) // 2)
    end_iy = min(len(y_edges) - 2, start_iy + cells_span - 1)
    start_iy = max(0, end_iy - cells_span + 1)

    return x_edges[start_ix], x_edges[end_ix + 1], y_edges[start_iy], y_edges[end_iy + 1]


def create_figure(folder_name: str, x_centers, y_centers, z, x_edges, y_edges, target_xy=None):
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_centers,
            y=y_centers,
            colorscale=CMAP,
            colorbar=dict(title="Mean power per cell [uW]"),
        )
    )
    fig.update_layout(
        title=f"{folder_name} | mean power per cell [uW]",
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        template="plotly_white",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)  # lock aspect ratio

    rect = _target_rect(target_xy, x_edges, y_edges)
    if rect:
        x0, x1, y0, y1 = rect
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            line=dict(color="cyan", width=3),
            fillcolor="rgba(0,255,255,0.25)",
        )
    return fig


def build_app(folder_name: str, folder_path: str, target_xy):
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H3(id="title"),
            dcc.Graph(id="heatmap"),
            dcc.Interval(id="refresh", interval=2000, n_intervals=0),
        ]
    )

    @app.callback(
        [Output("heatmap", "figure"), Output("title", "children")],
        [Input("refresh", "n_intervals")],
    )
    def update_heatmap(_):
        x_centers, y_centers, z, x_edges, y_edges = compute_plot_data(folder_path)
        fig = create_figure(folder_name, x_centers, y_centers, z, x_edges, y_edges, target_xy)
        timestamp = time.strftime("%H:%M:%S")
        return fig, f"{folder_name} (updated {timestamp})"

    return app


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Plotly heatmap that refreshes every 2 seconds."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help="Specific data subfolder to load (defaults to the newest in ../data)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host for the Dash server (default: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=8050, help="Port for the Dash server (default: 8050)")
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=[3.181, 1.774, 0.266],
        help="Target xyz to highlight with a ~0.1-wavelength rectangle (z ignored). Default: 3.181 1.774 0.266",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    folder_name, folder_path = resolve_folder_name(args.folder)
    print(f"Using folder: {folder_path}")

    target_xy = (args.target[0], args.target[1]) if args.target else None
    app = build_app(folder_name, folder_path, target_xy)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
