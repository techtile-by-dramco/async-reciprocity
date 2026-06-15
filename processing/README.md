### Processing

Helper scripts for preparing TX phases and visualizing measurements.

#### Plotting
- `plot-values-positions-2d.py`: per-folder matplotlib heatmap; set `FOLDER` inside the script.
- `plot_all_folders_heatmap.py`: aggregates folders under `../data`, saves `heatmap.png` into each folder, and shows the plot. Example:
  ```
  python plot_all_folders_heatmap.py --plot-all --plot-movement
  ```
- `plot_all_folders_heatmap_live.py`: Plotly/Dash heatmap that refreshes every 2s (defaults to newest folder).
- `print-heatmap-a3.py`: build a real-size tiled A3 PDF from `heatmap.csv` + `x_edges.csv` + `y_edges.csv` using physics-aware interpolation (IDW, `p=2` by default).
  The output includes a separate A3 page with the heatmap colorbar in `uW` (disable with `--no-colorbar-page`).
  Colorbar page defaults: `600 DPI` and denser ticks (configure with `--colorbar-dpi` and `--colorbar-tick-count`).
  The output also includes a separate A3 layout legend page showing where each tile page belongs (disable with `--no-layout-page`).
  Tiles include print bleed by default (`--bleed-mm 3.0`) and trim marks for easier cutting (`--no-trim-marks` to disable).
  A white cutting margin is included by default (`--margin-mm 6.0`).
  Borders/registration marks are off by default for borderless printing (`--draw-borders` to enable).
  Example for `RECI-merged`:
  ```
  python processing/print-heatmap-a3.py --folder data/RECI-merged
  ```
  Dry-run (layout only):
  ```
  python processing/print-heatmap-a3.py --folder data/RECI-merged --dry-run
  ```
  Print notes:
  - print at `100%` / `Actual size`
  - disable any `Fit to page` or scaling
  - use A3 paper; default keeps a small cutting margin (6 mm) and no drawn borders

#### Energy-ball post-processing
- `process-energy-ball.py`: summarizes `server/record/data/exp-*.yml`, plots iteration power, and rewrites `client/tx-phases-energy-ball.yml` with the best phases.

#### Phase generation
- `compute-tx-phases.py`: fetch tile geometry and write `client/tx-phases-friis.yml` and `client/tx-phases-benchmark.yml`.
- `compute-tx-weights-sionna.py`: generate Sionna-based weights/phases (`tx-weights-sionna.yml`, `tx-phases-sionna-<specular_order>SDR.yml`).
