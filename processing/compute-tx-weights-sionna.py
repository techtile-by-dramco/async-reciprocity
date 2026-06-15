# ============================================================
# compute-tx-weights.py
# Single RX at target_location
# Irregular TX array: one Transmitter per element position (all_pts)
# Sionna RT requires tx_array/rx_array -> set to 1x1 (element pattern only)
# Plywood box room via XML
# ============================================================
# %%
import os
from pathlib import Path
import subprocess
import argparse
import numpy as np
import requests
import yaml
from typing import Iterable

import mitsuba as mi

import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
WAVELENGTH = 3e8 / 920e6  # meters
GRID_RES = 0.08 * WAVELENGTH
GRID_BOUNDS = {
    "xmin": 2.66882,
    "xmax": 3.94708,
    "ymin": 1.12416,
    "ymax": 2.40242,
}
# Receiver location (single antenna RX)
# target_location = np.array([3.278115966796875, 1.778475341796875, 0.2522267761230469], dtype=np.float32)  # center of grid
# target_location = np.array([3.181, 1.774, 0.266], dtype=np.float32) #jarne location
# target_location = np.array(
#     [3.201299560546875, 1.70512451171875, 0.23005308532714844], dtype=np.float32
# )
# Specular multipath control

SPEC_ORDERS = [0,1,2,1,2]
SDRs = [False, False, False, True, True] 
active_tiles = ["A05", "A06", "A07", "A08", "A09", "A10", "B05", "B06", "B07", "B08", "B09", "B10", "C05", "C06", "C07", "C08", "C09", "C10", "D05", "D06", "D07", "D08", "D09", "D10", "E05", "E06", "E07", "E08", "E09", "E10", "F05", "F06", "F07", "F08", "F09", "F10", "G05", "G06", "G07", "G08", "G09", "G10"]
active_tiles_upper = {t.upper() for t in active_tiles}

positions_url = (
    "https://raw.githubusercontent.com/techtile-by-dramco/"
    "techtile-description/refs/heads/main/geometry/"
    "techtile_antenna_locations.yml"
)


# Narrowband frequency for CSI
fc = 920e6  # Hz


# XML scene path
script_dir = Path(__file__).resolve().parent
xml_path = script_dir / "room.xml"
experiment_settings_path = script_dir.parent / "experiment-settings.yaml"


def load_target_location(settings_path: Path) -> np.ndarray:
    """Load target_location from experiment-settings.yaml and return it as np.array."""
    if not settings_path.exists():
        raise FileNotFoundError(f"Experiment settings file not found at {settings_path}")

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    try:
        raw_value = settings["experiment_config"]["target_location"]
    except Exception as e:
        raise ValueError("target_location not found in experiment-settings.yaml") from e

    if isinstance(raw_value, str):
        parts = [p.strip() for p in raw_value.split(",") if p.strip()]
    elif isinstance(raw_value, Iterable):
        parts = list(raw_value)
    else:
        raise ValueError("target_location must be a string or iterable of three coordinates")

    if len(parts) != 3:
        raise ValueError("target_location must contain exactly three coordinates (x, y, z)")

    try:
        coords = [float(val) for val in parts]
    except (TypeError, ValueError) as e:
        raise ValueError("target_location coordinates must be numeric") from e

    return np.array(coords, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute TX weights using Sionna RT.")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU execution (avoids GPU memory issues).",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        help="Force a specific GPU index (overrides auto free-memory selection).",
    )
    parser.add_argument(
        "--grid-plane",
        action="store_true",
        help="Compute beamformed power on a 2D plane (uses GRID_BOUNDS and GRID_RES).",
    )
    parser.add_argument(
        "--plot-grid",
        action="store_true",
        help="If grid-plane is enabled, render and save a power heatmap PNG.",
    )
    parser.add_argument(
        "--target-location",
        type=str,
        help="Override target_location as 'x,y,z' (comma-separated).",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix appended to output filenames (e.g. 'run1').",
    )
    return parser.parse_args()


def parse_target_location(raw_value: str) -> np.ndarray:
    """Parse a 'x,y,z' string into a target_location array."""
    parts = [p.strip() for p in raw_value.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("target_location must contain exactly three coordinates (x, y, z)")
    try:
        coords = [float(val) for val in parts]
    except (TypeError, ValueError) as e:
        raise ValueError("target_location coordinates must be numeric") from e
    return np.array(coords, dtype=np.float32)


def pick_gpu_with_max_free_memory() -> int | None:
    """Return GPU index with most free memory via nvidia-smi; None if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        candidates = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                idx = int(parts[0])
                free = float(parts[1])
                candidates.append((idx, free))
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            print(f"Selecting GPU {best[0]} with ~{best[1]} MiB free (nvidia-smi).")
            return best[0]
    except Exception as exc:
        print(f"Could not query nvidia-smi for free memory, using default GPU selection: {exc}")
    return None


def configure_mitsuba_variant(cpu_only: bool, gpu_index: int | None) -> str:
    """Select Mitsuba CPU/GPU variant via mi.set_variant before importing Sionna."""
    env_cpu_only = os.environ.get("GBWPT_CPU_ONLY", "0") in {"1", "true", "True", "yes"}
    if cpu_only or env_cpu_only:
        mi.set_variant("llvm_ad_mono_polarized")
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        print(
            "Forcing CPU Mitsuba variant 'llvm_ad_mono_polarized' (flag or GBWPT_CPU_ONLY=1)."
        )
        return "llvm_ad_rgb"

    chosen_idx = None
    if gpu_index is not None and gpu_index >= 0:
        chosen_idx = gpu_index
        print(f"Using user-selected GPU index {chosen_idx}.")
    else:
        chosen_idx = pick_gpu_with_max_free_memory()
        if chosen_idx is not None:
            print(f"Selecting GPU {chosen_idx} with most free memory (nvidia-smi).")

    if chosen_idx is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen_idx)

    try:
        mi.set_variant("llvm_ad_mono_polarized")
        print("Using GPU Mitsuba variant 'cuda_ad_rgb'.")
        return "cuda_ad_rgb"
    except Exception as exc:
        print(f"Failed to set GPU variant, falling back to CPU Mitsuba variant 'llvm_ad_rgb': {exc}")
        mi.set_variant("llvm_ad_mono_polarized")
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return "llvm_ad_rgb"


args = parse_args()
if args.target_location:
    target_location = parse_target_location(args.target_location)
    print(f"Using target_location from args: {target_location.tolist()}")
else:
    target_location = load_target_location(experiment_settings_path)
    print(f"Using target_location from {experiment_settings_path}: {target_location.tolist()}")
configure_mitsuba_variant(cpu_only=args.cpu_only, gpu_index=args.gpu_index)

import sionna.rt as rt
from sionna.rt import load_scene

# ============================================================
# LOAD TX POSITIONS (channel 1 only) -> all_pts
# ============================================================

resp = requests.get(positions_url, timeout=20)
resp.raise_for_status()
config = yaml.safe_load(resp.content.decode("utf-8"))

tx_positions = []
tx_names = []
selected_entries = []

tile_lookup = {entry["tile"].upper(): entry for entry in config["antennes"]}


# only extract ceiling
for tile in active_tiles:
    entry = tile_lookup.get(tile.upper())
    if entry is None:
        raise RuntimeError(f"Tile '{tile}' not found in config['antennes']")
    selected_entries.append(entry)
    ch1 = entry["channels"][1]  # channel 1
    pos = np.array([ch1["x"], ch1["y"], ch1["z"]], dtype=np.float32)
    tx_positions.append(pos)
    tx_names.append(f"{entry['tile']}_ch1")

all_pts = np.stack(tx_positions, axis=0)  # (N,3)
num_tx = all_pts.shape[0]
print(f"Loaded {num_tx} TX elements into all_pts.")

# ============================================================
# ROOM SIZE
# ============================================================

Lx = 8.0
Ly = 4.0
Lz = 2.4
print(f"Room size: Lx={Lx:.2f} m, Ly={Ly:.2f} m, Lz={Lz:.2f} m")



solver = rt.PathSolver()

for specular_order, SDR in zip(SPEC_ORDERS, SDRs):
    print(f"\n=== Specular order: {specular_order}, SDR: {SDR} ===")
    grid_xs = grid_ys = None
    grid_shape = None

    # ============================================================
    # LOAD SCENE
    # ============================================================

    if specular_order == 0:
        scene = load_scene(None)
    else:
        # ============================================================
        # WRITE MITSUBA XML (PLYWOOD BOX)
        # ============================================================

        room_xml = f"""<?xml version="1.0"?>
        <scene version="3.0.0">
            <integrator type="path"/>

            <bsdf type="itu-radio-material" id="wall_mat">
                <string name="type" value="plywood"/>
            </bsdf>

            <shape type="rectangle">
                <transform name="to_world">
                    <scale x="{Lx/2}" y="{Ly/2}"/>
                    <translate x="{Lx/2}" y="{Ly/2}" z="0"/>
                </transform>
                <ref id="wall_mat"/>
            </shape>

            <shape type="rectangle">
                <transform name="to_world">
                    <rotate x="1" angle="180"/>
                    <scale x="{Lx/2}" y="{Ly/2}"/>
                    <translate x="{Lx/2}" y="{Ly/2}" z="{Lz}"/>
                </transform>
                <ref id="wall_mat"/>
            </shape>

            <shape type="rectangle">
                <transform name="to_world">
                    <rotate x="1" angle="90"/>
                    <scale x="{Lx/2}" y="{Lz/2}"/>
                    <translate x="{Lx/2}" y="0" z="{Lz/2}"/>
                </transform>
                <ref id="wall_mat"/>
            </shape>

            <shape type="rectangle">
                <transform name="to_world">
                    <rotate x="1" angle="-90"/>
                    <scale x="{Lx/2}" y="{Lz/2}"/>
                    <translate x="{Lx/2}" y="{Ly}" z="{Lz/2}"/>
                </transform>
                <ref id="wall_mat"/>
            </shape>

        </scene>

        """

        xml_path.write_text(room_xml, encoding="utf-8")
        print(f"Wrote room XML to {xml_path}")
        scene = load_scene(str(xml_path), merge_shapes=False)
    #

    # ============================================================
    # REQUIRED by Sionna RT (even for "single antennas"):
    # set 1x1 arrays to define element pattern / polarization
    # ============================================================

    scene.tx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.0,
        horizontal_spacing=0.0,
        pattern="dipole",
        polarization="V",
    )

    scene.rx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.0,
        horizontal_spacing=0.0,
        pattern="dipole",
        polarization="V",
    )

    # ============================================================
    # RX: primary receiver at target_location (+ optional grid receivers)
    # ============================================================

    rx = rt.Receiver(name="rx", position=target_location.tolist())
    scene.add(rx)

    grid_receivers = []
    if args.grid_plane:
        xs = np.arange(GRID_BOUNDS["xmin"], GRID_BOUNDS["xmax"] + GRID_RES / 2, GRID_RES)
        ys = np.arange(GRID_BOUNDS["ymin"], GRID_BOUNDS["ymax"] + GRID_RES / 2, GRID_RES)
        grid_xs, grid_ys = xs, ys
        grid_shape = (len(xs), len(ys))
        grid_z = float(target_location[2])
        for i_x, x in enumerate(xs):
            for i_y, y in enumerate(ys):
                name = f"rx_grid_{i_x}_{i_y}"
                r = rt.Receiver(name=name, position=[float(x), float(y), grid_z])
                scene.add(r)
                grid_receivers.append((name, i_x, i_y))

    # ============================================================
    # TX: irregular array -> one Transmitter per TX element position (all_pts)
    # ============================================================

    for name, pos in zip(tx_names, all_pts):
        scene.add(rt.Transmitter(name=name, position=pos.tolist()))
        scene.get(name).look_at(rx)

    # ============================================================
    # VISUALIZE SCENE ONLY
    # ============================================================

    # scene.preview()  # interactive 3D viewer

    # %%
    # ============================================================
    # RAY TRACING (PathSolver API)
    # ============================================================
    # %%

    paths = solver(
        scene=scene,
        max_depth=specular_order,
        los=True,
        specular_reflection=SDR,
        diffuse_reflection=SDR,
        refraction=False,
        synthetic_array=False,
        seed=1,
    )

    # %%
    # scene.preview(paths=paths, show_devices=True)  # interactive 3D viewer

    # # ============================================================
    # # NARROWBAND CSI PER TX ELEMENT
    # # ============================================================

    # %%
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")

    # Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    print("Shape of a: ", a.shape)

    # Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    print("Shape of tau: ", tau.shape)

    # %%
    a_abs = np.abs(a)[0, 0, 0, 0, :, 0]
    t = tau[0, 0, 0, 0, :] / 1e-9  # Scale to ns

    import matplotlib.pyplot as plt

    # And plot the CIR
    plt.figure()
    plt.title("Channel impulse response")
    plt.stem(t, a_abs)
    plt.xlabel(r"$\tau$ [ns]")
    plt.ylabel(r"$|a|$")

    # a = paths.a
    # tau = paths.tau

    # rot = tf.exp(-1j * 2.0 * np.pi * fc * tf.cast(tau, tf.complex64))
    # h_paths = a * rot
    # H_all = tf.reduce_sum(h_paths, axis=-1)  # sum over paths (assumed last axis)
    # H_all = tf.squeeze(H_all)

    # def extract_per_tx_vector(H, num_tx):
    #     H = tf.convert_to_tensor(H)
    #     shp = H.shape.as_list()

    #     if len(shp) == 1 and shp[0] == num_tx:
    #         return H

    #     for ax, s in enumerate(shp):
    #         if s == num_tx:
    #             x = H
    #             # single RX assumed: index any remaining non-singleton dims at 0
    #             for other_ax in reversed(range(len(shp))):
    #                 if (
    #                     other_ax != ax
    #                     and x.shape[other_ax] is not None
    #                     and x.shape[other_ax] > 1
    #                 ):
    #                     x = tf.gather(x, 0, axis=other_ax)
    #             x = tf.squeeze(x)

    #             if x.shape.rank == 1 and x.shape[0] == num_tx:
    #                 return x

    #             x = tf.experimental.numpy.moveaxis(x, ax, 0)
    #             x = tf.reshape(x, [num_tx, -1])
    #             return x[:, 0]

    #     raise RuntimeError(f"TX axis of length {num_tx} not found in shape {shp}")

    # H_tx = extract_per_tx_vector(H_all, num_tx)

    # csi_mag = tf.abs(H_tx).numpy()
    # csi_phase_rad = tf.math.angle(H_tx).numpy()
    # csi_phase_deg = np.rad2deg(csi_phase_rad)

    # print("H_tx shape:", H_tx.shape)
    # print("Per-TX CSI magnitude:", csi_mag)
    # print("Per-TX CSI phase [deg]:", csi_phase_deg)
    # print("Per-TX CSI phase unwrapped [deg]:", np.rad2deg(np.unwrap(csi_phase_rad)))

    # %%
    # ------------------------------------------------------------
    # Convert delays to narrowband frequency response at fc:
    #   H(fc) = sum_l a_l * exp(-j 2*pi*fc*tau_l)
    #
    # Use time step 0 (index 0). If you have mobility/time evolution,
    # loop over the last axis of a instead.
    # ------------------------------------------------------------
    # Select RX=0, RXant=0, TXant=0, time_step=0

    # Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    print("Shape of a: ", a.shape)

    # Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    print("Shape of tau: ", tau.shape)

    t = tau[0, 0, :, 0, :]
    ampl = np.abs(a)[0,0,:,0,:,0]

    h = ampl*np.exp(-1j * 2.0 * np.pi * fc * t)  # [num_tx, num_paths]
    w = np.sum(h, axis=-1)
    w_phase_deg = np.rad2deg(-np.angle(w))  # degrees
    bf_weights = 0.8 * np.exp(1j * np.deg2rad(w_phase_deg))

    # %%
    # ------------------------------------------------------------
    # Export to YAML in the same structure as your Friis script
    # tile -> [{ch:0, ampl:0, phase:0}, {ch:1, ampl:AMPLITUDE, phase:<deg>}]
    # ------------------------------------------------------------
    out_full_dict = {}
    out_dict = {}

    # IMPORTANT: This assumes the TX ordering in paths.cir() matches the ordering
    # you added Transmitters / the ordering in selected_entries (channel 1).
    if len(selected_entries) != len(w_phase_deg):
        raise RuntimeError(
            f"Mismatch: selected config has {len(selected_entries)} TX entries but CIR has {len(w_phase_deg)} TX."
        )

    for i, c in enumerate(selected_entries):
        ch1 = c["channels"][1]
        tile_name = c["tile"]

        # ensure list exists
        out_full_dict[tile_name] = []
        out_dict[tile_name] = (
            float(w_phase_deg[i])
        )  # done just to compare with my LoS computations

        # CH 0 forced to zero
        out_full_dict[tile_name].append({"ch": 0, "ampl": float(0.0), "phase": float(0.0)})

        # CH 1 uses Sionna-derived phase, fixed amplitude
        out_full_dict[tile_name].append(
            {"ch": 1, "ampl": float(0.8), "phase": float(w_phase_deg[i])}
        )

    suffix = f"-{args.output_suffix}" if args.output_suffix else ""
    output_path = f"../client/tx-weights-sionna{suffix}.yml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_full_dict, f, sort_keys=False)

        output_path = (
            f"../client/tx-phases-sionna-{specular_order}{'' if not SDR else 'SDR'}{suffix}.yml"
        )
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_dict, f, sort_keys=False)

    print(f"Wrote Sionna-based TX weights to: {output_path}")

    # ============================================================
    # OPTIONAL: BEAMFORMED POWER MAP ON 2D GRID
    # ============================================================
    if args.grid_plane and grid_shape is not None and len(grid_receivers) > 0:
        grid_power = np.full(grid_shape, np.nan, dtype=float)
        for r_idx, (name, i_x, i_y) in enumerate(grid_receivers, start=1):  # offset by 1 for target rx
            tau_r = tau[r_idx, 0, :, 0, :]  # [num_tx, num_paths]
            ampl_r = np.abs(a)[r_idx, 0, :, 0, :, 0]
            h_r = ampl_r * np.exp(-1j * 2.0 * np.pi * fc * tau_r)
            h_sum = np.sum(h_r, axis=-1)  # [num_tx]
            rx_field = np.sum(bf_weights * h_sum)
            grid_power[i_x, i_y] = float(np.abs(rx_field) ** 2)

        grid_out = script_dir / f"grid-power-{specular_order}{'SDR' if SDR else ''}.npy"
        np.save(grid_out, grid_power)
        print(f"Saved beamformed grid power map to {grid_out} with shape {grid_power.shape}")

        if args.plot_grid:
            extent = [
                GRID_BOUNDS["xmin"],
                GRID_BOUNDS["xmax"],
                GRID_BOUNDS["ymin"],
                GRID_BOUNDS["ymax"],
            ]
            plt.figure()
            plt.title(f"Beamformed power map | order {specular_order}{'SDR' if SDR else ''}")
            img = plt.imshow(
                grid_power.T,
                origin="lower",
                cmap="inferno",
                extent=extent,
            )
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            cbar = plt.colorbar(img)
            cbar.set_label("Power (arb. units)")
            png_out = script_dir / f"grid-power-{specular_order}{'SDR' if SDR else ''}.png"
            plt.tight_layout()
            plt.savefig(png_out)
            print(f"Saved grid power heatmap to {png_out}")
            plt.close()

    del paths, a, tau, scene

    # %%
