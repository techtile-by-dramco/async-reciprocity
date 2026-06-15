import numpy as np
import matplotlib.pyplot as plt
import requests
import yaml
from matplotlib.patches import Rectangle
from scipy.constants import c as v_c

#################### CONFIGURATIONS ####################
output_path_benchmark = "../client/tx-weights-benchmark.yml"
output_full_path_friis = "../client/tx-weights-friis.yml"
output_path_friis = "../client/tx-phases-friis.yml"
AMPLTIUDE = 0.8
########################################################


positions_url = r"https://raw.githubusercontent.com/techtile-by-dramco/techtile-description/refs/heads/main/geometry/techtile_antenna_locations.yml"

# Retrieve the file content from the URL
response = requests.get(positions_url, allow_redirects=True)
# Convert bytes to string
content = response.content.decode("utf-8")
# Load the yaml
config = yaml.safe_load(content)

antennas = dict()


# UE position (energy neutral device)
target_location = np.array([3.27905810546875, 1.7493585205078126, 0.2528133087158203])

# Constants
f = 920e6  # Antenna frequency (Hz)
lambda_ = v_c / f  # Wavelength (m)

out_full_dict = dict()
out_dict = dict()

for c in config["antennes"]:
    # only one antenna is used
    ch1 = c["channels"][0]
    tile_name = c["tile"]
    out_dict[tile_name] = []
    out_full_dict[tile_name] = []
    pos = [ch1["x"]+7.5/100, ch1["y"], ch1["z"]] #TODO #ALERT here added offset because not yet in file (+7.5cm in the x-direction)
    phase = 0
    ampl = AMPLTIUDE
    d_EN = np.linalg.norm(pos - target_location)  # Scalar distances (L x 1)

    t = d_EN / v_c  # Time delay (s)
    h = np.exp(-1j * 2 * np.pi *f *t)

    print(f"Delay tile {tile_name}: {(d_EN/v_c)/ 1e-9:.3f} ns")
    print(f"Complex value tile {tile_name}: {h:.3f}")

    # MRT weights
    w = np.conj(h)
    # CH 0 should be zero in this case
    out_dict[tile_name] = float(np.rad2deg(np.angle(w)))
    out_full_dict[tile_name].append({"ch": 0, "ampl": float(0.0), "phase": float(0.0)})

    out_full_dict[tile_name].append(
        {"ch": 1, "ampl": float(AMPLTIUDE), "phase": float(np.rad2deg(np.angle(w)))}
    )


with open(output_full_path_friis, "w", encoding="utf-8") as f:
    yaml.safe_dump(out_full_dict, f, sort_keys=False)


with open(output_path_friis, "w", encoding="utf-8") as f:
    yaml.safe_dump(out_dict, f, sort_keys=False)
