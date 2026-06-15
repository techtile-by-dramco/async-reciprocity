import sys, os

# Path setup for repo imports and data output
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_path = os.path.dirname(current_dir)
repo_root = os.path.dirname(parent_path)
sys.path.insert(0, repo_root)

import time
from lib.yaml_utils import read_yaml_file
from lib.ep import RFEP

settings = read_yaml_file("../../experiment-settings.yaml")
rfep = RFEP(settings["ep"]["ip"], settings["ep"]["port"])

try:
    while True:
        print(rfep.get_data())
        time.sleep(1)
finally:
    rfep.stop()