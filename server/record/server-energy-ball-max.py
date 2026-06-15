#!/usr/bin/python3
# usage: sync_server.py <delay> <num_subscribers>

# VALUE "num_subscribers" --> IMPORTANT --> The server waits until all subscribers have sent their "alive" or ready message before starting a measurement.

# to kill it: sudo fuser -k 5557/tcp

import argparse
import logging
import zmq
import time
import sys
import os
import signal
import subprocess
import shutil
from datetime import datetime, UTC, timezone
# from helper import *
import numpy as np

def _fmt_power(pw: float) -> str:
    """Format power in pW with scaling to nW/uW/mW."""
    if pw is None:
        return "n/a"
    try:
        val = float(pw)
    except (TypeError, ValueError):
        return str(pw)
    unit = "pW"
    abs_pw = abs(val)
    if abs_pw >= 1e9:
        unit = "mW"
        val = val / 1e9
    elif abs_pw >= 1e6:
        unit = "uW"
        val = val / 1e6
    elif abs_pw >= 1e3:
        unit = "nW"
        val = val / 1e3
    return f"{val:12.3f} {unit}"

# =============================================================================
#                           Experiment Configuration
# =============================================================================
DEFAULT_HOST = "*"          # Host address to bind to. "*" means all interfaces.
DEFAULT_SYNC_PORT = "5557"  # Port used for synchronization messages.
DEFAULT_ALIVE_PORT = "5558" # Port used for heartbeat/alive messages.
DEFAULT_DATA_PORT = "5559"  # Port used for data transmission.
DEFAULT_DELAY = 2           # Seconds to wait before sending SYNC.
DEFAULT_SUBS = 42           # Expected subscribers.
DEFAULT_WAIT_TIMEOUT = 60.0 * 10.0
# =============================================================================
# =============================================================================

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

def parse_args():
    parser = argparse.ArgumentParser(
        description="ZMQ energy-ball max server for GBWPT experiments."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind (default: *)")
    parser.add_argument(
        "--sync-port", default=DEFAULT_SYNC_PORT, help="Port for SYNC PUB (default: 5557)"
    )
    parser.add_argument(
        "--alive-port", default=DEFAULT_ALIVE_PORT, help="Port for alive/ready REP (default: 5558)"
    )
    parser.add_argument(
        "--data-port", default=DEFAULT_DATA_PORT, help="Port for data REP (default: 5559)"
    )
    parser.add_argument(
        "--delay", type=int, default=DEFAULT_DELAY, help="Delay before sending SYNC (seconds)"
    )
    parser.add_argument(
        "--num-subscribers",
        type=int,
        default=DEFAULT_SUBS,
        help="Expected subscribers before SYNC",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=DEFAULT_WAIT_TIMEOUT,
        help="Timeout in seconds to give up waiting for new messages once some arrived (default: 600s).",
    )
    return parser.parse_args()


args = parse_args()
delay = args.delay
num_subscribers = args.num_subscribers
host = args.host
sync_port = args.sync_port
alive_port = args.alive_port
data_port = args.data_port


# Ensure the sync port is free before binding sockets
def _free_port(port: str):
    """Attempt to kill any process bound to the given TCP port using fuser."""
    cmd = ["fuser", "-k", f"{port}/tcp"]
    if shutil.which(cmd[0]) is None:
        logger.warning("fuser not found; skipping port cleanup for %s/tcp", port)
        return
    try:
        subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Attempted to free port %s/tcp via fuser", port)
    except Exception as e:
        logger.warning("Failed to run fuser for %s/tcp: %s", port, e)


_free_port(sync_port)

# Creates a socket instance
context = zmq.Context()

sync_socket = context.socket(zmq.PUB)
# Binds the socket to a predefined port on localhost
sync_socket.bind("tcp://{}:{}".format(host, sync_port))

# Create a SUB socket to listen for subscribers
alive_socket = context.socket(zmq.REP)
alive_socket.bind("tcp://{}:{}".format(host, alive_port))

# Create a SUB socket to listen for subscribers
data_socket = context.socket(zmq.REP)
data_socket.bind("tcp://{}:{}".format(host, data_port))

# Measurement and experiment identifiers
meas_id = 0

# Unique ID for the experiment based on current UTC timestamp
unique_id = str(datetime.now(UTC).strftime("%Y%m%d%H%M%S"))

# Directory where this script is located
# script_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))

# ZeroMQ poller setup
poller = zmq.Poller()
poller.register(
    alive_socket, zmq.POLLIN
)  # Register the alive socket to monitor incoming messages

# Track time of the last received message

# Maximum time to wait for messages before breaking out of the inner loop
WAIT_TIMEOUT = args.wait_timeout

TIME_DIFF_MSG = 1.0  # warn if gap between received messages exceeds this (s)

# Inform the user that the experiment is starting
logger.info("Starting experiment: %s", unique_id)


# Path setup for repo imports and data output
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_path = os.path.dirname(current_dir)
repo_root = os.path.dirname(parent_path)
sys.path.insert(0, repo_root)
output_path = os.path.join(parent_path, f"record/data/exp-{unique_id}.yml")

from lib.yaml_utils import read_yaml_file
from lib.ep import RFEP

settings = read_yaml_file("experiment-settings.yaml")
rfep = RFEP(settings["ep"]["ip"], settings["ep"]["port"])


logger.info("Initial RFEP data: %s", rfep.get_data())

CAPTURE_POWER_TIME = 3  # s
prev_power = 0
stop_requested = False

# Log where the server is bound
logger.info(
    "Server listening on host '%s' (sync tcp://%s:%s, alive tcp://%s:%s, data tcp://%s:%s)",
    host,
    host,
    sync_port,
    host,
    alive_port,
    host,
    data_port,
)

def _handle_interrupt(signum=None, frame=None):
    global stop_requested
    stop_requested = True


def _handle_tstp(signum=None, frame=None):
    """Handle Ctrl+Z by triggering cleanup instead of suspend."""
    _handle_interrupt()
    cleanup()
    sys.exit(0)
def send_sync():
    """
    Synchronize measurement across all subscribers and handle their responses.
    
    Waits for incoming messages from all subscribers with a configurable timeout.
    Records received messages to a file and sends responses back. After all subscribers
    have reported or timeout is reached, broadcasts a synchronization signal with the
    current measurement ID and unique server identifier to trigger the next measurement cycle.
    
    Global variables modified:
        meas_id: Incremented after each synchronization cycle.
    """
    global meas_id, poller
    messages_received = 0
    new_msg_received = 0 

    while messages_received < num_subscribers and not stop_requested:
        try:
            # Poll the socket for incoming messages with a 1-second timeout
            _socks = dict(poller.poll(1000))
        except KeyboardInterrupt:
            _handle_interrupt()
            return False

        # If some messages were received but no new message comes within WAIT_TIMEOUT, break
        if messages_received > 2 and time.time() - new_msg_received > WAIT_TIMEOUT:
            logger.error("Timeout waiting for subscribers. Proceeding with %d/%d messages received.",
                         messages_received, num_subscribers)
            break

        if alive_socket in _socks and _socks[alive_socket] == zmq.POLLIN:
            # Record time when a new message is received
            new_msg_received = time.time()

            # Receive the message string from the subscriber
            _message = alive_socket.recv_string()
            messages_received += 1

            # Print received message and write it to the YAML file
            logger.info("%s (%d/%d)", _message, messages_received, num_subscribers)
            f.write(f"     - {_message}\n")

            # Process the request (example placeholder)
            response = "Response from server"

            # Send response back to the subscriber
            alive_socket.send_string(response)

    # Wait a fixed delay before sending the next SYNC signal
    logger.info("sending 'SYNC' message in %.2f ms...", delay * 1000.0)
    f.flush()
    time.sleep(delay)

    # Increment measurement ID for next iteration
    meas_id += 1

    # Broadcast synchronization message to all subscribers
    sync_socket.send_string(f"{meas_id} {unique_id}")  # str(meas_id)
    logger.info("SYNC %s", meas_id)
    return True


def collect_power(next_tx_in: float) -> float:
    max_samples = []


    time.sleep(next_tx_in) # no need for margin as we take the max power anyhow.

    start_time = time.time()
    logger.info("Collecting power measurements for %s seconds...", CAPTURE_POWER_TIME)
    try:
        while CAPTURE_POWER_TIME > time.time() - start_time and not stop_requested:
            d = rfep.get_data()
            if d is None:
                continue
            max_samples.append(d.pwr_pw)
    except KeyboardInterrupt:
        _handle_interrupt()

    # take median of the max 10 power samples, median to avoid outliers
    max_samples = sorted(max_samples, reverse=True)[:10]
    if not max_samples:
        logger.warning("No power samples captured.")
        return 0.0
    median_power = np.median(max_samples).item()  # np.array to scalar
    logger.info(
        "Power stats: samples=%d max=%s median=%s",
        len(max_samples),
        _fmt_power(max_samples[0]),
        _fmt_power(median_power),
    )
    return median_power


def wait_till_tx_done(is_stronger: bool, best_phase_per_host: dict[str, float]):
    # Wait for all subscribers to send a TX DONE MODE message
    logger.info("Waiting for %d subscribers to send a TX DONE Mode ...", num_subscribers)

    # Track number of messages received from subscribers
    messages_received = 0
    first_msg_received = 0.0
    new_msg_received = 0
    last_msg_time = None

    max_starting_in = 0.0
    tx_updates = []

    while messages_received < num_subscribers and not stop_requested:
        try:
            # Poll the socket for incoming messages with a 1-second timeout
            socks = dict(poller.poll(1000))
        except KeyboardInterrupt:
            _handle_interrupt()
            return max_starting_in, tx_updates, first_msg_received, messages_received

        # If some messages were received but no new message comes within WAIT_TIMEOUT, break
        if messages_received > 2 and time.time() - new_msg_received > WAIT_TIMEOUT:
            break

        if alive_socket in socks and socks[alive_socket] == zmq.POLLIN:
            # Record time when a new message is received
            new_msg_received = time.time()
            last_msg_time = new_msg_received

            if messages_received == 0:
                first_msg_received = time.time()

            # Receive the message string from the subscriber
            message = alive_socket.recv_string()
            messages_received += 1

            # Parse and log TX DONE payload: "<HOSTNAME> <applied_phase> <applied_delta>"
            parts = message.split()
            if len(parts) >= 4:
                host, applied_phase, applied_delta, starting_in = (
                    parts[0],
                    parts[1],
                    parts[2],
                    parts[3],
                )
                if messages_received == 1:
                    logger.info("idx   host           phase(rad)    delta(rad)    start(s)")
                logger.info(
                    "%3d/%-3d %-12s %12.6f %12.6f %10.2f",
                    messages_received,
                    num_subscribers,
                    host,
                    float(applied_phase),
                    float(applied_delta),
                    float(starting_in),
                )
                if float(starting_in) > max_starting_in:
                    max_starting_in = float(starting_in)
                tx_updates.append((host, applied_phase, applied_delta))

            else:
                logger.warning("%s (%d/%d)", message, messages_received, num_subscribers)

            # Send response back to the subscriber
            best_phase = best_phase_per_host.get(host, applied_phase)
            alive_socket.send_string(f"{is_stronger} {best_phase}")
    if last_msg_time and first_msg_received and (last_msg_time - first_msg_received) > TIME_DIFF_MSG:
        logger.warning(
            "Time difference first->last message exceeded: %.2fs (threshold %.2fs)",
            last_msg_time - first_msg_received,
            TIME_DIFF_MSG,
        )
    return max_starting_in, tx_updates, first_msg_received, messages_received

def cleanup():
    try:
        sync_socket.close()
        alive_socket.close()
        data_socket.close()
    except Exception:
        pass
    try:
        context.term()
    except Exception:
        pass

try:
    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTSTP, _handle_tstp)
    with open(output_path, "w") as f:
        # Write experiment metadata to the YAML file
        f.write(f"experiment: {unique_id}\n")
        f.write(f"num_subscribers: {num_subscribers}\n")
        f.write("measurments:\n")

        while not stop_requested:
            # Wait for all subscribers to send a message
            logger.info("Waiting for %d subscribers to send a message...", num_subscribers)

            # Start a new measurement entry in the YAML file
            f.write(f"  - meas_id: {meas_id}\n")
            f.write("    active_tiles:\n")

            # Track number of messages received from subscribers
            if not send_sync():
                break
            stronger = False
            prev_power = 0.0
            max_power = 0.0
            best_phase_per_host: dict[str, float] = {}
            f.write("    iterations:\n")

            for i in range(0, 100):
                next_tx_in, tx_updates, first_msg_received, messages_received = wait_till_tx_done(
                    is_stronger=stronger,
                    best_phase_per_host=best_phase_per_host,
                )

                time_to_sleep = (first_msg_received+next_tx_in)- time.time() 
                current_max_power = collect_power(time_to_sleep)

                stronger = current_max_power > prev_power

                max_power = max(max_power, current_max_power)

                logger.info(
                    "\n"
                    "=============================================\n"
                    "Iteration    : %3d\n"
                    "Subscribers  : %3d / %3d\n"
                    "Max power    : %s\n"
                    "Current power: %s\n"
                    "Status       : %s\n"
                    "=============================================",
                    i,
                    messages_received,
                    num_subscribers,
                    _fmt_power(max_power),
                    _fmt_power(current_max_power),
                    "STRONGER" if stronger else "WEAKER",
                )

                if current_max_power >= max_power: # equals as well as if the current > max, than max = current
                    for host, applied_phase, _ in tx_updates:
                        try:
                            best_phase_per_host[host] = float(applied_phase)
                        except ValueError:
                            best_phase_per_host[host] = 0.0

                prev_power = current_max_power

                f.write(f"      - iter: {i}\n")
                f.write(f"        power_pw: {current_max_power}\n")
                f.write("        clients:\n")
                for host, applied_phase, applied_delta in tx_updates:
                    f.write(f"          - host: {host}\n")
                    f.write(f"            applied_phase: {applied_phase}\n")
                    f.write(f"            applied_delta: {applied_delta}\n")
except KeyboardInterrupt:
    stop_requested = True
    logger.info("KeyboardInterrupt received, shutting down...")
finally:
    cleanup()
