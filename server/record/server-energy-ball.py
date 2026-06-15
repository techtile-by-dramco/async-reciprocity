#!/usr/bin/python3
# usage: sync_server.py <delay> <num_subscribers>

# VALUE "num_subscribers" --> IMPORTANT --> The server waits until all subscribers have sent their "alive" or ready message before starting a measurement.

# to kill it: sudo fuser -k 5557/tcp

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

# =============================================================================
#                           Experiment Configuration
# =============================================================================
host = "*"  # Host address to bind to. "*" means all available interfaces.
sync_port = "5557"  # Port used for synchronization messages.
alive_port = "5558"  # Port used for heartbeat/alive messages.
data_port = "5559"  # Port used for data transmission.
# =============================================================================
# =============================================================================

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

if len(sys.argv) > 1:
    delay = int(sys.argv[1])
    num_subscribers = int(sys.argv[2])
else:
    delay = 2
    num_subscribers = 42 


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

# Maximum time to wait for messages before breaking out of the inner loop (10 minutes)
WAIT_TIMEOUT = 10.0

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

CAPTURE_POWER_TIME = 5
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
    logger.info("sending 'SYNC' message in %ss...", delay)
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

    # sleep till TX+1.0
    time.sleep(next_tx_in + 1.0)

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
        max_samples[0],
        median_power,
    )
    return median_power


def wait_till_tx_done(is_stronger: bool):
    # Wait for all subscribers to send a TX DONE MODE message
    logger.info("Waiting for %d subscribers to send a TX DONE Mode ...", num_subscribers)

    # Track number of messages received from subscribers
    messages_received = 0
    first_msg_received = 0.0
    new_msg_received = 0

    max_starting_in = 0.0
    tx_updates = []

    while messages_received < num_subscribers and not stop_requested:
        try:
            # Poll the socket for incoming messages with a 1-second timeout
            socks = dict(poller.poll(1000))
        except KeyboardInterrupt:
            _handle_interrupt()
            return max_starting_in, tx_updates

        # If some messages were received but no new message comes within WAIT_TIMEOUT, break
        if messages_received > 2 and time.time() - new_msg_received > WAIT_TIMEOUT:
            break

        if alive_socket in socks and socks[alive_socket] == zmq.POLLIN:
            # Record time when a new message is received
            new_msg_received = time.time()

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
                logger.info(
                    "%s phase=%s delta=%s starting_in=%s (%d/%d)",
                    host,
                    applied_phase,
                    applied_delta,
                    starting_in,
                    messages_received,
                    num_subscribers,
                )
                if float(starting_in) > max_starting_in:
                    max_starting_in = float(starting_in)
                tx_updates.append((host, applied_phase, applied_delta))

            else:
                logger.warning("%s (%d/%d)", message, messages_received, num_subscribers)

            # Send response back to the subscriber
            alive_socket.send_string(str(is_stronger))
    return max_starting_in, tx_updates, first_msg_received

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
            f.write("    iterations:\n")

            for i in range(0, 100):
                next_tx_in, tx_updates, first_msg_received = wait_till_tx_done(
                    is_stronger=stronger
                )

                time_to_sleep = (first_msg_received+next_tx_in)- time.time() 
                current_max_power = collect_power(time_to_sleep)

                stronger = current_max_power > prev_power

                max_power = max(max_power, current_max_power)

                logger.info(
                    "Iteration %d: max_power=%s now_power=%s prev_power=%s stronger=%s",
                    i,
                    max_power,
                    current_max_power,
                    prev_power,
                    stronger,
                )

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
