from utils.client_com import Client
import signal
import time
import sys
import argparse, shlex
from datetime import datetime, timezone, timedelta
import uhd
import numpy as np
import yaml
import logging
import os
import queue
import threading
from typing import Optional

import utils.client_logger as client_logger
from utils.exit_codes import ExitCode
from utils.constants import *


# =============================================================================
#                           Parse the command line arguments
# =============================================================================
#
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str)

args = parser.parse_args()

logger = client_logger.get_logger()


# =============================================================================
#                           Loading configurations
# =============================================================================
# Loading configuration settings from 'cal-settings.yml' file.
# Defaults are loaded with the optional typing
# =============================================================================

RX_TX_SAME_CHANNEL: Optional[bool] = None
CLOCK_TIMEOUT: Optional[float] = None
INIT_DELAY: Optional[float] = None
RATE: Optional[float] = None
LOOPBACK_TX_GAIN: Optional[float] = None
FREE_TX_GAIN: Optional[float] = None
LOOPBACK_RX_GAIN: Optional[float] = None
REF_RX_GAIN: Optional[float] = None
FREQ: Optional[float] = None
CAPTURE_TIME: Optional[float] = None
TX_TIME: Optional[float] = None
SERVER_IP: Optional[str] = None

BEGIN_TIME = 5.0  # seconds from now to start the USRP

try:
    with open(args.config_file, "r", encoding="utf-8") as file:
        _vars = yaml.safe_load(file)
        globals().update(_vars)
        logger.debug("%s", _vars)
except FileNotFoundError:
    logger.error(
        "Calibration file '%s' not found in the current directory.", "cal-settings.yml"
    )
    sys.exit(ExitCode.CAL_FILE_NOT_FOUND)
except yaml.YAMLError as e:
    logger.error("Error parsing 'cal-settings.yml': %s", e)
    sys.exit(ExitCode.YAML_PARSING_ERROR)
except Exception as e:
    logger.error("Unexpected error while loading calibration settings: %s", e)
    sys.exit(ExitCode.CAL_FILE_UNEXPECTED_ERROR)


# =============================================================================
#                           SYNC HANDLER AND SIGNALS
# =============================================================================
#
# =============================================================================


client = Client(args.config_file)
got_sync = False
meas_id = ""

def handle_sync(command, args):
    logger.debug("Received SYNC command: %s %s", command, args)
    
    global got_sync
    global meas_id
    
    got_sync = True
    meas_id = args[0]


def handle_signal(signum, frame):
    logger.debug("Stopping client...")
    client.stop()


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

#TODO remove all constants here, everyhting needs to be imported from constants.py or config
CLOCK_TIMEOUT = 1000  # 1000mS timeout for external clock locking
REF_RX_CH = FREE_TX_CH = 0
LOOPBACK_RX_CH = LOOPBACK_TX_CH = 1
logger.debug("\nPLL REF → CH0 RX\nCH1 TX → CH1 RX\nCH0 TX →")

def starting_in(usrp, at_time):
    return f"Starting in {delta(usrp, at_time):.2f}s"


def delta(usrp, at_time):
    return at_time - usrp.get_time_now().get_real_secs()


def get_current_time(usrp):
    return usrp.get_time_now().get_real_secs()


def setup_usrp_clock(usrp, clock_src, num_mboards):
    usrp.set_clock_source(clock_src)

    end_time = datetime.now() + timedelta(milliseconds=CLOCK_TIMEOUT)

    logger.debug("Now confirming lock on clock signals...")

    # Lock onto clock signals for all mboards
    for i in range(num_mboards):
        is_locked = usrp.get_mboard_sensor("ref_locked", i)
        while (not is_locked) and (datetime.now() < end_time):
            time.sleep(1e-3)
            is_locked = usrp.get_mboard_sensor("ref_locked", i)
        if not is_locked:
            logger.debug("Unable to confirm clock signal locked on board %d", i)
            return False
        else:
            logger.debug("Clock signals are locked")
    return True


def setup_usrp_pps(usrp, pps):
    """Setup the PPS source"""
    usrp.set_time_source(pps)
    return True


def print_tune_result(tune_res):
    logger.debug(
        "Tune Result:\n    Target RF  Freq: %.6f (MHz)\n Actual RF  Freq: %.6f (MHz)\n Target DSP Freq: %.6f "
        "(MHz)\n "
        "Actual DSP Freq: %.6f (MHz)\n",
        (tune_res.target_rf_freq / 1e6),
        (tune_res.actual_rf_freq / 1e6),
        (tune_res.target_dsp_freq / 1e6),
        (tune_res.actual_dsp_freq / 1e6),
    )


def tune_usrp(usrp, freq, channels, at_time):
    """Synchronously set the device's frequency.
    If a channel is using an internal LO it will be tuned first
    and every other channel will be manually tuned based on the response.
    This is to account for the internal LO channel having an offset in the actual DSP frequency.
    Then all channels are synchronously tuned."""
    treq = uhd.types.TuneRequest(freq)
    usrp.set_command_time(uhd.types.TimeSpec(at_time))
    treq.dsp_freq = 0.0
    treq.target_freq = freq
    treq.rf_freq = freq
    treq.rf_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
    treq.dsp_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
    args = uhd.types.DeviceAddr("mode_n=integer")
    treq.args = args
    rx_freq = freq - 1e3
    rreq = uhd.types.TuneRequest(rx_freq)
    rreq.rf_freq = rx_freq
    rreq.target_freq = rx_freq
    rreq.dsp_freq = 0.0
    rreq.rf_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
    rreq.dsp_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
    rreq.args = uhd.types.DeviceAddr("mode_n=fractional")
    for chan in channels:
        print_tune_result(usrp.set_rx_freq(rreq, chan))
        print_tune_result(usrp.set_tx_freq(treq, chan))
    while not usrp.get_rx_sensor("lo_locked").to_bool():
        logger.debug(".")
        time.sleep(0.01)
    logger.info("RX LO is locked")
    while not usrp.get_tx_sensor("lo_locked").to_bool():
        logger.debug(".")
        time.sleep(0.01)
    logger.info("TX LO is locked")


def setup_usrp(usrp, connect=True):
    rate = RATE
    mcr = 20e6
    assert (
        mcr / rate
    ).is_integer(), f"The masterclock rate {mcr} should be an integer multiple of the sampling rate {rate}"
    # Manual selection of master clock rate may also be required to synchronize multiple B200 units in time.
    usrp.set_master_clock_rate(mcr)
    channels = [0, 1]
    setup_usrp_clock(usrp, "external", usrp.get_num_mboards())
    setup_usrp_pps(usrp, "external")
    # smallest as possible (https://files.ettus.com/manual/page_usrp_b200.html#b200_fe_bw)
    rx_bw = 200e3
    for chan in channels:
        usrp.set_rx_rate(rate, chan)
        usrp.set_tx_rate(rate, chan)
        # NOTE DC offset is enabled
        usrp.set_rx_dc_offset(True, chan)
        usrp.set_rx_bandwidth(rx_bw, chan)
        usrp.set_rx_agc(False, chan)
    # specific settings from loopback/REF PLL
    usrp.set_tx_gain(LOOPBACK_TX_GAIN, LOOPBACK_TX_CH)
    usrp.set_tx_gain(LOOPBACK_TX_GAIN, FREE_TX_CH)

    usrp.set_rx_gain(LOOPBACK_RX_GAIN, LOOPBACK_RX_CH)
    usrp.set_rx_gain(REF_RX_GAIN, REF_RX_CH)
    # streaming arguments
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = channels
    # streamers
    tx_streamer = usrp.get_tx_stream(st_args)
    rx_streamer = usrp.get_rx_stream(st_args)
    # Step1: wait for the last pps time to transition to catch the edge
    # Step2: set the time at the next pps (synchronous for all boards)
    # this is better than set_time_next_pps as we wait till the next PPS to transition and after that we set the time.
    # this ensures that the FPGA has enough time to clock in the new timespec (otherwise it could be too close to a PPS edge)
    logger.info("Waiting for server sync")
    while not got_sync:
        pass

    logger.info("Setting device timestamp to 0...")
    usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))

    usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
    logger.debug("[SYNC] Resetting time.")
    logger.info("RX GAIN PROFILE CH0: %s", usrp.get_rx_gain_names(0))
    logger.info("RX GAIN PROFILE CH1: %s", usrp.get_rx_gain_names(1))
    # we wait 2 seconds to ensure a PPS rising edge occurs and latches the 0.000s value to both USRPs.
    time.sleep(2)
    tune_usrp(usrp, FREQ, channels, at_time=BEGIN_TIME)
    logger.info(
        "USRP has been tuned and setup. (%s)", usrp.get_time_now().get_real_secs()
    )
    return tx_streamer, rx_streamer


def tx_thread(
    usrp, tx_streamer, quit_event, phase=[0, 0], amplitude=[0.8, 0.8], start_time=None
):
    tx_thr = threading.Thread(
        target=tx_ref,
        args=(usrp, tx_streamer, quit_event, phase, amplitude, start_time),
    )

    tx_thr.name = "TX_thread"
    tx_thr.start()

    return tx_thr


def rx_thread(usrp, rx_streamer, quit_event, duration, res, start_time=None):
    _rx_thread = threading.Thread(
        target=rx_ref,
        args=(
            usrp,
            rx_streamer,
            quit_event,
            duration,
            res,
            start_time,
        ),
    )
    _rx_thread.name = "RX_thread"
    _rx_thread.start()
    return _rx_thread


def measure_loopback(
    usrp, tx_streamer, rx_streamer, quit_event, result_queue, at_time=None
):
    # ------------------------------------------------------------
    # Function: measure_loopback
    # Purpose:
    #   This function performs a loopback measurement using a USRP device.
    #   It transmits a known signal on one channel and simultaneously
    #   receives it on another channel (loopback). The result is captured,
    #   stored, and processed later.
    # ------------------------------------------------------------

    logger.debug("########### Measure LOOPBACK ###########")

    # ------------------------------------------------------------
    # 1. Configure transmit signal amplitudes
    # ------------------------------------------------------------
    amplitudes = [0.0, 0.0]              # Initialize amplitude array for both channels
    amplitudes[LOOPBACK_TX_CH] = 0.8     # Enable TX on the selected loopback channel

    # ------------------------------------------------------------
    # 2. Set the transmission start time
    # ------------------------------------------------------------
    start_time = uhd.types.TimeSpec(at_time)
    logger.debug("%s", starting_in(usrp, at_time))

    # ------------------------------------------------------------
    # 3. (Legacy) Access user settings interface for low-level FPGA control
    #    Used to switch the USRP into "loopback mode" by writing to
    #    a register in the user settings interface.
    #    NOTE: This interface is no longer available in UHD 4.x.
    # ------------------------------------------------------------
    user_settings = None
    try:
        user_settings = usrp.get_user_settings_iface(1)
        if user_settings:
            # Read current register value (for debug)
            logger.debug("%s", user_settings.peek32(0))
            # Write a value to activate loopback mode
            user_settings.poke32(0, SWITCH_LOOPBACK_MODE)
            # Read again to verify the register value was updated
            logger.debug("%s", user_settings.peek32(0))
        else:
            logger.error("Cannot write to user settings.")
    except Exception as e:
        logger.error("%s", e)

    # ------------------------------------------------------------
    # 4. Start transmit (TX), metadata, and receive (RX) threads
    # ------------------------------------------------------------
    tx_thr = tx_thread(
        usrp,
        tx_streamer,
        quit_event,
        amplitude=amplitudes,
        phase=[0.0, 0.0],
        start_time=start_time,
    )

    # Thread responsible for handling TX metadata (timestamps, etc.)
    tx_meta_thr = tx_meta_thread(tx_streamer, quit_event)

    # Thread that captures received samples during loopback
    rx_thr = rx_thread(
        usrp,
        rx_streamer,
        quit_event,
        duration=CAPTURE_TIME,
        res=result_queue,
        start_time=start_time,
    )

    # ------------------------------------------------------------
    # 5. Wait for the capture duration plus some safety margin (delta)
    # ------------------------------------------------------------
    time.sleep(CAPTURE_TIME + delta(usrp, at_time))

    # ------------------------------------------------------------
    # 6. Signal all threads to stop and wait for them to finish
    # ------------------------------------------------------------
    quit_event.set()   # Triggers thread termination
    tx_thr.join()
    rx_thr.join()
    tx_meta_thr.join()

    # ------------------------------------------------------------
    # 7. Reset the RF switch control (disable loopback mode)
    # ------------------------------------------------------------
    if user_settings:
        user_settings.poke32(0, SWITCH_RESET_MODE)

    # ------------------------------------------------------------
    # 8. Clear the quit event flag to prepare for the next measurement
    # ------------------------------------------------------------
    quit_event.clear()


def tx(duration, tx_streamer, rate, channels):
    logger.debug("TX START")
    metadata = uhd.types.TXMetadata()

    buffer_samps = tx_streamer.get_max_num_samps()
    samps_to_send = int(rate*duration)

    tx_signal = np.ones((len(channels), buffer_samps), dtype=np.complex64)
    tx_signal *= (
        np.exp(1j * np.random.rand(len(channels), 1) * 2 * np.pi) * 0.8
    )  # 0.8 to not exceed to 1.0 threshold

    logger.debug("Signal sample: %s", tx_signal[:, 0])

    send_samps = 0

    while send_samps < samps_to_send:
        samples = tx_streamer.send(tx_signal, metadata)
        send_samps += samples
    # Send EOB to terminate Tx
    metadata.end_of_burst = True
    tx_streamer.send(np.zeros((len(channels), 1), dtype=np.complex64), metadata)
    logger.debug("TX END")
    # Help the garbage collection
    return send_samps

got_start = False

def handle_tx_start(command, args):
    #TODO update this function
    print("Received tx-start command:", command, args)

    global got_start
    global duration

    got_start = True
    _, _, val_str = args[0].partition("=")
    duration = int(val_str)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # FPGA file path
        fpga_path = os.path.join(script_dir, "usrp_b210_fpga_loopback.bin")

        # Initialize USRP device with custom FPGA image and integer mode
        usrp = uhd.usrp.MultiUSRP(
            "enable_user_regs, " \
            f"fpga={fpga_path}, " \
            "mode_n=integer"
        )
        logger.info("Using Device: %s", usrp.get_pp_string())

        client.on("SYNC", handle_sync)
        client.start()
        logger.debug("Client running...")

        # -------------------------------------------------------------------------
        # STEP 0: Preparations
        # -------------------------------------------------------------------------

        # Set up TX and RX streamers and establish connection
        tx_streamer, rx_streamer = setup_usrp(usrp, connect=True)

        logger.debug("Client hostname: %s", client.hostname)
        logger.debug("Measurement ID: %s", meas_id)
        file_name = f"data_{client.hostname}_{meas_id}.txt"

        try:
            data_file = open(file_name, "a")
        except Exception as e:
            logger.error(e)

        # Event used to control thread termination
        quit_event = threading.Event()

        margin = 5.0                     # Safety margin for timing
        cmd_time = CAPTURE_TIME + margin # Duration for one measurement step
        start_next_cmd = cmd_time        # Timestamp for the next scheduled command

        # Queue to collect measurement results and communicate between threads
        result_queue = queue.Queue()

        # -------------------------------------------------------------------------
        # STEP 0: Read CAL file and Weights
        # -------------------------------------------------------------------------

        weights_file = "tx-weights-benchmark.yml"  # TODO extract from config file?
        phi_offset = 0.0
        amplitude = 0.0
        with open(
            os.path.join(
                os.path.dirname(__file__), weights_file
            ), 
            "r",
            encoding="utf-8",
        ) as weights_yaml:
            try:
                weights_dict = yaml.safe_load(weights_yaml)
                if client.hostname in weights_dict.keys():
                    for c_weights in weights_dict[client.hostname]:
                        # TODO only now ch1 is considered to be updated in later versions!
                        if c_weights["ch"] == 1:
                            phi_offset = c_weights["phase"]
                            amplitude = c_weights["ampl"]
                    logger.debug("Applying phase weight: %s", phi_offset)
                    logger.debug("Applying amplitude weight: %s", amplitude)
                else:
                    logger.error("Weights are not found in %s", weights_file)
                    sys.exit(ExitCode.WEIGHTS_NOT_FOUND)
            except yaml.YAMLError as exc:
                logger.error("Error parsing '%s': %s", weights_file, exc)
                sys.exit(ExitCode.YAML_PARSING_ERROR)

        start_next_cmd += cmd_time + 1.0  # Schedule next command after delay

        # -------------------------------------------------------------------------
        # STEP 2: Perform internal loopback measurement with reference signal
        # -------------------------------------------------------------------------

        logger.debug("UHD version: %s", uhd.get_version_string())

        measure_loopback(
            usrp,
            tx_streamer,
            rx_streamer,
            quit_event,
            result_queue,
            at_time=start_next_cmd,
        )

        # Retrieve loopback phase result
        phi_LB = result_queue.get()

        # Print loopback phase
        logger.info("Phase pilot reference signal in rad: %s", phi_LB)
        logger.info("Phase pilot reference signal in degrees: %s", np.rad2deg(phi_LB))

        start_next_cmd += cmd_time + 2.0  # Schedule next command

        client.on("tx-start", handle_tx_start)
        client.start()
        logger.debug("Client running...")

        # TODO how to stop loop? Keyboard interrupt through Ansible?

        try:
            while client.running:
                if got_start:
                    got_start = False
                    tx(DUR, tx_streamer, rate, [channel])
                    client.send("tx-done")
                else:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass

        client.stop()
        client.join()
        logger.debug("Client terminated.")

    except Exception as e:
        logger.error("%s", e)
