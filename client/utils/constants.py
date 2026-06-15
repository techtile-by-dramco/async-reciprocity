"""Shared constants for client components."""

# FPGA user register values for loopback switching
SWITCH_LOOPBACK_MODE = 0x00000006  # binary 110
SWITCH_RESET_MODE = 0x00000000

# ZMQ ports
IQ_PUB_PORT = 50001
SYNC_PORT = 5557
ALIVE_PORT = 5558
TX_MODE_PORT = 5559

__all__ = [
    "SWITCH_LOOPBACK_MODE",
    "SWITCH_RESET_MODE",
    "IQ_PUB_PORT",
    "SYNC_PORT",
    "ALIVE_PORT",
    "TX_MODE_PORT",
]
