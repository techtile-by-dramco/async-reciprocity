import logging
import re
import socket
import threading
import time
import yaml
import zmq

# from client_logger import get_logger
from client_logger import *


class Client:
    def __init__(self, config_path="client_config.yaml"):
        self.logger = get_logger(__name__, level=logging.DEBUG)

        # Load YAML config
        self.logger.debug("Loading client config from %s", config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            experiment_settings = yaml.safe_load(f)

        server_settings = experiment_settings.get("server", {})
        host = server_settings.get("host", "")
        messaging_port = server_settings.get("messaging_port", "")
        sync_port = server_settings.get("sync_port", "")
        self.messaging_endpoint = f"tcp://{host}:{messaging_port}"
        self.sync_endpoint = f"tcp://{host}:{sync_port}"
        self.heartbeat_interval = experiment_settings.get("heartbeat_interval", 5)

        # Derive ID from hostname
        _hostname = socket.gethostname()
        m = re.match(r"rpi-(.+)", _hostname, re.IGNORECASE)
        # TODO stop because no valid hostname, we cannot continue
        if not m:
            raise ValueError(
                f"Hostname '{_hostname}' does not match expected pattern 'rpi-<ID>'"
            )
        self.hostname = _hostname[4:]
        self.client_id = m.group(1).encode()
        self.logger.debug("Client ID derived from hostname: %s", self.client_id)

        # State
        self.running = False
        self.thread = None

        # Setup ZMQ
        self.context = zmq.Context()
        self.messaging = self.context.socket(zmq.DEALER)
        self.messaging.setsockopt(zmq.IDENTITY, self.client_id)

        self.sync = self.context.socket(zmq.SUB)
        self.sync.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all topics

        # Robust reconnection handling
        self.messaging.setsockopt(zmq.RECONNECT_IVL, 1000)  # retry every 1s
        self.messaging.setsockopt(zmq.RECONNECT_IVL_MAX, 5000)  # up to 5s backoff
        self.messaging.setsockopt(zmq.HEARTBEAT_IVL, 3000)  # client heartbeats to server
        self.messaging.setsockopt(zmq.HEARTBEAT_TIMEOUT, 10000)
        self.messaging.setsockopt(zmq.HEARTBEAT_TTL, 30000)

        # Event handling
        self.callbacks = {}

        self.logger.debug(
            "Initialized client with messaging=%s, sync=%s, heartbeat=%ss",
            self.messaging_endpoint,
            self.sync_endpoint,
            self.heartbeat_interval,
        )

    def start(self):
        if self.running:
            self.logger.debug("Client already running; start() ignored")
            return
        self.running = True
        self.logger.debug("Client start requested")

        # Connect (non-blocking even if server is DOWN)
        self.logger.debug("Connecting messaging socket to %s", self.messaging_endpoint)
        self.messaging.connect(self.messaging_endpoint)
        self.logger.debug("Connecting sync socket to %s", self.sync_endpoint)
        self.sync.connect(self.sync_endpoint)

        # Launch background thread
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.logger.debug("Client background thread started")

    def stop(self):
        if not self.running:
            self.logger.debug("Client already stopped; stop() ignored")
            return
        self.running = False
        self.logger.debug("Client stop requested")

        try:
            self.messaging.close(0)
            self.sync.close(0)
        except Exception as exc:
            self.logger.debug("Socket close during stop raised: %s", exc)
        try:
            self.context.term()
        except Exception as exc:
            self.logger.debug("Context term during stop raised: %s", exc)

    def join(self):
        if self.thread:
            self.thread.join()
            self.logger.debug("Client background thread joined")

    def on(self, command, func):
        """Register a callback for a given server command."""
        self.callbacks[command] = func
        self.logger.debug("Registered callback for command '%s'", command)

    def send(self, msg_type, *payload_frames):
        """
        Send a message to the server.

        Parameters
        ----------
        msg_type : str
            Message type string (e.g., 'heartbeat', 'request', etc.).
        payload_frames : optional list of bytes or str
            Additional frames to send after the message type.
        """

        frames = [msg_type.encode()]
        for frame in payload_frames:
            if isinstance(frame, str):
                frame = frame.encode()
            frames.append(frame)

        self.messaging.send_multipart(frames)
        self.logger.debug("Sent message type '%s' with %d payload frames", msg_type, len(payload_frames))

    def _run(self):
        self.logger.debug("Client event loop starting")
        poller = zmq.Poller()
        poller.register(self.messaging, zmq.POLLIN)
        poller.register(self.sync, zmq.POLLIN)

        last_heartbeat = 0

        while self.running:
            now = time.time()

            # Send heartbeat (non-blocking)
            if now - last_heartbeat >= self.heartbeat_interval:
                try:
                    self.messaging.send_multipart([b"heartbeat", b"alive"], zmq.NOBLOCK)
                    self.logger.debug("Heartbeat sent")
                except zmq.Again:
                    self.logger.debug("Heartbeat send would block; server likely down")
                last_heartbeat = now

            # Poll server messages
            try:
                events = dict(poller.poll(timeout=100))
            except zmq.error.ZMQError as exc:
                self.logger.debug("Poller error, stopping: %s", exc)
                break

            if self.messaging in events:
                try:
                    frames = self.messaging.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    continue
                except zmq.ZMQError as exc:
                    self.logger.debug("Messaging recv error: %s", exc)
                    break

                self.logger.debug("Received messaging frames: %s", frames)
                self._handle_server_message(frames)
                
            if self.sync in events:
                try:
                    frames = self.sync.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    continue
                except zmq.ZMQError as exc:
                    self.logger.debug("Sync recv error: %s", exc)
                    break

                self.logger.debug("Received sync frames: %s", frames)
                self._handle_server_message(frames)

        # Cleanup
        try:
            self.messaging.close(0)
            self.sync.close(0)
        except Exception as exc:
            self.logger.debug("Socket cleanup raised: %s", exc)
        self.logger.debug("Client event loop exited")

    def _handle_server_message(self, frames):
        if not frames:
            self.logger.debug("Empty frame list received; ignoring")
            return

        command = frames[0].decode()
        args = [f.decode() for f in frames[1:]]

        # If a callback exists, call it
        if command in self.callbacks:
            try:
                self.callbacks[command](command, args)
            except Exception as e:
                self.logger.error("Callback error for %s: %s", command, e)
            return

        # Default built-in handlers
        if command == "ping":
            try:
                self.messaging.send_multipart([b"pong", b"ok"], zmq.NOBLOCK)
                self.logger.debug("Responded to ping with pong")
            except zmq.Again:
                self.logger.debug("Failed to respond to ping; send would block")
        else:
            try:
                self.messaging.send_multipart([b"error", b"unknown_command"], zmq.NOBLOCK)
                self.logger.debug("Sent unknown_command error for %s", command)
            except zmq.Again:
                self.logger.debug("Failed to send unknown_command; send would block")
