"""Non-blocking local IPC for sharing PiPER joint feedback.

The process that already owns the PiPER SDK publishes its cached feedback over
an abstract Unix datagram endpoint.  Perception processes receive the feedback
without opening another CAN or socketcand connection.
"""

from __future__ import annotations

import math
import os
import socket
import struct
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union


DEFAULT_JOINT_FEEDBACK_ENDPOINT = "@piper_joint_feedback_v1"

_MAGIC = b"PJF1"
_PACKET = struct.Struct("<4sIQdd6d")
def _unix_address(endpoint: str) -> Union[str, bytes]:
    if not endpoint:
        raise ValueError("joint-feedback endpoint must not be empty")
    if endpoint.startswith("@"):
        return b"\0" + endpoint[1:].encode("utf-8")
    return endpoint


@dataclass(frozen=True)
class JointFeedbackPacket:
    publisher_pid: int
    sequence: int
    feedback_timestamp_s: float
    publisher_monotonic_s: float
    joint_degrees: Tuple[float, float, float, float, float, float]


def encode_joint_feedback(
    *,
    publisher_pid: int,
    sequence: int,
    feedback_timestamp_s: float,
    publisher_monotonic_s: float,
    joint_degrees: Iterable[float],
) -> bytes:
    joints = tuple(float(value) for value in joint_degrees)
    if len(joints) != 6:
        raise ValueError("PiPER feedback must contain exactly six joints")
    numeric_values = (feedback_timestamp_s, publisher_monotonic_s, *joints)
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("PiPER feedback contains a non-finite value")
    return _PACKET.pack(
        _MAGIC,
        int(publisher_pid),
        int(sequence),
        float(feedback_timestamp_s),
        float(publisher_monotonic_s),
        *joints,
    )


def decode_joint_feedback(payload: bytes) -> JointFeedbackPacket:
    if len(payload) != _PACKET.size:
        raise ValueError(
            f"invalid PiPER feedback packet size: {len(payload)} != {_PACKET.size}"
        )
    unpacked = _PACKET.unpack(payload)
    if unpacked[0] != _MAGIC:
        raise ValueError("invalid PiPER feedback packet magic")
    packet = JointFeedbackPacket(
        publisher_pid=int(unpacked[1]),
        sequence=int(unpacked[2]),
        feedback_timestamp_s=float(unpacked[3]),
        publisher_monotonic_s=float(unpacked[4]),
        joint_degrees=tuple(float(value) for value in unpacked[5:]),
    )
    numeric_values = (
        packet.feedback_timestamp_s,
        packet.publisher_monotonic_s,
        *packet.joint_degrees,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("PiPER feedback packet contains a non-finite value")
    return packet


class PiperJointFeedbackPublisher:
    """Best-effort publisher that can never block the arm-control loop."""

    def __init__(self, endpoint: str = DEFAULT_JOINT_FEEDBACK_ENDPOINT) -> None:
        self.endpoint = endpoint
        self._address = _unix_address(endpoint)
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._sequence = 0
        self.last_error: Optional[Exception] = None

    def publish(
        self,
        feedback_timestamp_s: float,
        joint_degrees: Iterable[float],
    ) -> bool:
        self._sequence += 1
        try:
            payload = encode_joint_feedback(
                publisher_pid=os.getpid(),
                sequence=self._sequence,
                feedback_timestamp_s=feedback_timestamp_s,
                publisher_monotonic_s=time.monotonic(),
                joint_degrees=joint_degrees,
            )
            self._socket.sendto(payload, self._address)
        except (OSError, ValueError, OverflowError, struct.error) as exc:
            self.last_error = exc
            return False
        self.last_error = None
        return True

    def close(self) -> None:
        self._socket.close()


class PiperJointFeedbackReceiver:
    """Single-consumer receiver for the local PiPER feedback stream."""

    def __init__(self, endpoint: str = DEFAULT_JOINT_FEEDBACK_ENDPOINT) -> None:
        self.endpoint = endpoint
        self._address = _unix_address(endpoint)
        self._socket: Optional[socket.socket] = None

    def start(self) -> None:
        if self._socket is not None:
            raise RuntimeError("PiPER feedback receiver is already started")
        receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            receiver.bind(self._address)
            receiver.settimeout(0.1)
        except BaseException:
            receiver.close()
            raise
        self._socket = receiver

    def receive(self) -> Optional[JointFeedbackPacket]:
        receiver = self._socket
        if receiver is None:
            raise RuntimeError("PiPER feedback receiver is not started")
        try:
            payload = receiver.recv(_PACKET.size + 1)
        except socket.timeout:
            return None
        return decode_joint_feedback(payload)

    def close(self) -> None:
        receiver = self._socket
        self._socket = None
        if receiver is not None:
            receiver.close()


__all__ = [
    "DEFAULT_JOINT_FEEDBACK_ENDPOINT",
    "JointFeedbackPacket",
    "PiperJointFeedbackPublisher",
    "PiperJointFeedbackReceiver",
    "decode_joint_feedback",
    "encode_joint_feedback",
]
