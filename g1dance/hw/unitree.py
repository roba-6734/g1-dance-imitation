"""Unitree SDK2 (Python) I/O. Imports unitree_sdk2py lazily so the offline
pipeline and tests never need it.

Interface (unitree_sdk2_python example/g1/high_level/g1_arm7_sdk_dds_example.py):
  * state:   subscribe "rt/lowstate"  (unitree_hg LowState_)
  * command: publish   "rt/arm_sdk"   (unitree_hg LowCmd_), motor_cmd[i].{q,dq,kp,kd,tau}
             for i in 12..28, motor_cmd[29].q = blend weight 0..1, crc set.
Leg entries 0..11 of the LowCmd are never written (left at their zero
defaults: q=dq=kp=kd=tau=0). Whether the firmware ignores them in arm_sdk
mode is an operator verification item (docs/HARDWARE.md).
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ..joints import ARM_SDK_TOPIC, ARM_STATE_TOPIC, N_UPPER, NUM_MOTORS, UPPER_IDS, WEIGHT_INDEX, assert_upper_only
from ..monitor import RobotState


def _sdk():
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    except ImportError as e:  # pragma: no cover
        raise SystemExit("unitree_sdk2py is not installed. See docs/HARDWARE.md (pip install -e unitree_sdk2_python).") from e
    return ChannelFactoryInitialize, ChannelSubscriber, LowState_


_FACTORY_DONE = False


def init_dds(iface: str):
    global _FACTORY_DONE
    ChannelFactoryInitialize, _, _ = _sdk()
    if not _FACTORY_DONE:
        ChannelFactoryInitialize(0, iface)
        _FACTORY_DONE = True


class LowStateReader:
    """Read-only subscriber to rt/lowstate."""

    def __init__(self):
        _, ChannelSubscriber, LowState_ = _sdk()
        self._lock = threading.Lock()
        self._latest: RobotState | None = None
        self.count = 0
        self.stamps: list[float] = []
        self.sub = ChannelSubscriber(ARM_STATE_TOPIC, LowState_)
        self.sub.Init(self._on_msg, 10)

    def _on_msg(self, msg):
        now = time.monotonic()
        q = np.array([msg.motor_state[i].q for i in range(NUM_MOTORS)], float)
        dq = np.array([msg.motor_state[i].dq for i in range(NUM_MOTORS)], float)
        rpy = np.array(msg.imu_state.rpy, float)
        st = RobotState(stamp=now, q=q, dq=dq, imu_rpy=rpy, mode_machine=int(msg.mode_machine))
        with self._lock:
            self._latest = st
            self.count += 1
            if len(self.stamps) < 20000:
                self.stamps.append(now)

    def latest(self) -> RobotState | None:
        with self._lock:
            return self._latest

    def wait_first(self, timeout: float = 3.0) -> RobotState:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            st = self.latest()
            if st is not None:
                return st
            time.sleep(0.01)
        raise TimeoutError(f"no {ARM_STATE_TOPIC} message within {timeout}s (interface / network?)")


class TopicActivityProbe:
    """Read-only subscriber that counts messages on rt/arm_sdk (someone else commanding?)."""

    def __init__(self):
        _, ChannelSubscriber, _ = _sdk()
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        self.count = 0
        self.last_weight = None
        self.sub = ChannelSubscriber(ARM_SDK_TOPIC, LowCmd_)
        self.sub.Init(self._on, 10)

    def _on(self, msg):
        self.count += 1
        self.last_weight = float(msg.motor_cmd[WEIGHT_INDEX].q)


class UnitreeBackend:
    """Live backend. Publishing requires ``arm(confirmed=True)`` from the trial CLI."""
    name = "unitree_arm_sdk"
    data_kind = "physical"

    def __init__(self, iface: str, control_dt: float = 0.02):
        init_dds(iface)
        self.reader = LowStateReader()
        self.reader.wait_first()
        self.dt = control_dt
        self._pub = None
        self._next = time.monotonic()
        self.overruns = 0

    def arm(self, confirmed: bool):
        if not confirmed:
            raise PermissionError("live motor commands need --enable-motion and operator confirmation")
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        self._pub = ChannelPublisher(ARM_SDK_TOPIC, LowCmd_)
        self._pub.Init()
        self._msg = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self._next = time.monotonic()

    def read_state(self) -> RobotState:
        st = self.reader.latest()
        assert st is not None
        return st

    def send(self, q, kp, kd, weight):
        if self._pub is None:
            raise PermissionError("backend not armed; refusing to publish")
        q, kp, kd = (np.asarray(x, float) for x in (q, kp, kd))
        if q.shape != (N_UPPER,) or not np.all(np.isfinite(q)):
            raise ValueError("bad command vector")
        assert_upper_only(UPPER_IDS)
        m = self._msg
        for c, i in enumerate(UPPER_IDS):
            mc = m.motor_cmd[i]
            mc.q, mc.dq, mc.kp, mc.kd, mc.tau = float(q[c]), 0.0, float(kp[c]), float(kd[c]), 0.0
        m.motor_cmd[WEIGHT_INDEX].q = float(np.clip(weight, 0.0, 1.0))
        m.crc = self._crc.Crc(m)
        self._pub.Write(m)

    def wait_tick(self):
        self._next += self.dt
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self.overruns += 1
            self._next = time.monotonic()

    def now(self) -> float:
        return time.monotonic()
