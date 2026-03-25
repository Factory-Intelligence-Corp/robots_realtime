"""WebRTC-based VR controller teleoperation agent for I2RT YAM.

Receives controller pose and button data from a Meta Quest headset via
the ``teleop-bridge`` WebRTC pipeline, converts to robot end-effector
targets, runs IK through PyRoki, and outputs joint commands compatible
with both simulation and real I2RT YAM hardware.

Video streaming back to the headset is handled entirely by the bridge's
aiortc H.264 video tracks — the agent does not manage camera frames.

Requirements:
  - ``teleop-bridge`` package installed with a transport extra
    (e.g. ``uv sync --extra webrtc_teleop --extra webrtc_aiortc``)
  - A running signaling server (e.g. on Fly.io)
  - Quest browser connecting to the signaling server URL

Button mapping (Quest Touch controllers):
  - Trigger: engage teleoperation (controller pose drives the arm)
  - Grip (squeeze): close gripper while held, open on release
  - A button (right) / X button (left): reset BOTH arms to home position
  - B button (right) / Y button (left): open gripper
"""

import asyncio
import importlib
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy
from dm_env.specs import Array

from robots_realtime.agents.agent import Agent
from robots_realtime.agents.teleoperation.quest_vr_agent import (
    _DANGER_ZONE_RAD,
    _GRIPPER_CLOSED,
    _GRIPPER_OPEN,
    _MAX_JOINT_VEL_RAD_PER_S,
    _YAM_JOINT_LIMITS,
    _clamp_to_limits,
    _CriticallyDampedFilter,
    _deadzone,
    _extract_pos_quat_from_col_major,
    _limit_joint_velocity,
    _quat_conj_wxyz,
    _quat_mul_wxyz,
    _slerp_wxyz,
    _solve_ik_teleop,
    _webxr_mat_to_robot_quat,
    _webxr_to_robot_pos,
)
from robots_realtime.utils.portal_utils import remote

logger = logging.getLogger(__name__)

# Maximum age (seconds) of controller data before we consider it stale
# and hold the current joint positions instead of commanding new ones.
_STALE_DATA_TIMEOUT = 0.5


class WebRTCTeleopAgent(Agent):
    """Teleoperation agent driven by Quest controllers via teleop-bridge WebRTC.

    Uses **delta-based control**: when the trigger is first squeezed, the
    controller position and the robot end-effector position are recorded as
    anchors.  Subsequent hand movements are applied as scaled offsets from
    the EE anchor, giving intuitive 1:1 spatial control regardless of where
    the user is standing in the room.

    The teleop-bridge runs in a background daemon thread and exposes
    controller state via a thread-safe ``CtrlStateStore``.  The ``act()``
    method reads the latest snapshot each tick.

    Args:
        bimanual: Drive both arms (left controller -> left arm, right -> right).
        transport: Name of the WebRTC transport backend (e.g. "aiortc", "gstreamer").
            The corresponding ``bridge.<transport>_transport`` module is imported
            dynamically so only the installed transport's dependencies are required.
        signaling_url: URL of the signaling server for SDP exchange.
        signaling_token: Bearer token for signaling server auth.
        ice_servers: List of ICE server dicts (STUN/TURN configuration).
        bridge_config_path: Path to a bridge YAML config file. If provided,
            inline parameters (signaling_url, etc.) override values from file.
        position_scale: Multiplier applied to the controller delta.
        smoothing_omega: Natural frequency for the critically-damped filter.
        smoothing_alpha: SLERP factor for orientation smoothing.
        max_joint_vel: Maximum joint velocity in rad/s.
        danger_zone_margin: Radians to stay away from hard joint limits.
        track_orientation: If True, map controller rotation to EE orientation.
        default_orientation_wxyz: Fixed EE orientation when track_orientation=False.
    """

    def __init__(
        self,
        bimanual: bool = False,
        bimanual_combined_key: Optional[str] = None,
        # Bridge / WebRTC params
        transport: str = "aiortc",
        signaling_url: str = "http://localhost:8000",
        signaling_token: str = "",
        ice_servers: Optional[List[Dict[str, Any]]] = None,
        bridge_config_path: Optional[str] = None,
        camera_devices: Optional[List[int]] = None,
        camera_resolution: Optional[List[int]] = None,
        camera_fps: int = 30,
        max_velocity: Optional[float] = 5.0,
        # Motion / IK params (same as QuestVRAgent)
        position_scale: float = 1.0,
        smoothing_omega: float = 8.0,
        smoothing_alpha: float = 0.4,
        max_joint_vel: float = _MAX_JOINT_VEL_RAD_PER_S,
        danger_zone_margin: float = _DANGER_ZONE_RAD,
        deadzone_m: float = 0.004,
        track_orientation: bool = False,
        default_orientation_wxyz: Optional[List[float]] = None,
        ik_pos_weight: float = 50.0,
        ik_ori_weight: float = 20.0,
        ik_rest_weight: float = 0.1,
        workspace_radius: float = 0.42,
        home_joints: Optional[List[float]] = None,
        home_vel: float = 0.8,
        debug_mapping: bool = False,
        # Effort-based collision detection
        effort_limit: Optional[float] = None,
        effort_obs_key: str = "joint_efforts",
    ) -> None:
        # -- Deferred import of teleop-bridge (optional dependency) ----------
        from bridge.bridge import BridgeConfig, TeleopBridge

        # Dynamically import the transport module to trigger @register_transport
        try:
            importlib.import_module(f"bridge.{transport}_transport")
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"Transport '{transport}' not found. Install the matching extra "
                f"(e.g. `uv sync --extra webrtc_{transport}`) and ensure the "
                f"bridge.{transport}_transport module exists."
            ) from exc

        # -- Store motion params --------------------------------------------
        self.bimanual = bimanual
        self.bimanual_combined_key = bimanual_combined_key
        self.position_scale = position_scale
        self._smoothing_omega = smoothing_omega
        self.smoothing_alpha = smoothing_alpha
        self.max_joint_vel = max_joint_vel
        self.danger_zone_margin = danger_zone_margin
        self._deadzone_m = deadzone_m
        self._track_orientation = track_orientation
        self._workspace_radius = workspace_radius
        self._debug_mapping = debug_mapping
        self._debug_counter = 0
        self._debug_anchor_raw: np.ndarray = np.zeros(3)

        # IK weights as traced JAX arrays
        self._ik_pos_weight = jnp.array(ik_pos_weight, dtype=jnp.float32)
        self._ik_ori_weight = jnp.array(ik_ori_weight, dtype=jnp.float32)
        self._ik_rest_weight = jnp.array(ik_rest_weight, dtype=jnp.float32)

        # Home position
        if home_joints is not None:
            self._home_joints = np.array(home_joints, dtype=np.float64)
        else:
            self._home_joints = np.zeros(6, dtype=np.float64)
        self._home_vel = home_vel
        self._homing = False

        # Effort-based collision detection
        self._effort_limit = effort_limit
        self._effort_obs_key = effort_obs_key
        self._effort_blocked: Dict[str, bool] = {}

        # Default orientation
        if default_orientation_wxyz is not None:
            self._default_wxyz = np.array(default_orientation_wxyz, dtype=np.float64)
        else:
            self._default_wxyz = np.array([-0.5, -0.5, -0.5, -0.5])

        # -- Load URDF and build PyRoki robot for IK + FK -------------------
        current_path = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(
            current_path, "..", "..", "..", "dependencies", "i2rt", "i2rt", "robot_models", "yam", "yam.urdf"
        )
        mesh_dir = os.path.join(
            current_path, "..", "..", "..", "dependencies", "i2rt", "i2rt", "robot_models", "yam", "assets"
        )
        self._urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)
        self._pk_robot = pk.Robot.from_urdf(self._urdf)
        self._target_link = "link_6"
        self._target_link_index = self._pk_robot.links.names.index(self._target_link)
        self._n_joints = 6

        # -- Per-arm state --------------------------------------------------
        sides = ["left", "right"] if bimanual else ["left"]
        self._sides = sides
        self._joints: Dict[str, np.ndarray] = {s: np.zeros(self._n_joints) for s in sides}
        self._gripper: Dict[str, float] = {s: _GRIPPER_OPEN for s in sides}
        self._pos_filter: Dict[str, _CriticallyDampedFilter] = {
            s: _CriticallyDampedFilter(omega=smoothing_omega, dim=3) for s in sides
        }
        self._smoothed_target_wxyz: Dict[str, Optional[np.ndarray]] = {s: None for s in sides}
        self._effort_blocked = {s: False for s in sides}
        self._last_act_time = time.time()

        # Delta-control anchors
        self._anchor_ctrl_pos: Dict[str, Optional[np.ndarray]] = {s: None for s in sides}
        self._anchor_ee_pos: Dict[str, Optional[np.ndarray]] = {s: None for s in sides}
        self._anchor_ctrl_wxyz: Dict[str, Optional[np.ndarray]] = {s: None for s in sides}
        self._anchor_ee_wxyz: Dict[str, Optional[np.ndarray]] = {s: None for s in sides}
        self._trigger_prev: Dict[str, bool] = {s: False for s in sides}

        # -- Warm up JAX / PyRoki JIT --------------------------------------
        self._compute_ee_position(np.zeros(self._n_joints))
        _solve_ik_teleop(
            self._pk_robot,
            jnp.array(self._target_link_index),
            jnp.array(self._default_wxyz, dtype=jnp.float32),
            jnp.array([0.1, 0.0, 0.15], dtype=jnp.float32),
            jnp.zeros(self._n_joints, dtype=jnp.float32),
            self._ik_pos_weight,
            self._ik_ori_weight,
            self._ik_rest_weight,
        )

        # -- Build bridge config --------------------------------------------
        bridge_kwargs: Dict[str, Any] = {}
        if bridge_config_path is not None:
            import yaml

            with open(bridge_config_path) as f:
                bridge_kwargs = yaml.safe_load(f) or {}

        # Inline params override file values
        bridge_kwargs["transport"] = transport
        bridge_kwargs["signaling_url"] = signaling_url
        bridge_kwargs["signaling_token"] = signaling_token
        if ice_servers is not None:
            bridge_kwargs["ice_servers"] = ice_servers
        if max_velocity is not None:
            bridge_kwargs["max_velocity"] = max_velocity
        if camera_devices is not None:
            bridge_kwargs["camera_devices"] = camera_devices
        if camera_resolution is not None:
            bridge_kwargs["camera_resolution"] = camera_resolution
        bridge_kwargs["camera_fps"] = camera_fps

        self._bridge_config = BridgeConfig(**bridge_kwargs)
        self._bridge = TeleopBridge(self._bridge_config)

        # -- Start bridge in background thread ------------------------------
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._bridge_thread = threading.Thread(target=self._run_bridge, daemon=True)
        self._bridge_thread.start()

        logger.info(
            "WebRTCTeleopAgent started (delta control). "
            "Bridge connecting to signaling at %s. "
            "Squeeze trigger to engage, move hand to control EE.",
            signaling_url,
        )

    # ------------------------------------------------------------------ #
    # Bridge background thread
    # ------------------------------------------------------------------ #

    def _run_bridge(self) -> None:
        """Run the teleop-bridge event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._bridge.start())
            # Kick off the status heartbeat as a concurrent task
            self._loop.create_task(self._bridge.run_status_loop())
            self._loop.run_forever()
        except Exception:
            logger.exception("Bridge event loop crashed")
        finally:
            self._loop.close()

    # ------------------------------------------------------------------ #
    # Forward kinematics helper
    # ------------------------------------------------------------------ #

    def _compute_ee_pose(self, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position [x,y,z], quaternion wxyz) for the given joints."""
        Ts = self._pk_robot.forward_kinematics(jnp.array(joints, dtype=jnp.float32))
        ee_pose = jaxlie.SE3(Ts[self._target_link_index])
        pos = np.array(ee_pose.translation())
        wxyz = np.array(ee_pose.rotation().wxyz)
        return pos, wxyz

    def _compute_ee_position(self, joints: np.ndarray) -> np.ndarray:
        """Return the end-effector position [x, y, z] for the given joints."""
        return self._compute_ee_pose(joints)[0]

    # ------------------------------------------------------------------ #
    # Effort-based collision detection
    # ------------------------------------------------------------------ #

    def _check_effort(self, obs: Dict[str, Any], side: str) -> bool:
        """Return True if motion should be blocked due to excessive effort."""
        if self._effort_limit is None:
            return False

        arm_obs = obs.get(side, {})
        if not isinstance(arm_obs, dict):
            return False
        efforts = arm_obs.get(self._effort_obs_key)
        if efforts is None:
            return False

        efforts = np.asarray(efforts)
        max_effort = np.max(np.abs(efforts))

        if self._effort_blocked[side]:
            if max_effort < self._effort_limit * 0.8:
                self._effort_blocked[side] = False
                logger.info("%s arm: effort dropped below threshold, resuming", side)
        elif max_effort > self._effort_limit:
            self._effort_blocked[side] = True
            logger.warning(
                "%s arm: effort %.2f exceeds limit %.2f — freezing motion",
                side,
                max_effort,
                self._effort_limit,
            )

        return self._effort_blocked[side]

    # ------------------------------------------------------------------ #
    # Agent protocol
    # ------------------------------------------------------------------ #

    def act(self, obs: Dict[str, Any]) -> Any:
        now = time.time()
        dt = now - self._last_act_time
        self._last_act_time = now

        # Read controller state from the bridge (already thread-safe)
        ctrl = self._bridge.ctrl_state.read()

        # If no data yet or data is stale, hold current position
        if not ctrl or self._bridge.ctrl_state.last_update_age > _STALE_DATA_TIMEOUT:
            action: Dict[str, Dict[str, np.ndarray]] = {}
            for side in self._sides:
                action[side] = {
                    "pos": np.concatenate(
                        [
                            np.flip(self._joints[side]),
                            [self._gripper[side]],
                        ]
                    ).astype(np.float32),
                }
            return self._maybe_combine_bimanual(action)

        action = {}

        # Check if A (right) or X (left) was pressed → home ALL arms
        for side in self._sides:
            state_key = f"{side}State"
            buttons = ctrl.get(state_key, {})
            if buttons.get("aButton", False) or buttons.get("xButton", False):
                if not self._homing:
                    self._homing = True
                    for s in self._sides:
                        self._anchor_ctrl_pos[s] = None
                        self._anchor_ee_pos[s] = None
                        self._anchor_ctrl_wxyz[s] = None
                        self._anchor_ee_wxyz[s] = None
                        self._smoothed_target_wxyz[s] = None
                    logger.info("Home reset triggered — returning all arms to home position")
                break

        # If homing, smoothly move all arms toward home
        if self._homing:
            all_home = True
            for side in self._sides:
                homed = _limit_joint_velocity(
                    self._home_joints,
                    self._joints[side],
                    dt,
                    self._home_vel,
                )
                self._joints[side] = homed
                if np.max(np.abs(homed - self._home_joints)) > 0.01:
                    all_home = False

                state_key = f"{side}State"
                buttons = ctrl.get(state_key, {})
                grip_pressed = bool(buttons.get("squeeze", False) or buttons.get("grip", False))
                self._gripper[side] = _GRIPPER_CLOSED if grip_pressed else _GRIPPER_OPEN

                action[side] = {
                    "pos": np.concatenate(
                        [
                            np.flip(self._joints[side]),
                            [self._gripper[side]],
                        ]
                    ).astype(np.float32),
                }

            if all_home:
                self._homing = False
                logger.info("All arms reached home position")

            return self._maybe_combine_bimanual(action)

        # -- Main teleoperation loop ----------------------------------------
        for side in self._sides:
            ctrl_key = side
            state_key = f"{side}State"

            mat16 = ctrl.get(ctrl_key)
            buttons = ctrl.get(state_key, {})

            trigger_pressed = bool(buttons.get("trigger", False))
            trigger_just_pressed = trigger_pressed and not self._trigger_prev[side]
            trigger_just_released = not trigger_pressed and self._trigger_prev[side]
            self._trigger_prev[side] = trigger_pressed

            if mat16 is not None and len(mat16) == 16:
                raw_pos, _ = _extract_pos_quat_from_col_major(mat16)
                ctrl_pos_robot = _webxr_to_robot_pos(raw_pos)
                ctrl_wxyz_robot = _webxr_mat_to_robot_quat(mat16)

                if trigger_just_pressed:
                    self._anchor_ctrl_pos[side] = ctrl_pos_robot.copy()
                    ee_pos, ee_wxyz = self._compute_ee_pose(self._joints[side])
                    self._anchor_ee_pos[side] = ee_pos
                    self._pos_filter[side].reset(ee_pos)
                    if self._track_orientation:
                        self._anchor_ctrl_wxyz[side] = ctrl_wxyz_robot.copy()
                        self._anchor_ee_wxyz[side] = ee_wxyz.copy()
                        self._smoothed_target_wxyz[side] = ee_wxyz.copy()
                    logger.info(
                        "%s trigger pressed — anchor EE at %s",
                        side,
                        self._anchor_ee_pos[side],
                    )
                    if self._debug_mapping:
                        print(f"\n[DEBUG] {side} ANCHOR SET", flush=True)
                        print(
                            f"  WebXR raw pos : x={raw_pos[0]:+.4f}  y={raw_pos[1]:+.4f}  z={raw_pos[2]:+.4f}",
                            flush=True,
                        )
                        print(
                            f"  Robot frame   : x={ctrl_pos_robot[0]:+.4f}  y={ctrl_pos_robot[1]:+.4f}  z={ctrl_pos_robot[2]:+.4f}",
                            flush=True,
                        )
                        print(f"  EE position   : {ee_pos}", flush=True)
                        print(f"  Current joints: {self._joints[side]}", flush=True)
                        self._debug_anchor_raw = raw_pos.copy()

                if trigger_pressed and self._anchor_ctrl_pos[side] is not None:
                    if self._check_effort(obs, side):
                        pass  # Effort exceeded — hold current joints
                    else:
                        raw_delta = (ctrl_pos_robot - self._anchor_ctrl_pos[side]) * self.position_scale
                        delta = _deadzone(raw_delta, self._deadzone_m)
                        target_pos = self._anchor_ee_pos[side] + delta

                        # Clamp to reachable workspace
                        if self._workspace_radius is not None:
                            dist = np.linalg.norm(target_pos)
                            if dist > self._workspace_radius:
                                target_pos = target_pos * (self._workspace_radius / dist)

                        if self._debug_mapping:
                            self._debug_counter += 1
                            if self._debug_counter % 50 == 0:
                                webxr_delta = raw_pos - self._debug_anchor_raw
                                print(f"\n[DEBUG] {side} MAPPING (every 1s)", flush=True)
                                print(
                                    f"  WebXR delta (raw): x={webxr_delta[0]:+.4f}  y={webxr_delta[1]:+.4f}  z={webxr_delta[2]:+.4f}",
                                    flush=True,
                                )
                                print(
                                    f"  Robot delta      : x={delta[0]:+.4f}  y={delta[1]:+.4f}  z={delta[2]:+.4f}",
                                    flush=True,
                                )
                                print(f"  IK target pos    : {target_pos}", flush=True)
                                ee_now = self._compute_ee_position(self._joints[side])
                                print(f"  Actual EE pos    : {ee_now}", flush=True)
                                print(f"  Joints (rad)     : {np.round(self._joints[side], 3)}", flush=True)

                        ik_pos = self._pos_filter[side].step(target_pos, dt)
                        ik_wxyz = self._default_wxyz

                        if self._track_orientation and self._anchor_ctrl_wxyz[side] is not None:
                            delta_q = _quat_mul_wxyz(
                                ctrl_wxyz_robot,
                                _quat_conj_wxyz(self._anchor_ctrl_wxyz[side]),
                            )
                            delta_q = np.array(delta_q)
                            target_wxyz = _quat_mul_wxyz(delta_q, self._anchor_ee_wxyz[side])
                            target_wxyz /= np.linalg.norm(target_wxyz)

                            self._smoothed_target_wxyz[side] = _slerp_wxyz(
                                self._smoothed_target_wxyz[side],
                                target_wxyz,
                                self.smoothing_alpha,
                            )
                            ik_wxyz = self._smoothed_target_wxyz[side]

                            if self._debug_mapping and self._debug_counter % 50 == 0:
                                _, ee_wxyz_now = self._compute_ee_pose(self._joints[side])
                                print(f"  Target ori (wxyz): {np.round(ik_wxyz, 3)}", flush=True)
                                print(f"  Actual ori (wxyz): {np.round(ee_wxyz_now, 3)}", flush=True)

                        try:
                            raw_joints = np.array(
                                _solve_ik_teleop(
                                    self._pk_robot,
                                    jnp.array(self._target_link_index),
                                    jnp.array(ik_wxyz, dtype=jnp.float32),
                                    jnp.array(ik_pos, dtype=jnp.float32),
                                    jnp.array(self._joints[side], dtype=jnp.float32),
                                    self._ik_pos_weight,
                                    self._ik_ori_weight,
                                    self._ik_rest_weight,
                                )
                            )
                            raw_joints = _clamp_to_limits(
                                raw_joints,
                                _YAM_JOINT_LIMITS,
                                self.danger_zone_margin,
                            )
                            raw_joints = _limit_joint_velocity(
                                raw_joints,
                                self._joints[side],
                                dt,
                                self.max_joint_vel,
                            )
                            self._joints[side] = raw_joints
                        except Exception:
                            logger.warning(
                                "IK solve failed for %s arm, holding previous joints",
                                side,
                            )

            if trigger_just_released:
                self._anchor_ctrl_pos[side] = None
                self._anchor_ee_pos[side] = None
                self._anchor_ctrl_wxyz[side] = None
                self._anchor_ee_wxyz[side] = None
                self._smoothed_target_wxyz[side] = None

            # Gripper
            grip_pressed = bool(buttons.get("squeeze", False) or buttons.get("grip", False))
            b_pressed = buttons.get("bButton", False)
            if grip_pressed:
                self._gripper[side] = _GRIPPER_CLOSED
            elif b_pressed or not grip_pressed:
                self._gripper[side] = _GRIPPER_OPEN

            action[side] = {
                "pos": np.concatenate(
                    [
                        np.flip(self._joints[side]),
                        [self._gripper[side]],
                    ]
                ).astype(np.float32),
            }

        return self._maybe_combine_bimanual(action)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _maybe_combine_bimanual(
        self,
        action: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Merge left+right into a single 14-DOF vector if in combined mode."""
        if self.bimanual and self.bimanual_combined_key is not None:
            left_pos = action.get("left", {}).get("pos", np.zeros(7, dtype=np.float32))
            right_pos = action.get("right", {}).get("pos", np.zeros(7, dtype=np.float32))
            return {
                self.bimanual_combined_key: {
                    "pos": np.concatenate([left_pos, right_pos]),
                }
            }
        return action

    @remote(serialization_needed=True)
    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        if self.bimanual and self.bimanual_combined_key is not None:
            return {
                self.bimanual_combined_key: {"pos": Array(shape=(14,), dtype=np.float32)},
            }
        spec: Dict[str, Dict[str, Array]] = {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)},
        }
        if self.bimanual:
            spec["right"] = {"pos": Array(shape=(7,), dtype=np.float32)}
        return spec

    def close(self) -> None:
        """Shut down the bridge and its event loop."""
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._bridge.stop(), self._loop)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._bridge_thread.is_alive():
            self._bridge_thread.join(timeout=5.0)
        logger.info("WebRTCTeleopAgent shutting down.")
