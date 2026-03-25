# Robot Realtime Control Interfaces

Robots Realtime is a research codebase supporting modular software stacks for realtime control, teleoperation, and policy integration on real-world robot embodiments including bi-manual I2RT YAM arms, Franka Panda, (more to come...).

It provides extensible pythonic infrastructure for low-latency joint command streaming, agent-based policy control, visualization, and integration with inverse kinematics solvers like [pyroki](https://github.com/chungmin99/pyroki) developed by [Chung-Min Kim](https://chungmin99.github.io/)! 

Examples:

<img src="media/yam_realtime.gif" width="500">
<img src="media/franka_realtime2.gif" width="500">
<img src="media/yam_active_leader.gif" width="500">
<!-- ![yam_realtime](media/yam_realtime.gif) -->
<!-- ![franka_realtime](media/franka_realtime.gif) -->
<!-- ![franka_realtime2](media/franka_realtime2.gif) -->

## Installation
Clone the repository and initialize submodules:
```bash
git clone --recurse-submodules https://github.com/uynitsuj/robots_realtime.git
# Or if already cloned without --recurse-submodules, run:
git submodule update --init --recursive
```
Install the main package and I2RT repo for CAN driver interface using uv:
```bash
cd robots_realtime
curl -LsSf https://astral.sh/uv/install.sh | sh
source .venv/bin/activate

uv venv --python 3.11
uv pip install -e .
```

### Working with the `teleop-bridge` submodule
When working with making changes in the submodule `dependencies/teleop-bridge`, make sure changes are made in the teleop-bridge repository first

Get the updated version of the module by running
```bash
git submodule update --remote dependencies/teleop-bridge
```

## Configuration
If using YAM arms, configure YAM arms CAN chain according to instructions from the [I2RT repo](https://github.com/i2rt-robotics/i2rt)

## Launch
Then run the launch entrypoint script with an appropriate robot config file.
For Bimanual YAMS:
```bash
uv run robots_realtime/envs/launch.py --config_path configs/yam/yam_viser_bimanual.yaml
```
For Franka Panda (with default panda gripper):
```bash
uv sync --extra sensors --extra franka_panda
uv run robots_realtime/envs/launch.py --config_path configs/franka/franka_viser_osc.yaml
```
or for robotiq gripper instead of default panda grippers (ensure flange orientation is correct):
```bash
uv run robots_realtime/envs/launch.py --config_path configs/franka/franka_robotiq_viser.yaml
```

## Quest VR Teleoperation

Teleoperate I2RT YAM arms using Meta Quest 3 / 3S controllers via WebXR. Works with both real hardware and MuJoCo simulation. The camera feed (sim or real) can be streamed live into the Quest headset.

### Prerequisites

- Linux workstation with Python 3.11+
- Meta Quest 3 or 3S headset
- USB-C cable connecting the Quest to the workstation
- ADB installed (`sudo apt install adb`)
- For real hardware: CAN interfaces configured for your YAM arm(s) (see [I2RT repo](https://github.com/i2rt-robotics/i2rt))

### 1. Install dependencies

```bash
source .venv/bin/activate
uv pip install vuer
```

### 2. Connect Quest via USB and set up ADB port forwarding

Plug the Quest into your workstation with a USB-C cable. Enable Developer Mode on the Quest if you haven't already (Settings > System > Developer), then verify it's detected:

```bash
adb devices
```

You should see your device listed. If you get `unauthorized`, put on the headset and accept the **"Allow USB debugging"** prompt.

Forward the Vuer WebXR port so the Quest browser can reach your workstation:

```bash
adb reverse tcp:8012 tcp:8012
```

### 3. Launch

#### MuJoCo Simulation (no hardware needed)

```bash
# Bimanual — two arms + red cube pick task
uv run python robots_realtime/envs/launch.py \
    --config-path configs/yam/yam_quest_pick_red_cube_sim.yaml

# Single arm — plain MuJoCo sim
uv run python robots_realtime/envs/launch.py \
    --config-path configs/yam/yam_quest_mujoco_sim.yaml
```

#### Real I2RT YAM Hardware

Make sure your CAN interfaces are up and the arm(s) are powered on before launching.

```bash
# Single arm (left)
uv run python robots_realtime/envs/launch.py \
    --config-path configs/yam/yam_quest_single_arm.yaml

# Bimanual (left + right)
uv run python robots_realtime/envs/launch.py \
    --config-path configs/yam/yam_quest_bimanual.yaml
```

> **Safety note (real hardware):** The real hardware configs default to `max_joint_vel: 0.3` rad/s (lower than sim) for safer operation. Increase gradually once you're comfortable with the mapping. You can also enable `effort_limit` in the config for collision detection — uncomment and set an appropriate threshold for your setup.

To stream a real camera to the Quest headset, uncomment the `sensors.cameras` section and the `stream_camera` / `camera_key` lines in the config. Set `device_path` to your camera device (e.g., `/dev/video4`).

### 4. Open the VR scene on the Quest headset

Put on the Quest headset and open the **Meta Quest Browser**. Navigate to:

```
https://vuer.ai/?ws=ws://localhost:8012
```

> The `adb reverse` tunnel makes `localhost:8012` on the Quest reach your workstation directly over USB. No ngrok or HTTPS tunnel is needed.

When the page loads you will see the Vuer WebXR scene with a blue grid floor and (if camera streaming is enabled) the camera feed in the background.

### 5. Enter VR and start teleoperating

1. Click the **"Enter VR"** button in the browser (VR goggles icon, bottom-right).
2. If prompted, enable **Passthrough** so you can see your physical surroundings overlaid with the VR scene.
3. Pick up your Quest controllers:
   - **Left controller** drives the **left arm**
   - **Right controller** drives the **right arm** (bimanual only)
4. **Squeeze the trigger** and move your hand — the robot arm follows.

### Controls

| Action | Control |
|---|---|
| **Move arm** | Hold **trigger** and move your hand (position is relative to where you first squeezed) |
| **Release / re-anchor** | Release trigger, reposition your hand, squeeze again to resume |
| **Close gripper** | Squeeze **grip** button |
| **Open gripper** | Release grip, or press **B** (right) / **Y** (left) |
| **Reset to home** | Press **A** (right) or **X** (left) — resets ALL arms |
| **Wrist orientation** | While holding trigger, tilt/rotate your wrist — the gripper follows your roll, pitch, and yaw |

### Tips

- **Clutching:** Release the trigger to "pick up" your hand and reposition without moving the robot. Squeeze again to resume from the arm's current position.
- **Smooth motion:** The system uses a critically-damped trajectory filter — move your hand at a natural pace.
- **Reach:** `position_scale` controls hand-to-robot movement ratio. Default is `0.5` — increase for larger reach, decrease for finer control.
- **Tracking quality:** Keep controllers in front of you, roughly chest-to-waist height, for the best Quest tracking.
- **Real hardware first run:** Start with `debug_mapping: true` in the config so you can verify the coordinate mapping in the terminal before committing to large motions.

### Available configs

| Config | Mode | Description |
|---|---|---|
| `yam_quest_single_arm.yaml` | Real hardware | Single YAM arm with Quest VR |
| `yam_quest_bimanual.yaml` | Real hardware | Bimanual YAM arms with Quest VR |
| `yam_quest_pick_red_cube_sim.yaml` | Simulation | Bimanual YAM + red cube pick task (camera streaming on) |
| `yam_quest_mujoco_sim.yaml` | Simulation | Single YAM arm in plain MuJoCo sim (camera streaming on) |
| `yam_webrtc_bimanual_mujoco_sim.yaml` | Simulation | Bimanual YAM + Remote Teleop WebRTC connection | 

### Tunable parameters (in YAML config under `agent:`)

| Parameter | Default | Description |
|---|---|---|
| `position_scale` | `0.5` | Multiplier on hand-to-robot movement. Increase for larger reach, decrease for finer control |
| `smoothing_omega` | `8.0` | Trajectory filter speed. Higher = more responsive but less smooth |
| `smoothing_alpha` | `0.4` | Orientation smoothing (0 = frozen, 1 = instant) |
| `max_joint_vel` | `0.3` (real) / `0.5` (sim) | Max joint velocity in rad/s |
| `danger_zone_margin` | `0.05` | Radians to stay away from hard joint limits |
| `deadzone_m` | `0.004` | Hand tremor rejection threshold in meters |
| `track_orientation` | `true` | Map controller roll/pitch/yaw to gripper orientation |
| `workspace_radius` | `0.42` | Max EE distance from robot base (meters) |
| `effort_limit` | disabled | Motor effort threshold for collision detection (enable for real hardware) |
| `stream_camera` | `false` (real) / `true` (sim) | Stream camera feed to Quest headset |
| `stream_fps` | `30.0` | Camera stream framerate to Quest headset |

## Extending with Custom Agents
To integrate your own controller or policy:

Subclass the base agent interface:
```python
from robots_realtime.agents.agent import Agent

class MyAgent(Agent):
    ...
```
Add your agent to your YAML config so the launcher knows which controller to instantiate.

Examples of agents you might implement:
- Leader arm or VR controller teleoperation
- Learned policy (e.g., Diffusion Policy, ACT, PI0)
- Offline motion-planner + scripted trajectory player

## Linting
If contributing, please use ruff (automatically installed) for linting (https://docs.astral.sh/ruff/tutorial/#getting-started)
```bash
ruff check # lint
ruff check --fix # lint and fix anything fixable
ruff format # code format
```

## Roadmap/Todos

- [ ] Add data logging infrastructure
- [ ] Implement a [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) agent controller
- [ ] Implement a [Physical Intelligence π0](https://www.physicalintelligence.company/blog/pi0) agent controller
- [ ] Implement bimanual arms reset mechanism for teleop
