# OpenPI π₀.₅ WebSocket Policy

This policy communicates with an external OpenPI π₀.₅ model running on a WebSocket server instead of loading the model locally.

## Installation

The WebSocket policy requires the `websockets` package:

```bash
pip install websockets
```

## Usage

### Basic Example

```python
from lerobot.policies.factory import make_policy
from lerobot.policies.pi05_websocket.configuration_pi05_websocket import PI05WebSocketConfig
from lerobot.robots import make_robot_from_config, RobotConfig

# Configure the WebSocket policy
policy_config = PI05WebSocketConfig(
    websocket_url="ws://localhost:8000",  # Your OpenPI server URL
    timeout=5.0,
    action_horizon=8
)

# Create the policy
policy = make_policy(policy_config)

# Use with robot control
# ... robot setup code ...
```

### Command Line Usage

```bash
python lerobot/scripts/control_s0101_with_openpi.py \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras='{
        "wrist": {"type": "opencv", "index_or_path": 0, "width": 224, "height": 224, "fps": 30},
        "realsense": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}
    }' \
    --policy.type=pi05_websocket \
    --policy.websocket_url=ws://localhost:8000 \
    --control.single_task="pick up the red block and place it in the container" \
    --control.control_time_s=60.0 \
    --control.fps=10.0
```

## Configuration

The WebSocket policy accepts the following configuration parameters:

- `websocket_url`: URL of the WebSocket server running OpenPI π₀.₅ (default: "ws://localhost:8000")
- `timeout`: Timeout for WebSocket requests in seconds (default: 5.0)
- `action_horizon`: Number of actions to predict per inference call (default: 8)
- `device`: Device for tensor operations (not used for WebSocket policy, default: "cpu")

## WebSocket Protocol

The policy sends observations to the server in the following format:

```json
{
    "type": "inference",
    "observation": {
        "observation.images.wrist": [[[0.1, 0.2, ...], ...], ...],
        "observation.images.realsense": [[[0.3, 0.4, ...], ...], ...],
        "observation.state.shoulder_pan.pos": 0.5,
        "observation.state.shoulder_lift.pos": -0.2,
        ...
    }
}
```

The server should respond with:

```json
{
    "success": true,
    "actions": [
        [0.1, -0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, -0.1, 0.4, 0.3, 0.6, 0.5],
        ...
    ]
}
```

## Error Handling

The policy includes robust error handling:

- If the WebSocket connection fails, it returns zero actions
- If the server returns an error, it logs the error and returns zero actions
- If no actions are received, it returns zero actions
- The policy maintains an action queue for smooth execution

## Safety Features

- Maximum relative movement limits can be applied to prevent sudden large movements
- Action queue management ensures smooth execution
- Graceful fallback to zero actions on errors
