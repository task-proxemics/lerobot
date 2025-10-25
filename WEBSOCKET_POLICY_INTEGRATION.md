# WebSocket Policy Integration with OpenPI π₀.₅

This document describes the integration of the PI05 WebSocket Policy with the OpenPI WebSocket server for robot control.

## Overview

The PI05 WebSocket Policy allows LeRobot to communicate with an external OpenPI π₀.₅ model running on a WebSocket server instead of loading the model locally. This enables:

- Offloading computation to more powerful servers
- Using cloud-based policy inference
- Integrating with external AI services
- Real-time robot control via WebSocket communication

## Data Format Compatibility

### WebSocket Server Protocol (OpenPI)

The OpenPI WebSocket server expects:

1. **Connection**: Client connects to WebSocket server
2. **Metadata**: Server sends metadata first (using MessagePack)
3. **Observation**: Client sends observation data (using MessagePack)
4. **Action**: Server returns action prediction (using MessagePack)

### Message Format

**Server Response Format:**
```python
{
    "actions": numpy_array,  # (horizon, action_dim) - 6-DOF actions
    "server_timing": {
        "infer_ms": 123.45,
        "prev_total_ms": 456.78
    }
}
```

**Client Observation Format:**
```python
{
    "observation/images/wrist2": numpy_array,     # (224, 224, 3) HWC, uint8
    "observation/images/realsense": numpy_array,  # (480, 640, 3) HWC, uint8
    "observation/state": numpy_array,             # (6,) float32
    "prompt": str                                 # Task description
}
```

## Key Changes Made

### 1. Policy Implementation (`modeling_pi05_websocket.py`)

- **MessagePack Support**: Replaced JSON with MessagePack for data serialization
- **Protocol Compliance**: Implemented OpenPI server protocol (metadata first, then observation)
- **Data Format**: Convert tensors to numpy arrays (OpenPI expects numpy, not lists)
- **Response Handling**: Parse OpenPI response format with `action` field
- **Error Handling**: Graceful fallback to zero actions on communication failure

### 2. Dependencies (`pyproject.toml`)

Added new policy dependency:
```toml
pi05_websocket = ["websockets>=12.0", "msgpack>=1.0.0", "msgpack-numpy>=0.4.0"]
```

### 3. Factory Integration (`factory.py`)

- Added `pi05_websocket` case in `get_policy_class()`
- Added `pi05_websocket` case in `make_policy_config()`
- Added import for `PI05WebSocketConfig`

### 4. Policy Registry (`__init__.py`)

- Added `PI05WebSocketConfig` import

## Usage

### 1. Install Dependencies

```bash
pip install lerobot[pi05_websocket]
# or
pip install websockets msgpack msgpack-numpy
```

### 2. Start OpenPI WebSocket Server

```python
from openpi_client import base_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

# Create your OpenPI policy
policy = YourOpenPIPolicy()

# Start WebSocket server
server = WebsocketPolicyServer(policy, host="0.0.0.0", port=8000)
server.serve_forever()
```

### 3. Use in LeRobot

```python
from lerobot.policies.factory import make_policy
from lerobot.policies.pi05_websocket.configuration_pi05_websocket import PI05WebSocketConfig

# Configure policy
config = PI05WebSocketConfig(
    websocket_url="ws://localhost:8000",
    timeout=5.0,
    action_horizon=8
)

# Create policy
policy = make_policy(config)

# Use in control loop
observation = robot.get_observation()
action = policy.select_action(observation)
robot.send_action(action)
```

## Control Loop Integration

The control loop in `examples/control_s0101_with_openpi.py` works as follows:

1. **Get Observation**: Robot provides joint positions and camera images
2. **Format Data**: Convert images to tensors and normalize to [0,1]
3. **Send to Server**: WebSocket policy sends observation to OpenPI server
4. **Receive Action**: Server returns 6-DOF action vector
5. **Apply Action**: Convert action to robot commands and send to robot
6. **Repeat**: Continue at specified control frequency

## Data Flow

```
S0101 ARM → Joint Positions → Policy Observation
     ↓
Wrist Camera → Image Data → Policy Observation  
     ↓
RealSense Camera → Image Data → Policy Observation
     ↓
WebSocket Server → Action Predictions → Robot Commands
     ↓
S0101 ARM ← Joint Commands ← Action Processing
```

## Error Handling

- **Connection Failures**: Graceful fallback to zero actions
- **Server Errors**: Log errors and continue with zero actions
- **Invalid Responses**: Validate action format and fallback if needed
- **Timeout**: Configurable timeout for WebSocket requests

## Testing

Run the test script to verify the policy works:

```bash
python test_websocket_policy.py
```

This will test:
- Policy configuration
- Observation format creation
- Error handling (without server)
- Basic functionality

## Compatibility

The implementation is compatible with:
- OpenPI π₀.₅ WebSocket server
- S0101 ARM robot
- Intel RealSense cameras
- OpenCV cameras
- LeRobot control loops

## Future Improvements

- Support for action horizons > 1
- Better error recovery
- Connection pooling
- Metrics and monitoring
- Authentication support
