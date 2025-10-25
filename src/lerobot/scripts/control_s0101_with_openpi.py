#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Control script for S0101 arm with OpenPI π₀.₅ WebSocket Policy.

This script demonstrates how to control an S0101 arm using an external
OpenPI π₀.₅ model running on a WebSocket server. It supports both wrist
camera and Intel RealSense camera.

Example usage:
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
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.pi05_websocket.configuration_pi05_websocket import PI05WebSocketConfig
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    so101_follower,
)
from lerobot.utils.control_utils import predict_action, get_safe_torch_device
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data


@dataclass
class ControlConfig:
    """Configuration for S0101 control with OpenPI WebSocket policy."""
    
    # Robot configuration
    robot: RobotConfig
    
    # Policy configuration
    policy: PreTrainedConfig = PI05WebSocketConfig()
    
    # Control parameters
    control_time_s: float = 60.0
    fps: float = 10.0
    single_task: str = "pick up the red block and place it in the container"
    display_data: bool = True
    
    # Safety parameters
    max_relative_target: float = 0.1  # Maximum relative movement per step


def control_loop_with_openpi(
    robot: Robot,
    policy: PreTrainedPolicy,
    config: ControlConfig,
    events: dict = None
):
    """Main control loop for S0101 arm with OpenPI WebSocket policy.
    
    Args:
        robot: S0101 robot instance
        policy: WebSocket policy instance
        config: Control configuration
        events: Event dictionary for early exit
    """
    if events is None:
        events = {"exit_early": False}
    
    logger.info(f"Starting control loop for {config.control_time_s} seconds")
    logger.info(f"Task: {config.single_task}")
    logger.info(f"Control frequency: {config.fps} Hz")
    
    # Initialize policy
    policy.reset()
    
    # Initialize rerun for visualization
    if config.display_data:
        _init_rerun("S0101 OpenPI Control")
        rr.log("task", rr.Text(config.single_task), static=True)
    
    start_time = time.perf_counter()
    step_count = 0
    
    try:
        while time.perf_counter() - start_time < config.control_time_s:
            loop_start = time.perf_counter()
            
            # Check for early exit
            if events["exit_early"]:
                logger.info("Early exit requested")
                break
            
            # Get observation from robot
            try:
                observation = robot.get_observation()
                logger.debug(f"Step {step_count}: Got observation with keys: {list(observation.keys())}")
            except Exception as e:
                logger.error(f"Failed to get observation: {e}")
                time.sleep(0.1)
                continue
            
            # Prepare observation for policy
            # Convert images to proper format and add batch dimension
            policy_observation = {}
            for key, value in observation.items():
                if key in robot.cameras:  # Camera data
                    # Convert to tensor and normalize
                    import torch
                    import numpy as np
                    
                    if isinstance(value, np.ndarray):
                        # Convert to tensor and normalize to [0, 1]
                        img_tensor = torch.from_numpy(value).float() / 255.0
                        # Convert from HWC to CHW format
                        img_tensor = img_tensor.permute(2, 0, 1)
                        
                        # Map camera keys to OpenPI server format
                        if key == "wrist":
                            policy_observation["observation/images/wrist2"] = img_tensor
                        elif key == "realsense":
                            policy_observation["observation/images/realsense"] = img_tensor
                        else:
                            policy_observation[f"observation/images/{key}"] = img_tensor
                    else:
                        # Map camera keys to OpenPI server format
                        if key == "wrist":
                            policy_observation["observation/images/wrist2"] = value
                        elif key == "realsense":
                            policy_observation["observation/images/realsense"] = value
                        else:
                            policy_observation[f"observation/images/{key}"] = value
                else:  # Joint positions
                    policy_observation[f"observation/state"] = value
            
            # Add prompt for OpenPI server
            policy_observation["prompt"] = config.single_task
            
            # Get action from policy
            try:
                action_tensor = policy.select_action(policy_observation)
                logger.debug(f"Step {step_count}: Got action tensor: {action_tensor}")
            except Exception as e:
                logger.error(f"Failed to get action from policy: {e}")
                # Use zero action as fallback
                import torch
                action_tensor = torch.zeros(6, dtype=torch.float32)
            
            # Convert action tensor to robot action format
            action_dict = {}
            joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
            
            for i, joint_name in enumerate(joint_names):
                if i < len(action_tensor):
                    action_dict[f"{joint_name}.pos"] = action_tensor[i].item()
                else:
                    action_dict[f"{joint_name}.pos"] = 0.0
            
            # Apply safety constraints
            if config.max_relative_target is not None:
                # Get current joint positions
                current_positions = robot.bus.sync_read("Present_Position")
                
                # Apply relative movement limits
                for joint_name in joint_names:
                    if joint_name in current_positions:
                        current_pos = current_positions[joint_name]
                        target_pos = action_dict[f"{joint_name}.pos"]
                        
                        # Calculate relative movement
                        relative_movement = target_pos - current_pos
                        
                        # Clamp relative movement
                        if abs(relative_movement) > config.max_relative_target:
                            relative_movement = config.max_relative_target * (1 if relative_movement > 0 else -1)
                            action_dict[f"{joint_name}.pos"] = current_pos + relative_movement
            
            # Send action to robot
            try:
                sent_action = robot.send_action(action_dict)
                logger.debug(f"Step {step_count}: Sent action: {sent_action}")
            except Exception as e:
                logger.error(f"Failed to send action to robot: {e}")
                continue
            
            # Log data for visualization
            if config.display_data:
                log_rerun_data(observation, {"action": action_tensor}, config.single_task)
            
            step_count += 1
            
            # Maintain control frequency
            loop_time = time.perf_counter() - loop_start
            busy_wait(max(0, 1.0 / config.fps - loop_time))
            
            # Log performance
            if step_count % 10 == 0:
                actual_fps = step_count / (time.perf_counter() - start_time)
                logger.info(f"Step {step_count}: Actual FPS: {actual_fps:.2f}, Loop time: {loop_time*1000:.1f}ms")
    
    except KeyboardInterrupt:
        logger.info("Control loop interrupted by user")
    except Exception as e:
        logger.error(f"Control loop error: {e}")
        raise
    finally:
        logger.info(f"Control loop finished after {step_count} steps")


@draccus.wrap()
def main(cfg: ControlConfig):
    """Main function for S0101 control with OpenPI WebSocket policy."""
    init_logging()
    logger.info("Starting S0101 control with OpenPI WebSocket policy")
    logger.info(pformat(asdict(cfg)))
    
    # Create robot
    robot = make_robot_from_config(cfg.robot)
    
    # Create policy
    policy = make_policy(cfg.policy)
    
    # Events for early exit
    events = {"exit_early": False}
    
    try:
        # Connect to robot
        logger.info("Connecting to robot...")
        robot.connect()
        logger.info("Robot connected successfully")
        
        # Run control loop
        control_loop_with_openpi(robot, policy, cfg, events)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
    finally:
        # Cleanup
        if robot.is_connected:
            robot.disconnect()
            logger.info("Robot disconnected")


if __name__ == "__main__":
    main()