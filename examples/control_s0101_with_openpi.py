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
Example script for controlling S0101 arm with OpenPI π₀.₅ WebSocket Policy.

This script demonstrates how to control an S0101 arm using an external
OpenPI π₀.₅ model running on a WebSocket server.

Example usage:
    python examples/control_s0101_with_openpi.py \
        --robot.port=/dev/tty.usbmodem58760431541 \
        --policy.websocket_url=ws://localhost:8000 \
        --control.single_task="pick up the red block and place it in the container"
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from lerobot.policies.factory import make_policy
from lerobot.policies.pi05_websocket.configuration_pi05_websocket import PI05WebSocketConfig
from lerobot.robots import make_robot_from_config, RobotConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

logger = logging.getLogger(__name__)


@dataclass
class ExampleConfig:
    """Configuration for S0101 control example with OpenPI WebSocket policy."""
    
    # Robot configuration
    robot: RobotConfig = RobotConfig(
        type="so101_follower",
        port="/dev/tty.usbmodem58760431541",
        cameras={
            "wrist": OpenCVCameraConfig(
                type="opencv",
                index_or_path=0,
                width=224,
                height=224,
                fps=30
            ),
            "realsense": OpenCVCameraConfig(
                type="opencv", 
                index_or_path=1,
                width=640,
                height=480,
                fps=30
            )
        }
    )
    
    # Policy configuration
    policy: PI05WebSocketConfig = PI05WebSocketConfig(
        websocket_url="ws://localhost:8000",
        timeout=5.0,
        action_horizon=8
    )
    
    # Control parameters
    control_time_s: float = 30.0
    fps: float = 10.0
    single_task: str = "pick up the red block and place it in the container"
    display_data: bool = True


def main():
    """Main function for S0101 control example with OpenPI WebSocket policy."""
    logging.basicConfig(level=logging.INFO)
    
    # Parse configuration
    cfg = ExampleConfig()
    
    logger.info("Starting S0101 control example with OpenPI WebSocket policy")
    logger.info(pformat(asdict(cfg)))
    
    # Create robot
    robot = make_robot_from_config(cfg.robot)
    
    # Create policy
    policy = make_policy(cfg.policy)
    
    try:
        # Connect to robot
        logger.info("Connecting to robot...")
        robot.connect()
        logger.info("Robot connected successfully")
        
        # Run control loop
        logger.info(f"Starting control loop for {cfg.control_time_s} seconds")
        logger.info(f"Task: {cfg.single_task}")
        logger.info(f"Control frequency: {cfg.fps} Hz")
        
        # Initialize policy
        policy.reset()
        
        start_time = time.perf_counter()
        step_count = 0
        
        while time.perf_counter() - start_time < cfg.control_time_s:
            loop_start = time.perf_counter()
            
            # Get observation from robot
            try:
                observation = robot.get_observation()
                logger.debug(f"Step {step_count}: Got observation with keys: {list(observation.keys())}")
            except Exception as e:
                logger.error(f"Failed to get observation: {e}")
                time.sleep(0.1)
                continue
            
            # Prepare observation for policy
            policy_observation = {}
            for key, value in observation.items():
                if key in robot.cameras:  # Camera data
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
            policy_observation["prompt"] = cfg.single_task
            
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
            
            # Send action to robot
            try:
                sent_action = robot.send_action(action_dict)
                logger.debug(f"Step {step_count}: Sent action: {sent_action}")
            except Exception as e:
                logger.error(f"Failed to send action to robot: {e}")
                continue
            
            step_count += 1
            
            # Maintain control frequency
            loop_time = time.perf_counter() - loop_start
            sleep_time = max(0, 1.0 / cfg.fps - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Log performance
            if step_count % 10 == 0:
                actual_fps = step_count / (time.perf_counter() - start_time)
                logger.info(f"Step {step_count}: Actual FPS: {actual_fps:.2f}, Loop time: {loop_time*1000:.1f}ms")
        
        logger.info(f"Control loop finished after {step_count} steps")
        
    except KeyboardInterrupt:
        logger.info("Control loop interrupted by user")
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
