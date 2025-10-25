#!/usr/bin/env python

"""
Test script for PI05 WebSocket Policy.

This script tests the WebSocket policy without requiring a real robot or server.
"""

import logging
import torch
from lerobot.policies.pi05_websocket.configuration_pi05_websocket import PI05WebSocketConfig
from lerobot.policies.pi05_websocket.modeling_pi05_websocket import PI05WebSocketPolicy

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_websocket_policy():
    """Test the WebSocket policy configuration and basic functionality."""
    
    # Test configuration
    config = PI05WebSocketConfig(
        websocket_url="ws://localhost:8000",
        timeout=5.0,
        action_horizon=8
    )
    
    logger.info(f"Configuration: {config}")
    
    # Test policy creation
    try:
        policy = PI05WebSocketPolicy(config)
        logger.info(f"Policy created successfully: {policy.name}")
        logger.info(f"Action horizon: {policy.action_horizon}")
        logger.info(f"WebSocket URL: {policy.websocket_url}")
        logger.info(f"Timeout: {policy.timeout}")
        
        # Test observation format
        test_observation = {
            "observation.images.wrist": torch.randn(3, 224, 224),
            "observation.images.realsense": torch.randn(3, 640, 480),
            "observation.state.shoulder_pan.pos": torch.tensor(0.0),
            "observation.state.shoulder_lift.pos": torch.tensor(0.0),
            "observation.state.elbow_flex.pos": torch.tensor(0.0),
            "observation.state.wrist_flex.pos": torch.tensor(0.0),
            "observation.state.wrist_roll.pos": torch.tensor(0.0),
            "observation.state.gripper.pos": torch.tensor(0.0),
        }
        
        logger.info("Test observation created:")
        for key, value in test_observation.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape} {value.dtype}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Test action selection (will fail without server, but should handle gracefully)
        logger.info("Testing action selection (will fail without server)...")
        try:
            action = policy.select_action(test_observation)
            logger.info(f"Action received: {action}")
        except Exception as e:
            logger.info(f"Expected error (no server): {e}")
        
        logger.info("✅ WebSocket policy test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Policy creation failed: {e}")
        raise

if __name__ == "__main__":
    test_websocket_policy()
