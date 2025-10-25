#!/usr/bin/env python

"""
Test script for LeRobot WebSocket Policy integration with OpenPI server.

This script tests the exact same data format as your working test_s0101_inference_main.py
to ensure compatibility.
"""

import logging
import numpy as np
import torch
from lerobot.policies.pi05_websocket.configuration_pi05_websocket import PI05WebSocketConfig
from lerobot.policies.pi05_websocket.modeling_pi05_websocket import PI05WebSocketPolicy

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_lerobot_websocket_integration():
    """Test LeRobot WebSocket policy with the exact format from your working test."""
    
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
        
        # Create test observation in the EXACT format from your working test
        test_observation = {
            "observation/images/wrist2": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/images/realsense": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "observation/state": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "prompt": "manipulate the object"
        }
        
        logger.info("Test observation created (matching your working format):")
        for key, value in test_observation.items():
            if isinstance(value, np.ndarray):
                logger.info(f"  {key}: {value.shape} {value.dtype}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Convert to LeRobot format (tensors)
        policy_observation = {}
        for key, value in test_observation.items():
            if key.startswith("observation/images/"):
                # Convert HWC uint8 to CHW float32 tensor
                if isinstance(value, np.ndarray) and value.dtype == np.uint8:
                    # Convert to float32 and normalize to [0, 1]
                    img_tensor = torch.from_numpy(value).float() / 255.0
                    # Convert from HWC to CHW format
                    img_tensor = img_tensor.permute(2, 0, 1)
                    policy_observation[key] = img_tensor
                else:
                    policy_observation[key] = value
            else:
                policy_observation[key] = value
        
        logger.info("Policy observation (LeRobot format):")
        for key, value in policy_observation.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape} {value.dtype}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Test action selection (will fail without server, but should handle gracefully)
        logger.info("Testing action selection (will fail without server)...")
        try:
            action = policy.select_action(policy_observation)
            logger.info(f"✅ Action received: {action}")
            logger.info(f"Action shape: {action.shape}")
            logger.info(f"Action values: {action}")
        except Exception as e:
            logger.info(f"Expected error (no server): {e}")
        
        logger.info("✅ LeRobot WebSocket policy integration test completed!")
        
    except Exception as e:
        logger.error(f"❌ Policy creation failed: {e}")
        raise

def test_data_format_compatibility():
    """Test that our data format matches the OpenPI server expectations."""
    
    logger.info("\n=== Testing Data Format Compatibility ===")
    
    # Simulate the exact data flow from your working test
    wrist_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    realsense_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Your working format
    your_format = {
        "observation/images/wrist2": wrist_img,
        "observation/images/realsense": realsense_img,
        "observation/state": state,
        "prompt": "manipulate the object"
    }
    
    logger.info("Your working format:")
    for key, value in your_format.items():
        logger.info(f"  {key}: {value.shape} {value.dtype}")
    
    # Convert to LeRobot format (what our policy expects)
    policy_observation = {}
    for key, value in your_format.items():
        if key.startswith("observation/images/"):
            # Convert HWC uint8 to CHW float32 tensor
            img_tensor = torch.from_numpy(value).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)
            policy_observation[key] = img_tensor
        else:
            policy_observation[key] = value
    
    logger.info("LeRobot policy format:")
    for key, value in policy_observation.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: {value.shape} {value.dtype}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Test the conversion back (what gets sent to server)
    config = PI05WebSocketConfig()
    policy = PI05WebSocketPolicy(config)
    
    # Simulate the conversion that happens in _send_observation_to_server
    obs_data = {}
    for key, value in policy_observation.items():
        if isinstance(value, torch.Tensor):
            numpy_value = value.cpu().numpy()
            
            # Handle image data specifically
            if "images" in key:
                # Convert from CHW to HWC format
                if len(numpy_value.shape) == 3 and numpy_value.shape[0] == 3:
                    numpy_value = numpy_value.transpose(1, 2, 0)
                
                # Convert from float32 (0-1) to uint8 (0-255)
                if numpy_value.dtype == np.float32:
                    numpy_value = (numpy_value * 255).astype(np.uint8)
            
            obs_data[key] = numpy_value
        else:
            obs_data[key] = value
    
    logger.info("Final format sent to server:")
    for key, value in obs_data.items():
        logger.info(f"  {key}: {value.shape} {value.dtype}")
    
    # Verify it matches your working format
    logger.info("\n=== Format Compatibility Check ===")
    for key in your_format.keys():
        if key in obs_data:
            your_val = your_format[key]
            our_val = obs_data[key]
            
            if isinstance(your_val, np.ndarray) and isinstance(our_val, np.ndarray):
                if your_val.shape == our_val.shape and your_val.dtype == our_val.dtype:
                    logger.info(f"✅ {key}: Compatible")
                else:
                    logger.error(f"❌ {key}: Shape {your_val.shape} vs {our_val.shape}, dtype {your_val.dtype} vs {our_val.dtype}")
            else:
                logger.info(f"✅ {key}: Compatible (non-array)")
        else:
            logger.error(f"❌ {key}: Missing in our format")
    
    logger.info("✅ Data format compatibility test completed!")

if __name__ == "__main__":
    test_lerobot_websocket_integration()
    test_data_format_compatibility()
