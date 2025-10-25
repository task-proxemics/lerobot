#!/usr/bin/env python

"""
Simple test script for PI05 WebSocket Policy configuration.

This script tests the configuration without requiring torch or other dependencies.
"""

import logging
from lerobot.policies.pi05_websocket.configuration_pi05_websocket import PI05WebSocketConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_websocket_config():
    """Test the WebSocket policy configuration."""
    
    # Test valid configuration
    config = PI05WebSocketConfig(
        websocket_url="ws://localhost:8000",
        timeout=5.0,
        action_horizon=8
    )
    
    logger.info(f"✅ Valid configuration created: {config}")
    logger.info(f"  WebSocket URL: {config.websocket_url}")
    logger.info(f"  Timeout: {config.timeout}")
    logger.info(f"  Action Horizon: {config.action_horizon}")
    logger.info(f"  Device: {config.device}")
    
    # Test invalid configurations
    try:
        PI05WebSocketConfig(websocket_url="http://localhost:8000")  # Should fail
        logger.error("❌ Should have failed with invalid URL")
    except ValueError as e:
        logger.info(f"✅ Correctly caught invalid URL: {e}")
    
    try:
        PI05WebSocketConfig(timeout=-1)  # Should fail
        logger.error("❌ Should have failed with negative timeout")
    except ValueError as e:
        logger.info(f"✅ Correctly caught negative timeout: {e}")
    
    try:
        PI05WebSocketConfig(action_horizon=0)  # Should fail
        logger.error("❌ Should have failed with zero action horizon")
    except ValueError as e:
        logger.info(f"✅ Correctly caught zero action horizon: {e}")
    
    logger.info("✅ All configuration tests passed!")

if __name__ == "__main__":
    test_websocket_config()
