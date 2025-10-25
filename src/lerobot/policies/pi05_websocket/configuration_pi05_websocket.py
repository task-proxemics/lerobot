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
Configuration for OpenPI π₀.₅ WebSocket Policy.

This policy communicates with an external OpenPI π₀.₅ model running
on a WebSocket server instead of loading the model locally.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("pi05_websocket")
@dataclass
class PI05WebSocketConfig(PreTrainedConfig):
    """Configuration for OpenPI π₀.₅ WebSocket policy.
    
    This policy communicates with an external OpenPI π₀.₅ model running
    on a WebSocket server instead of loading the model locally.
    
    Args:
        websocket_url: URL of the WebSocket server running OpenPI π₀.₅
        timeout: Timeout for WebSocket requests in seconds
        action_horizon: Number of actions to predict per inference call
        device: Device for tensor operations (not used for WebSocket policy)
    """
    
    websocket_url: str = "ws://localhost:8000"
    timeout: float = 5.0
    action_horizon: int = 8
    device: str = "cpu"  # Not used for WebSocket policy but required by base class
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.websocket_url.startswith(("ws://", "wss://")):
            raise ValueError("websocket_url must start with 'ws://' or 'wss://'")
        
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
