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
OpenPI π₀.₅ WebSocket Policy implementation.

This policy communicates with an external OpenPI π₀.₅ model running
on a WebSocket server instead of loading the model locally.
"""

import asyncio
import logging
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor

try:
    import websockets
    import msgpack
    import msgpack_numpy
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None
    msgpack = None
    msgpack_numpy = None 

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.pi05_websocket.configuration_pi05_websocket import PI05WebSocketConfig

logger = logging.getLogger(__name__)


class PI05WebSocketPolicy(PreTrainedPolicy):
    """OpenPI π₀.₅ WebSocket Policy for robot control.

    This policy communicates with an external OpenPI π₀.₅ model running
    on a WebSocket server. It sends observations to the server and receives
    action predictions back.
    """

    config_class: type[PI05WebSocketConfig] = PI05WebSocketConfig
    
    def __init__(self, config: PI05WebSocketConfig):
        super().__init__(config)
        self.name = "pi05_websocket"
        
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError(
                "websockets package is required for PI05WebSocketPolicy. "
                "Install it with: pip install websockets"
            )
        
        self.config = config
        self.websocket_url = config.websocket_url
        self.timeout = config.timeout
        self.action_horizon = config.action_horizon
        
        # Action queue for managing action sequences
        self._action_queue = deque()
        self._current_action_index = 0
        
        logger.info(f"Initialized PI05WebSocketPolicy with server: {self.websocket_url}")
    
    def reset(self):
        """Reset the policy state."""
        self._action_queue.clear()
        self._current_action_index = 0
        logger.debug("Policy reset")
    
    def get_optim_params(self):
        """Return empty list as this policy doesn't have trainable parameters."""
        return []
    
    async def _send_observation_to_server(self, observation: Dict[str, Tensor]) -> Dict[str, Any]:
        """Send observation to OpenPI WebSocket server and get action prediction.
        
        Args:
            observation: Dictionary containing observation tensors
            
        Returns:
            Dictionary containing action predictions from the server
        """
        try:
            if websockets is None or msgpack is None or msgpack_numpy is None:
                raise ImportError("websockets, msgpack, and msgpack_numpy packages are required")
            
            # Convert observation to format expected by OpenPI server
            # OpenPI expects HWC numpy arrays, uint8 format
            obs_data = {}
            for key, value in observation.items():
                if isinstance(value, Tensor):
                    # Convert to numpy array
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
            
            async with websockets.connect(self.websocket_url, timeout=self.timeout) as websocket:
                # Receive metadata first (as per OpenPI server protocol)
                metadata = msgpack_numpy.unpackb(await websocket.recv())
                logger.debug(f"Received metadata from server: {metadata}")
                
                # Send observation using MessagePack (as per OpenPI server protocol)
                packer = msgpack_numpy.Packer()
                await websocket.send(packer.pack(obs_data))
                
                # Receive action prediction
                response = await websocket.recv()
                result = msgpack_numpy.unpackb(response)
                
                return result
                
        except Exception as e:
            logger.error(f"WebSocket communication failed: {e}")
            # Return zero actions as fallback (OpenPI format)
            return {
                "actions": np.zeros((self.action_horizon, 6), dtype=np.float32),
                "success": False,
                "error": str(e)
            }
    
    def _run_async_inference(self, observation: Dict[str, Tensor]) -> Dict[str, Any]:
        """Run async WebSocket inference in sync context.
        
        Args:
            observation: Dictionary containing observation tensors
            
        Returns:
            Dictionary containing action predictions from the server
        """
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._send_observation_to_server(observation))
    
    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor], noise: Optional[Tensor] = None) -> Tensor:
        """Select action using WebSocket communication with OpenPI π₀.₅ server.
        
        Args:
            batch: Dictionary containing observation tensors
            noise: Optional noise tensor (not used)
            
        Returns:
            Action tensor for the robot
        """
        # If we have actions in queue, return the next one
        if self._action_queue and self._current_action_index < len(self._action_queue):
            action = self._action_queue[self._current_action_index]
            self._current_action_index += 1
            
            # Convert to tensor
            action_tensor = torch.tensor(action, dtype=torch.float32)
            logger.debug(f"Using queued action {self._current_action_index-1}/{len(self._action_queue)}")
            return action_tensor
        
        # Get new action sequence from server
        logger.debug("Requesting new action sequence from WebSocket server")
        result = self._run_async_inference(batch)
        
        # OpenPI server returns actions in "actions" field as numpy array
        # The server response format is: {"actions": numpy_array, "server_timing": {...}}
        if "actions" in result:
            actions = result["actions"]
            if isinstance(actions, np.ndarray) and len(actions.shape) >= 2:
                # actions is shape (horizon, action_dim) - take first action
                first_action = actions[0] if len(actions) > 0 else np.zeros(6)
                
                # Take first 6 elements for S0101 (6-DOF)
                s0101_action = first_action[:6] if len(first_action) >= 6 else np.zeros(6)
                
                # Store all actions in queue for action horizon
                action_list = [actions[i][:6] if len(actions[i]) >= 6 else np.zeros(6) for i in range(len(actions))]
                self._action_queue = deque(action_list)
                self._current_action_index = 1
                
                # Return the first action
                action_tensor = torch.tensor(s0101_action, dtype=torch.float32)
                logger.debug(f"Received {len(actions)} actions from server, first: {s0101_action}")
                return action_tensor
            else:
                logger.warning(f"Invalid actions format received: {actions}")
                return torch.zeros(6, dtype=torch.float32)
        else:
            logger.warning(f"No actions in server response: {result}")
            return torch.zeros(6, dtype=torch.float32)
    
    def forward(self, batch: Dict[str, Tensor]) -> tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass - not used for WebSocket policy."""
        raise NotImplementedError("Forward pass not implemented for WebSocket policy")
    
    def get_loss(self, batch: Dict[str, Tensor]) -> tuple[Tensor, Dict[str, Tensor]]:
        """Get loss - not used for WebSocket policy."""
        raise NotImplementedError("Loss computation not implemented for WebSocket policy")
