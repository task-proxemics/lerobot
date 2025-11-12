#!/bin/bash

# Activate the env_isaaclab conda environment (first deactivate any existing environment)
source /home/savio/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate env_isaaclab

# Set ROS2 variables
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/savio/anaconda3/envs/env_isaacsim/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib

# Launch IsaacSim (Python Environment Installation)
isaacsim