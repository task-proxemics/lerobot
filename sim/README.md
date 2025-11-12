# IsaacSim & IsaacLab Docs

## Installation

**Versions**
- Ubuntu 24.04
- IsaacSim 5.0.0
- ROS2 Jazzy

### Create conda environment
- Install anaconda by following the instructions [here](https://www.anaconda.com/docs/getting-started/anaconda/install#macos-linux-installation)
- Create and activate a new environment for IsaacSim based on python 3.11
    ```
    conda create -n env_isaaclab python=3.11
    conda activate env_isaaclab
    ```

### Install IsaacSim within the conda environment
- [Reference Link](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html)
- Within the `env_isaaclab` environment run the following
    ```
    pip install --upgrade pip
    pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
    ```
- IsaacSim gets added as a command line-tool within the environment. To launch it, simply run
    ```
    isaacsim
    ```
- Note: You'll need to accept the Omniverse License Agreement (EULA). The first time IsaacSim opens it takes time to load (~2-3 minutes)

### Install IsaacLab

```
sudo apt install cmake build-essential
```

```
git clone git@github.com:isaac-sim/IsaacLab.git
```


For isaacsim5.0.0, it is recommended to use `v2.2.1` of IsaacLab 

```
cd IsaacLab
git checkout v2.2.1
./isaaclab.sh --install
```

### Install ROS2
- Install ROS2 Jazzy by following the instructions [here](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
- Note: Install `ros-jazzy-desktop`
- After the installation alos install the `teleop-twist-keyboard` package.
    ```
    sudo apt-get install ros-$ROS_DISTRO-teleop-twist-keyboard
    ```

### Launching IsaacSim with ROS2
- To interface IsaacSim with ROS2 some environment variables need to be setup. I've put all of these in the script `launch_isaacsim.bash`. Close any existing IsaacSim instance, open a new terminal and run 
    ```
    ./launch_isaacsim.bash
    ```
- In IsaacSim, open the `kaya_omniwheel_example/kaya_omniwheel_example.usd` scene. (*File -> Open*)
- Run the simulation (Play button in the left panel)
- In another terminal, source your ROS2 installation
    ```
    source /opt/ros/jazzy/setup.bash
    ```
- Run the `teleop-twist-keyboard` ROS node. 
    ```
    ros2 run teleop_twist_keyboard teleop_twist_keyboard
    ```
- Follow the `teleop` instructions to control the robot and ensure that its moving accordingly in IsaacSim. Note: Use the `Holonomic mode (strafing)` instructions.

### Explanation: 
- In the `kaya_omniwheel_example.usd` file I have setup an ActionGraph. The ROS2 commands come in as an input on the `/cmd_vel` topic which is read by the `ROS2 Subscribe Twist Node`. 
- The `linear` and `angular` velocities are then passed on to the `Holonomic Controller` which converts them to joint velocity commands. 
- The joint velocity commands are then passed to the `Articulation Controller` which actually controls the torque applied at each individual joint. 

![alt text](action_graph.png)
- References: 
    - https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_drive_turtlebot.html 
    - https://medium.com/@kabilankb2003/mastering-holonomic-control-with-nvidia-isaac-sim-and-ros2-a-guide-for-the-kaya-robot-0b8ad6706eaa
