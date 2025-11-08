import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

device = torch.device("cuda")  # or "cuda" or "cpu"
model_id = "ptizzza/act_so101_test3"
model = ACTPolicy.from_pretrained(model_id)

dataset_id = "pr0tos/so101_single_tasks" 
# This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
dataset_metadata = LeRobotDatasetMetadata(dataset_id)
preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)

# # find ports using lerobot-find-port
follower_port = "/dev/ttyACM1"  # something like "/dev/tty.usbmodem58760431631"

# # the robot ids are used the load the right calibration files
follower_id = "so101_follower"  # something like "follower_so100"

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20

# Robot and environment configuration
# Camera keys must match the name and resolutions of the ones used for training!
# You can check the camera keys expected by a model in the info.json card on the model card on the Hub
camera_config = {
    "wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30, fourcc="MJPG"),
    "up": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=30, fourcc="MJPG"),
}

robot_cfg = SO101FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
robot = SO101Follower(robot_cfg)
robot.connect()

for _ in range(MAX_EPISODES):
    for _ in range(MAX_STEPS_PER_EPISODE):
        obs = robot.get_observation()
        obs_frame = build_inference_frame(
            observation=obs, ds_features=dataset_metadata.features, device=device
        )

        obs = preprocess(obs_frame)

        action = model.select_action(obs)
        action = postprocess(action)

        action = make_robot_action(action, dataset_metadata.features)

        robot.send_action(action)

    print("Episode finished! Starting new episode...")
