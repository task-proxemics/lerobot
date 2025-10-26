rm -rf /home/teamproxemics/.cache/huggingface/lerobot/xle/tactile_data_test
python lerobot_record.py --robot.type=so101_follower \
    --robot.port=/dev/ttyACM3 \
    --robot.id=xle_right_follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=xle_right_leader \
    --dataset.repo_id='xle/tactile_data_test' \
    --display_data=true \
    --dataset.push_to_hub=False \
    --dataset.num_episodes=2 \
    --dataset.episode_time_s=300 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Test tactile data pilot"


# --robot.cameras="{wrist1: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, realsense: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}} "  \
