#rm -rf /home/teamproxemics/.cache/huggingface/lerobot/t-nukala/asgard_training_data
python lerobot_record.py --robot.type=so101_follower \
    --robot.port=/dev/ttyACM3 \
    --robot.id=xle_right_follower \
    --robot.cameras="{wrist1: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, realsense: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}} "  \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=xle_right_leader \
    --display_data=false \
    --dataset.push_to_hub=False \
    --dataset.num_episodes=10 \
    --dataset.repo_id='t-nukala/asgard_training_data_potato3' \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Pick up potato and hand it to the person"
