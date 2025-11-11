
```
python lerobot_record.py \ 
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=xle_right_follower \
  --robot.cameras="{wrist: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30, fourcc: MJPG},
           up: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG}} " \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM3 \
  --teleop.id=xle_right_leader \
  --dataset.repo_id=jasonmeaux/so101_teleop_test \
  --dataset.push_to_hub=True \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="Describe task - LATER " \ 
  --dataset.video_encoding_batch_size={desired_batch_size}
```

## Recording Teleop Data

Navigate to the scripts folder

```shell
cd src/lerobot/scripts
```

```shell
python lerobot_record.py \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACMx \
  --robot.id=xle_right_follower \
  --robot.cameras="{wrist: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30, fourcc: MJPG}, 
                    up: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: YUYV}} " \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACMx \
  --teleop.id=xle_right_leader \
  --dataset.repo_id=${HF_USERNAME}/${DATASET_NAME} \
  --dataset.push_to_hub=True \   
  --dataset.num_episodes=?? \
  --dataset.episode_time_s=?? \
  --dataset.reset_time_s=?? \
  --dataset.single_task="Describe task here" \
  --dataset.video_encoding_batch_size={desired_batch_size}
```

### Notes:
- The Intel Realsense camera doesn't support recording in the `MJPG` format. Hence, we set its format to `YUYV`. To know which formats are supported by a camera run the following command in a terminal `v4l2-ctl -d /dev/video{x} --list-formats-ext`.
- Set `dataset.push_to_hub` to `False` if you only want to save the dataset locally.
- In order to collect teleop data more efficiently, set the `video_encoding_batch_size` param to a number higher than 1 to avoid encoding at each telop episode.


## Replaying Teleop Data

```shell
python lerobot_replay.py \ 
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACMx \
    --robot.id=xle_right_follower \
    --dataset.repo_id=${HF_USERNAME}/${DATASET_NAME} \
    --dataset.episode=xx
```

## Training VLA 


## Deploying VLA

## Troubleshooting


1. `No status packet error.`

    **Cause**: The motor ids are not being found potentially

    **Solution**: Unplug power and data cable on driver board.

2. `TimeoutError: Timed out waiting for frame from camera OpenCVCamera(0) after 200 ms. Read thread alive: True.`

    **Cause**: Don't ask. Don't Tell. 

    **Solution**: In the `wrist` and `up` camera configs, set the `fourcc` param to `MJPG`