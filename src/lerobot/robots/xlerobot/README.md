# XleRobot Instructions

## 1. Running joycon teleop

- Install the joycon-robotics package

```
git clone https://github.com/box2ai-robotics/joycon-robotics.git
cd joycon-robotics

pip install -e .
sudo apt-get update
sudo apt-get install -y dkms libevdev-dev libudev-dev cmake git
make install
```


- Running

```shell
conda activate lerobot
cd examples
python 7_xlerobot_teleop_joycon.py
```

## 2. Running speech to text recognition script

- Prerequisites

```
sudo apt install build-essential portaudio19-dev libportaudio2 python3-dev swig
```

```
pip install pyaudio speech_recognition pocketsphinx pyttsx3
```

- Running

```
cd src/lerobot/robots/xlerobot
python speech_recog.py
```

## 3. Anyskin Tactile Demo



## 4. Running audio navigation policy:

- Prerequisites:

Set the correct ports for XleRobot in `config_xlerobot.py`

```
pip install pyusb
```

```shell
python audio_navigation.py
```

