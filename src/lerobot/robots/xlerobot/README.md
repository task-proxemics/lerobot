# XleRobot Instructions

## Running audio navigation policy:

Set the correct ports for XleRobot

```shell
python audio_navigation.py
```

## Running speech to text recognition script

- Prerequisites

```
sudo apt install build-essential portaudio19-dev libportaudio2 python3-dev
```

```
pip install pyaudio speech_recognition pocketsphinx pyttsx3
```

- Running

```
cd src/lerobot/robots/xlerobot
python speech_recog.py
```