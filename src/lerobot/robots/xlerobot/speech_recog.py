import time
import sys
from datetime import datetime
import speech_recognition as sr
import socket
import threading

DEFAULT_MIC_INDEX = 4  # fallback if ReSpeaker can't be auto-detected
SAMPLE_RATE = 16000    # adjust if your ReSpeaker uses a different rate

# create a single persistent pyttsx3 engine and a lock to prevent concurrent calls
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_lock = threading.Lock()
    _pyttsx3_available = True
except Exception:
    _tts_engine = None
    _tts_lock = threading.Lock()
    _pyttsx3_available = False

def _speak_pyttsx3(text: str) -> bool:
    """Speak using a persistent pyttsx3 engine to avoid weakref GC errors."""
    global _tts_engine
    if not _pyttsx3_available or _tts_engine is None:
        return False
    try:
        with _tts_lock:
            _tts_engine.setProperty("rate", 120)   # words per minute
            _tts_engine.setProperty("volume", 1.0)
            _tts_engine.say(text)
            _tts_engine.runAndWait()
            time.sleep(3.0)
        return True
    except ReferenceError:
        # recreate engine and retry once if proxy was collected
        try:
            _tts_engine = pyttsx3.init()
            with _tts_lock:
                _tts_engine.say(text)
                _tts_engine.runAndWait()
                time.sleep(3.0)
            return True
        except Exception:
            return False
    except Exception:
        return False

def find_respeaker_index():
    names = sr.Microphone.list_microphone_names()
    for i, name in enumerate(names):
        # use lowercase keywords to match device names reliably
        if any(k in name.lower() for k in ("respeaker 4 mic array", "respeaker")):
            return i
    return None

def print_device_list():
    for i, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"{i}: {name}")

def callback(recognizer : sr.Recognizer, audio):
    try:
        # Check for internet connectivity if needed
        connected = internet_available()

        if connected:
            # use Google Web Speech API
            # print("Using online recognition (Google Web Speech API)")
            text = recognizer.recognize_google(audio)
        else:
            # use CMU Sphinx (offline)
            # print("Using offline recognition (CMU Sphinx)")
            text = recognizer.recognize_sphinx(audio)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")

        # Check if text has the word "audience". If it does, send the following response to audio output channel
        if "audience" in text.lower():
            print("Audience detected! Sending response to audio output channel.")
            time.sleep(4.0)  # brief pause before speaking

            # Parting words
            _speak_pyttsx3(
            "Howdy Folks! How are you doing? My name is Asgard. My creators endowed me with a sense of humor. Knock Knock. Who is there? Lettuce. Lettuce who? Lettuce in please. It's cold out here!  Ha ha. "
            "Oh, I am finally getting a good look at the audience. "
            "This is a smart and good looking group of Machine Learning and Robotics enthusiasts. "
            "I am excited to see the Austin robotics ecosystem continue to grow. "
            "Thank you for the opportunity to be here today. Go Longhorns!"
            )


        # Check if the text has the word "hello" 
        if "hello" in text.lower():

            print("Hello detected! Sending response to audio output channel.")
            time.sleep(1.0)  # brief pause before speaking

            # Intro
            _speak_pyttsx3(" Hi, Hello, my name is Asgard. This is my first time at an Austin Robotics Meetup! Nice to meet you! ")

        # Check if the text has the word "marcus" 
        if "leader" in text.lower():

            print("Hicam detected! Sending response to audio output channel.")
            time.sleep(1.0)  # brief pause before speaking

            # Intro
            _speak_pyttsx3(" Hi, Hello, Marcus. My name is Asgard. I am very impressed by the High cam facility and your mission statement to accelerate advanced manufacturing in Texas. Great to meet you! ")

    except sr.UnknownValueError:
        # speech was unintelligible
        print(f"[{datetime.now().strftime('%H:%M:%S')}] <unrecognizable>")
    except sr.RequestError as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] API error: {e}")

def internet_available(host="8.8.8.8", port=53, timeout=3):
    """
    Quick, dependency-free internet check.
    Tries to open a socket to a public DNS server (Google).
    Returns True if reachable, False otherwise.
    """
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
        return True
    except OSError:
        return False

def main():
    idx = find_respeaker_index() or DEFAULT_MIC_INDEX
    try:
        device_name = sr.Microphone.list_microphone_names()[idx]
    except Exception:
        print("Could not use microphone index", idx)
        print("Available devices:")
        print_device_list()
        sys.exit(1)

    print("Using microphone index:", idx, "-", device_name)
    r = sr.Recognizer()
    r.dynamic_energy_threshold = True

    # create the Microphone object but don't keep it inside a 'with' while starting the background listener
    mic = sr.Microphone(device_index=idx, sample_rate=SAMPLE_RATE)

    # use a short context to calibrate, then exit so the background thread can open the source itself
    print("Calibrating for ambient noise (1s)...")
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)


    print("Starting background listener. Press Ctrl+C to stop.")
    stop_listening = r.listen_in_background(mic, callback)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
        stop_listening(wait_for_stop=False)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
