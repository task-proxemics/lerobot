import time
import sys
from datetime import datetime
import speech_recognition as sr
import socket
from functools import partial

DEFAULT_MIC_INDEX = 4  # fallback if ReSpeaker can't be auto-detected
SAMPLE_RATE = 16000    # adjust if your ReSpeaker uses a different rate

def _speak_pyttsx3(text: str) -> bool:
    try:
        import pyttsx3
    except Exception:
        return False
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)     # words per minute
    engine.setProperty("volume", 1.0)   # 0.0 to 1.0
    engine.say(text)
    engine.runAndWait()
    return True

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
            time.sleep(2.0)  # brief pause before speaking
            # Here you can add code to send the response to the audio output channel
            # For example, you might use a text-to-speech library or send a message to another system
            _speak_pyttsx3("Howdy Folks! How are you doing?")


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
