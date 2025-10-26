import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json

# Configuration
MODEL_PATH = "model"  # Path to your downloaded Vosk model
SAMPLE_RATE = 16000  # Sample rate for audio (must match model)
CHANNELS = 1         # Number of audio channels (mono)

# Create a queue to store audio data
q = queue.Queue()

# Callback function for sounddevice
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# Main script
try:
    # Load the Vosk model
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)

    print("Listening... Press Ctrl+C to stop.")

    # Start audio stream from microphone
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000,
                           dtype='int16', channels=CHANNELS, callback=callback):
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                if result.get("text", ""):
                    print("You said:", result["text"])
            else:
                partial_result = json.loads(recognizer.PartialResult())
                if partial_result.get("partial", ""):
                    print("Partial:", partial_result["partial"])

except KeyboardInterrupt:
    print("\nStopped recording.")
except Exception as e:
    print(f"An error occurred: {e}")