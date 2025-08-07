import speech_recognition as sr
import os
from openai import OpenAI
import time

from dotenv import load_dotenv
load_dotenv()

# Set up your API key
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_KEY)

# ❗ IMPORTANT: Set this to your microphone's device index ❗
# Based on your previous checks, this might be 2 or 3.
MIC_INDEX = 3

def print_mic_transcriptions():
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone(device_index=MIC_INDEX) as source:
            print(f"✅ Listening on mic index {MIC_INDEX}. Say a phrase now...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while True:
                try:
                    # Listen for up to 3 seconds
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
                    print("➡️ Audio captured. Sending to Whisper...")

                    temp_audio_file = "test_audio.wav"
                    with open(temp_audio_file, "wb") as f:
                        f.write(audio.get_wav_data())

                    with open(temp_audio_file, "rb") as audio_file:
                        response = openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            language="ta",
                            response_format="text",
                            timeout=20
                        )
                    
                    transcription = response.strip()
                    if not transcription:
                        print(f"❌ Whisper returned an empty transcription.")
                    else:
                        print(f"✅ Whisper heard: '{transcription}'")

                except sr.WaitTimeoutError:
                    print("⏱️ No speech detected. Listening again...")
                except Exception as e:
                    print(f"❌ An API or transcription error occurred: {e}")
                
                time.sleep(1) # Wait a bit before the next listen
    
    except Exception as e:
        print(f"❌ An initial error occurred with the microphone setup: {e}")

if __name__ == "__main__":
    print_mic_transcriptions()