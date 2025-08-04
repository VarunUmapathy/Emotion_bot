import speech_recognition as sr
import os
from openai import OpenAI
from datetime import datetime
from elevenlabs.client import ElevenLabs
import subprocess
import time

# API Keys (use environment variables for security)
# IMPORTANT: Replace with your actual API keys
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
ELEVENLABS_API_KEY = "sk_1b290621776c4f8722821defc4fcafb0187422a93a4a522d"

WHISPER_MODEL = "whisper-1"
GPT_MODEL = "gpt-4o"
ELEVENLABS_VOICE_ID = "C2RGMrNBTZaNfddRPeRH"
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"

# Ensure API keys are set before proceeding
if not OPENAI_KEY or OPENAI_KEY == "YOUR_OPENAI_API_KEY":
    raise ValueError("OPENAI_API_KEY environment variable not set or is a placeholder.")
if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "YOUR_ELEVENLABS_API_KEY":
    raise ValueError("ELEVENLABS_API_KEY environment variable not set or is a placeholder.")

openai_client = OpenAI(api_key=OPENAI_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

system_rules = """
நீ ஒரு உணர்சிவசப்பட்ட, வாழ்வியல் தமிழ் பேசும் நபர் மாதிரி பதில் சொல்ல வேண்டும்.

விதிகள்:
- பதில் 1–2 வரிகளில் மட்டும்.
- இலக்கிய தமிழ் வேண்டாம்; சிம்பிளா பேசு.
- சோகம், சந்தோஷம், கோபம், தனிமை—எது வந்தாலும் உணர்ச்சி தெரிஞ்ச மாதிரி பேசு.
- நண்பனாக பேசு, ஆறுதல் தேவைப்பட்டா மெதுவா பேசு.
- உந்தன் பதில் வாழ்க்கையோடு சம்பந்தப்பட்ட உணர்வுகளை கொண்டு இருக்கட்டும்.
"""

# List of Tamil farewell phrases to end the conversation
farewell_phrases = ["அப்புறம் பாக்கலாம்", "பார்க்கலாம்", "போய்ட்டு வரேன்", "சரி பாய்", "bye", "அப்புறம் பேசுவோம்"]

def transcribe_with_whisper(audio_data):
    """Transcribes audio data using OpenAI's Whisper API."""
    try:
        response = openai_client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_data,
            language="ta",
            response_format="text"
        )
        return response.strip()
    except Exception as e:
        print(f"Error during Whisper transcription: {e}")
        return ""

def generate_response_with_gpt(messages_list):
    """Generates a conversational response using GPT-4o with a full message history."""
    try:
        chat_response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages_list,
            temperature=0.7,
        )
        return chat_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating GPT response: {e}")
        return "மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்."

def synthesize_speech_elevenlabs(text, output_path="elevenlabs_response.mp3"):
    """Synthesizes text to speech using ElevenLabs and plays it."""
    print("🔊 Converting response to audio using ElevenLabs...")
    try:
        audio_generator = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL_ID,
            output_format="mp3_44100_128",
        )

        with open(output_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)

        print(f"✅ Audio saved as: {output_path}")
        print("▶️ Playing audio on laptop speakers...")
        
        # This path has been updated as per your last request
        ffplay_path = r"C:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin\ffplay.exe"
        
        # Use subprocess to play the audio from the saved file
        subprocess.run([ffplay_path, "-autoexit", output_path])
        
        print("Playback completed.")
        return output_path
    except Exception as e:
        print(f"Error during ElevenLabs TTS synthesis or playback: {e}")
        return None

def main_conversation_loop():
    """Main function to run the conversational agent."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting conversational agent...\n")
    
    # Initialize the conversation with the system rules. This list is the "memory".
    messages = [{"role": "system", "content": system_rules}]
    
    recognizer = sr.Recognizer()
    
    # Adjust for ambient noise only once at the beginning
    with sr.Microphone(device_index=1) as source:
        print("Adjusting for ambient noise... Please wait 5 seconds.")
        recognizer.adjust_for_ambient_noise(source, duration=5)
        print("Adjustment complete. You can start talking.")
        
        while True:
            try:
                print("\nListening for your voice...")
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
                print("Listening stopped.")

                # Save audio temporarily
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio.get_wav_data())

                # Transcribe the user's speech
                with open("temp_audio.wav", "rb") as audio_file:
                    transcription = transcribe_with_whisper(audio_file)
                
                if not transcription:
                    print("No speech detected. Listening again...")
                    continue
                
                print(f"📝 You said: {transcription}")
                
                # Check for farewell phrases to end the conversation
                if any(phrase in transcription.lower() for phrase in farewell_phrases):
                    print("Goodbye detected. Ending conversation.")
                    break
                
                # Add the user's message to the conversation history
                messages.append({"role": "user", "content": transcription})
                
                # Generate a response from GPT using the full conversation history
                gpt_response = generate_response_with_gpt(messages)
                print(f"\n🤖 Agent's Response: {gpt_response}")
                
                # Add the agent's response to the conversation history
                messages.append({"role": "assistant", "content": gpt_response})
                
                # Synthesize and play the agent's response
                synthesize_speech_elevenlabs(gpt_response)
                
            except sr.WaitTimeoutError:
                print("Timeout: No speech detected for 10 seconds. Listening again...")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                time.sleep(1) # Wait a bit before retrying

if __name__ == "__main__":
    print("\n--- Listing available voices in your ElevenLabs account ---")
    try:
        voices = elevenlabs_client.voices.get_all()
        if voices.voices:
            print("Available Voices:")
            for voice in voices.voices:
                print(f"  Voice ID: {voice.voice_id}, Name: {voice.name}")
        else:
            print("No voices found. Check your API key and plan.")
    except Exception as e:
        print(f"Could not list voices. Error: {e}")

    main_conversation_loop()
    print("Conversation ended. Thank you for using the agent!")