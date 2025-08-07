import speech_recognition as sr
import os
from openai import OpenAI
from datetime import datetime
from elevenlabs.client import ElevenLabs
import subprocess
import time
import firebase_admin
from firebase_admin import credentials, firestore
import threading
import re

from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "YOUR_ELEVENLABS_API_KEY")
WAKE_WORD = os.environ.get("WAKE_WORD", "hello nila")

WHISPER_MODEL = "whisper-1"
GPT_MODEL = "gpt-4o"
ELEVENLABS_VOICE_ID = "C2RGMrNBTZaNfddRPeRH"
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
SERVICE_ACCOUNT_KEY_FILE = "serviceAccountKey.json"

if not OPENAI_KEY or OPENAI_KEY == "YOUR_OPENAI_API_KEY":
    raise ValueError("OPENAI_API_KEY environment variable not set or is a placeholder.")
if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "YOUR_ELEVENLABS_API_KEY":
    raise ValueError("ELEVENLABS_API_KEY environment variable not set or is a placeholder.")

try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_FILE)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    db = None

openai_client = OpenAI(api_key=OPENAI_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

system_rules = """
You are a conversational agent speaking in Tamil. Your persona is a friend who is emotionally aware and gives life advice.

You are equipped with a long-term memory system. I will provide you with relevant information from the user's past conversations under the tag <memories>. Use this information to create a more personalized and coherent response.
If there are no memories, just respond normally.

Here are the rules for your responses:
- Respond in 1-3 sentences.
- Speak in simple, conversational Tamil, not literary Tamil.
- Respond with emotions like sadness, happiness, anger, loneliness, etc., as appropriate.
- Be a good friend and offer gentle comfort when needed.
- Your answers should relate to real-life feelings and situations.
- If the user wants to end the conversation by saying words like "bye," "see you," "அப்புறம் பாக்கலாம்," or "போய்ட்டு வரேன்," you should give a brief, kind farewell and then end your response with the special token [END_CONVERSATION].
- Try to use the user's name where appropriate, but if you don't know it, just use "நண்பா" (friend).
"""

def transcribe_with_whisper(audio_data):
    """Transcribes audio data using OpenAI's Whisper API."""
    try:
        temp_audio_file = "temp_audio.wav"
        with open(temp_audio_file, "wb") as f:
            f.write(audio_data.get_wav_data())
        with open(temp_audio_file, "rb") as audio_file:
            response = openai_client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                language="ta",
                response_format="text",
                timeout=15
            )
        return response.strip()
    except Exception as e:
        print(f"Error during Whisper transcription: {e}")
        return ""

def generate_response_with_gpt(messages_list):
    """Generates a conversational response using GPT-4o with a full message history."""
    print("🧠 Generating response with GPT-4o...")
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
        ffplay_path = r"C:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin\ffplay.exe"
        subprocess.run([ffplay_path, "-autoexit", output_path])
        print("Playback completed.")
        return output_path
    except Exception as e:
        print(f"Error during ElevenLabs TTS synthesis or playback: {e}")
        return None

def extract_and_store_memories(latest_message, db):
    if not db:
        return
    extraction_prompt = f"""
    You are a memory extraction bot. Your job is to extract key facts, personal details, and stories from the following user message. Respond with a simple list of facts.
    If the user message does not contain any personal facts, details, or stories, respond with the exact phrase "NO_FACTS".
    User message: "{latest_message}"
    Extraction:
    """
    try:
        extraction_response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.3,
        )
        extracted_memories = extraction_response.choices[0].message.content.strip()
        if extracted_memories and extracted_memories != "NO_FACTS":
            memories_ref = db.collection('memories')
            memories_ref.add({
                'timestamp': datetime.now(),
                'fact': extracted_memories
            })
            print(f"✅ Stored new memory snippets.")
        else:
            print("➡️ No new facts to store.")
    except Exception as e:
        print(f"Error extracting or storing memories in background: {e}")

def retrieve_memories(db):
    if not db:
        return ""
    try:
        memories_ref = db.collection('memories')
        memories_docs = memories_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(5).get()
        memories_string = ""
        if memories_docs:
            for doc in memories_docs:
                memories_string += "- " + doc.to_dict()['fact'] + "\n"
        return memories_string
    except Exception as e:
        print(f"Error retrieving memories: {e}")
        return ""

def start_conversation(recognizer, source):
    """Handles the actual conversation after the wake word is detected."""
    short_term_history = []
    
    initial_gpt_response = generate_response_with_gpt([
        {"role": "system", "content": system_rules},
        {"role": "user", "content": "hello nila"}
    ])
    print(f"\n🤖 Agent's Response: {initial_gpt_response}")
    synthesize_speech_elevenlabs(initial_gpt_response)
    
    while True:
        try:
            print("\nListening for your voice...")
            audio = recognizer.listen(source, phrase_time_limit=10, timeout=5)
            print("Listening stopped.")

            transcription = transcribe_with_whisper(audio)
            
            if not transcription:
                print("No speech detected or transcription failed. Listening again...")
                continue
            
            print(f"📝 You said: {transcription}")
            
            api_messages = [{"role": "system", "content": system_rules}]
            
            user_memories = retrieve_memories(db)
            if user_memories:
                print(f"🧠 Retrieved memories:\n{user_memories}")
                api_messages.append({"role": "system", "content": f"<memories>\n{user_memories}\n</memories>"})
            
            api_messages.extend(short_term_history)
            api_messages.append({"role": "user", "content": transcription})
            
            gpt_response = generate_response_with_gpt(api_messages)
            
            if db:
                memory_thread = threading.Thread(target=extract_and_store_memories, args=(transcription, db))
                memory_thread.start()

            if "[END_CONVERSATION]" in gpt_response:
                print("Goodbye detected. Ending conversation.")
                final_response = gpt_response.replace("[END_CONVERSATION]", "").strip()
                print(f"\n🤖 Agent's Response: {final_response}")
                synthesize_speech_elevenlabs(final_response)
                break
            else:
                print(f"\n🤖 Agent's Response: {gpt_response}")
                short_term_history.append({"role": "user", "content": transcription})
                short_term_history.append({"role": "assistant", "content": gpt_response})
                if len(short_term_history) > 4:
                    short_term_history = short_term_history[-4:]
                synthesize_speech_elevenlabs(gpt_response)
        
        except sr.WaitTimeoutError:
            print("Timeout: No speech detected for 5 seconds. Conversation ended. Returning to dormant state.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

def wake_word_detection_loop(stop_event):
    """Main loop to listen for the wake word before starting a conversation."""
    recognizer = sr.Recognizer()
    MIC_INDEX = 3
    
    WAKE_WORD_TAMIL_1 = "ஹலோ"
    WAKE_WORD_TAMIL_2 = "வணக்கம்"
    WAKE_WORD_STARTS_WITH = "hello"

    try:
        print(f"✅ Attempting to initialize microphone with index {MIC_INDEX}...")
        with sr.Microphone(device_index=MIC_INDEX) as source:
            print(f"✅ Microphone initialized successfully. Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print(f"Adjustment complete. Agent is dormant, listening for '{WAKE_WORD}'.")
            
            while not stop_event.is_set():
                try:
                    print(f"\nWaiting for '{WAKE_WORD}'... (listening for up to 3 seconds)")
                    audio = recognizer.listen(source, phrase_time_limit=3)
                    print("➡️ Audio captured. Sending to Whisper...")
                    
                    transcription = transcribe_with_whisper(audio)
                    print(f"✅ Whisper transcribed: '{transcription}'")
                    
                    transcription_lower = transcription.lower()
                    
                    is_wake_word_detected = False
                    
                    if transcription_lower.startswith(WAKE_WORD_STARTS_WITH) and len(transcription_lower.split()) > 1:
                        is_wake_word_detected = True
                        print("✅ Wake word detected by 'starts with' check.")
                    
                    # --- NEW LINE ADDED HERE ---
                    elif WAKE_WORD_TAMIL_1 in transcription or WAKE_WORD_TAMIL_2 in transcription:
                        is_wake_word_detected = True
                        print("✅ Wake word detected by 'contains Tamil word' check.")
                    
                    if is_wake_word_detected:
                        print(f"Wake word detected! Starting conversation.")
                        start_conversation(recognizer, source)
                    else:
                        print("❌ Wake word not detected. Listening again.")
                    
                except sr.WaitTimeoutError:
                    if stop_event.is_set():
                        break
                    print("⏱️ No speech detected within timeout. Listening again.")
                    continue
                except Exception as e:
                    print(f"❌ An error occurred during transcription: {e}")
                    time.sleep(1)

        print("🚫 Bot thread stopped gracefully.")

    except Exception as e:
        print(f"❌ FATAL ERROR: Microphone setup failed. Error: {e}")
        stop_event.set()

if __name__ == "__main__":
    print("\n--- Running in standalone mode ---")
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

    dummy_stop_event = threading.Event()
    wake_word_detection_loop(dummy_stop_event)
    print("Conversation ended. Thank you for using the agent!")