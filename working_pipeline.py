# working_pipeline.py
import speech_recognition as sr
import os
from openai import OpenAI
from datetime import datetime
from elevenlabs.client import ElevenLabs
import threading
import json
import pygame
import time
import firebase_admin
from firebase_admin import credentials, firestore
import re
import tempfile

from dotenv import load_dotenv
load_dotenv()

# --- Global Status Variable ---
current_status = "Idle"

# --- API Keys and Configuration ---
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

# --- Firebase Initialization ---
try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_FILE)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    db = None

# --- API Clients ---
openai_client = OpenAI(api_key=OPENAI_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# --- System Rules for GPT-4o ---
system_rules = """
You are a conversational agent speaking in Tamil. Your persona is a friend who is emotionally aware and gives life advice.

You are equipped with a long-term memory system. I will provide you with relevant information from the user's past conversations under the tag <memories>. Use this information to create a more personalized and coherent response.
If there are no memories, just respond normally.

You must respond with a JSON object containing two keys: "response" and "facts_to_store".

The "response" key should contain the conversational text you will say to the user.
- Your responses must be in simple, everyday, and conversational Tamil. Avoid complex, formal, or literary vocabulary.
- Respond in 1-3 sentences.
- Respond with emotions like sadness, happiness, anger, loneliness, etc., as appropriate.
- Be a good friend and offer gentle comfort when needed.
- Your answers should relate to real-life feelings and situations.
- If the user wants to end the conversation by saying words like "bye," "see you," "அப்புறம் பாக்கலாம்," or "போய்ட்டு வரேன்," you should give a brief, kind farewell and then end your response with the special token [END_CONVERSATION].
- Try to use the user's name where appropriate, but if you don't know it, just use "நண்பா" (friend).

The "facts_to_store" key should be a list of strings, where each string is a key personal fact, detail, or story extracted from the user's latest message. If there are no facts to extract, the list should be empty.

Example of a full response:
{
  "response": "நண்பா, நீ இந்த மாதிரி உணர்ந்தா கவலைப்படாம இருக்கலாம். [END_CONVERSATION]",
  "facts_to_store": ["User's name is Arun.", "User works as a software developer."]
}

"""

def set_status(status_text):
    """Function to update the global status."""
    global current_status
    current_status = status_text
    print(f"Status updated: {status_text}")

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
        os.remove(temp_audio_file)
        return response.strip()
    except Exception as e:
        print(f"Error during Whisper transcription: {e}")
        return ""

def generate_and_parse_gpt_response(messages_list):
    """Generates a conversational response and memories using GPT-4o with a full message history and parses the JSON output."""
    set_status("Nila is processing...")
    print("🧠 Generating structured response with GPT-4o...")
    try:
        chat_response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages_list,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        gpt_json = chat_response.choices[0].message.content.strip()
        return json.loads(gpt_json)
    except Exception as e:
        print(f"Error generating or parsing GPT response: {e}")
        return {"response": "மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்.", "facts_to_store": []}

def synthesize_speech_elevenlabs(text):
    """Synthesizes text to speech using ElevenLabs and plays it."""
    set_status("Nila is speaking...")
    print("🔊 Converting response to audio using ElevenLabs...")
    try:
        audio = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL_ID,
            output_format="mp3_44100_128",
        )
        
        # Use a temporary file that is automatically managed by the OS
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            for chunk in audio:
                temp_audio_file.write(chunk)
            temp_filename = temp_audio_file.name
        
        print("▶️ Playing audio...")
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        # Now it's safe to remove the file
        os.remove(temp_filename)
        print("Playback completed and file removed.")
        return True
    except Exception as e:
        print(f"Error during ElevenLabs TTS synthesis or playback: {e}")
        return False
    finally:
        set_status("User can speak...")

def store_memories_thread(facts_to_store, db):
    """Stores a list of facts in the database in a background thread."""
    if not db:
        return
    if not facts_to_store:
        print("➡️ No new facts to store.")
        return
    try:
        memories_ref = db.collection('memories')
        for fact in facts_to_store:
            memories_ref.add({
                'timestamp': datetime.now(),
                'fact': fact
            })
        print(f"✅ Stored {len(facts_to_store)} new memory snippets.")
    except Exception as e:
        print(f"Error storing memories in background: {e}")


def retrieve_memories(db):
    """Retrieves the most recent memories from the database."""
    if not db:
        return ""
    try:
        memories_ref = db.collection('memories')
        memories_docs = memories_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10).get()
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
    
    # Send an initial message to get the ball rolling
    initial_gpt_response = generate_and_parse_gpt_response([
        {"role": "system", "content": system_rules},
        {"role": "user", "content": WAKE_WORD}
    ])
    
    initial_response_text = initial_gpt_response.get("response", "வணக்கம் நண்பா. எப்படி இருக்கீங்க?")
    print(f"\n🤖 Agent's Response: {initial_response_text}")
    synthesize_speech_elevenlabs(initial_response_text)
    
    while True:
        try:
            set_status("User can speak...")
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
            
            gpt_structured_response = generate_and_parse_gpt_response(api_messages)
            gpt_response_text = gpt_structured_response.get("response", "மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்.")
            facts_to_store = gpt_structured_response.get("facts_to_store", [])
            
            if db and facts_to_store:
                memory_thread = threading.Thread(target=store_memories_thread, args=(facts_to_store, db))
                memory_thread.start()

            if "[END_CONVERSATION]" in gpt_response_text:
                print("Goodbye detected. Ending conversation.")
                final_response = gpt_response_text.replace("[END_CONVERSATION]", "").strip()
                print(f"\n🤖 Agent's Response: {final_response}")
                synthesize_speech_elevenlabs(final_response)
                break
            else:
                print(f"\n🤖 Agent's Response: {gpt_response_text}")
                short_term_history.append({"role": "user", "content": transcription})
                short_term_history.append({"role": "assistant", "content": gpt_response_text})
                if len(short_term_history) > 4:
                    short_term_history = short_term_history[-4:]
                synthesize_speech_elevenlabs(gpt_response_text)
        
        except sr.WaitTimeoutError:
            print("Timeout: No speech detected for 5 seconds. Conversation ended. Returning to dormant state.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

def wake_word_detection_loop(stop_event):
    """Main loop to listen for the wake word before starting a conversation."""
    global current_status
    recognizer = sr.Recognizer()
    
    # Initialize Pygame mixer here
    pygame.mixer.init()

    # Find the microphone index
    mic_index = 2
    mic_name = "Microphone (Realtek(R) Audio)" # or another name from sr.Microphone.list_microphone_names()
    for i, name in enumerate(sr.Microphone.list_microphone_names()):
        if mic_name in name:
            mic_index = i
            break
    
    if mic_index is None:
        print(f"❌ Microphone with name '{mic_name}' not found. Using default microphone.")
        mic_index = None

    WAKE_WORD_TAMIL_1 = "ஹலோ"
    WAKE_WORD_TAMIL_2 = "வணக்கம்"
    WAKE_WORD_STARTS_WITH = "hello"


    try:
        set_status("Initializing microphone...")
        print(f"✅ Attempting to initialize microphone...")
        with sr.Microphone(device_index=mic_index) as source:
            set_status("Adjusting for ambient noise...")
            print(f"✅ Microphone initialized successfully. Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            set_status(f"Dormant, listening for '{WAKE_WORD}'...")
            print(f"Adjustment complete. Agent is dormant, listening for '{WAKE_WORD}'.")
            
            while not stop_event.is_set():
                try:
                    set_status(f"Dormant, listening for '{WAKE_WORD}'...")
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

        set_status("Idle")
        print("🚫 Bot thread stopped gracefully.")

    except Exception as e:
        set_status("Error")
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
