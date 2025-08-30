import speech_recognition as sr
import os
from openai import OpenAI
from datetime import datetime
from elevenlabs.client import ElevenLabs
import threading
import json
import pygame
import time
import re
import tempfile
from llama_cpp import Llama
from pymongo import MongoClient

from dotenv import load_dotenv
load_dotenv()

# --- Global Status Variable ---
current_status = "Idle"
llm = None

# --- API Keys and Configuration ---
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "YOUR_ELEVENLABS_API_KEY")
WAKE_WORD = os.environ.get("WAKE_WORD", "hello nila")

WHISPER_MODEL = "whisper-1"
GPT_MODEL = "gpt-4o"
ELEVENLABS_VOICE_ID = "C2RGMrNBTZaNfddRPeRH"
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
LOCAL_LLM_MODEL = "tamil-llama-7b-v0.1-q4_k_m.gguf"

if not OPENAI_KEY or OPENAI_KEY == "YOUR_OPENAI_API_KEY":
    raise ValueError("OPENAI_API_KEY environment variable not set or is a placeholder.")
if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "YOUR_ELEVENLABS_API_KEY":
    raise ValueError("ELEVENLABS_API_KEY environment variable not set or is a placeholder.")
if not os.path.exists(LOCAL_LLM_MODEL):
    raise FileNotFoundError(f"Local LLM model file '{LOCAL_LLM_MODEL}' not found. Please download it.")

# --- MongoDB Initialization ---
try:
    MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    DATABASE_NAME = "nila_memory_db"
    
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_db = mongo_client[DATABASE_NAME]
    db = mongo_db.memories
    print("✅ MongoDB initialized successfully.")
except Exception as e:
    print(f"Error initializing MongoDB: {e}")
    db = None

openai_client = OpenAI(api_key=OPENAI_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# --- System Rules for Local LLM ---
system_rules_response = """You are a conversational agent speaking in Tamil. Your persona is a friend who is emotionally aware and gives life advice.
- Your responses must be in simple, everyday, and conversational Tamil. Avoid complex, formal, or literary vocabulary.
- Respond in 1-3 sentences.
- Respond with emotions like sadness, happiness, anger, loneliness, etc., as appropriate.
- Be a good friend and offer gentle comfort when needed.
- Your answers should relate to real-life feelings and situations.
- If the user wants to end the conversation, give a brief, kind farewell and end with the special token [END_CONVERSATION].
- Try to use the user's name where appropriate, but if you don't know it, just use "நண்பா" (friend).
"""
system_rules_facts = """You are a memory extraction bot. Your job is to extract key personal facts, details, and stories from the following user message. Respond with a simple list of facts.
If the user message does not contain any personal facts, details, or stories, respond with the exact phrase "NO_FACTS".
Example:
User message: "My name is Arun and I work as a software developer. My favorite food is dosa."
Extraction:
- User's name is Arun.
- User works as a software developer.
- User's favorite food is dosa.
"""

def set_status(status_text):
    global current_status
    current_status = status_text
    print(f"Status updated: {status_text}")

def transcribe_with_whisper(audio_data):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio_data.get_wav_data())
            temp_filename = temp_audio_file.name

        with open(temp_filename, "rb") as audio_file:
            response = openai_client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                language="ta",
                response_format="text",
                timeout=15
            )
        os.remove(temp_filename)
        return response.strip()
    except Exception as e:
        print(f"Error during Whisper transcription: {e}")
        return ""

def get_llm_response_and_facts_multi_step(prompt, short_term_history, memories):
    global llm
    if not llm:
        return {"response": "மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்.", "facts_to_store": []}
    set_status("Nila is processing...")
    print("\n🧠 Step 1: Generating conversational response...")
    
    prompt_response = f"""{system_rules_response}
### Instruction:
{prompt}
### Response:"""
    
    try:
        response_completion = llm.create_completion(
            prompt=prompt_response,
            temperature=0.7,
            max_tokens=200,
            stream=False,
            stop=["### Instruction:", "### Response:"]
        )
        conversational_response = response_completion['choices'][0]['text'].strip()
    except Exception as e:
        print(f"❌ Error getting conversational response: {e}")
        conversational_response = "மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்."

    set_status("Nila is processing...")
    print("🧠 Step 2: Extracting facts...")
    
    prompt_facts = f"""{system_rules_facts}
### Instruction:
{prompt}
### Response:"""

    try:
        facts_completion = llm.create_completion(
            prompt=prompt_facts,
            temperature=0.3,
            max_tokens=200,
            stream=False,
            stop=["### Instruction:", "### Response:"]
        )
        extracted_memories = facts_completion['choices'][0]['text'].strip()
        
        if extracted_memories == "NO_FACTS" or not extracted_memories:
            facts_to_store = []
        else:
            facts_to_store = extracted_memories.split('\n')
    except Exception as e:
        print(f"❌ Error extracting facts: {e}")
        facts_to_store = []

    final_response = {
        "response": conversational_response,
        "facts_to_store": facts_to_store
    }
    
    return final_response

def synthesize_speech_elevenlabs(text):
    set_status("Nila is speaking...")
    print("🔊 Converting response to audio using ElevenLabs...")
    try:
        audio = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL_ID,
            output_format="mp3_44100_128",
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            for chunk in audio:
                temp_audio_file.write(chunk)
            temp_filename = temp_audio_file.name

        print("▶️ Playing audio...")
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        os.remove(temp_filename)
        print("Playback completed and file removed.")
        return True
    except Exception as e:
        print(f"Error during ElevenLabs TTS synthesis or playback: {e}")
        return False
    finally:
        set_status("User can speak...")

def store_memories_thread(facts_to_store, db):
    if db is not None:
        if not facts_to_store:
            print("➡️ No new facts to store.")
            return
        try:
            for fact in facts_to_store:
                db.insert_one({
                    'timestamp': datetime.now(),
                    'fact': fact
                })
            print(f"✅ Stored {len(facts_to_store)} new memory snippets.")
        except Exception as e:
            print(f"Error storing memories in background: {e}")

def retrieve_memories(db):
    if db is not None:
        try:
            memories_docs = db.find().sort("timestamp", -1).limit(10)
            memories_string = ""
            if memories_docs:
                for doc in memories_docs:
                    memories_string += "- " + doc['fact'] + "\n"
            return memories_string
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return ""
    return ""

def start_conversation(recognizer, source):
    short_term_history = []
    initial_response = get_llm_response_and_facts_multi_step(WAKE_WORD, short_term_history, "")
    initial_response_text = initial_response.get("response", "வணக்கம் நண்பா. எப்படி இருக்கீங்க?")
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
            user_memories = retrieve_memories(db)
            local_llm_structured_response = get_llm_response_and_facts_multi_step(transcription, short_term_history, user_memories)
            gpt_response_text = local_llm_structured_response.get("response", "மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்.")
            facts_to_store = local_llm_structured_response.get("facts_to_store", [])
            if db is not None and facts_to_store:
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
    global current_status, llm
    recognizer = sr.Recognizer()
    pygame.mixer.init()
    mic_index = None
    mic_name = "Microphone (Realtek(R) Audio)"
    for i, name in enumerate(sr.Microphone.list_microphone_names()):
        if mic_name and mic_name in name:
            mic_index = i
            break
    if mic_index is None:
        print(f"✅ Using default microphone.")
    else:
        print(f"✅ Using microphone at index {mic_index}.")
    WAKE_WORD_TAMIL_1 = "ஹலோ"
    WAKE_WORD_TAMIL_2 = "வணக்கம்"
    WAKE_WORD_STARTS_WITH = "hello"
    if llm is None:
        try:
            llm = Llama(model_path=LOCAL_LLM_MODEL, n_ctx=2048, verbose=False)
            print("✅ Local LLM loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading local LLM model: {e}")
            set_status("Error")
            return
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