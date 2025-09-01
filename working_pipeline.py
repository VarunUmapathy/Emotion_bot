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
import re
import tempfile
from pymongo import MongoClient
from bson import ObjectId

# === NEW: speaker verification
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import io
import soundfile as sf

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

if not OPENAI_KEY or OPENAI_KEY == "YOUR_OPENAI_API_KEY":
    raise ValueError("OPENAI_API_KEY environment variable not set or is a placeholder.")
if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "YOUR_ELEVENLABS_API_KEY":
    raise ValueError("ELEVENLABS_API_KEY environment variable not set or is a placeholder.")

# --- MongoDB Initialization ---
try:
    MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "nila_memory_db")

    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[MONGO_DB_NAME]   # DB: nila_memory_db
    print("✅ MongoDB initialized successfully.")
except Exception as e:
    print(f"Error initializing MongoDB: {e}")
    db = None

# --- API Clients ---
openai_client = OpenAI(api_key=OPENAI_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# === NEW: Speaker encoder (lazy init to speed cold start)
voice_encoder = None
def get_voice_encoder():
    global voice_encoder
    if voice_encoder is None:
        print("🎙️ Initializing speaker encoder...")
        voice_encoder = VoiceEncoder()  # CPU ok
    return voice_encoder

# === NEW: Speaker matching parameters
SPEAKER_MIN_SECONDS = 2.5          # min speech for reliable embedding
SPEAKER_SIM_THRESHOLD = 0.78       # tune 0.75–0.85 for your room/noise
SPEAKER_TOP_K = 3

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
- If the user wants to end the conversation by saying words like "bye," "see you," "அப்புறம் பாக்கலாம்," or "போய்ட்டு வரேன்," you should give a brief, kind farewell and then end your response with the special token [END_CONVERSATION].
- Try to use the user's name where appropriate, but if you don't know it, just use "நண்பா" (friend).

The "facts_to_store" key should be a list of strings, where each string is a key personal fact, detail, or story extracted from the user's latest message. If there are no facts to extract, the list should be empty.
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

# === NEW: utilities to extract raw waveform from speech_recognition.AudioData
def audio_data_to_wav_float32(audio_data: sr.AudioData):
    """
    Returns (wav_float32, sample_rate). We rely on the WAV bytes from speech_recognition and decode with soundfile.
    """
    wav_bytes = audio_data.get_wav_data()
    data, sr_hz = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return data, sr_hz

def compute_speaker_embedding(audio_data: sr.AudioData):
    """
    Compute speaker embedding from an utterance. Returns np.array shape (256,) or None if too short.
    """
    try:
        wav, sr_hz = audio_data_to_wav_float32(audio_data)
        duration = len(wav) / float(sr_hz)
        if duration < SPEAKER_MIN_SECONDS:
            print(f"🔇 Utterance too short for speaker embedding ({duration:.2f}s).")
            return None
        encoder = get_voice_encoder()
        # preprocess_wav will resample + normalize
        wav_pp = preprocess_wav(wav, source_sr=sr_hz)
        emb = encoder.embed_utterance(wav_pp)
        return emb.astype(np.float32)
    except Exception as e:
        print(f"Speaker embedding error: {e}")
        return None

# === NEW: Mongo helpers for users & embeddings
def users_col():
    return db["users"]

def memories_col():
    return db["memories"]

def fetch_all_users():
    return list(users_col().find({}, {"_id": 1, "name": 1, "embedding": 1}))

def enroll_user(name: str, embedding: np.ndarray):
    doc = {
        "name": name.strip(),
        "embedding": embedding.tolist(),
        "created_at": datetime.utcnow()
    }
    res = users_col().insert_one(doc)
    print(f"✅ Enrolled new user '{name}' with _id={res.inserted_id}")
    return res.inserted_id

def cosine_sim(a: np.ndarray, b: np.ndarray):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

def identify_speaker(embedding: np.ndarray):
    """
    Returns (matched_user_id:ObjectId|None, matched_user_name:str|None, score:float)
    """
    try:
        candidates = fetch_all_users()
        if not candidates:
            return (None, None, 0.0)
        # compute sims
        scored = []
        for u in candidates:
            emb_u = np.array(u["embedding"], dtype=np.float32)
            s = cosine_sim(embedding, emb_u)
            scored.append((u["_id"], u["name"], s))
        # sort desc
        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[0]
        print("🔎 Speaker top matches:", [(str(_id), name, round(sc, 3)) for _id, name, sc in scored[:SPEAKER_TOP_K]])
        if top[2] >= SPEAKER_SIM_THRESHOLD:
            return top  # (id, name, score)
        return (None, None, top[2])
    except Exception as e:
        print(f"Speaker identification error: {e}")
        return (None, None, 0.0)

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

# === CHANGED: store/retrieve memories now scoped to user_id
def store_memories_thread(facts_to_store, db, user_id: ObjectId | None):
    """Stores a list of facts in MongoDB in a background thread."""
    if db is None:
        return
    if not facts_to_store:
        print("➡️ No new facts to store.")
        return
    try:
        docs = [{
            "timestamp": datetime.utcnow(),
            "user_id": user_id,
            "fact": fact
        } for fact in facts_to_store]
        memories_col().insert_many(docs)
        print(f"✅ Stored {len(facts_to_store)} new memory snippets for user_id={user_id}.")
    except Exception as e:
        print(f"Error storing memories in background: {e}")

def retrieve_memories(db, user_id: ObjectId | None):
    """Retrieves the most recent memories from MongoDB for a user."""
    if db is None:
        return ""
    try:
        q = {"user_id": user_id} if user_id else {"user_id": None}
        docs = memories_col().find(q).sort("timestamp", -1).limit(10)
        memories_string = "".join([f"- {doc['fact']}\n" for doc in docs])
        return memories_string
    except Exception as e:
        print(f"Error retrieving memories: {e}")
        return ""

# === NEW: simple Tamil prompts
TAMIL_UNKNOWN_SPEAKER = "யாருனு தெரியலே. உங்க பெயர் சொல்லுவீங்களா? உங்க குரல் நினைவில் வைச்சுக்கலாமா?"
TAMIL_ENROLLED = lambda name: f"{name}, உங்க குரலை நான் நினைவில் வைச்சுக்கிட்டேன். இனிமேல் உங்களை உடனடியாக அறிஞ்சுக்குவேன்!"
TAMIL_WELCOME_BACK = lambda name: f"{name}, வணக்கம்! உங்க பழைய நினைவுகள் எல்லாம் லோட் பண்றேன்."
TAMIL_NOT_SURE = "குரல் கொஞ்சம் தெளிவா இல்லை. ஒரே முறை தெளிவா பேச முடியுமா?"

def start_conversation(recognizer, source):
    """Handles the actual conversation after the wake word is detected."""
    short_term_history = []
    current_user_id = None
    current_user_name = None

    # === NEW: Ask user to say anything for 3–4 sec to capture voiceprint
    set_status("Listening for speaker identification sample...")
    print("\n🎧 Please speak for 3–4 seconds for voice identification...")
    try:
        audio_for_id = recognizer.listen(source, phrase_time_limit=4, timeout=5)
        emb = compute_speaker_embedding(audio_for_id)
        if emb is not None:
            user_id, user_name, score = identify_speaker(emb)
            if user_id is not None:
                current_user_id = user_id
                current_user_name = user_name
                print(f"✅ Recognized speaker: {user_name} (score={score:.3f})")
                synthesize_speech_elevenlabs(TAMIL_WELCOME_BACK(user_name))
            else:
                print(f"👤 Unknown speaker (best score={score:.3f}).")
                synthesize_speech_elevenlabs(TAMIL_UNKNOWN_SPEAKER)
                # get their name
                try:
                    name_audio = recognizer.listen(source, phrase_time_limit=4, timeout=5)
                    # transcribe name quickly (Tamil or English names okay)
                    said_name = transcribe_with_whisper(name_audio)
                    said_name = re.sub(r"[^A-Za-z\u0B80-\u0BFF\s]", "", said_name).strip()
                    if said_name:
                        # enroll with earlier embedding if available, else compute on name audio
                        enroll_emb = emb if emb is not None else compute_speaker_embedding(name_audio)
                        if enroll_emb is not None:
                            current_user_id = enroll_user(said_name, enroll_emb)
                            current_user_name = said_name
                            synthesize_speech_elevenlabs(TAMIL_ENROLLED(said_name))
                        else:
                            synthesize_speech_elevenlabs(TAMIL_NOT_SURE)
                    else:
                        synthesize_speech_elevenlabs(TAMIL_NOT_SURE)
                except Exception as e:
                    print(f"Name capture error: {e}")
                    synthesize_speech_elevenlabs(TAMIL_NOT_SURE)
        else:
            print("⚠️ Could not compute embedding; proceeding anonymously.")
    except sr.WaitTimeoutError:
        print("No speech for ID sample; proceeding anonymously.")

    # kick off first response (same as before)
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

            user_memories = retrieve_memories(db, current_user_id)
            if user_memories:
                print(f"🧠 Retrieved memories for user_id={current_user_id}:\n{user_memories}")
                api_messages.append({"role": "system", "content": f"<memories>\n{user_memories}\n</memories>"})

            # optionally tell model the user's name if we have it
            if current_user_name:
                api_messages.append({"role": "system", "content": f"The user's name is {current_user_name}."})

            api_messages.extend(short_term_history)
            api_messages.append({"role": "user", "content": transcription})

            gpt_structured_response = generate_and_parse_gpt_response(api_messages)
            gpt_response_text = gpt_structured_response.get("response", "மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்.")
            facts_to_store = gpt_structured_response.get("facts_to_store", [])

            if db is not None and facts_to_store:
                memory_thread = threading.Thread(target=store_memories_thread, args=(facts_to_store, db, current_user_id))
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
    pygame.mixer.init()

    mic_index = 3
    mic_name = "Microphone (Realtek(R) Audio)"
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
