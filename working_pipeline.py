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

from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "YOUR_ELEVENLABS_API_KEY")

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
- Respond in 1-2 sentences.
- Speak in simple, conversational Tamil, not literary Tamil.
- Respond with emotions like sadness, happiness, anger, loneliness, etc., as appropriate.
- Be a good friend and offer gentle comfort when needed.
- Your answers should relate to real-life feelings and situations.
- If the user wants to end the conversation by saying words like "bye," "see you," "அப்புறம் பாக்கலாம்," or "போய்ட்டு வரேன்," you should give a brief, kind farewell and then end your response with the special token [END_CONVERSATION].
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
        
        ffplay_path = r"C:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin\ffplay.exe"
        subprocess.run([ffplay_path, "-autoexit", output_path])
        
        print("Playback completed.")
        return output_path
    except Exception as e:
        print(f"Error during ElevenLabs TTS synthesis or playback: {e}")
        return None

# --- MODIFIED: Refined the prompt for better extraction and filtering ---
def extract_and_store_memories(latest_message, db):
    if not db:
        return
    
    extraction_prompt = f"""
    You are a memory extraction bot. Your job is to extract key facts, personal details, and stories from the following user message. Respond with a simple list of facts.

    If the user message does not contain any personal facts, details, or stories, respond with the exact phrase "NO_FACTS".

    Example 1:
    User message: "என் பெயர் ராதா. நான் நேற்று தான் வேலை தேடுறதுக்கு ஒரு கம்பெனிக்கு போயிருந்தேன்."
    Extraction:
    - User's name is ராதா.
    - User was looking for a job yesterday.
    
    Example 2:
    User message: "இன்று நான் கால்பந்து விளையாடினேன்."
    Extraction:
    - User played football today.
    
    Example 3:
    User message: "காலையில் சாப்பிட்டேன்."
    Extraction:
    NO_FACTS
    
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
        
        # --- NEW: Filter out non-essential responses ---
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

def main_conversation_loop():
    """Main function to run the conversational agent."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting conversational agent with long-term memory...\n")
    
    short_term_history = []
    
    recognizer = sr.Recognizer()
    
    with sr.Microphone(device_index=1) as source:
        print("Adjusting for ambient noise... Please wait 5 seconds.")
        recognizer.adjust_for_ambient_noise(source, duration=5)
        print("Adjustment complete. You can start talking.")
        
        while True:
            try:
                print("\nListening for your voice...")
                audio = recognizer.listen(source, phrase_time_limit=10, timeout=3)
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
                
                # --- NEW: Start a separate thread for database writes ---
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
                print("Timeout: No speech detected for 3 seconds. Listening again...")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                time.sleep(1)

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