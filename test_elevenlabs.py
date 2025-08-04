import os
from elevenlabs.client import ElevenLabs
from elevenlabs import save

# Set your ElevenLabs API key (use environment variable or hardcode for testing)
ELEVENLABS_API_KEY = "sk_1b290621776c4f8722821defc4fcafb0187422a93a4a522d"
ELEVENLABS_VOICE_ID = "C2RGMrNBTZaNfddRPeRH"  # Replace with a valid voice ID if needed
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"

# Initialize the ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Text to convert to speech (sample Tamil text)
sample_text = "நான் உன்னை நேசிக்கிறேன், நல்லா இரு!"  # "I love you, be well!"

# Output file path
output_file = "test_elevenlabs_output.mp3"

# Function to synthesize speech
def synthesize_speech(text, output_path):
    print(f"🔊 Converting text to audio using ElevenLabs: {text}")
    try:
        audio = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL_ID,
            output_format="mp3_44100_128",
        )

        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        print(f"✅ Audio saved as: {output_path}")
        return True
    except Exception as e:
        print(f"Error during ElevenLabs TTS synthesis: {e}")
        return False

# Run the synthesis
if __name__ == "__main__":
    success = synthesize_speech(sample_text, output_file)
    if success:
        print("Test completed successfully. Play the output file to verify.")
    else:
        print("Test failed. Check your API key and configuration.")