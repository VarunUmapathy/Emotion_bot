# test_llm.py
import os
import json
from llama_cpp import Llama

# --- Configuration ---
# Set the path to your downloaded GGUF model file.
MODEL_FILE = "tamil-llama-7b-v0.1-q4_k_m.gguf"

if not os.path.exists(MODEL_FILE):
    print(f"Error: The model file '{MODEL_FILE}' was not found.")
    print("Please download the .gguf model file and place it in the same directory as this script.")
    exit()

# --- Initialize the local LLM ---
print("Initializing local LLM model...")
try:
    llm = Llama(model_path=MODEL_FILE, n_ctx=2048, verbose=False)
    print("✅ Local LLM loaded successfully.")
except Exception as e:
    print(f"❌ Error loading local LLM model: {e}")
    exit()

def get_local_llm_response(prompt):
    """Generates a response from the local LLM in two separate, sequential calls."""
    
    # --- Step 1: Get the conversational response first (plain text) ---
    # We use a system prompt focused ONLY on the conversational task.
    messages_response = [
        {"role": "system", "content": """You are a conversational agent speaking in Tamil. Your persona is a friend who is emotionally aware and gives life advice.
- Your responses must be in simple, everyday, and conversational Tamil.
- Respond in 1-3 sentences.
- Respond with emotions like sadness, happiness, anger, loneliness, etc., as appropriate.
- Be a good friend and offer gentle comfort when needed.
- Your answers should relate to real-life feelings and situations.
- If the user wants to end the conversation, give a brief, kind farewell and end with the special token [END_CONVERSATION].
- Try to use the user's name where appropriate, but if you don't know it, just use "நண்பா" (friend).
"""},
        {"role": "user", "content": prompt}
    ]
    
    print(f"\n🧠 Step 1: Getting conversational response for prompt: '{prompt}'")
    try:
        response_completion = llm.create_chat_completion(
            messages=messages_response,
            temperature=0.7,
            max_tokens=200,
            stream=False
        )
        conversational_response = response_completion['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"❌ Error getting conversational response: {e}")
        conversational_response = "மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்."

    # --- Step 2: Extract facts with a separate, focused prompt ---
    # This prompt is highly structured to force a consistent output.
    messages_facts = [
        {"role": "system", "content": """You are a memory extraction bot. Your job is to extract key personal facts, details, and stories from the following user message. Respond with a simple list of facts.
If the user message does not contain any personal facts, details, or stories, respond with the exact phrase "NO_FACTS".
Example:
User message: "My name is Arun and I work as a software developer. My favorite food is dosa."
Extraction:
- User's name is Arun.
- User works as a software developer.
- User's favorite food is dosa.
"""},
        {"role": "user", "content": prompt}
    ]

    print(f"\n🧠 Step 2: Extracting facts from prompt: '{prompt}'")
    try:
        facts_completion = llm.create_chat_completion(
            messages=messages_facts,
            temperature=0.3,
            max_tokens=200,
            stream=False
        )
        extracted_memories = facts_completion['choices'][0]['message']['content'].strip()
        
        # Filter "NO_FACTS" and format the output
        if extracted_memories == "NO_FACTS" or not extracted_memories:
            facts_to_store = []
        else:
            facts_to_store = extracted_memories.split('\n')
    except Exception as e:
        print(f"❌ Error extracting facts: {e}")
        facts_to_store = []

    # --- Combine the results into the final JSON object ---
    final_response = {
        "response": conversational_response,
        "facts_to_store": facts_to_store
    }
    
    return json.dumps(final_response, ensure_ascii=False)

# --- Main execution ---
if __name__ == "__main__":
    test_prompt = "I feel sad today."
    
    llm_response_json = get_local_llm_response(test_prompt)

    if llm_response_json:
        try:
            response_data = json.loads(llm_response_json)
            print("\n✅ Response received:")
            print(json.dumps(response_data, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("\n❌ Received an invalid JSON response:")
            print(llm_response_json)
    else:
        print("\n❌ No response received from the local model.")
