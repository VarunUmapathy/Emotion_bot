from flask import Flask, render_template, jsonify
import threading
import os
import sys

# You will import your main bot script here.
# Make sure this is in the same folder as app.py
import working_pipeline

# Add the project's root directory to the sys.path
# This helps Flask find the templates and static folders
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

app = Flask(__name__)

# A flag and event to manage the bot's thread lifecycle
bot_thread = None
stop_event = threading.Event()

@app.route('/')
def index():
    """Serves the main web page."""
    return render_template('index.html')

@app.route('/start_bot', methods=['POST'])
def start_bot():
    """Starts the conversational bot in a new thread."""
    global bot_thread, stop_event
    if bot_thread and bot_thread.is_alive():
        return jsonify({"status": "Bot is already running"})
    
    print("Starting conversational agent...")
    stop_event.clear() # Clear the stop event to allow the bot to start
    bot_thread = threading.Thread(target=working_pipeline.wake_word_detection_loop, args=(stop_event,))
    bot_thread.start()
    return jsonify({"status": "Bot started"})

@app.route('/stop_bot', methods=['POST'])
def stop_bot():
    """Stops the conversational bot."""
    global bot_thread, stop_event
    if bot_thread and bot_thread.is_alive():
        print("Stopping conversational agent...")
        stop_event.set() # Set the event to signal the thread to stop
        bot_thread.join(timeout=10) # Wait for the thread to finish gracefully
        return jsonify({"status": "Bot stopping"})
    return jsonify({"status": "Bot is not running"})


if __name__ == '__main__':
    # You can set debug=False for production, but it's good for development.
    app.run(debug=True, host='0.0.0.0', port=5000)