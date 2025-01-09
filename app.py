from flask import Flask, render_template, request, jsonify
from query_data import process_query, process_image
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Store chat history
chat_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    try:
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400

        # Use the existing process_query function from query_data.py
        global chat_history
        result, chat_history = process_query(user_input, chat_history)
        
        if "error" in result:
            return jsonify({'error': result['error']}), 500
            
        return jsonify({'response': result['response']})
    except Exception as e:
        print(f"Error: {str(e)}")  # For debugging
        return jsonify({'error': str(e)}), 500

@app.route('/process_image', methods=['POST'])
def handle_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No image selected'}), 400
            
        # Save the uploaded image temporarily
        temp_path = "temp_image.png"
        image.save(temp_path)
        
        # Process the image using existing function
        result = process_image(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        if "error" in result:
            return jsonify({'error': result['error']}), 500
            
        return jsonify(result)
    except Exception as e:
        print(f"Error: {str(e)}")  # For debugging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)