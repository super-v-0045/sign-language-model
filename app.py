import base64
import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Define AI Variables (Empty at startup) ---
model = None
mp_hands = None
hands = None
MAX_LEN = 84 

# --- Global variables for text generation ---
generated_text = ""
last_prediction = None
frames_held = 0
REQUIRED_FRAMES = 10  

# --- LAZY LOADER: Only runs on the first frame received ---
def init_ai_models():
    global model, mp_hands, hands
    if model is None:
        print("Initializing AI Models for the first time...")
        model_dict = pickle.load(open('model.p', 'rb'))
        model = model_dict['model']
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=2, 
            min_detection_confidence=0.3
        )
        print("AI Models successfully loaded into memory!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global generated_text, last_prediction, frames_held
    
    # 1. Ensure models are loaded before processing
    init_ai_models()
    
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data received'}), 400
        
    try:
        header, encoded = data['image'].split(',', 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Decoding failed'}), 400

        data_aux = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        predicted_character = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
                
                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            if len(data_aux) < MAX_LEN:
                data_aux.extend([0.0] * (MAX_LEN - len(data_aux)))
            else:
                data_aux = data_aux[:MAX_LEN]

            prediction = model.predict([np.asarray(data_aux)]) 
            predicted_character = str(prediction[0])

            if predicted_character == last_prediction:
                frames_held += 1
            else:
                frames_held = 0
                last_prediction = predicted_character

            if frames_held == REQUIRED_FRAMES:
                generated_text += predicted_character
        else:
            last_prediction = None
            frames_held = 0

        return jsonify({
            'prediction': predicted_character,
            'accumulated_text': generated_text
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_text')
def get_text():
    return jsonify({'text': generated_text})

@app.route('/clear_text', methods=['POST'])
def clear_text():
    global generated_text
    generated_text = ""
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True)
