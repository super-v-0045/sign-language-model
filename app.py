import base64
import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 1. Load the model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# 2. Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.3
)

MAX_LEN = 84 

# Global tracking variables
generated_text = ""
last_prediction = None
frames_held = 0
REQUIRED_FRAMES = 15 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global generated_text, last_prediction, frames_held
    
    try:
        # 1. Catch missing or malformed JSON payloads safely
        data = request.json
        if not data or 'image' not in data or not data['image']:
            return jsonify({'processed_image': '', 'text': generated_text})
            
        image_data = data['image'].split(",")[1]
        
        # Decode base64 back into an OpenCV image matrix
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 2. CRITICAL SAFETY CHECK: If the image didn't decode or is blank, skip execution
        if frame is None or frame.size == 0:
            return jsonify({'processed_image': '', 'text': generated_text})
        
        # Mirroring the frame horizontally
        frame = cv2.flip(frame, 1)

        data_aux = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        predicted_character = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
                
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            # Guard tracking length thresholds
            if len(data_aux) < MAX_LEN:
                data_aux.extend([0.0] * (MAX_LEN - len(data_aux)))
            else:
                data_aux = data_aux[:MAX_LEN]

            # Predict sign
            prediction = model.predict([np.asarray(data_aux)]) 
            predicted_character = str(prediction[0])

            # Text accumulation logic
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

        # Draw current live text onto the frame
        cv2.putText(frame, f"Sign: {predicted_character if predicted_character else 'None'}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode processed image back to base64 string
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'processed_image': f"data:image/jpeg;base64,{jpg_as_text}",
            'text': generated_text
        })

    except Exception as e:
        # Catch unexpected runtime glitches without killing the Render container thread
        print(f"Server caught an inner handling error: {str(e)}")
        return jsonify({'processed_image': '', 'text': generated_text})

@app.route('/clear_text', methods=['POST'])
def clear_text():
    global generated_text
    generated_text = ""
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True)
