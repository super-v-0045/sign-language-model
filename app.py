import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify

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

# --- Global variables for text generation ---
generated_text = ""
last_prediction = None
frames_held = 0
REQUIRED_FRAMES = 15  # The user must hold the sign for 15 frames (~0.5 seconds) to type a letter

def generate_frames():
    global generated_text, last_prediction, frames_held
    
    # Open local webcam connection
    cap = cv2.VideoCapture(0)
    
    # IF CAMERA IS NOT FOUND (e.g., Cloud Server environment)
    if not cap.isOpened():
        print("Webcam hardware not detected. Streaming empty server canvas.")
        while True:
            # Generate a clean black frame as a placeholder on production
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "Server Live: Use API or Frontend for Streaming", (30, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # IF CAMERA IS FOUND (e.g., Local laptop development)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        data_aux = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

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

            if len(data_aux) < MAX_LEN:
                data_aux.extend([0.0] * (MAX_LEN - len(data_aux)))
            else:
                data_aux = data_aux[:MAX_LEN]

            prediction = model.predict([np.asarray(data_aux)]) 
            predicted_character = str(prediction[0])

            # --- TEXT ACCUMULATION LOGIC ---
            if predicted_character == last_prediction:
                frames_held += 1
            else:
                frames_held = 0
                last_prediction = predicted_character

            # If the sign has been held consistently for REQUIRED_FRAMES, append it
            if frames_held == REQUIRED_FRAMES:
                generated_text += predicted_character

            # Display both current sign and the generated word on screen
            cv2.putText(frame, f"Sign: {predicted_character}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Text: {generated_text}", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)
        
        else:
            # If no hands are detected, reset tracker
            last_prediction = None
            frames_held = 0

        # Encode and stream output
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


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
