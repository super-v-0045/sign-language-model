import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)

# Configured to allow cross-origin requests securely from your Render deployment
CORS(app, resources={
    r"/*": {
        "origins": ["https://sign-language-model-fbyr.onrender.com"],
        "supports_credentials": True
    }
})

# Load your classification model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

MAX_LEN = 84

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'landmarks' not in data:
            return jsonify({'error': 'No landmarks provided'}), 400

        landmarks = data['landmarks']  # Array of {x, y, (optional z)} from frontend
        
        data_aux = []
        # Safely extract x and y coordinates, ignoring z if present
        x_ = [lm['x'] for lm in landmarks if 'x' in lm]
        y_ = [lm['y'] for lm in landmarks if 'y' in lm]
        
        if not x_ or not y_:
            return jsonify({'error': 'Malformed landmarks dataset'}), 400

        # Relative coordinate normalisation against bounding box minimums
        for lm in landmarks:
            if 'x' in lm and 'y' in lm:
                data_aux.append(lm['x'] - min(x_))
                data_aux.append(lm['y'] - min(y_))

        # Enforce structural matching to MAX_LEN requirement expected by your model
        if len(data_aux) < MAX_LEN:
            data_aux.extend([0.0] * (MAX_LEN - len(data_aux)))
        else:
            data_aux = data_aux[:MAX_LEN]

        # Machine Learning Inference pass
        prediction = model.predict([np.asarray(data_aux)])
        return jsonify({'prediction': str(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
