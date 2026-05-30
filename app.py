import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

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

        landmarks = data['landmarks']  # list of {x, y} objects
        
        data_aux = []
        x_ = [lm['x'] for lm in landmarks]
        y_ = [lm['y'] for lm in landmarks]
        for lm in landmarks:
            data_aux.append(lm['x'] - min(x_))
            data_aux.append(lm['y'] - min(y_))

        if len(data_aux) < MAX_LEN:
            data_aux.extend([0.0] * (MAX_LEN - len(data_aux)))
        else:
            data_aux = data_aux[:MAX_LEN]

        prediction = model.predict([np.asarray(data_aux)])
        return jsonify({'prediction': str(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)