from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import librosa
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model  # Correct way to load H5 model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved model (ensure the correct path)
MODEL_PATH = "audio.h5"
model = load_model(MODEL_PATH)  # Use TensorFlow/Keras to load the H5 file

# Define an upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to extract features from audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean.reshape(1, -1)

@app.route('/')
def home():
    return "Deepfake Voice Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Extract features and predict
    features = extract_features(file_path)
    prediction = model.predict(features).flatten()[0]  # Ensure a single scalar value
    
    result = {"prediction": "Deepfake" if prediction >= 0.5 else "Real"}
    print(result)  # Debugging output
    return jsonify(result)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port
    app.run(host='0.0.0.0', port=port)
