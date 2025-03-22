from flask import Flask, request, jsonify
from flask_cors import CORS  
import numpy as np
import librosa
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model  

app = Flask(__name__)
CORS(app)  # Enable CORS

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model globally to prevent reloading on every request
model = load_model("audio.h5", compile=False)

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
    
    features = extract_features(file_path)
    prediction = model.predict(features).flatten()[0]  
    result = {"prediction": "Deepfake" if prediction >= 0.5 else "Real"}
    
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000)) 
    app.run(host='0.0.0.0', port=port)
