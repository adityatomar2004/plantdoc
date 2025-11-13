import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io

print(f"TensorFlow Version: {tf.__version__}")

# --- 1. Initialize Flask App ---
app = Flask(__name__)
CORS(app) 

# --- 2. Load Model and Class Names ---
MODEL_PATH = 'plant_disease_model.h5'
CLASS_NAMES_PATH = 'plant_disease_classes.txt'
IMG_SIZE = 224 # Must be the same size as training

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [name.strip() for name in f.readlines()]
    print("\n*** Model and class names loaded successfully. ***")
except Exception as e:
    print(f"Error loading model or class names: {e}")
    model = None
    class_names = []

# --- 3. Preprocessing Function ---
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# --- 4. Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or not class_names:
        return jsonify({'error': 'Model not loaded'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction (this is fast on a CPU)
        prediction = model.predict(processed_image)
        predicted_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        predicted_class = class_names[predicted_index]
        
        # Format class name
        formatted_class = predicted_class.replace('___', ' ')
        
        return jsonify({
            'disease': formatted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': f'Prediction error: {e}'}), 500

# --- 5. Run the App ---
if __name__ == '__main__':
    print("\n*** Starting Flask server at http://127.0.0.1:5000 ***")
    app.run(debug=False, port=5000)