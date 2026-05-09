import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from config import Config

# Load Labels
try:
    with open('labels.json', 'r') as f:
        labels_dict = json.load(f)
except Exception as e:
    # Fallback to defaults
    labels_dict = {"0": "Aloe_Vera", "1": "Neem", "2": "Tulsi"}

# Attempt to load model globally to avoid loading on every request
try:
    if os.path.exists(Config.MODEL_PATH):
        model = load_model(Config.MODEL_PATH)
        print("Model loaded successfully.")
    else:
        model = None
        print(f"Warning: Model not found at {Config.MODEL_PATH}. Will run in mock mode.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Apply InceptionV3 preprocessing
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array

def predict_plant(img_path):
    """
    Returns prediction dict: {'predicted_class': 'PlantName', 'confidence': 0.95, 'is_mock': False}
    """
    if model is None:
        # MOCK MODE - For demonstrating UI without training a full model
        class_idx = str(random.choice(list(labels_dict.keys())))
        confidence = round(random.uniform(0.75, 0.99), 4)
        return {
            'predicted_class': labels_dict[class_idx],
            'confidence': confidence,
            'is_mock': True
        }

    # REAL PREDICTION
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    
    predicted_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_idx])
    
    predicted_class = labels_dict.get(str(predicted_idx), "Unknown")
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'is_mock': False
    }
