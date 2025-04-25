# app.py
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__, 
           template_folder='frontend',
           static_folder='frontend',
           static_url_path='')

# Load your trained model
model = tf.keras.models.load_model('saved_models/best_model.h5')

# Define disease classes (update these based on your model's classes)
disease_classes = ['Healthy', 'Citrus Greening', 'Black Spot', 'Canker']  # Example classes

def preprocess_image(image):
    # Resize to match your model's input size (e.g., 224x224)
    image = image.resize((224, 224))
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')
        
        # Save uploaded image (optional)
        upload_dir = os.path.join('frontend', 'uploads')
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        image.save(os.path.join(upload_dir, 'uploaded_image.jpg'))
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        # Prepare response
        result = {
            'disease': disease_classes[predicted_class],
            'confidence': confidence
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
