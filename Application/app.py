from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf
import os

from scripts.predict_single import predict_image_from_object  # ✅ import your new function

app = Flask(__name__, 
           template_folder='frontend',
           static_folder='frontend',
           static_url_path='')

# Load model and metadata
model = tf.keras.models.load_model('saved_models/best_model.h5')
img_size = 224  # set based on config
disease_classes = ['Blackspot', 'Canker', 'Healthy', 'Citrus Grenning']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')

        # Save uploaded image (optional)
        upload_dir = os.path.join('frontend', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        image.save(os.path.join(upload_dir, 'uploaded_image.jpg'))

        # ✅ Use the function from predict_single.py
        predicted_class, confidence, _ = predict_image_from_object(model, image, img_size, disease_classes)

        return jsonify({
            'disease': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
