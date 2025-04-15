import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model_utils import load_config

def load_and_preprocess_image(img_path, img_size):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_single_image(model, img_path, img_size, class_names):
    img_array = load_and_preprocess_image(img_path, img_size)
    predictions = model.predict(img_array)[0]
    pred_index = np.argmax(predictions)
    return class_names[pred_index], float(predictions[pred_index]), dict(zip(class_names, map(float, predictions)))

def predict_batch(config_path, model_path, input_dir):
    config = load_config(config_path)
    model, img_size, class_names = load_model(model_path), config['model']['img_size'], sorted(os.listdir(config['data']['train_path']))
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    os.makedirs('predictions', exist_ok=True)
    
    results = [{
        'file_name': os.path.basename(img_path),
        'predicted_class': pred_class,
        'confidence': confidence,
        'probabilities': probabilities
    } for img_path in image_files for pred_class, confidence, probabilities in [predict_single_image(model, img_path, img_size, class_names)]]
    
    with open('predictions/prediction_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Processed {len(results)} images. Results saved to prediction_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with VGG model")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model", default="saved_models/best_model.h5", help="Path to saved model")
    parser.add_argument("--input", required=True, help="Directory containing images to predict")
    args = parser.parse_args()
    predict_batch(args.config, args.model, args.input)
