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
    """Load and preprocess a single image for model prediction."""
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_single_image(model, img_path, img_size, class_names):
    """Predict the class of a single image."""
    img_array = load_and_preprocess_image(img_path, img_size)
    predictions = model.predict(img_array)[0]
    pred_index = np.argmax(predictions)
    return class_names[pred_index], float(predictions[pred_index]), dict(zip(class_names, map(float, predictions)))

def main(config_path, model_path, img_path):
    """Load model, process image, make prediction, and save results."""
    if not os.path.isfile(img_path):
        print(f"Error: The file '{img_path}' does not exist.")
        return

    config = load_config(config_path)
    model = load_model(model_path)
    img_size = config['model']['img_size']
    class_names = sorted(os.listdir(config['data']['train_path']))

    pred_class, confidence, probabilities = predict_single_image(model, img_path, img_size, class_names)
    
    result = {
        'file_name': os.path.basename(img_path),
        'predicted_class': pred_class,
        'confidence': confidence,
        'probabilities': probabilities
    }

    print(json.dumps(result, indent=4))  # Print result
    os.makedirs('predictions', exist_ok=True)  # Create directory if it doesn't exist
    # Save the result to a JSON file
    with open('predictions/single_prediction_result.json', 'w') as f:
        json.dump(result, f, indent=4)

    print("Prediction saved to single_prediction_result.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a single image using a trained model")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model", default="saved_models/best_model.h5", help="Path to saved model")
    parser.add_argument("--input", required=True, help="Path to the single image file")
    args = parser.parse_args()

    main(args.config, args.model, args.input)
