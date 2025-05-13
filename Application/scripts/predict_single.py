import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model_utils import load_config


def preprocess_image(image_obj, img_size):
    """
    Preprocess a PIL Image for model prediction.
    
    Args:
        image_obj (PIL.Image): Image to preprocess
        img_size (int): Target image size (width and height)

    Returns:
        np.array: Preprocessed image array ready for prediction
    """
    image_resized = image_obj.resize((img_size, img_size))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_image_from_object(model, image_obj, img_size, class_names):
    
    preprocessed = preprocess_image(image_obj, img_size)
    predictions = model.predict(preprocessed)[0]
    pred_index = np.argmax(predictions)
    return (
        class_names[pred_index],
        float(predictions[pred_index]),
        dict(zip(class_names, map(float, predictions)))
    )


def predict_image_from_path(model, img_path, img_size, class_names):
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    image = Image.open(img_path).convert("RGB")
    pred_class, confidence, probs = predict_image_from_object(model, image, img_size, class_names)

    return {
        "file_name": os.path.basename(img_path),
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": probs
    }


def main(config_path, model_path, img_path):
    """
    CLI-based usage for predicting a single image.
    """
    config = load_config(config_path)
    model = tf.keras.models.load_model(model_path)
    img_size = config['model']['img_size']
    class_names = sorted(os.listdir(config['data']['train_path']))

    result = predict_image_from_path(model, img_path, img_size, class_names)

    # Print and save result
    print(json.dumps(result, indent=4))
    os.makedirs("predictions", exist_ok=True)
    with open("predictions/single_prediction_result.json", "w") as f:
        json.dump(result, f, indent=4)
    print("Prediction saved to predictions/single_prediction_result.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a single image using a trained model")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model", default="saved_models/best_model.h5", help="Path to saved model")
    parser.add_argument("--input", required=True, help="Path to the single image file")
    args = parser.parse_args()

    main(args.config, args.model, args.input)
