import argparse
import os
import sys
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_loader import create_data_generators
from utils.model_utils import load_config, evaluate_model

def evaluate(config_path, model_path):
    """Evaluate a trained model."""
    config = load_config(config_path)
    train_gen, test_gen, class_names = create_data_generators(config)
    model = load_model(model_path)
    
    print("Evaluating model...")
    metrics = evaluate_model(model, test_gen, class_names)
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Evaluation metrics saved to evaluation_metrics.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model", default="saved_models/best_model.h5", help="Path to saved model")
    args = parser.parse_args()
    evaluate(args.config, args.model)
