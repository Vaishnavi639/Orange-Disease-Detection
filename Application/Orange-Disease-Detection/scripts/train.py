import os
import sys
import argparse
import tensorflow as tf
import mlflow
import mlflow.keras
from tensorflow.keras.optimizers import Adam, SGD
from models.cnn_model import create_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_loader import create_data_generators
from utils.model_utils import load_config, create_callbacks, evaluate_model

def train_model(config_path):
    """Train the model based on the provided config."""
    config = load_config(config_path)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    train_gen, test_gen, class_names = create_data_generators(config)
    model = create_model(config['model']['version'], config['model']['img_size'], config['model']['num_classes'])
    
    optimizer = Adam(config['training']['learning_rate']) if config['training']['optimizer'].lower() == 'adam' \
                else SGD(config['training']['learning_rate'], momentum=0.9)
    
    model.compile(optimizer=optimizer, loss=config['training']['loss'], metrics=['accuracy'])
    
    callbacks = create_callbacks(config)
    with mlflow.start_run():
        mlflow.log_params({key: config['model'][key] for key in ['architecture', 'img_size', 'num_classes']})
        mlflow.log_params({key: config['training'][key] for key in ['batch_size', 'learning_rate', 'optimizer', 'loss']})
        
        model.fit(train_gen, epochs=config['training']['epochs'], callbacks=callbacks)
        
        metrics = evaluate_model(model, test_gen, class_names)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.keras.log_model(model, "model")
        model.save(os.path.join(config['paths']['model_save_path'], 'final_model.h5'))
        print(f"Training completed. Test accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    train_model(args.config)
