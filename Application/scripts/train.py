import tensorflow as tf
import mlflow
import mlflow.keras
import os
import sys
import argparse
from tensorflow.keras.optimizers import Adam, SGD
from models.cnn_model import create_model
from dotenv import load_dotenv
from models.cnn_model import create_model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_loader import create_data_generators
from utils.model_utils import load_config, create_callbacks, evaluate_model

load_dotenv()

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
print(f"MLflow URI: {mlflow_tracking_uri}")

def train_model(config_path):
    """Train the model based on the provided config."""
    config = load_config(config_path)
<<<<<<< HEAD

    # Make sure tracking directory exists if using file-based MLflow store
    os.makedirs(config['mlflow']['artifact_location'], exist_ok=True)

    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

=======

    # Get tracking URI from environment or config
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config['mlflow']['tracking_uri'])
    mlflow.set_tracking_uri(tracking_uri)

    # Print the tracking URI being used
    print(f"Tracking URI used: {tracking_uri}")
# Get experiment name from environment or config
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", config['mlflow']['experiment_name'])
    mlflow.set_experiment(experiment_name)

>>>>>>> a3941852 (mlflow)
    train_gen, test_gen, class_names = create_data_generators(config)

    model = create_model(config['model']['version'], config['model']['img_size'], config['model']['num_classes'])

    optimizer = Adam(config['training']['learning_rate']) if config['training']['optimizer'].lower() == 'adam' \
                else SGD(config['training']['learning_rate'], momentum=0.9)

    model.compile(optimizer=optimizer, loss=config['training']['loss'], metrics=['accuracy'])
<<<<<<< HEAD

    callbacks = create_callbacks(config)

    with mlflow.start_run():
        # Log all parameters
        mlflow.log_params({
            'architecture': config['model']['architecture'],
            'img_size': config['model']['img_size'],
            'num_classes': config['model']['num_classes'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'optimizer': config['training']['optimizer'],
            'loss': config['training']['loss'],
            'epochs': config['training']['epochs']
        })

        # Train model
        model.fit(train_gen, epochs=config['training']['epochs'], callbacks=callbacks)

        # Evaluate
        metrics = evaluate_model(model, test_gen, class_names)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

        # Log model in MLflow
        mlflow.keras.log_model(model, "model")
# Save model locally too
        os.makedirs(config['paths']['model_save_path'], exist_ok=True)
        model.save(os.path.join(config['paths']['model_save_path'], 'final_model.h5'))
=======
    callbacks = create_callbacks(config)

    with mlflow.start_run(run_name=f"orange_disease_{config['model']['architecture']}_{config['model']['version']}"):
        # Log model parameters
        mlflow.log_params({key: config['model'][key] for key in ['architecture', 'img_size', 'num_classes']})
        mlflow.log_params({key: config['training'][key] for key in ['batch_size', 'learning_rate', 'optimizer', 'loss']})

        # Train the model and capture history
        history = model.fit(
            train_gen,
            epochs=config['training']['epochs'],
            validation_data=test_gen,
            callbacks=callbacks
        )

        # Log training history metrics
        for epoch, (acc, loss, val_acc, val_loss) in enumerate(zip(
            history.history['accuracy'],
            history.history['loss'],
            history.history.get('val_accuracy', []),
            history.history.get('val_loss', [])
        )):
            mlflow.log_metric("accuracy", acc, step=epoch)
            mlflow.log_metric("loss", loss, step=epoch)
            if val_acc:
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            if val_loss:
                mlflow.log_metric("val_loss", val_loss, step=epoch)

        # Evaluate and log metrics
        metrics = evaluate_model(model, test_gen, class_names)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

        # Log the model to MLflow
        print("Logging model to MLflow...")
        mlflow.keras.log_model(model, "model")
        print("Model logged!")

        # Save model locally
        os.makedirs(config['paths']['model_save_path'], exist_ok=True)
        model_save_path = os.path.join(config['paths']['model_save_path'], 'final_model.h5')
        model.save(model_save_path)

        # Log the saved model file as an artifact
        mlflow.log_artifact(model_save_path)

        # Log class names as artifact
        class_names_path = os.path.join(config['paths']['model_save_path'], 'class_names.txt')
        with open(class_names_path, 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")
        mlflow.log_artifact(class_names_path)
>>>>>>> a3941852 (mlflow)

        print(f"Training completed. Test accuracy: {metrics['accuracy']:.4f}")
        print(f"Model tracked in MLflow with run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    train_model(args.config)

                                                                                                                                                                      149,0-1       Bot

