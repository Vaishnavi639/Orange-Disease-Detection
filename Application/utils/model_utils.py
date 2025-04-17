import os
import yaml
import mlflow
import mlflow.keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_callbacks(config):
    """
    Create training callbacks.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        list: List of Keras callbacks
    """
    callbacks = []
    
    # Create directories if they don't exist
    model_dir = config['paths']['model_save_path']
    logs_dir = config['paths']['logs_path']
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Model checkpoint to save the best model
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_dir, 'best_model.h5'),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(model_checkpoint)
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['training']['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=config['training']['reduce_lr_factor'],
        patience=config['training']['reduce_lr_patience'],
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard logging
    tensorboard = TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1,
        write_graph=True
    )
    callbacks.append(tensorboard)
    
    return callbacks

def evaluate_model(model, test_generator, class_names):
    """
    Evaluate the model and generate evaluation metrics.
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        class_names: List of class names
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Get predictions
    test_generator.reset()
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # True labels
    y_true = test_generator.classes
    
    # Calculate metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the confusion matrix
    confusion_matrix_path = 'confusion_matrix.png'  # Path to your confusion matrix file
    plt.savefig(confusion_matrix_path)
    plt.close()  # Close the plot to avoid issues

    # Log the confusion matrix as an artifact in MLflow
    mlflow.log_artifact(confusion_matrix_path)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': float(report['accuracy']),
        'precision_macro': float(report['macro avg']['precision']),
        'recall_macro': float(report['macro avg']['recall']),
        'f1_score_macro': float(report['macro avg']['f1-score']),
    }

    # Flatten per-class metrics
    for class_name in class_names:
        metrics[f"{class_name}_precision"] = float(report[class_name]['precision'])
        metrics[f"{class_name}_recall"] = float(report[class_name]['recall'])
        metrics[f"{class_name}_f1_score"] = float(report[class_name]['f1-score'])

    return metrics
