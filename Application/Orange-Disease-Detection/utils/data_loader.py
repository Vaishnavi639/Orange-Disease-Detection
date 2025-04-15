import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(config):
    """
    Create data generators for training, validation, and testing.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (train_generator, test_generator, class_names)
    """
    # Data augmentation for training
    if config['data']['augmentation']:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Only rescaling for validation and test
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    img_size = config['model']['img_size']
    batch_size = config['training']['batch_size']
    
    train_generator = train_datagen.flow_from_directory(
        config['data']['train_path'],
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    
    test_generator = test_datagen.flow_from_directory(
        config['data']['test_path'],
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get class names from the training generator
    class_names = list(train_generator.class_indices.keys())
    
    return train_generator, test_generator, class_names