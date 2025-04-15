import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model


def create_model(version='16', img_size=224, num_classes=4):
    if version == '16':
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
    elif version == '19':
        base_model = VGG19(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
    else:
        raise ValueError("Invalid VGG version. Choose '16' or '19'.")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False  # Freeze pre-trained layers

    return model
