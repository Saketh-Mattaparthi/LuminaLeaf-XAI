import os
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json

# Configuration
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 20
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/validation'
MODEL_SAVE_PATH = 'saved_model/medical_plant_model.keras'

def build_dataset(directory):
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} not found. Cannot load dataset.")
        return None, None

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        label_mode='categorical'
    )
    class_names = dataset.class_names
    
    # Prefetching for performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, class_names

def create_model(num_classes):
    # Data Augmentation layer
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal_and_vertical'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.1),
    ])

    # Base model InceptionV3
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    
    # Freeze the base model
    base_model.trainable = False

    # Create new model on top
    inputs = tf.keras.Input(shape=(299, 299, 3))
    x = data_augmentation(inputs)
    
    # Preprocessing expected by InceptionV3
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def main():
    print("Loading datasets...")
    train_ds, class_names = build_dataset(TRAIN_DIR)
    val_ds, _ = build_dataset(VAL_DIR)

    if train_ds is None:
        print("Exiting training script. Please place data in 'data/train' and 'data/validation'.")
        return

    # Save labels mapping
    labels_dict = {i: name for i, name in enumerate(class_names)}
    with open('labels.json', 'w') as f:
        json.dump(labels_dict, f, indent=4)
    print(f"Saved class labels to labels.json: {labels_dict}")

    model = create_model(num_classes=len(class_names))
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy')
    ]

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Optionally, write code here to plot and save accuracy/loss curves using matplotlib

if __name__ == '__main__':
    main()
