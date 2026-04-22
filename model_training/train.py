import os
import tensorflow as tf
import keras
from keras.applications import EfficientNetB0
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 50
DATA_DIR = "model_training/data/processed"  # Use processed data from prepare_data.py

def build_model(num_classes):
    """Builds a model using EfficientNetB0 for transfer learning."""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze the base model
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found. Run prepare_data.py first.")
        return

    # Data Augmentation - This is CRITICAL for high accuracy on skin images
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    test_valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = test_valid_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'valid'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    num_classes = train_generator.num_classes
    model = build_model(num_classes)

    # Callbacks
    checkpoint = ModelCheckpoint('best_skin_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    print("--- Starting Initial Training (Base Model Frozen) ---")
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # FINE TUNING: Unfreeze the last few layers of EfficientNet
    print("--- Starting Fine Tuning (Base Model Unfrozen) ---")
    for layer in model.layers[-20:]:  # Unfreeze last 20 layers
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Lower LR for fine tuning
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history_fine = model.fit(
        train_generator,
        epochs=EPOCHS - 15,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # Save final model (replace any existing model)
    import os
    new_model_path = 'new_model.h5'
    if os.path.exists(new_model_path):
        os.remove(new_model_path)
    model.save(new_model_path)
    print(f"Training finished. Model saved as '{new_model_path}'")

    # Plot results
    plot_history(history, history_fine)

def plot_history(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig('training_report.png')
    plt.show()

if __name__ == "__main__":
    train()
