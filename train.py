import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import shutil
from sklearn.model_selection import train_test_split
import multiprocessing
from multiprocessing import freeze_support

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Data paths
base_dir = 'datasets'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Create train/val/test directories if they don't exist
for dir_path in [train_dir, val_dir, test_dir]:
    for class_name in ['belly', 'not_belly']:
        os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)

# Verify dataset structure
if not os.path.exists(base_dir):
    raise FileNotFoundError(f"Dataset directory {base_dir} not found")

if not os.path.exists(os.path.join(base_dir, 'belly')) or not os.path.exists(os.path.join(base_dir, 'not_belly')):
    raise FileNotFoundError("Required subdirectories 'belly' and 'not_belly' not found in dataset directory")

# Split data into train/val/test
def split_data(source_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.15, 0.15)):
    for class_name in ['belly', 'not_belly']:
        source_class_dir = os.path.join(source_dir, class_name)
        files = [f for f in os.listdir(source_class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Split files into train, val, and test
        train_files, temp_files = train_test_split(files, train_size=split_ratio[0], random_state=42)
        val_files, test_files = train_test_split(temp_files, train_size=split_ratio[1]/(split_ratio[1]+split_ratio[2]), random_state=42)
        
        # Copy files to respective directories
        for file in train_files:
            shutil.copy2(os.path.join(source_class_dir, file), 
                        os.path.join(train_dir, class_name, file))
        for file in val_files:
            shutil.copy2(os.path.join(source_class_dir, file), 
                        os.path.join(val_dir, class_name, file))
        for file in test_files:
            shutil.copy2(os.path.join(source_class_dir, file), 
                        os.path.join(test_dir, class_name, file))

def main():
    # Split the data
    split_data(base_dir, train_dir, val_dir, test_dir)

    # Data preprocessing with more augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Create data generators
    try:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=True,
            seed=42
        )

        validation_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=True,
            seed=42
        )

        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
        
        logger.info(f"Found {train_generator.samples} training images")
        logger.info(f"Found {validation_generator.samples} validation images")
        logger.info(f"Found {test_generator.samples} test images")
        
    except Exception as e:
        logger.error(f"Error creating data generators: {str(e)}")
        raise

    # Create CNN model with regularization
    model = Sequential([
        Input(shape=(224, 224, 3)),
        
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        
        # Second Conv Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        
        # Third Conv Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        
        Flatten(),
        
        # Dense layers with regularization
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(1, activation='sigmoid')
    ])

    # Compile the model with a higher learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        min_delta=0.001
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        min_delta=0.001
    )

    model_checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Train the model
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // BATCH_SIZE,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            workers=1,  # Reduced to 1 worker for Windows compatibility
            use_multiprocessing=False  # Disabled multiprocessing
        )
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')

    # Save the final model
    model.save('baby_sleep_classifier.h5')

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_generator)
    logger.info(f'\nTest accuracy: {test_accuracy:.4f}')
    logger.info(f'Test loss: {test_loss:.4f}')

    # Generate predictions on test set
    y_pred = model.predict(test_generator)
    y_pred = (y_pred > 0.5).astype(int)
    y_true = test_generator.classes

    # Generate classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_true, y_pred, target_names=['Not Belly', 'Belly']))

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Belly', 'Belly'],
                yticklabels=['Not Belly', 'Belly'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')

if __name__ == '__main__':
    freeze_support()
    main()
