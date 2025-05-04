import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import random

# Load the trained model
try:
    model = tf.keras.models.load_model('baby_sleep_classifier.keras')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Test directory paths
test_dir = 'datasets/test'
belly_dir = os.path.join(test_dir, 'belly')
not_belly_dir = os.path.join(test_dir, 'not_belly')

# Verify directories exist
if not os.path.exists(belly_dir) or not os.path.exists(not_belly_dir):
    print(f"Error: Test directories not found. Please check if {test_dir} exists with 'belly' and 'not_belly' subdirectories")
    exit(1)

# Function to preprocess image
def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

# Function to invert prediction
def invert_prediction(prediction):
    return 1 - prediction

# Function to display prediction
def display_prediction(image, prediction, true_label):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    prediction_value = prediction[0][0]
    # Invert the prediction value
    inverted_prediction = invert_prediction(prediction_value)
    predicted_class = "Belly" if inverted_prediction > 0.5 else "Not Belly"
    confidence = inverted_prediction if predicted_class == "Belly" else 1 - inverted_prediction
    plt.title(f'True Label: {true_label}\nPrediction: {predicted_class} (Confidence: {confidence:.2%})')
    plt.axis('off')
    plt.show()

# Test random images from each class
def test_random_images(num_images=5):
    # Get random images from each class
    belly_images = random.sample(os.listdir(belly_dir), min(num_images, len(os.listdir(belly_dir))))
    not_belly_images = random.sample(os.listdir(not_belly_dir), min(num_images, len(os.listdir(not_belly_dir))))
    
    # Test belly images
    print("\nTesting Belly images:")
    for img_name in belly_images:
        img_path = os.path.join(belly_dir, img_name)
        img_array, img = preprocess_image(img_path)
        if img_array is None:
            continue
            
        prediction = model.predict(img_array, verbose=0)
        inverted_prediction = invert_prediction(prediction[0][0])
        print(f"\nImage: {img_name}")
        print(f"Raw prediction value: {prediction[0][0]:.4f}")
        print(f"Inverted prediction value: {inverted_prediction:.4f}")
        print(f"Prediction: {'Belly' if inverted_prediction > 0.5 else 'Not Belly'}")
        display_prediction(img, prediction, "Belly")
        print("-" * 50)
    
    # Test not belly images
    print("\nTesting Not Belly images:")
    for img_name in not_belly_images:
        img_path = os.path.join(not_belly_dir, img_name)
        img_array, img = preprocess_image(img_path)
        if img_array is None:
            continue
            
        prediction = model.predict(img_array, verbose=0)
        inverted_prediction = invert_prediction(prediction[0][0])
        print(f"\nImage: {img_name}")
        print(f"Raw prediction value: {prediction[0][0]:.4f}")
        print(f"Inverted prediction value: {inverted_prediction:.4f}")
        print(f"Prediction: {'Belly' if inverted_prediction > 0.5 else 'Not Belly'}")
        display_prediction(img, prediction, "Not Belly")
        print("-" * 50)

# Test all images and calculate accuracy
def test_all_images():
    correct = 0
    total = 0
    belly_correct = 0
    belly_total = 0
    not_belly_correct = 0
    not_belly_total = 0
    
    # Test belly images
    for img_name in os.listdir(belly_dir):
        img_path = os.path.join(belly_dir, img_name)
        img_array, _ = preprocess_image(img_path)
        if img_array is None:
            continue
            
        prediction = model.predict(img_array, verbose=0)
        inverted_prediction = invert_prediction(prediction[0][0])
        if inverted_prediction > 0.5:
            belly_correct += 1
        belly_total += 1
    
    # Test not belly images
    for img_name in os.listdir(not_belly_dir):
        img_path = os.path.join(not_belly_dir, img_name)
        img_array, _ = preprocess_image(img_path)
        if img_array is None:
            continue
            
        prediction = model.predict(img_array, verbose=0)
        inverted_prediction = invert_prediction(prediction[0][0])
        if inverted_prediction <= 0.5:
            not_belly_correct += 1
        not_belly_total += 1
    
    total = belly_total + not_belly_total
    correct = belly_correct + not_belly_correct
    
    print("\nDetailed Test Results:")
    print(f"Belly Class:")
    print(f"  Correct: {belly_correct}/{belly_total} ({belly_correct/belly_total:.2%})")
    print(f"Not Belly Class:")
    print(f"  Correct: {not_belly_correct}/{not_belly_total} ({not_belly_correct/not_belly_total:.2%})")
    print(f"\nOverall Accuracy: {correct/total:.2%}")
    print(f"Total Images Tested: {total}")

if __name__ == "__main__":
    print("Testing model on random images from test set...")
    test_random_images(5)  # Test 5 random images from each class
    
    print("\nTesting model on all images in test set...")
    test_all_images() 