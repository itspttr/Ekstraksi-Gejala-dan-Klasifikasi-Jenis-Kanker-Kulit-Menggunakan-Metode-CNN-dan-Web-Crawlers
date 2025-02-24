import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model and label encoder
try:
    model = load_model('skin_cancer_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    print("Model and label encoder loaded successfully.")
    print(f"Classes: {le.classes_}")
except FileNotFoundError:
    model, le = None, None
    print("Model file or label encoder file not found. Please ensure 'skin_cancer_model.h5' and 'label_encoder.pkl' are in the correct directory.")
except Exception as e:
    model, le = None, None
    print(f"Error loading the model or label encoder: {e}")

# Path to test image
image_path = 'C:/xampp/htdocs/ta_project/dataset/training/melanoma/dataset/training/melanoma/ISIC_0000013-150x150.jpg.jpg'  # Ganti dengan path absolut gambar yang ingin Anda uji

# Check if the file exists
if not os.path.exists(image_path):
    print(f"File not found: {image_path}")
else:
    # Preprocess the image
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded. Please check the file path.")
        image = cv2.resize(image, (150, 150))  # Ensure this matches the input size of your model
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image.astype('float32') / 255.0
        print("Image preprocessed successfully.")
    except Exception as e:
        print(f"Error processing image: {e}")

    # Predict the class
    try:
        if model is None or le is None:
            print("Model or label encoder not loaded properly.")
        else:
            prediction = model.predict(image)
            print(f"Prediction values: {prediction}")
            predicted_class = np.argmax(prediction, axis=1)
            print(f"Predicted class index: {predicted_class[0]}")
            skin_cancer_type = le.inverse_transform(predicted_class)[0]
            print(f"Predicted class label: {skin_cancer_type}")
    except Exception as e:
        print(f"Error predicting image class: {e}")
