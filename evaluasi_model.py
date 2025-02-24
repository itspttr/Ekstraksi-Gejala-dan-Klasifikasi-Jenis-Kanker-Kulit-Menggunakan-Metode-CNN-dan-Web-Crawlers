import numpy as np
import pickle
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

# Load model and label encoder
try:
    model = load_model('skin_cancer_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    print("Model and label encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    raise

def preprocess_image(image_path):
    print(f"Loading image from: {image_path}")
    if not os.path.isfile(image_path):
        raise ValueError(f"File tidak ditemukan di path: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded. Please check the file path.")
    print(f"Image loaded successfully. Shape: {image.shape}")

    image = cv2.resize(image, (150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.0
    return image

def evaluate_model(test_data):
    # Evaluasi model
    loss, accuracy = model.evaluate(test_data)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    # Prediksi
    y_pred_prob = model.predict(test_data)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Ambil label dari dataset
    class_names = list(le.classes_)

    # Ambil label asli dari dataset
    y_true = test_data.classes

    # Hitung confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=class_names)

    print("Confusion Matrix:")
    print(conf_matrix)

    print("Classification Report:")
    print(class_report)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    # Definisikan direktori data pengujian
    test_data_dir = 'C:/xampp/htdocs/ta_project/dataset/testing'  # Ganti dengan path direktori data pengujian Anda

    # Load test data
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_data = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',  # Use 'sparse' for integer labels, 'categorical' for one-hot encoding
        shuffle=False
    )

    # Evaluasi model dan tampilkan confusion matrix
    try:
        evaluate_model(test_data)
    except Exception as e:
        print(f"Error: {e}")
