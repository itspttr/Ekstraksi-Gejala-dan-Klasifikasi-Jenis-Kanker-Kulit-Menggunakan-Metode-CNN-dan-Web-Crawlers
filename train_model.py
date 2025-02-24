import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report

# Fungsi untuk plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.show()

# Fungsi untuk menampilkan classification report
def print_classification_report(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

# Menyusun data latih dan validasi
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/training',
    image_size=(150, 150),  # Sesuaikan ukuran gambar
    batch_size=16,
    shuffle=True
)

validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/validasi',
    image_size=(150, 150),  # Sesuaikan ukuran gambar
    batch_size=16,
    shuffle=False
)

# Membangun model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Untuk klasifikasi biner
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Summary model
model.summary()

# Latih model
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=50
)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

# Save the plots
plt.savefig('training_plots.png')
plt.show()

# Simpan model
model.save('skin_cancer_model.h5')

if __name__ == "__main__":
    # Load test data
    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        'dataset/testing',
        image_size=(150, 150),  # Sesuaikan ukuran gambar
        batch_size=16,
        shuffle=False
    )

    # Evaluasi model pada test data
    y_true = np.concatenate([y for x, y in test_data], axis=0)
    y_pred_prob = model.predict(test_data)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Menghitung dan menampilkan confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, ['carcinoma', 'melanoma'])

    # Menampilkan classification report
    print_classification_report(y_true, y_pred, ['carcinoma', 'melanoma'])
