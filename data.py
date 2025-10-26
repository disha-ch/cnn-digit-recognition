"""
data.py
Train and save a compact, high‑accuracy CNN for MNIST digit recognition (Apple Silicon‑ready).
"""
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os

print("TensorFlow:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())

# --- Load MNIST ---
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# --- Model ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

es = callbacks.EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test), callbacks=[es])

path = os.path.join(os.path.dirname(__file__), "mnist_cnn_model.keras")
model.save(path)
print("Model saved at:", path)
