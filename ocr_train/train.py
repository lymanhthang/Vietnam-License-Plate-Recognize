import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os


DATASET_DIR = 'data'
IMG_HEIGHT = 28
IMG_WIDTH = 28
BATCH_SIZE = 32
EPOCHS = 50
BEST_MODEL_NAME = 'best_ocr_model.h5'

print("--- Bước 1: Tải dữ liệu ---")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    shuffle=True
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, 'validation'),
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
NUM_CLASSES = len(class_names)
print(f"Đã tìm thấy {NUM_CLASSES} lớp (ký tự): {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("\n--- Bước 2: Xây dựng kiến trúc mô hình ---")
model = keras.Sequential([
    layers.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
model.summary()

print("\n--- Bước 3: Compile mô hình ---")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_NAME,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

print("\n--- Bước 4: Bắt đầu quá trình huấn luyện ---")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, model_checkpoint]
)

print("\n--- Bước 5: Đánh giá hiệu suất mô hình tốt nhất ---")

val_loss, val_accuracy = model.evaluate(validation_dataset)
print(f"\nĐộ chính xác trên tập validation: {val_accuracy * 100:.2f}%")
print(f"Loss trên tập validation: {val_loss:.4f}")

print(f"\nMô hình tốt nhất đã được lưu vào file: '{BEST_MODEL_NAME}'")

actual_epochs = len(history.history['loss'])
epochs_range = range(actual_epochs)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('training_history.png')
print("Biểu đồ quá trình huấn luyện đã được lưu vào file: 'training_history.png'")