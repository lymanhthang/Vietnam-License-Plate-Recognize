from tensorflow import keras

model = keras.models.load_model(r'best_ocr_model.keras')
model.save("best_ocr_model.h5")
