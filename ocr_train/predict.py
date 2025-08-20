import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os


MODEL_PATH = 'best_ocr_model.keras'


IMAGE_PATH = 'data/validation/C/C_1f53059b.jpg'

IMG_HEIGHT = 28
IMG_WIDTH = 28

CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N','O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z']


def predict_single_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy file ảnh tại '{image_path}'")
        return None, None

    # 1. Tải ảnh từ đường dẫn
    img = keras.utils.load_img(
        image_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale'  # Tải dưới dạng ảnh xám
    )

    # 2. Chuyển ảnh thành một mảng numpy
    img_array = keras.utils.img_to_array(img)

    # 3. Mở rộng chiều để tạo thành một "lô" chỉ có 1 ảnh
    # Kích thước từ (28, 28, 1) -> (1, 28, 28, 1)
    img_array = tf.expand_dims(img_array, 0)

    # 4. Thực hiện dự đoán
    predictions = model.predict(img_array)

    # 5. Lấy ra kết quả
    # Lấy chỉ số có xác suất cao nhất
    score_index = np.argmax(predictions[0])
    # Lấy giá trị xác suất (độ tin cậy)
    confidence = 100 * np.max(predictions[0])
    # Lấy tên lớp tương ứng
    predicted_class = CLASS_NAMES[score_index]

    return predicted_class, confidence, img


# --- Main execution ---
if __name__ == '__main__':
    # Tải model đã được huấn luyện
    try:
        print(f"Đang tải model từ '{MODEL_PATH}'...")
        loaded_model = keras.models.load_model(MODEL_PATH)
        print("Tải model thành công.")
    except Exception as e:
        print(f"LỖI: Không thể tải model. Vui lòng kiểm tra lại đường dẫn.\n{e}")
        exit()

    # Thực hiện dự đoán
    predicted_char, confidence, original_image = predict_single_image(loaded_model, IMAGE_PATH)

    # In và hiển thị kết quả
    if predicted_char is not None:
        print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
        print(f"Ký tự dự đoán: {predicted_char}")
        print(f"Độ tin cậy: {confidence:.2f}%")

        # Hiển thị ảnh và kết quả
        plt.figure(figsize=(4, 4))
        plt.imshow(original_image, cmap='gray')
        plt.title(f"Dự đoán: {predicted_char} ({confidence:.2f}%)")
        plt.axis("off")
        plt.show()