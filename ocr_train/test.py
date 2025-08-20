import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

MODEL_PATH = 'best_ocr_model.keras'
IMAGE_PATH = 'test_img.png'

IMG_HEIGHT = 28
IMG_WIDTH = 28

CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N','O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z']


try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Đã tải model thành công từ '{MODEL_PATH}'")
except Exception as e:
    print(f"LỖI: Không thể tải model. Vui lòng kiểm tra lại đường dẫn.\n{e}")
    exit()


def sort_bounding_boxes_top_to_bottom_then_left_to_right(boxes, y_threshold=15):
    boxes = sorted(boxes, key=lambda b: b[1])  # Sắp theo y (từ trên xuống)
    rows = []
    current_row = [boxes[0]]

    for box in boxes[1:]:
        if abs(box[1] - current_row[0][1]) <= y_threshold:
            current_row.append(box)
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))  # trái sang phải
            current_row = [box]
    rows.append(sorted(current_row, key=lambda b: b[0]))

    # Gộp tất cả các hàng lại
    sorted_boxes = [box for row in rows for box in row]
    return sorted_boxes

def ocr_pipeline(image_path):
    if not os.path.exists(image_path):
        print(f"LỖI: Không tìm thấy ảnh tại '{image_path}'")
        return None, None

    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w > 5 and h > 15) and (w < 50 and h < 50):
            bounding_boxes.append((x, y, w, h))

    if len(bounding_boxes) == 0:
        print("Không tìm thấy ký tự nào.")
        return "", original_image

    # Sắp xếp thông minh: Ưu tiên từ trên xuống, sau đó trái sang
    bounding_boxes = sort_bounding_boxes_top_to_bottom_then_left_to_right(bounding_boxes)

    recognized_string = ""

    for box in bounding_boxes:
        x, y, w, h = box
        char_image = gray_image[y:y + h, x:x + w]
        char_image = cv2.resize(char_image, (IMG_WIDTH, IMG_HEIGHT))

        char_image = np.expand_dims(char_image, axis=-1)  # (28, 28, 1)
        char_image = np.expand_dims(char_image, axis=0)   # (1, 28, 28, 1)

        prediction = model.predict(char_image, verbose=0)
        predicted_char = CLASS_NAMES[np.argmax(prediction)]
        recognized_string += predicted_char

        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(original_image, predicted_char, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return recognized_string, original_image


if __name__ == '__main__':

    final_text, visualized_image = ocr_pipeline(IMAGE_PATH)

    if final_text is not None:
        print(f"KẾT QUẢ NHẬN DẠNG: {final_text}")

        cv2.imshow("Ket qua nhan dang", visualized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
