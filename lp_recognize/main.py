from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time

IMG_WIDTH = 28
IMG_HEIGHT = 28

# Danh sách các ký tự model OCR nhận dạng
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z']


def load_models(lp_path, ocr_path):
    lp_model = YOLO(lp_path)
    
    ocr_model = keras.models.load_model(ocr_path)

    print("Nạp model thành công.")
    return lp_model, ocr_model


def crop_lp_from_image(lp_model, img_path):
    img = cv2.imread(img_path)
    cropped_plates = []

    if img is None:
        print(f"Lỗi: Không tìm thấy ảnh tại '{img_path}'")
        return cropped_plates

    results = lp_model(img)[0]  # Chạy model phát hiện

    # Lặp qua các biển số phát hiện được
    for box in results.boxes:
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, coords)
        # Cắt ảnh biển số
        cropped_license_plate = img[y1:y2, x1:x2]
        cropped_plates.append(cropped_license_plate)

    return cropped_plates


def sort_char_bboxes(boxes, margin=10):
    if not boxes:
        return []

        # Lấy 3 bbox có y nhỏ nhất
    boxes_sorted_by_y = sorted(boxes, key=lambda b: b[1])
    top_boxes = boxes_sorted_by_y[:3]

    # Tính trung tâm của 4 bbox
    points = []
    for (x, y, w, h) in top_boxes:
        cx = x + w / 2
        cy = y + h / 2
        points.append((cx, cy))

    points = np.array(points)

    # Tạo pt đường thẳng: y = m*x + b để phân biệt 2 hàng của biển số
    m, b = np.polyfit(points[:, 0], points[:, 1], 1)

    # Chia thành 2 hàng
    top_row = []
    bottom_row = []
    for (x, y, w, h) in boxes:
        cx = x + w / 2
        cy = y + h / 2
        y_line = m * cx + b
        if cy <= y_line + margin:
            top_row.append((x, y, w, h))
        else:
            bottom_row.append((x, y, w, h))

    # Sắp xếp từng hàng theo x
    top_row = sorted(top_row, key=lambda b: b[0])
    bottom_row = sorted(bottom_row, key=lambda b: b[0])

    # Gộp lại
    return top_row + bottom_row


def ocr_plate(ocr_model, cropped_lp_image, min_area_ratio=0.018, max_area_ratio=0.06):
    # 1. Tiền xử lý ảnh
    gray_image = cv2.cvtColor(cropped_lp_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Ngưỡng hóa ảnh để tách ký tự ra khỏi nền
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Tìm các đường viền của ký tự
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = gray_image.shape
    lp_area = img_h * img_w
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Lọc các box quá nhỏ hoặc quá lớn để loại bỏ nhiễu
        box_area = w * h
        area_ratio = box_area / lp_area
        # aspect_ratio = h / w
        # if 1.2 < aspect_ratio < 5.0 and 5 < w < 30 and 15 < h < 100:
        if min_area_ratio <= area_ratio <= max_area_ratio:
            bounding_boxes.append((x, y, w, h))

    if not bounding_boxes:
        print("Không tìm thấy ký tự nào hợp lệ.")
        return "", []

    # 3. Sắp xếp các ký tự theo đúng thứ tự
    sorted_bounding_boxes = sort_char_bboxes(bounding_boxes)

    recognized_string = ""

    # 4. Nhận dạng từng ký tự
    for box in sorted_bounding_boxes:
        x, y, w, h = box
        # Cắt ảnh của từng ký tự
        char_image = gray_image[y:y + h, x:x + w]

        # Resize về đúng kích thước đầu vào của model OCR
        char_image_resized = cv2.resize(char_image, (IMG_WIDTH, IMG_HEIGHT))

        # Chuẩn bị ảnh cho model (thêm batch dimension và channel dimension)
        char_image_processed = np.expand_dims(char_image_resized, axis=-1)  # (28, 28, 1)
        char_image_processed = np.expand_dims(char_image_processed, axis=0)  # (1, 28, 28, 1)

        # Dự đoán ký tự
        prediction = ocr_model.predict(char_image_processed, verbose=0)
        predicted_char_index = np.argmax(prediction)
        predicted_char = CLASS_NAMES[predicted_char_index]

        recognized_string += predicted_char

    return recognized_string, sorted_bounding_boxes


if __name__ == '__main__':
    lp_model_path = '../lp_train/best.pt'
    ocr_model_path = '../ocr_train/best_ocr_model.h5'

    lp_detector_model, ocr_reader_model = load_models(lp_model_path, ocr_model_path)

    t_start = time.perf_counter()
    input_image_path = 'test_img/test_img.jpg'

    cropped_license_plates = crop_lp_from_image(lp_detector_model, input_image_path)

    if not cropped_license_plates:
        print("Không phát hiện được biển số xe nào.")
    else:
        print(f"Phát hiện được {len(cropped_license_plates)} biển số xe.")
        for i, lp_img in enumerate(cropped_license_plates):
            recognized_text, char_boxes = ocr_plate(ocr_reader_model, lp_img)
            img_h, img_w = lp_img.shape[:2]
            for (x, y, w, h) in char_boxes:
                cv2.rectangle(lp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(f"KẾT QUẢ CHO BIỂN SỐ THỨ {i + 1}:")
            print(f"Ký tự nhận dạng được: {recognized_text}")

		
            cv2.imshow(f'Bien so {i + 1}', lp_img)
    t_end = time.perf_counter()
    print(f'Runtime: {t_end - t_start:.4f}s')
