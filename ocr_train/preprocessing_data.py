import os
import os
import cv2
import uuid
import csv


with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Tạo dictionary bỏ qua 'BACKGROUND' (ở vị trí 0)
class_id_to_char = {i - 1: label for i, label in enumerate(labels) if i != 0}

print(class_id_to_char)



folder_path = 'yolo_plate_ocr_dataset'
output_folder = os.path.join(folder_path, 'data')
os.makedirs(output_folder, exist_ok=True)

output_csv_file = os.path.join(output_folder, 'labels.csv')


# Ghi nhãn vào file CSV
with open(output_csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'words'])  # header

    # Duyệt qua các file ảnh
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            txt_path = os.path.splitext(img_path)[0] + '.txt'

            if not os.path.exists(txt_path):
                continue

            # Đọc ảnh
            image = cv2.imread(img_path)
            if image is None:
                continue
            h, w = image.shape[:2]

            # Đọc bounding boxes
            boxes = []
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x_center, y_center, box_width, box_height = map(float, parts)
                    class_id = int(class_id)

                    # Chuyển về toạ độ pixel
                    x = int((x_center - box_width / 2) * w)
                    y = int((y_center - box_height / 2) * h)
                    bw = int(box_width * w)
                    bh = int(box_height * h)

                    boxes.append((x, y, bw, bh, class_id))

            # Sắp xếp bbox theo trục hoành (trái sang phải)
            boxes.sort(key=lambda b: b[0])

            # Crop và lưu ảnh
            for x, y, bw, bh, class_id in boxes:
                char_img = image[y:y+bh, x:x+bw]
                char = class_id_to_char.get(class_id, 'UNK')
                filename_crop = f"{char}_{str(uuid.uuid4())[:8]}.jpg"
                crop_path = os.path.join(output_folder, filename_crop)
                cv2.imwrite(crop_path, char_img)
                writer.writerow([filename_crop, char])
                



