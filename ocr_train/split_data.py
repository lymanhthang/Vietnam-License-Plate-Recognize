import os
import shutil
import csv
from sklearn.model_selection import train_test_split


source_images_dir = 'yolo_plate_ocr_dataset/data'

labels_filename = 'yolo_plate_ocr_dataset/data/labels.csv'

train_dir = 'train'
val_dir = 'val'

# Tỷ lệ tách dữ liệu (0.2 tương đương 20% cho tập validation)
validation_split_ratio = 0.2


print("Bắt đầu quá trình chia dữ liệu...")

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    print(f"Đã tạo thư mục: {train_dir}")

if not os.path.exists(val_dir):
    os.makedirs(val_dir)
    print(f"Đã tạo thư mục: {val_dir}")


print(f"Đang đọc file nhãn gốc từ: {labels_filename}")

all_data_records = []
try:
    with open(labels_filename, mode='r', newline='', encoding='utf-8') as f:
        # Sử dụng csv.DictReader để tự động xử lý dòng header
        reader = csv.DictReader(f)
        all_data_records = list(reader)

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{labels_filename}'. Vui lòng kiểm tra lại đường dẫn và tên file.")
    exit()
except KeyError:
    print("Lỗi: File CSV dường như không có header 'filename' và 'words'. Vui lòng kiểm tra lại file.")
    exit()

if not all_data_records:
    print("Lỗi: Không có dữ liệu nào được đọc từ file CSV. File có thể bị rỗng hoặc sai định dạng.")
    exit()

train_records, val_records = train_test_split(all_data_records, test_size=validation_split_ratio, random_state=42)

print(f"\nTổng số mẫu đọc được (sau header): {len(all_data_records)}")
print(f"Số mẫu Train: {len(train_records)} ({len(train_records)/len(all_data_records):.0%})")
print(f"Số mẫu Validation: {len(val_records)} ({len(val_records)/len(all_data_records):.0%})")


def process_split(records, output_dir):
    output_csv_path = os.path.join(output_dir, 'labels.csv')

    try:
        # Định nghĩa tên các cột (header)
        fieldnames = ['filename', 'words']
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)

            # Ghi dòng header vào file CSV
            writer.writeheader()

            # Ghi từng dòng dữ liệu
            writer.writerows(records)

    except Exception as e:
        print(f"Đã có lỗi xảy ra khi ghi file {output_csv_path}: {e}")
        return

    # Sao chép các file ảnh vào thư mục tương ứng
    for record in records:
        img_filename = record['filename'].strip()
        source_img_path = os.path.join(source_images_dir, img_filename)
        dest_img_path = os.path.join(output_dir, img_filename)

        if os.path.exists(source_img_path):
            shutil.copy(source_img_path, dest_img_path)
        else:
            print(f"Không tìm thấy file ảnh '{source_img_path}'")

print("\nĐang xử lý tập train...")
process_split(train_records, train_dir)

print("Đang xử lý tập validation...")
process_split(val_records, val_dir)

print("\nHoàn tất! Dữ liệu đã được chia thành công vào hai thư mục 'train' và 'val' với file 'labels.csv' đi kèm.")