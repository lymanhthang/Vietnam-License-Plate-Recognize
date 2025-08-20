import os
import shutil


SOURCE_DATA_DIR = 'character_data'

OUTPUT_DATASET_DIR = 'data'

def reorganize_dataset():

    print("Bắt đầu quá trình tổ chức lại dataset cho TensorFlow...")

    output_train_dir = os.path.join(OUTPUT_DATASET_DIR, 'train')
    output_val_dir = os.path.join(OUTPUT_DATASET_DIR, 'validation')

    if os.path.exists(OUTPUT_DATASET_DIR):
        print(f"Thư mục '{OUTPUT_DATASET_DIR}' đã tồn tại. Xóa và tạo lại...")
        shutil.rmtree(OUTPUT_DATASET_DIR)

    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)
    print(f"Đã tạo các thư mục output.")

    def process_directory(source_dir, destination_dir):
        if not os.path.isdir(source_dir):
            print(f"LỖI: Không tìm thấy thư mục nguồn '{source_dir}'.")
            return 0

        copied_count = 0
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}

        for filename in os.listdir(source_dir):
            # Bỏ qua các file không phải là ảnh
            if os.path.splitext(filename)[1].lower() not in image_extensions:
                continue

            # Tạo nhãn từ ký tự đầu tiên của tên file
            label = filename[0]

            # Tạo thư mục con cho nhãn nếu chưa có
            label_dir = os.path.join(destination_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            # Đường dẫn nguồn và đích
            source_path = os.path.join(source_dir, filename)
            destination_path = os.path.join(label_dir, filename)

            # Sao chép file
            shutil.copy(source_path, destination_path)
            copied_count += 1

        return copied_count

    source_train_dir = os.path.join(SOURCE_DATA_DIR, 'train')
    print(f"\nĐang xử lý thư mục '{source_train_dir}'...")
    num_train = process_directory(source_train_dir, output_train_dir)
    print(f"-> Đã sao chép {num_train} ảnh train.")

    source_val_dir = os.path.join(SOURCE_DATA_DIR, 'val')
    print(f"\nĐang xử lý thư mục '{source_val_dir}'...")
    num_val = process_directory(source_val_dir, output_val_dir)
    print(f"-> Đã sao chép {num_val} ảnh validation.")

    print(f"\nDataset đã được tổ chức lại trong thư mục '{OUTPUT_DATASET_DIR}'.")


if __name__ == '__main__':
    reorganize_dataset()