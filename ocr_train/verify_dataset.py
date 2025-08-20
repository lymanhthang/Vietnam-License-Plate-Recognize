import os
import csv

def verify_dataset(directory_path):
    """
    Kiểm tra xem các file ảnh trong thư mục có khớp với các nhãn trong file labels.csv không.
    """
    print(f"\n--- Bắt đầu kiểm tra thư mục: '{directory_path}' ---")

    labels_file_path = os.path.join(directory_path, 'labels.csv')

    # 1. Kiểm tra xem file labels.csv có tồn tại không
    if not os.path.exists(labels_file_path):
        print(f"❌ Lỗi: Không tìm thấy file '{labels_file_path}'")
        return False

    # 2. Lấy danh sách tên file ảnh từ trong thư mục
    try:
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
        images_in_dir = {f for f in os.listdir(directory_path) if os.path.splitext(f)[1].lower() in image_extensions}
        print(f"Tìm thấy {len(images_in_dir)} file ảnh trong thư mục.")
    except Exception as e:
        print(f"❌ Lỗi khi đọc thư mục ảnh: {e}")
        return False

    # 3. Lấy danh sách tên file từ file labels.csv
    try:
        filenames_in_csv = set()
        with open(labels_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filenames_in_csv.add(row['filename'])
        print(f"Tìm thấy {len(filenames_in_csv)} nhãn trong file labels.csv.")
    except Exception as e:
        print(f"❌ Lỗi khi đọc file CSV: {e}")
        return False

    # 4. So sánh hai danh sách
    missing_in_csv = images_in_dir - filenames_in_csv
    missing_in_dir = filenames_in_csv - images_in_dir

    is_ok = True
    if not missing_in_csv and not missing_in_dir:
        print("✅ THÀNH CÔNG! Tất cả ảnh và nhãn đều khớp nhau hoàn hảo.")
    else:
        is_ok = False
        if missing_in_csv:
            print(f"❌ LỖI: {len(missing_in_csv)} ảnh có trong thư mục nhưng không có nhãn trong CSV:")
            for filename in list(missing_in_csv)[:5]: # In ra 5 file đầu tiên
                print(f"  - {filename}")

        if missing_in_dir:
            print(f"❌ LỖI: {len(missing_in_dir)} nhãn có trong CSV nhưng không tìm thấy ảnh tương ứng:")
            for filename in list(missing_in_dir)[:5]: # In ra 5 file đầu tiên
                print(f"  - {filename}")

    return is_ok

# --- Thực hiện kiểm tra ---
if __name__ == "__main__":
    is_train_ok = verify_dataset('character_data/train')
    is_val_ok = verify_dataset('character_data/val')

    if is_train_ok and is_val_ok:
        print("\n🎉🎉🎉 Tổng kết: Cả hai tập train và val đều hợp lệ. Bạn có thể bắt đầu fine-tune!")
    else:
        print("\n🚨🚨🚨 Tổng kết: Có lỗi trong việc chia dữ liệu. Vui lòng kiểm tra lại.")