import kagglehub
import os
import shutil

# 1️⃣ Tải dataset
path = kagglehub.dataset_download("mohammedabdeldayem/gstrb-dataset")
print("Đã tải về:", path)

# 2️⃣ Di chuyển hoặc copy đến thư mục ./data
dest = "./data/gstrb-dataset"
os.makedirs(dest, exist_ok=True)

# Copy toàn bộ dataset sang thư mục data
shutil.copytree(path, dest, dirs_exist_ok=True)

print("Đã sao chép dataset vào:", dest)
