# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from src import utils, enhancement, restoration, segmentation, morphology

def process_traffic_sign(image_path, output_dir="results/"):
    """
    Pipeline hoàn chỉnh để xử lý và phân vùng biển báo giao thông từ một ảnh.
    Args:
        image_path (str): Đường dẫn đến ảnh đầu vào.
        output_dir (str): Thư mục để lưu các kết quả trung gian và cuối cùng.
    """
    # --- 1. Tải ảnh và Chuẩn bị ---
    base_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(base_filename)
    
    img_original = utils.load_image(image_path)
    if img_original is None:
        return
        
    print(f"--- Bắt đầu xử lý ảnh: {base_filename} ---")
    
    # --- 2. Tăng cường ảnh ---
    # Tăng cường độ tương phản để làm nổi bật biển báo
    img_enhanced = enhancement.contrast_stretching(img_original)
    
    # --- 3. Khôi phục ảnh ---
    # Giảm nhiễu bằng bộ lọc trung vị, bảo vệ cạnh tốt hơn bộ lọc trung bình
    img_restored = restoration.apply_median_filter(img_enhanced, kernel_size=3)
    
    # --- 4. Phân vùng ảnh ---
    # Sử dụng phân vùng màu để tìm các vùng có màu đỏ và xanh (màu phổ biến của biển báo)
    # Dải màu đỏ trong HSV (có 2 dải vì H quay vòng)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Dải màu xanh trong HSV
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    mask_red1 = segmentation.color_threshold_segmentation(img_restored, lower_red1, upper_red1)
    mask_red2 = segmentation.color_threshold_segmentation(img_restored, lower_red2, upper_red2)
    mask_blue = segmentation.color_threshold_segmentation(img_restored, lower_blue, upper_blue)
    
    # Kết hợp các mask màu lại
    mask_segmented = cv2.bitwise_or(mask_red1, mask_red2)
    mask_segmented = cv2.bitwise_or(mask_segmented, mask_blue)

    # --- 5. Xử lý hình thái ---
    # Áp dụng phép mở để loại bỏ các điểm nhiễu nhỏ
    mask_opened = morphology.apply_opening(mask_segmented, kernel_size=(3, 3))
    
    # Áp dụng phép đóng để lấp các lỗ hổng bên trong biển báo và làm liền mạch
    mask_closed = morphology.apply_closing(mask_opened, kernel_size=(7, 7))
    
    # --- 6. Trích xuất đối tượng ---
    # Tìm các đường bao (contours) trong mask cuối cùng
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_final_result = img_original.copy()
    
    if contours:
        # Giả sử biển báo là đối tượng có diện tích lớn nhất
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Lấy hình chữ nhật bao quanh đối tượng lớn nhất
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Vẽ hình chữ nhật lên ảnh kết quả
        cv2.rectangle(img_final_result, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print(f"Đã phát hiện đối tượng tại: (x={x}, y={y}, w={w}, h={h})")
    else:
        print("Không tìm thấy đối tượng nào phù hợp.")

    # --- 7. Lưu và Hiển thị Kết quả ---
    # Lưu các ảnh trung gian
    utils.save_image(img_enhanced, os.path.join)