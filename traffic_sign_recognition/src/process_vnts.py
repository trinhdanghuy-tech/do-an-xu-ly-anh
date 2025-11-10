import cv2
import numpy as np
import os
from pathlib import Path


# =====================================================
# 🧩 MODULE 1 – ĐỌC ẢNH VÀ CHUẨN HÓA
# =====================================================
def load_image(image_path):
    """Đọc ảnh và chuyển sang định dạng BGR chuẩn."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[LỖI] Không thể đọc {image_path}")
        return None, None

    # Ảnh đen trắng -> chuyển sang BGR để đồng nhất
    if len(img.shape) == 2 or img.shape[2] == 1:
        gray = img
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img_color, gray


# =====================================================
# 🧩 MODULE 2 – TĂNG CƯỜNG ẢNH
# =====================================================
def enhance_image(gray):
    """Áp dụng CLAHE + sharpen + tăng tương phản."""
    # 1️⃣ CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # 2️⃣ Làm sắc nét
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_gray, -1, kernel)

    # 3️⃣ Tăng tương phản & làm mịn
    enhanced_final = cv2.convertScaleAbs(sharpened, alpha=1.3, beta=10)
    enhanced_final = cv2.GaussianBlur(enhanced_final, (3, 3), 0)

    return enhanced_final


# =====================================================
# 🧩 MODULE 3 – PHÂN VÙNG MÀU BIỂN BÁO
# =====================================================
def segment_sign_colors(img_color):
    """Phân vùng màu đỏ và xanh lam trong ảnh."""
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    # --- Màu đỏ ---
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                              cv2.inRange(hsv, lower_red2, upper_red2))

    # --- Màu xanh ---
    lower_blue = np.array([90, 70, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # --- Kết hợp ---
    mask = cv2.bitwise_or(mask_red, mask_blue)

    # --- Dọn nhiễu ---
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    return mask


# =====================================================
# 🧩 MODULE 4 – TÌM VÙNG BIỂN BÁO (ROI)
# =====================================================
def extract_roi(img_color, mask):
    """Tìm vùng biển báo (contour lớn nhất) và cắt ROI."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contour = img_color.copy()
    roi = None

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        if w * h > 100:  # bỏ nhiễu nhỏ
            cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = img_color[y:y + h, x:x + w]

    return img_contour, roi


# =====================================================
# 🧩 MODULE 5 – LƯU KẾT QUẢ
# =====================================================
def save_results(img_name, results_dir, img_color, enhanced_gray, mask, img_contour, roi):
    """Lưu toàn bộ các bước xử lý vào thư mục riêng."""
    img_result_dir = Path(results_dir) / img_name
    img_result_dir.mkdir(parents=True, exist_ok=True)

    # Lưu từng bước
    cv2.imwrite(str(img_result_dir / "step1_original.png"), img_color)
    cv2.imwrite(str(img_result_dir / "step2_enhanced.png"), enhanced_gray)
    cv2.imwrite(str(img_result_dir / "step3_mask.png"), mask)
    cv2.imwrite(str(img_result_dir / "step4_detected.png"), img_contour)

    if roi is not None:
        cv2.imwrite(str(img_result_dir / "step5_roi.png"), roi)

    print(f"✅ Đã lưu kết quả {img_name} → {img_result_dir}")


# =====================================================
# 🧩 MODULE 6 – XỬ LÝ MỘT ẢNH DUY NHẤT
# =====================================================
def process_image(image_path, results_dir):
    """Chạy toàn bộ pipeline cho một ảnh."""
    img_name = Path(image_path).stem
    img_color, gray = load_image(image_path)
    if img_color is None:
        return

    enhanced_gray = enhance_image(gray)
    mask = segment_sign_colors(img_color)
    img_contour, roi = extract_roi(img_color, mask)
    save_results(img_name, results_dir, img_color, enhanced_gray, mask, img_contour, roi)


# =====================================================
# 🧩 MODULE 7 – XỬ LÝ TOÀN BỘ THƯ MỤC
# =====================================================
def process_all_images(input_dir, results_dir):
    """Duyệt qua tất cả ảnh trong thư mục input."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.ppm', '.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, file)
            process_image(image_path, results_dir)


# =====================================================
# 🚀 MAIN – CHẠY CHƯƠNG TRÌNH
# =====================================================
if __name__ == "__main__":
    INPUT_DIR = "C:\\DoAnXuLyAnh\\traffic_sign_recognition\\data\\gstrb-dataset\\gtsrb\\0"
    RESULTS_DIR = "C:\\DoAnXuLyAnh\\traffic_sign_recognition\\results"

    process_all_images(INPUT_DIR, RESULTS_DIR)
