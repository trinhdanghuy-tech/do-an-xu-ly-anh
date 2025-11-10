import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


# =====================================================
# 🧩 MODULE 1 – ĐỌC ẢNH VÀ CHUẨN HÓA
# =====================================================
def load_image(image_path):
    """Đọc ảnh và chuyển sang định dạng BGR + GRAY."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[LỖI] Không thể đọc ảnh: {image_path}")
        return None, None

    # Nếu ảnh là grayscale → chuyển sang BGR
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
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_gray, -1, kernel)

    # Tăng tương phản và giảm nhiễu
    enhanced_final = cv2.convertScaleAbs(sharpened, alpha=1.3, beta=10)
    enhanced_final = cv2.GaussianBlur(enhanced_final, (3, 3), 0)

    return enhanced_final


# =====================================================
# 🧩 MODULE 3 – PHÂN VÙNG MÀU BIỂN BÁO
# =====================================================
def segment_sign_colors(img_color):
    """Phân vùng màu đỏ và xanh lam (màu biển báo)."""
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    # Màu đỏ
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                              cv2.inRange(hsv, lower_red2, upper_red2))

    # Màu xanh
    lower_blue = np.array([90, 70, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Kết hợp
    mask = cv2.bitwise_or(mask_red, mask_blue)

    # Dọn nhiễu
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    return mask


# =====================================================
# 🧩 MODULE 4 – TRÍCH XUẤT VÙNG BIỂN BÁO (ROI)
# =====================================================
def extract_roi(img_color, mask):
    """Tìm vùng biển báo lớn nhất và cắt ROI."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contour = img_color.copy()
    roi = None

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        if w * h > 100:  # bỏ vùng nhiễu nhỏ
            cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = img_color[y:y + h, x:x + w]

    return img_contour, roi


# =====================================================
# 🧩 MODULE 5 – LƯU KẾT QUẢ
# =====================================================
def save_results(img_name, results_dir, img_color, enhanced_gray, mask, img_contour, roi):
    """Lưu từng bước xử lý vào thư mục riêng."""
    img_result_dir = Path(results_dir) / img_name
    img_result_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(img_result_dir / "step1_original.png"), img_color)
    cv2.imwrite(str(img_result_dir / "step2_enhanced.png"), enhanced_gray)
    cv2.imwrite(str(img_result_dir / "step3_mask.png"), mask)
    cv2.imwrite(str(img_result_dir / "step4_detected.png"), img_contour)

    if roi is not None:
        cv2.imwrite(str(img_result_dir / "step5_roi.png"), roi)

    print(f"✅ Đã lưu kết quả cho ảnh: {img_name} → {img_result_dir}")


# =====================================================
# 🧩 MODULE 6 – HIỂN THỊ KẾT QUẢ TRỰC QUAN
# =====================================================
def show_results(img_color, enhanced_gray, mask, img_contour, roi, results_dir):
    """
    Hiển thị tất cả các bước xử lý trên cùng 1 tab (subplot) và lưu kết quả.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Danh sách các ảnh và tiêu đề
    steps = [
        ("Original", img_color),
        ("Enhanced", enhanced_gray),
        ("Mask", mask),
        ("Detected", img_contour)
    ]
    if roi is not None:
        steps.append(("ROI", roi))

    # Tạo figure hiển thị tất cả ảnh trên cùng một dòng
    n = len(steps)
    plt.figure(figsize=(5 * n, 5))  # mỗi ảnh rộng 5 inch

    for i, (title, img) in enumerate(steps, 1):
        plt.subplot(1, n, i)
        if len(img.shape) == 2:  # grayscale
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')

        # Lưu ảnh từng bước (nếu cần)
        cv2.imwrite(os.path.join(results_dir, f"{i:02d}_{title.lower()}.png"), img)

    plt.tight_layout()
    plt.show()




# =====================================================
# 🧩 MODULE 7 – XỬ LÝ 1 ẢNH DUY NHẤT
# =====================================================
def process_single_image(image_path, results_dir):
    """Pipeline xử lý cho 1 ảnh duy nhất."""
    img_name = Path(image_path).stem
    img_color, gray = load_image(image_path)
    if img_color is None:
        return

    enhanced_gray = enhance_image(gray)
    mask = segment_sign_colors(img_color)
    img_contour, roi = extract_roi(img_color, mask)
    save_results(img_name, results_dir, img_color, enhanced_gray, mask, img_contour, roi)
    show_results(img_color, enhanced_gray, mask, img_contour, roi, results_dir)


# =====================================================
# 🚀 MAIN – CHẠY CHO 1 ẢNH
# =====================================================
if __name__ == "__main__":
    IMAGE_PATH = "C:\\DoAnXuLyAnh\\traffic_sign_recognition\\data\\gstrb-dataset\\gtsrb\\1\\00000_00029.ppm"
    RESULTS_DIR = "C:\\DoAnXuLyAnh\\traffic_sign_recognition\\results"

    process_single_image(IMAGE_PATH, RESULTS_DIR)
