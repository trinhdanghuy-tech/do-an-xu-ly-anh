import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(image_path):
    """
    Tải ảnh và áp dụng CLAHE trên kênh V (Value) của HSV.
    Đây là Giai đoạn 1.
    """
    # 1. Tải ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể tải ảnh từ {image_path}")
        return None, None

    # 2. Chuyển sang HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 3. Áp dụng CLAHE cho kênh V (Value)
    # Đây là kỹ thuật cốt lõi từ Bài 4 để tăng cường tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)

    # 4. Gộp lại và chuyển về BGR
    hsv_clahe = cv2.merge([h, s, v_clahe])
    enhanced_img = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    
    print("Giai đoạn 1: Đã áp dụng CLAHE thành công.")
    return img, enhanced_img, hsv_clahe, v_clahe

def locate_traffic_signs(hsv_clahe, v_clahe):
    """
    Sử dụng lọc màu HSV và Biến đổi Hough (GHT) để định vị biển báo.
    Đây là Giai đoạn 2.
    """
    # 1. Lọc màu HSV (Kết hợp ý tưởng từ Bài 2)
    # Ngưỡng màu Đỏ (nằm ở 2 đầu của thang HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv_clahe, lower_red1, upper_red1)
    
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv_clahe, lower_red2, upper_red2)
    
    # Ngưỡng màu Xanh (Blue)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv_clahe, lower_blue, upper_blue)

    # Kết hợp các mask
    color_mask = cv2.bitwise_or(mask_red1, cv2.bitwise_or(mask_red2, mask_blue))

    # 2. Chỉ giữ lại các vùng có màu phù hợp trên ảnh grayscale (kênh V)
    gray_for_hough = cv2.bitwise_and(v_clahe, v_clahe, mask=color_mask)
    
    # Làm mờ để giảm nhiễu trước khi tìm vòng tròn
    gray_for_hough_blurred = cv2.medianBlur(gray_for_hough, 5)

    # 3. Áp dụng Biến đổi Hough cho Hình tròn (Thay thế cho GHT)
    # Các tham số này (1.2, 100, 30, 1, 15, 30) cần được tinh chỉnh
    circles = cv2.HoughCircles(
        gray_for_hough_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,         # Khoảng cách tối thiểu giữa các vòng tròn
        param1=100,          # Ngưỡng cao của Canny
        param2=30,           # Ngưỡng phát hiện tâm
        minRadius=15,        # Bán kính tối thiểu
        maxRadius=100        # Bán kính tối đa
    )
    
    print(f"Giai đoạn 2: Phát hiện {len(circles[0]) if circles is not None else 0} biển báo (vòng tròn).")
    return circles

def classify_signs(model, original_image, circles):
    """
    Cắt các ROI từ ảnh và đưa vào ResNet để phân loại.
    Đây là Giai đoạn 3.
    """
    results = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0, :]:
            # Lấy tọa độ và bán kính (x, y, r)
            center_x, center_y, radius = i[0], i[1], i[2]
            
            # Tính toán Bounding Box (ROI)
            # Thêm một chút đệm (padding)
            x1 = max(0, center_x - radius - 5)
            y1 = max(0, center_y - radius - 5)
            x2 = min(original_image.shape[1], center_x + radius + 5)
            y2 = min(original_image.shape[0], center_y + radius + 5)
            
            roi = original_image[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # 1. Tiền xử lý ROI cho ResNet
            roi_resized = cv2.resize(roi, (224, 224))
            img_array = image.img_to_array(roi_resized)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_batch)

            # 2. Phân loại
            predictions = model.predict(img_preprocessed)
            
            # 3. Giải mã kết quả (sử dụng ImageNet)
            # LƯU Ý: Đây là nơi bạn sẽ thay bằng mô hình đã fine-tune
            decoded = decode_predictions(predictions, top=1)[0]
            results.append(((x1, y1, x2, y2), decoded[0]))
            
            print(f"Giai đoạn 3: ROI tại {(center_x, center_y)} - Dự đoán: {decoded[0]}")

    return results

def draw_results(image, results):
    """Vẽ kết quả lên ảnh gốc."""
    for (box, pred) in results:
        (x1, y1, x2, y2) = box
        label, name, prob = pred
        
        # Vẽ Bounding Box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Ghi nhãn
        # LƯU Ý: 'name' ở đây là từ ImageNet (ví dụ: 'volcano', 'fireplug')
        # Khi bạn có mô hình thật, nó sẽ là ('stop_sign', 'speed_50'...)
        text = f"{name}: {prob*100:.2f}%"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return image

# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    # 1. Khởi tạo mô hình ResNet (chỉ 1 lần)
    print("Đang tải mô hình ResNet50 (pre-trained on ImageNet)...")
    resnet_model = ResNet50(weights='imagenet')
    print("Mô hình đã tải xong.")

    # 2. Đường dẫn ảnh đầu vào
    IMAGE_PATH = 'C:\\DoAnXuLyAnh\\traffic_sign_recognition\\data\\gstrb-dataset\\gtsrb\\0\\00000_00029.ppm' # <-- THAY ĐỔI ĐƯỜNG DẪN NÀY

    # 3. Chạy Giai đoạn 1: Tiền xử lý + CLAHE
    original_img, enhanced_img, hsv_clahe, v_clahe = load_and_preprocess_image(IMAGE_PATH)

    if original_img is not None:
        # 4. Chạy Giai đoạn 2: Định vị (HSV + GHT)
        detected_circles = locate_traffic_signs(hsv_clahe, v_clahe)
        
        # 5. Chạy Giai đoạn 3: Phân loại (ResNet)
        # Chúng ta dùng ảnh GỐC (chưa tăng cường) để phân loại
        # Hoặc bạn có thể thử với 'enhanced_img'
        classification_results = classify_signs(resnet_model, original_img, detected_circles)
        
        # 6. Vẽ kết quả
        final_image = draw_results(original_img, classification_results)
        
        # 7. Hiển thị kết quả
        cv2.imshow('Ảnh Gốc', original_img)
        cv2.imshow('Ảnh đã tăng cường CLAHE', enhanced_img)
        cv2.imshow('Kết quả Cuối cùng', final_image)
        
        print("\n--- ĐÃ HOÀN THÀNH ---")
        print("LƯU Ý: Kết quả phân loại (ví dụ: 'volcano') là từ ImageNet.")
        print("Bạn cần 'fine-tune' ResNet trên bộ dữ liệu biển báo để có kết quả chính xác.")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()