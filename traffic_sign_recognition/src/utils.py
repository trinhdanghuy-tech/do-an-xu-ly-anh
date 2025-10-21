import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def load_image(image_path, color_mode=cv2.IMREAD_COLOR):
    """
    Tải ảnh từ đường dẫn được chỉ định.
    Args:
        image_path (str): Đường dẫn tới file ảnh.
        color_mode (int): Chế độ màu để đọc ảnh (mặc định là ảnh màu).
    Returns:
        numpy.ndarray: Mảng numpy chứa dữ liệu ảnh, hoặc None nếu không tải được.
    """
    if not os.path.exists(image_path):
        print(f"Lỗi: File không tồn tại tại '{image_path}'")
        return None
    
    img = cv2.imread(image_path, color_mode)
    
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ '{image_path}'. File có thể bị hỏng hoặc không phải là định dạng ảnh được hỗ trợ.")
    
    return img

def save_image(image, save_path):
    """
    Lưu một ảnh vào đường dẫn được chỉ định.
    Tạo thư mục nếu nó chưa tồn tại.
    Args:
        image (numpy.ndarray): Ảnh cần lưu.
        save_path (str): Đường dẫn đầy đủ để lưu ảnh (bao gồm tên file và phần mở rộng).
    """
    # Tạo thư mục cha nếu nó chưa tồn tại
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)
    print(f"Đã lưu ảnh vào: {save_path}")

def display_image(title, image):
    """
    Hiển thị một ảnh duy nhất bằng Matplotlib.
    Args:
        title (str): Tiêu đề của cửa sổ hiển thị.
        image (numpy.ndarray): Ảnh cần hiển thị.
    """
    if image is None:
        print(f"Không thể hiển thị ảnh '{title}' vì nó không có dữ liệu.")
        return
        
    plt.figure(figsize=(6, 6))
    
    # Kiểm tra xem ảnh là ảnh xám hay ảnh màu
    if len(image.shape) == 2 or image.shape[2] == 1:
        plt.imshow(image, cmap='gray')
    else:
        # OpenCV đọc ảnh theo định dạng BGR, Matplotlib hiển thị theo RGB
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
    plt.title(title)
    plt.axis('off') # Ẩn các trục tọa độ
    plt.show()

def display_images(titles, images, figsize=(15, 5)):
    """
    Hiển thị một danh sách các ảnh trên cùng một figure.
    Args:
        titles (list of str): Danh sách các tiêu đề tương ứng với mỗi ảnh.
        images (list of numpy.ndarray): Danh sách các ảnh cần hiển thị.
        figsize (tuple): Kích thước của figure hiển thị.
    """
    if len(titles) != len(images):
        print("Lỗi: Số lượng tiêu đề và ảnh phải bằng nhau.")
        return

    n = len(images)
    plt.figure(figsize=figsize)
    
    for i in range(n):
        plt.subplot(1, n, i + 1)
        
        current_image = images[i]
        if current_image is None:
            plt.title(f"{titles[i]} (No data)")
            plt.axis('off')
            continue

        if len(current_image.shape) == 2:
             plt.imshow(current_image, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
            
        plt.title(titles[i])
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()