import cv2
import numpy as np

def color_threshold_segmentation(image, lower_bound, upper_bound):
    """
    Phân vùng ảnh dựa trên ngưỡng màu trong không gian màu HSV.
    Args:
        image (numpy.ndarray): Ảnh màu BGR đầu vào.
        lower_bound (numpy.ndarray): Mảng numpy chứa ngưỡng dưới của màu (H, S, V).
        upper_bound (numpy.ndarray): Mảng numpy chứa ngưỡng trên của màu (H, S, V).
    Returns:
        numpy.ndarray: Mask nhị phân, các pixel trong ngưỡng màu sẽ có giá trị 255.
    """
    # Chuyển ảnh sang không gian màu HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Tạo mask dựa trên dải màu
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return mask

def otsu_threshold_segmentation(image):
    """
    Phân vùng ảnh bằng phương pháp ngưỡng tự động Otsu trên ảnh xám.
    Args:
        image (numpy.ndarray): Ảnh đầu vào (màu hoặc xám).
    Returns:
        numpy.ndarray: Mask nhị phân.
    """
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
        
    # Áp dụng bộ lọc Gaussian nhẹ để giảm nhiễu có thể ảnh hưởng đến kết quả của Otsu
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Áp dụng ngưỡng Otsu
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def canny_edge_detection(image, low_threshold, high_threshold):
    """
    Phát hiện biên trong ảnh bằng thuật toán Canny.
    Args:
        image (numpy.ndarray): Ảnh đầu vào (màu hoặc xám).
        low_threshold (int): Ngưỡng dưới cho hysteresis.
        high_threshold (int): Ngưỡng trên cho hysteresis.
    Returns:
        numpy.ndarray: Ảnh chứa các cạnh đã được phát hiện.
    """
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
        
    # Làm mịn ảnh trước khi áp dụng Canny là rất quan trọng
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges