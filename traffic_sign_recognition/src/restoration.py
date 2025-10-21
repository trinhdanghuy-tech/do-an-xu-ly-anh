import cv2

def apply_mean_filter(image, kernel_size=(3, 3)):
    """
    Áp dụng bộ lọc trung bình số học để làm mờ và giảm nhiễu Gaussian.
    Args:
        image (numpy.ndarray): Ảnh đầu vào.
        kernel_size (tuple): Kích thước của kernel (phải là số lẻ).
    Returns:
        numpy.ndarray: Ảnh đã được lọc.
    """
    blurred_image = cv2.blur(image, kernel_size)
    return blurred_image

def apply_median_filter(image, kernel_size=3):
    """
    Áp dụng bộ lọc trung vị, hiệu quả để loại bỏ nhiễu "muối tiêu".
    Args:
        image (numpy.ndarray): Ảnh đầu vào.
        kernel_size (int): Kích thước của kernel (phải là số lẻ, ví dụ: 3, 5, 7).
    Returns:
        numpy.ndarray: Ảnh đã được lọc.
    """
    # Kích thước kernel phải là số lẻ
    if kernel_size % 2 == 0:
        kernel_size += 1
    median_filtered_image = cv2.medianBlur(image, kernel_size)
    return median_filtered_image

def apply_gaussian_blur(image, kernel_size=(5, 5), sigmaX=0):
    """
    Áp dụng bộ lọc Gaussian để làm mịn ảnh. Thường được dùng trước Canny.
    Args:
        image (numpy.ndarray): Ảnh đầu vào.
        kernel_size (tuple): Kích thước kernel (phải là số lẻ).
        sigmaX (float): Độ lệch chuẩn theo trục X.
    Returns:
        numpy.ndarray: Ảnh đã được làm mịn.
    """
    # Kích thước kernel phải gồm các số lẻ
    k_w = kernel_size[0] if kernel_size[0] % 2 != 0 else kernel_size[0] + 1
    k_h = kernel_size[1] if kernel_size[1] % 2 != 0 else kernel_size[1] + 1
    blurred_image = cv2.GaussianBlur(image, (k_w, k_h), sigmaX)
    return blurred_image