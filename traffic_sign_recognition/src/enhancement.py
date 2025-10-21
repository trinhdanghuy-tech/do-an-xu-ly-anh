import cv2
import numpy as np

def histogram_equalization(image):
    """
    Áp dụng cân bằng lược đồ xám (histogram equalization).
    Xử lý được cả ảnh màu và ảnh xám.
    Args:
        image (numpy.ndarray): Ảnh đầu vào.
    Returns:
        numpy.ndarray: Ảnh đã được cân bằng.
    """
    if len(image.shape) > 2: # Nếu là ảnh màu
        # Chuyển sang không gian màu YCrCb
        # Kênh Y đại diện cho độ sáng (luminance)
        img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # Chỉ cân bằng trên kênh độ sáng
        img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])
        # Chuyển ngược lại không gian màu BGR
        equalized_image = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
    else: # Nếu là ảnh xám
        equalized_image = cv2.equalizeHist(image)
    return equalized_image

def contrast_stretching(image, min_out=0, max_out=255):
    """
    Áp dụng giãn độ tương phản (contrast stretching).
    Args:
        image (numpy.ndarray): Ảnh đầu vào.
        min_out (int): Giá trị pixel nhỏ nhất của ảnh đầu ra.
        max_out (int): Giá trị pixel lớn nhất của ảnh đầu ra.
    Returns:
        numpy.ndarray: Ảnh đã được giãn độ tương phản.
    """
    stretched_image = image.copy()
    channels = cv2.split(image) if len(image.shape) > 2 else [image]
    processed_channels = []

    for channel in channels:
        min_in = np.min(channel)
        max_in = np.max(channel)
        
        # Tránh chia cho 0 nếu ảnh có một màu duy nhất
        if max_in == min_in:
             processed_channels.append(channel)
             continue
        
        # Áp dụng công thức giãn: g(x,y) = (f(x,y) - min_in) * (max_out - min_out) / (max_in - min_in) + min_out
        stretched_channel = ((channel - min_in) / (max_in - min_in)) * (max_out - min_out) + min_out
        processed_channels.append(stretched_channel.astype(np.uint8))

    if len(processed_channels) > 1:
        stretched_image = cv2.merge(processed_channels)
    else:
        stretched_image = processed_channels[0]

    return stretched_image

def power_law_transform(image, gamma=1.0):
    """
    Áp dụng biến đổi hàm mũ (gamma correction) để điều chỉnh độ sáng.
    gamma < 1: làm sáng ảnh.
    gamma > 1: làm tối ảnh.
    Args:
        image (numpy.ndarray): Ảnh đầu vào.
        gamma (float): Hệ số gamma.
    Returns:
        numpy.ndarray: Ảnh đã được hiệu chỉnh gamma.
    """
    # Xây dựng bảng tra cứu (lookup table) để tăng tốc độ
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Áp dụng bảng tra cứu cho ảnh
    gamma_corrected_image = cv2.LUT(image, table)
    return gamma_corrected_image