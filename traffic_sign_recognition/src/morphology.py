# -*- coding: utf-8 -*-
import cv2
import numpy as np

def apply_erosion(binary_image, kernel_size=(3, 3), iterations=1):
    """
    Áp dụng phép co (erosion) để làm mỏng hoặc loại bỏ các đối tượng nhỏ.
    Args:
        binary_image (numpy.ndarray): Ảnh nhị phân đầu vào.
        kernel_size (tuple): Kích thước của phần tử cấu trúc.
        iterations (int): Số lần lặp lại phép co.
    Returns:
        numpy.ndarray: Ảnh nhị phân sau khi co.
    """
    kernel = np.ones(kernel_size, np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=iterations)
    return eroded_image

def apply_dilation(binary_image, kernel_size=(3, 3), iterations=1):
    """
    Áp dụng phép giãn (dilation) để làm dày hoặc nối liền các phần bị đứt.
    Args:
        binary_image (numpy.ndarray): Ảnh nhị phân đầu vào.
        kernel_size (tuple): Kích thước của phần tử cấu trúc.
        iterations (int): Số lần lặp lại phép giãn.
    Returns:
        numpy.ndarray: Ảnh nhị phân sau khi giãn.
    """
    kernel = np.ones(kernel_size, np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=iterations)
    return dilated_image

def apply_opening(binary_image, kernel_size=(3, 3)):
    """
    Áp dụng phép mở (opening = erosion -> dilation).
    Hữu ích để loại bỏ nhiễu "muối" (các chấm trắng nhỏ).
    Args:
        binary_image (numpy.ndarray): Ảnh nhị phân đầu vào.
        kernel_size (tuple): Kích thước của phần tử cấu trúc.
    Returns:
        numpy.ndarray: Ảnh nhị phân sau khi mở.
    """
    kernel = np.ones(kernel_size, np.uint8)
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    return opened_image

def apply_closing(binary_image, kernel_size=(3, 3)):
    """
    Áp dụng phép đóng (closing = dilation -> erosion).
    Hữu ích để lấp các lỗ nhỏ ("tiêu") bên trong đối tượng.
    Args:
        binary_image (numpy.ndarray): Ảnh nhị phân đầu vào.
        kernel_size (tuple): Kích thước của phần tử cấu trúc.
    Returns:
        numpy.ndarray: Ảnh nhị phân sau khi đóng.
    """
    kernel = np.ones(kernel_size, np.uint8)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return closed_image