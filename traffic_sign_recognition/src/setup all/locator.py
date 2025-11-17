import cv2
import numpy as np

class TrafficSignLocator:
    """Định vị biển báo bằng HSV mask + Hough Circles"""
    def __init__(self):
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([140, 255, 255])

    def locate(self, hsv_clahe, v_clahe):
        mask_red1 = cv2.inRange(hsv_clahe, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv_clahe, self.lower_red2, self.upper_red2)
        mask_blue = cv2.inRange(hsv_clahe, self.lower_blue, self.upper_blue)
        color_mask = cv2.bitwise_or(mask_red1, cv2.bitwise_or(mask_red2, mask_blue))

        gray = cv2.bitwise_and(v_clahe, v_clahe, mask=color_mask)
        gray_blurred = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=30,
            minRadius=15,
            maxRadius=100
        )

        num_circles = len(circles[0]) if circles is not None else 0
        print(f"Đã phát hiện {num_circles} biển báo.")
        return circles
