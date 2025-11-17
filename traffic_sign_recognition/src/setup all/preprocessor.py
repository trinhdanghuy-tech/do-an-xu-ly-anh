import cv2
import numpy as np

class ImagePreprocessor:
    """Tiền xử lý ảnh: CLAHE trên kênh V của HSV"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def process(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Không thể tải ảnh từ {img_path}")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        v_clahe = clahe.apply(v)

        hsv_clahe = cv2.merge([h, s, v_clahe])
        enhanced_img = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

        return img, enhanced_img, hsv_clahe, v_clahe
