import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

class TrafficSignClassifier:
    """Phân loại ROI bằng ResNet50"""
    def __init__(self, model=None):
        if model is None:
            print("Đang tải ResNet50 pretrained trên ImageNet...")
            self.model = ResNet50(weights='imagenet')
            print("Đã tải xong.")
        else:
            self.model = model

    def classify(self, original_img, circles, resize_dim=(224,224)):
        results = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                x1 = max(0, x-r-5)
                y1 = max(0, y-r-5)
                x2 = min(original_img.shape[1], x+r+5)
                y2 = min(original_img.shape[0], y+r+5)

                roi = original_img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                roi_resized = cv2.resize(roi, resize_dim)
                img_array = image.img_to_array(roi_resized)
                img_batch = np.expand_dims(img_array, axis=0)
                img_preprocessed = preprocess_input(img_batch)

                preds = self.model.predict(img_preprocessed, batch_size=1)
                decoded = decode_predictions(preds, top=1)[0]
                results.append(((x1, y1, x2, y2), decoded[0]))
                print(f"ROI tại {(x, y)} -> Dự đoán: {decoded[0]}")
        return results
