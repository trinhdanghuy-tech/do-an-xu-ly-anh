import cv2
from preprocessor import ImagePreprocessor
from locator import TrafficSignLocator
from classifier import TrafficSignClassifier

class TrafficSignSystem:
    """Pipeline tổng hợp"""
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.locator = TrafficSignLocator()
        self.classifier = TrafficSignClassifier()

    def run(self, img_path):
        original_img, enhanced_img, hsv_clahe, v_clahe = self.preprocessor.process(img_path)
        circles = self.locator.locate(hsv_clahe, v_clahe)
        results = self.classifier.classify(original_img, circles)

        # Vẽ kết quả
        final_img = original_img.copy()
        for (box, pred) in results:
            x1, y1, x2, y2 = box
            label, name, prob = pred
            cv2.rectangle(final_img, (x1, y1), (x2, y2), (0,255,0), 2)
            text = f"{name}: {prob*100:.2f}%"
            cv2.putText(final_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow('Ảnh gốc', original_img)
        cv2.imshow('Ảnh CLAHE', enhanced_img)
        cv2.imshow('Kết quả', final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
