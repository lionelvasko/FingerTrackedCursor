import cv2
import numpy as np
import onnxruntime as ort

class HandDetector:
    def __init__(self, onnx_path="hand_landmark.onnx"):
        self.lower = np.array([0, 0, 0], dtype="uint8")
        self.upper = np.array([40, 255, 255], dtype="uint8")
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)
        self.prev_contour = None
        self.stability_counter = 0
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def detect_hand_by_onnx(self, frame):
        # 1. Gyors kézdetektálás (pl. bőrszín maszk + legnagyobb kontúr)
        mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), self.lower, self.upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return frame, None

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 3000:
            return frame, None

        x, y, w, h = cv2.boundingRect(cnt)
        roi = frame[y:y+h, x:x+w]
        
        # 2. Resize ROI a modellnek
        roi_resized = cv2.resize(roi, (224, 224))
        x_input = roi_resized.astype(np.float32).transpose(2,0,1)
        x_input = np.expand_dims(x_input, 0)

        # 3. Modell futtatás
        y_pred = self.session.run([self.output_name], {self.input_name: x_input})[0][0]  # (21,2)

        # 4. Landmark koordináták átszámítása az eredeti képbe
        landmarks = []
        for pt in y_pred:
            px = int(pt[0] * w) + x
            py = int(pt[1] * h) + y
            landmarks.append((px, py))
            cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

        return frame, np.array(landmarks)
