import cv2
import numpy as np
import imutils

class HandDetector:
    def __init__(self):
        self.lower = np.array([0, 0, 0], dtype="uint8")
        self.upper = np.array([40, 255, 255], dtype="uint8")
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)
        self.prev_contour = None
        self.stability_counter = 0

    def detect_hand(self, frame):
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        im = cv2.GaussianBlur(im, (5, 5), sigmaX=2.0, sigmaY=2.0)
        im = cv2.inRange(
                im,
                np.array([ 0, 40, 80 ], dtype="uint8"),
                np.array([255, 255, 255 ], dtype="uint8"),
        )
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        im= cv2.erode(im, element)
        im= cv2.erode(im, element)
        im= cv2.dilate(im, element)
        skeleton = np.zeros(im.shape, dtype="uint8")
        counter = 0
        while(counter<500):
            eroded = cv2.erode(im, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(im, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            im = eroded.copy()
            counter+=1
        return skeleton

    def detect_hand_by_hls_adaptive_threshold(self, frame):
        """HLS színtér alapú kézdetektálás adaptív küszöböléssel"""
        # Kép előfeldolgozás
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS_FULL)
        im = cv2.inRange(im, self.lower, self.upper)

        im = cv2.GaussianBlur(im, (7, 7), 3)

        # Adaptív küszöbölés és invertálás
        thresh = cv2.adaptiveThreshold(
            im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        thresh = cv2.bitwise_not(thresh)

        # Fix küszöbölés is (ha az adaptív nem elég)
        _, th = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)

        # Kontúrok keresése
        contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        if len(contours) == 0:
            return frame, 0  # nincs kéz

        # Legnagyobb kontúr (legvalószínűbb kéz)
        cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

        # Convex hull + defects számítás
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            return frame, 0  # túl kevés pont, nincs kéz

        hull = cv2.convexHull(approx, returnPoints=False)
        if hull is None or len(hull) < 3:
            return frame, 0

        defects = cv2.convexityDefects(approx, hull)
        if defects is None:
            return frame, 0

        return frame

    def detect_hand_by_edge_and_skin_color(self, frame):
        """Éldetektálás (Canny) és HSV bőrszín alapú kézdetektálás"""
        # Szürkeárnyalat + GaussianBlur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Élkeresés
        edges = cv2.Canny(blur, 50, 150)

        # HSV bőrszín maszk
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # Kombináljuk az éldetektálást és a bőrszűrést
        combined = cv2.bitwise_and(edges, mask)

        # Kontúrok keresése
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3000:  # zaj kiszűrése
                # Rajzoljuk ki a kontúrt
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

                # Konvex burok
                hull = cv2.convexHull(cnt)
                cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)

                # Körülrajzolás téglalappal
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        return frame


    def detect_hand_by_motion_and_skin_color(self, frame):
        """Mozgásérzékelés (MOG2) és HSV bőrszín alapú kézdetektálás stabilitással"""
        frame_copy = frame.copy()
        h, w = frame_copy.shape[:2]

        # --- 1. Mozgásmaszk ---
        fg_mask = self.bg_subtractor.apply(frame_copy)
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        fg_mask = cv2.erode(fg_mask, None, iterations=1)
        fg_mask = cv2.dilate(fg_mask, None, iterations=2)

        # --- 2. Bőrszín maszk ---
        hsv = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, self.lower, self.upper)
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

        # --- 3. Maszkok kombinálása ---
        combined_mask = cv2.bitwise_and(fg_mask, skin_mask)
        _, thresh = cv2.threshold(combined_mask, 50, 255, cv2.THRESH_BINARY)

        # --- 4. Kontúrok ---
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_contour = None
        if contours:
            detected_contour = max(contours, key=lambda x: cv2.contourArea(x))
            if cv2.contourArea(detected_contour) < 3000:  # minimum méret
                detected_contour = None

        # --- 5. Stabilizálás ---
        if detected_contour is not None:
            self.prev_contour = detected_contour
            self.stability_counter = 3
        else:
            self.stability_counter -= 1
            if self.stability_counter <= 0:
                self.prev_contour = None

        # --- 6. Rajzolás ---
        if self.prev_contour is not None:
            cv2.drawContours(frame_copy, [self.prev_contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(self.prev_contour)
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame_copy, "Hand", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame_copy
