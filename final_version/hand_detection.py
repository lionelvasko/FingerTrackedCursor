import cv2
import numpy as np
import imutils
from cv2 import typing

class HandDetector:
    def __init__(self):
        self.lower = np.array([0, 0, 0], dtype="uint8")
        self.upper = np.array([40, 255, 255], dtype="uint8")
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)
        self.prev_contour = None
        self.stability_counter = 0

    def preprocess(self, frame)-> typing.MatLike:
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

        return im
    
    def to_skeleton(self, im, element=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), counter_max=200) -> typing.MatLike: 
        skeleton = np.zeros(im.shape, dtype="uint8")
        counter = 0
        while(counter<counter_max):
            eroded = cv2.erode(im, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(im, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            im = eroded.copy()
            counter+=1
        return skeleton
    
    def edge_detection(self, im) -> typing.MatLike:
        im = cv2.Laplacian(im, cv2.CV_8U, ksize=3)
        return im
    
    def add_images(self, im1, im2) -> typing.MatLike:
        return cv2.add(im1, im2)

    def create_circular_mask(self, h, w, center=None, radius=None):

        if center is None: 
            center = (int(w/2), int(h/2))
        if radius is None: 
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask
    
    def max_distance_from(self, skeleton: typing.MatLike, edges: typing.MatLike, distance = 15) -> typing.MatLike:
        skeleton =  cv2.pyrDown(cv2.pyrDown(skeleton))
        edges =  cv2.pyrDown(cv2.pyrDown(edges))

        (x, y) = np.nonzero(skeleton)

        x_float32 = x[:].astype(np.float32)
        y_float32 = y[:].astype(np.float32)

        points = np.array(list(zip(x_float32, y_float32)))
        corrent_points = np.zeros(skeleton.shape)

        countours = np.array(edges).reshape((-1,1,2)).astype(np.int32)


        for point in points: #type: ignore
            eredmeny = cv2.pointPolygonTest(countours, point, True)
            if abs(eredmeny) < distance:
                corrent_points[int(point[0])][int(point[1])] = 255
                
        return cv2.pyrUp(cv2.pyrUp(corrent_points))

    def finger(self, im):
        positions = np.nonzero(im)

        if positions[0].size ==0:
            return (0, 0, 0,0)

        top = positions[0].min()
        bottom = positions[0].max()
        left = positions[1].min()
        right = positions[1].max()

        return (left, top, right, bottom)

    def draw_rect(self, im, coordinates):
        im = cv2.rectangle(im, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), (0, 255, 0), 1)
        return im


        


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
