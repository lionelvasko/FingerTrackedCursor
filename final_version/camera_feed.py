import cv2
import asyncio
import functools

class CameraFeed:
    def __init__(self):
        self.vc = cv2.VideoCapture(0)

        if self.vc.isOpened(): # try to get the first frame
            self.rval, self.frame = self.vc.read()
        else:
            self.rval = False

    async def show(self, frame = None):
        if frame is not None:
            self.frame = frame
        cv2.namedWindow("Camera Feed")
        while self.rval:
            # flip horizontally for a mirrored camera view
            flipped = cv2.flip(self.frame, 1)
            cv2.imshow("Camera Feed", flipped)
            self.rval, self.frame = self.vc.read()
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                self.close()
        cv2.destroyWindow("Camera Feed")

    def close(self):
        if self.vc.isOpened():
            self.vc.release()

    def get_frame(self):
        if self.rval:
            return self.frame
        else:
            raise Exception("No frame available, maybe camera was already closed.")