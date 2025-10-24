import cv2

class CameraFeed:
    displayed_frame = None
    read_frame = None

    def __init__(self, fps):
        self.vc = cv2.VideoCapture(0)
        self.fps = fps

        if self.vc.isOpened(): # try to get the first frame
            self.there_is_frame, self.read_frame = self.vc.read()
        else:
            self.there_is_frame = False

    async def show(self, frame = None):
        if frame is not None:
            self.read_frame = frame
        cv2.namedWindow("Camera Feed")
        while self.there_is_frame:
            # flip horizontally for a mirrored camera view
            if self.displayed_frame is None:
                self.displayed_frame = self.read_frame
            flipped = cv2.flip(self.displayed_frame, 1) #type:ignore
            cv2.imshow("Camera Feed", flipped)
            self.there_is_frame, self.read_frame = self.vc.read()
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                self.close()
        cv2.destroyWindow("Camera Feed")

    def close(self):
        if self.vc.isOpened():
            self.vc.release()

    def get_frame(self):
        if self.there_is_frame:
            return self.read_frame
        else:
            raise Exception("No frame available, maybe camera was already closed.")
