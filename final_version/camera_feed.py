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

    async def show(self):
        cv2.namedWindow("Camera Feed")
        while self.rval:
            cv2.imshow("Camera Feed", self.frame)
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
    

# Function for example usage
async def main(fps=10):
    camera_feed = CameraFeed()

    # You can start a the camera feed display in a separate thread this way
    asyncio.create_task(asyncio.to_thread(functools.partial(asyncio.run, camera_feed.show())))

    while True:
        frame = camera_feed.get_frame()
        await asyncio.sleep(1 / fps)  # Simulate processing at given fps

if __name__ == "__main__":
    asyncio.run(main())