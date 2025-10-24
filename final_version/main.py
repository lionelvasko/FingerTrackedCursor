import functools
import asyncio
from hand_detection import HandDetector
from camera_feed import CameraFeed


camera_feed = CameraFeed(10)
hand_detector = HandDetector()

async def main(fps=10):
    asyncio.create_task(asyncio.to_thread(functools.partial(asyncio.run, camera_feed.show())))
    try:
        while True:
            frame = camera_feed.get_frame()
            frame_with_hand = hand_detector.detect_hand(frame)
            camera_feed.displayed_frame = frame_with_hand
            await asyncio.sleep(1/fps)  # Simulate processing at given fps
            
    except KeyboardInterrupt:
        camera_feed.close()
    finally:
        camera_feed.close()

if __name__ == "__main__":
   asyncio.run(main())
