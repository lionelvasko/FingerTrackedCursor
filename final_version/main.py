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
            
            frame_with_edge = hand_detector.edge_detection(hand_detector.preprocess(frame))
            frame_with_skeleton = hand_detector.to_skeleton(hand_detector.preprocess(frame))
            
            camera_feed.displayed_frame = hand_detector.max_distance_from(frame_with_skeleton,frame_with_edge)
            camera_feed.displayed_frame2 = frame_with_skeleton
            await asyncio.sleep(1/fps)  # Simulate processing at given fps
            
    except KeyboardInterrupt:
        camera_feed.close()
    finally:
        camera_feed.close()

if __name__ == "__main__":
   asyncio.run(main())
