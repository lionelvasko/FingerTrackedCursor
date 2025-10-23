import cv2
import time
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from threading import Lock
from pathlib import Path


# === Alapbeállítások ===
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtils = mp.solutions.drawing_utils
DrawingStyles = mp.solutions.drawing_styles

p = Path(__file__).resolve().parent.parent / "mediapipe_version" / "resources" / "hand_landmarker.task"
if not p.exists():
    raise FileNotFoundError(f"Model file not found at {p}")
MODEL_PATH = str(p)

latest_result = None
result_lock = Lock()

def handle_result(result, output_image, timestamp_ms):
    global latest_result
    with result_lock:
        latest_result = result

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

  return annotated_image

OPTIONS = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    result_callback=handle_result
)

with HandLandmarker.create_from_options(OPTIONS) as landmarker:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Nem sikerült megnyitni a kamerát.")
        exit()

    print("📷 Kamera elindítva. Nyomj ESC-et a kilépéshez.")
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            with result_lock:
                result = latest_result

            if result:
                frame = draw_landmarks_on_image(frame, result)

            cv2.imshow("Hand Landmarks (LIVE_STREAM mode)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
