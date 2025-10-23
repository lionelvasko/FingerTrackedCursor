import json
import cv2
import time
import actions
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from threading import Lock
from gestures import detect_gesture
from pathlib import Path

with open("gesture-map.json", "r") as f:
    GESTURE_MAP = json.load(f)

# === Alapbeállítások ===
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtils = mp.solutions.drawing_utils
DrawingStyles = mp.solutions.drawing_styles
mp_draw = mp.solutions.drawing_utils

p = Path(__file__).resolve().parent.parent / "mediapipe-version" / "resources" / "hand-landmarker.task"
if not p.exists():
    raise FileNotFoundError(f"Model file not found at {p}")
with open(p, "rb") as f:
    MODEL_BYTES = f.read()

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
    base_options=BaseOptions(model_asset_buffer=MODEL_BYTES),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    result_callback=handle_result
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

with HandLandmarker.create_from_options(OPTIONS) as landmarker:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        exit()

    print("📷 Camera started. Press ESC to exit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            screen_w, screen_h = pyautogui.size()

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            with result_lock:
                result = latest_result

            if result and result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    landmark_proto = landmark_pb2.NormalizedLandmarkList(
                        landmark=[
                            landmark_pb2.NormalizedLandmark(
                                x=lm.x, y=lm.y, z=lm.z
                            ) for lm in hand_landmarks
                        ]
                    )
                    mp_draw.draw_landmarks(frame, landmark_proto, mp_hands.HAND_CONNECTIONS)

                    gesture = detect_gesture(hand_landmarks)
                    if gesture and gesture in GESTURE_MAP:
                        action_name = GESTURE_MAP[gesture]
                        action_fn = getattr(actions, action_name, None)
                        if action_fn:
                            if action_name == "move":
                                x = int(hand_landmarks[8].x * screen_w)
                                y = int(hand_landmarks[8].y * screen_h)
                                action_fn(x, y)
                            else:
                                action_fn()
                
            cv2.imshow("Hand Landmarks (LIVE_STREAM mode)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
