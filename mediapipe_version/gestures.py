import math

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def detect_gesture(hand_landmarks):
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]
    ring_tip = hand_landmarks[16]
    pinky_tip = hand_landmarks[20]

    thumb_index = distance(thumb_tip, index_tip)
    index_middle = distance(index_tip, middle_tip)

    if thumb_index < 0.05:
        return "pinch"
    elif all(tip.y < hand_landmarks[0].y for tip in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "open_palm"
    elif all(distance(hand_landmarks[i], hand_landmarks[i+2]) < 0.03 for i in [4,8,12,16,20]):
        return "fist"
    elif index_tip.y < middle_tip.y and middle_tip.y < ring_tip.y:
        return "two_fingers"
    else:
        return None
