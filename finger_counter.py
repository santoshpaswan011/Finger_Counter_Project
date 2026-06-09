import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import time
import os

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("ERROR: Model file 'hand_landmarker.task' not found.")
    print("Run: curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
    exit()

# ── Hand connections (hardcoded) ──────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
COL_PANEL  = (35,  35,  35)
COL_GREEN  = (0,  210,  90)
COL_RED    = (0,   60, 220)
COL_WHITE  = (240, 240, 240)
COL_YELLOW = (0,  220, 220)
COL_ORANGE = (0,  165, 255)
COL_CYAN   = (200, 200,   0)


def count_fingers(landmarks):
    up = []

    # ── Thumb: compare tip (4) vs MCP joint (2) using x distance from wrist ──
    wrist     = landmarks[0]
    thumb_mcp = landmarks[2]
    thumb_tip = landmarks[4]

    # Distance from wrist to tip vs wrist to MCP
    dist_tip = abs(thumb_tip.x - wrist.x)
    dist_mcp = abs(thumb_mcp.x - wrist.x)
    up.append(dist_tip > dist_mcp)   # tip further from wrist → thumb open

    # ── Other 4 fingers: tip y < PIP y → finger up ───────────────────────────
    tips  = [8, 12, 16, 20]
    pips  = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        up.append(landmarks[tip].y < landmarks[pip].y)

    return up


def draw_skeleton(frame, hand_landmarks_list):
    h, w = frame.shape[:2]
    for hand_lm in hand_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
        for s, e in HAND_CONNECTIONS:
            cv2.line(frame, pts[s], pts[e], COL_CYAN, 2)
        for cx, cy in pts:
            cv2.circle(frame, (cx, cy), 5, COL_WHITE, -1)
            cv2.circle(frame, (cx, cy), 5, COL_GREEN,  1)


def draw_panel(frame, detection_result, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, h), COL_PANEL, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_YELLOW, 2)

    if not detection_result or not detection_result.hand_landmarks:
        cv2.putText(frame, "No hand detected", (10, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_RED, 1)
        return

    draw_skeleton(frame, detection_result.hand_landmarks)

    total_fingers = 0
    for idx, (hand_lm, handedness) in enumerate(
        zip(detection_result.hand_landmarks, detection_result.handedness)
    ):
        label     = handedness[0].display_name
        finger_up = count_fingers(hand_lm)
        count     = sum(finger_up)
        total_fingers += count

        y = 70 + idx * 175
        cv2.putText(frame, f"{label} hand", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_ORANGE, 2)
        cv2.putText(frame, f"Count: {count}", (8, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COL_GREEN, 2)
        for fi, (name, up) in enumerate(zip(FINGER_NAMES, finger_up)):
            col  = COL_GREEN if up else COL_RED
            icon = "^" if up else "v"
            cv2.putText(frame, f" {icon} {name}", (8, y + 58 + fi * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    label_str = f"Total: {total_fingers}"
    tw = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_DUPLEX, 1.8, 3)[0][0]
    cv2.putText(frame, label_str, ((w - tw) // 2, h - 20),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, COL_WHITE, 3)


# ── Build HandLandmarker ──────────────────────────────────────────────────────
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# ── Main loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
prev_time = time.time()

print("Finger Counter running — press Q to quit.")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame    = cv2.flip(frame, 1)
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = landmarker.detect(mp_image)

    now       = time.time()
    fps       = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    draw_panel(frame, result, fps)
    cv2.imshow("Finger Counter  |  press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()