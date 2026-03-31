import cv2
import numpy as np
import os

# ── CHANGE THESE ──────────────────────────────────────
VIDEO_PATH   = "input/your_video.mp4"
OUTPUT_DIR   = "output/matched_frames"
TARGET_COLOR = "red"   # red / blue / green / yellow / white / black
FRAME_SKIP   = 10
MIN_PIXELS   = 5000
# ──────────────────────────────────────────────────────

COLOR_RANGES = {
    "red":    [((0,80,50),(10,255,255)), ((170,80,50),(180,255,255))],
    "blue":   [((100,80,50),(130,255,255))],
    "green":  [((40,60,50),(85,255,255))],
    "yellow": [((20,100,100),(35,255,255))],
    "white":  [((0,0,180),(180,40,255))],
    "black":  [((0,0,0),(180,255,50))],
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Could not open video at {VIDEO_PATH}")
    exit()

frame_num = 0
saved     = 0

print(f"Searching for '{TARGET_COLOR}' clothing in video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_num % FRAME_SKIP == 0:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for rng in COLOR_RANGES[TARGET_COLOR]:
            lo, hi = np.array(rng[0]), np.array(rng[1])
            mask  |= cv2.inRange(hsv, lo, hi)

        if cv2.countNonZero(mask) >= MIN_PIXELS:
            fname = f"{OUTPUT_DIR}/frame_{frame_num:06d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"  Saved → {fname}")
            saved += 1

    frame_num += 1

cap.release()
print(f"\nFinished! {saved} frames saved to '{OUTPUT_DIR}/'")