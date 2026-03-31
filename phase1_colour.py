import cv2
import numpy as np
import os

COLOR_RANGES = {
    "red":    [((0,180,120),(10,255,255)), ((170,180,120),(180,255,255))],
    "blue":   [((100,180,80),(130,255,255))],
    "green":  [((40,120,80),(85,255,255))],
    "yellow": [((20,180,150),(35,255,255))],
    "white":  [((0,0,210),(180,20,255))],
    "black":  [((0,0,0),(180,255,35))],
    "pink":   [((160,100,150),(180,255,255)), ((0,100,150),(10,255,255))],
    "orange": [((10,180,120),(20,255,255))],
    "purple": [((130,120,80),(160,255,255))],
    "brown":  [((10,120,50),(20,160,120))],
    "grey":   [((0,0,110),(180,20,170))],
    "cyan":   [((85,120,80),(100,255,255))],
}

BOX_COLORS = {
    "red":    (0, 0, 255),
    "blue":   (255, 0, 0),
    "green":  (0, 255, 0),
    "yellow": (0, 255, 255),
    "white":  (255, 255, 255),
    "black":  (50, 50, 50),
    "pink":   (203, 192, 255),
    "orange": (0, 165, 255),
    "purple": (255, 0, 255),
    "brown":  (42, 42, 165),
    "grey":   (128, 128, 128),
    "cyan":   (255, 255, 0),
}

SKIN_LOW  = np.array([0,  20, 70],  dtype=np.uint8)
SKIN_HIGH = np.array([20, 150, 255], dtype=np.uint8)

# ── ASK USER FOR INPUT ──────────────────────────────
VIDEO_PATH   = input("Enter video filename (e.g. two_people_shirt.mp4): ")
VIDEO_PATH   = f"input/{VIDEO_PATH}"
TARGET_COLOR = input(f"Enter colour to search {list(COLOR_RANGES.keys())}: ").strip().lower()
FRAME_SKIP   = 10
MIN_PIXELS   = 12000
OUTPUT_DIR   = f"output/{TARGET_COLOR}_frames"
# ────────────────────────────────────────────────────

if TARGET_COLOR not in COLOR_RANGES:
    print(f"ERROR: '{TARGET_COLOR}' is not supported. Choose from {list(COLOR_RANGES.keys())}")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Could not open video at {VIDEO_PATH}")
    exit()

frame_num = 0
saved     = 0

print(f"\nSearching for '{TARGET_COLOR}' clothing in video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_num % FRAME_SKIP == 0:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Build colour mask
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for rng in COLOR_RANGES[TARGET_COLOR]:
            lo, hi = np.array(rng[0]), np.array(rng[1])
            mask  |= cv2.inRange(hsv, lo, hi)

        # Remove skin tone
        skin_mask = cv2.inRange(hsv, SKIN_LOW, SKIN_HIGH)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin_mask))

        # Remove top 40% of frame (ignore face/head area)
        height = frame.shape[0]
        mask[:int(height * 0.40), :] = 0

        # Merge nearby regions into one big area
        kernel = np.ones((60, 60), np.uint8)
        mask   = cv2.dilate(mask, kernel, iterations=3)
        mask   = cv2.erode(mask, kernel, iterations=1)

        if cv2.countNonZero(mask) >= MIN_PIXELS:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                area    = cv2.contourArea(largest)

                if area > 5000:
                    frame_with_box = frame.copy()
                    x, y, w, h     = cv2.boundingRect(largest)
                    box_color      = BOX_COLORS.get(TARGET_COLOR, (0, 255, 0))

                    cv2.rectangle(frame_with_box, (x, y), (x+w, y+h), box_color, 3)
                    cv2.putText(frame_with_box, f"{TARGET_COLOR} clothing",
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

                    fname = f"{OUTPUT_DIR}/frame_{frame_num:06d}.jpg"
                    cv2.imwrite(fname, frame_with_box)
                    print(f"  Saved → {fname}")
                    saved += 1

    frame_num += 1



cap.release()
print(f"\nFinished! {saved} frames saved to '{OUTPUT_DIR}/'")