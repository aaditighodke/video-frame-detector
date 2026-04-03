import cv2
import numpy as np
import os
from ultralytics import YOLO

# ── COLOUR RANGES (HSV) ─────────────────────────────
COLOR_RANGES = {
    "red":    [((0,120,70),(10,255,255)), ((170,120,70),(180,255,255))],
    "blue":   [((100,150,80),(130,255,255))],
    "green":  [((40,100,80),(85,255,255))],
    "yellow": [((20,150,150),(35,255,255))],
    "white":  [((0,0,200),(180,30,255))],
    "black":  [((0,0,0),(180,255,40))],
    "pink":   [((160,80,150),(180,255,255)), ((0,80,150),(10,255,255))],
    "orange": [((10,150,100),(20,255,255))],
    "purple": [((130,100,80),(160,255,255))],
    "brown":  [((10,100,50),(20,180,130))],
    "grey":   [((0,0,100),(180,25,180))],
    "cyan":   [((85,100,80),(100,255,255))],
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

# ── ASK USER FOR INPUT ──────────────────────────────
VIDEO_PATH   = input("Enter video filename (e.g. street.mp4): ")
VIDEO_PATH   = f"input/{VIDEO_PATH}"
TARGET_COLOR = input(f"Enter colour to search {list(COLOR_RANGES.keys())}: ").strip().lower()
FRAME_SKIP   = 10
OUTPUT_DIR   = f"output/phase2_{TARGET_COLOR}_frames"
# ────────────────────────────────────────────────────

if TARGET_COLOR not in COLOR_RANGES:
    print(f"ERROR: '{TARGET_COLOR}' not supported!")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 model (downloads automatically first time)
print("\nLoading YOLO model...")
model = YOLO("yolov8m.pt")  # n = nano (smallest, fastest)
print("YOLO model loaded! ✅")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Could not open video at {VIDEO_PATH}")
    exit()

frame_num = 0
saved     = 0

print(f"\nSearching for persons wearing '{TARGET_COLOR}' clothing...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_num % FRAME_SKIP == 0:
        # Step 1 — YOLO detects all persons in frame
        results = model(frame, classes=[0], verbose=False)  # class 0 = person

        frame_with_boxes = frame.copy()
        box_drawn        = False
        hsv              = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for result in results:
            for box in result.boxes:
                # Get person bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Step 2 — Crop the person region
                person_crop = hsv[y1:y2, x1:x2]

                if person_crop.size == 0:
                    continue

                # Step 3 — Check clothing colour (bottom 60% of person = body/clothing)
                body_height  = y2 - y1
                clothing_crop = person_crop[int(body_height * 0.25):, :]

                # Step 4 — Build colour mask on clothing only
                mask = np.zeros(clothing_crop.shape[:2], dtype=np.uint8)
                for rng in COLOR_RANGES[TARGET_COLOR]:
                    lo, hi = np.array(rng[0]), np.array(rng[1])
                    mask  |= cv2.inRange(clothing_crop, lo, hi)

                colour_pixels = cv2.countNonZero(mask)
                total_pixels  = clothing_crop.shape[0] * clothing_crop.shape[1]

                # Step 5 — If more than 10% of clothing area matches colour → save
                if total_pixels > 0 and (colour_pixels / total_pixels) > 0.20:
                    box_color = BOX_COLORS.get(TARGET_COLOR, (0, 255, 0))
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), box_color, 3)
                    cv2.putText(frame_with_boxes, f"{TARGET_COLOR} clothing",
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                    box_drawn = True

        if box_drawn:
            fname = f"{OUTPUT_DIR}/frame_{frame_num:06d}.jpg"
            cv2.imwrite(fname, frame_with_boxes)
            print(f"  Saved → {fname}")
            saved += 1

    frame_num += 1

cap.release()
print(f"\nFinished! {saved} frames saved to '{OUTPUT_DIR}/'")