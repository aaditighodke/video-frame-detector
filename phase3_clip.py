import cv2
import numpy as np
import os
import clip
import torch
from PIL import Image
from ultralytics import YOLO

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

# ── ASK USER FOR INPUT ──────────────────────────────
VIDEO_PATH = input("Enter video filename (e.g. street.mp4): ")
VIDEO_PATH = f"input/{VIDEO_PATH}"
PROMPT     = input("Enter what to search (e.g. woman with yellow bag): ").strip()
COLOR      = input(f"Enter the colour of the object {list(COLOR_RANGES.keys())} (or press Enter to skip): ").strip().lower()
FRAME_SKIP = 15
OUTPUT_DIR = f"output/phase3_frames"
THRESHOLD  = 0.20
# ────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO
print("\nLoading YOLO model...")
model = YOLO("yolov8m.pt")
print("YOLO loaded! ✅")

# Load CLIP
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Multiple prompts for better accuracy
prompts = [
    PROMPT,
    f"a photo of {PROMPT}",
]
text_tokens = clip.tokenize(prompts).to(device)
print("CLIP loaded! ✅")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Could not open video at {VIDEO_PATH}")
    exit()

frame_num = 0
saved     = 0

print(f"\nSearching for: '{PROMPT}'...")

def check_colour(crop, colour):
    if colour not in COLOR_RANGES:
        return True  # if no colour specified skip colour check
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for rng in COLOR_RANGES[colour]:
        lo, hi = np.array(rng[0]), np.array(rng[1])
        mask  |= cv2.inRange(hsv, lo, hi)
    colour_pixels = cv2.countNonZero(mask)
    total_pixels  = hsv.shape[0] * hsv.shape[1]
    return total_pixels > 0 and (colour_pixels / total_pixels) > 0.08

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_num % FRAME_SKIP == 0:
        results          = model(frame, classes=[0], verbose=False)
        frame_with_boxes = frame.copy()
        box_drawn        = False

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                # Step 1 — CLIP checks if person matches prompt
                pil_image   = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                image_input = preprocess(pil_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input)
                    text_features  = clip_model.encode_text(text_tokens)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features  /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (image_features @ text_features.T).mean().item()

                # Step 2 — Colour check if colour was specified
                colour_match = check_colour(person_crop, COLOR)

                # Step 3 — Both CLIP and colour must match
                if similarity > THRESHOLD and colour_match:
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame_with_boxes, f"{PROMPT} ({similarity:.2f})",
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    box_drawn = True

        if box_drawn:
            fname = f"{OUTPUT_DIR}/frame_{frame_num:06d}.jpg"
            cv2.imwrite(fname, frame_with_boxes)
            print(f"  Saved → {fname}")
            saved += 1

    frame_num += 1

cap.release()
print(f"\nFinished! {saved} frames saved to '{OUTPUT_DIR}/'")