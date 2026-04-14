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
    "white":  [((0,0,200),(180,40,255))],
    "black":  [((0,0,0),(180,255,40))],
    "pink":   [((160,80,150),(180,255,255)), ((0,80,150),(10,255,255))],
    "orange": [((10,150,100),(20,255,255))],
    "purple": [((130,100,80),(160,255,255))],
    "brown":  [((10,100,50),(20,180,130))],
    "grey":   [((0,0,100),(180,25,180))],
    "cyan":   [((85,100,80),(100,255,255))],
}

# ALL detectable YOLO classes
YOLO_CLASSES = {
    "person":   0,
    "bicycle":  1,
    "car":      2,
    "motorcycle": 3,
    "bus":      5,
    "truck":    7,
    "dog":      16,
    "cat":      15,
    "horse":    17,
    "backpack": 24,
    "handbag":  26,
    "suitcase": 28,
}

def detect_colour_in_crop(crop, colour):
    if colour not in COLOR_RANGES:
        return True
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for rng in COLOR_RANGES[colour]:
        lo, hi = np.array(rng[0]), np.array(rng[1])
        mask  |= cv2.inRange(hsv, lo, hi)
    colour_pixels = cv2.countNonZero(mask)
    total_pixels  = hsv.shape[0] * hsv.shape[1]
    return total_pixels > 0 and (colour_pixels / total_pixels) > 0.10

def extract_colour_from_prompt(prompt):
    prompt_lower = prompt.lower()
    for colour in COLOR_RANGES.keys():
        if colour in prompt_lower:
            return colour
    return None

def run_detection(video_path, prompt, threshold, frame_skip, output_dir, progress_callback=None):
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    model      = YOLO("yolov8m.pt")
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Extract colour from prompt automatically
    colour = extract_colour_from_prompt(prompt)

    # Build CLIP prompts
    prompts = [
        prompt,
        f"a photo of {prompt}",
        f"a {prompt}",
        f"a clear image of {prompt}",
    ]
    text_tokens = clip.tokenize(prompts).to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num    = 0
    saved        = 0
    saved_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if progress_callback:
            progress_callback(frame_num, total_frames)

        if frame_num % frame_skip == 0:
            # Detect ALL objects
            results          = model(frame, verbose=False)
            frame_with_boxes = frame.copy()
            box_drawn        = False

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2-x1, y2-y1
                    if w < 30 or h < 30:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # CLIP check
                    pil_image   = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    image_input = preprocess(pil_image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        image_features = clip_model.encode_image(image_input)
                        text_features  = clip_model.encode_text(text_tokens)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features  /= text_features.norm(dim=-1, keepdim=True)
                        similarity = (image_features @ text_features.T).max().item()

                    # Colour check
                    colour_match = detect_colour_in_crop(crop, colour) if colour else True

                    if similarity > threshold and colour_match:
                        cv2.rectangle(frame_with_boxes, (x1,y1), (x2,y2), (0,255,0), 3)
                        cv2.putText(frame_with_boxes, f"{prompt[:30]} ({similarity:.2f})",
                                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        box_drawn = True

            if box_drawn:
                fname = f"{output_dir}/frame_{frame_num:06d}.jpg"
                cv2.imwrite(fname, frame_with_boxes)
                saved_frames.append(fname)
                saved += 1

        frame_num += 1

    cap.release()
    return saved, saved_frames