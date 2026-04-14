import cv2
import os
from twelvelabs import TwelveLabs
from twelvelabs.models.task import Task
import time

# ========== USER INPUT ==========
VIDEO_PATH = input("Enter video filename (e.g. street.mp4): ")
VIDEO_PATH = f"input/{VIDEO_PATH}"
PROMPT = input("Enter what to search (e.g., a lady with a blue bag): ").strip()
THRESHOLD = float(input("Enter confidence threshold (0.5 to 0.9, default 0.7): ") or "0.7")
OUTPUT_DIR = "output/marengo_frames"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("input", exist_ok=True)

# ========== INITIALIZE MARENGO ==========
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key

print("\n" + "="*50)
print("🚀 INITIALIZING MARENGO 2.6...")
print("="*50)

client = TwelveLabs(api_key=API_KEY)

# ========== CREATE INDEX ==========
print("\n📁 Creating index for video...")

# Create an index for your video (UPDATED for latest SDK)
try:
    index = client.indexes.create(
        name="cctv_search_index",
        engines=[
            {
                "name": "marengo2.6",
                "options": ["visual", "conversation", "text_in_video"],
            },
        ],
    )
    print(f"✅ Index created: {index.id}")
except Exception as e:
    print(f"❌ Error creating index: {e}")
    exit()

# ========== UPLOAD AND INDEX VIDEO ==========
print(f"\n📤 Uploading and indexing video: {VIDEO_PATH}")
print("⏳ This may take a few minutes depending on video length...")

# Create task for video upload and indexing (UPDATED for latest SDK)
try:
    task = client.tasks.create(
        index_id=index.id,
        file=VIDEO_PATH,
        language="en"
    )
    print(f"Task created: {task.id}")

    # Monitor progress (UPDATED for latest SDK)
    def on_task_update(task: Task):
        print(f"  Status={task.status}")
    
    task.wait_for_done(callback=on_task_update)
    
    if task.status != "ready":
        print(f"❌ Indexing failed with status {task.status}")
        exit()
    
    print("✅ Video indexing complete!")
    print(f"Video ID: {task.video_id}")
    
except Exception as e:
    print(f"❌ Error uploading video: {e}")
    exit()

# ========== SEARCH WITH YOUR PROMPT ==========
print(f"\n🔍 Searching for: '{PROMPT}'")
print(f"   Confidence threshold: {THRESHOLD}")

# Search using text query (UPDATED for latest SDK)
try:
    search_results = client.search.query(
        index_id=index.id,
        query_text=PROMPT,
        options=["visual", "conversation", "text_in_video"]
    )
    
    # Collect results (UPDATED for latest SDK - results are paginated)
    results_list = []
    for page in search_results:
        for clip in page:
            results_list.append(clip)
    
    print(f"\n📊 Found {len(results_list)} matching clips!")
    
except Exception as e:
    print(f"❌ Error during search: {e}")
    exit()

# ========== SAVE MATCHING FRAMES ==========
if len(results_list) == 0:
    print("\n❌ No matches found. Try lowering the threshold.")
    exit()

print(f"\n💾 Saving matching frames to: {OUTPUT_DIR}/")

cap = cv2.VideoCapture(VIDEO_PATH)
saved_count = 0

for i, clip in enumerate(results_list):
    timestamp = clip.start  # seconds
    score = clip.score  # Use score instead of confidence (newer API)
    
    # Skip if score is below threshold
    if score < THRESHOLD * 100:  # score is 0-100 scale
        continue
    
    # Jump to the timestamp
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    
    if ret:
        # Save frame
        filename = f"{OUTPUT_DIR}/match_{i+1:03d}_{timestamp:.2f}s_score_{score:.1f}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Saved: {filename}")
        saved_count += 1
        
        # Save metadata
        with open(f"{OUTPUT_DIR}/metadata.txt", "a") as f:
            f.write(f"{filename} | Timestamp: {timestamp:.2f}s | Score: {score:.1f}\n")
    else:
        print(f"⚠️ Could not read frame at {timestamp:.2f}s")

cap.release()

# ========== SUMMARY ==========
print("\n" + "="*50)
print("🎉 COMPLETE!")
print("="*50)
print(f"📹 Video processed: {VIDEO_PATH}")
print(f"🔍 Search prompt: '{PROMPT}'")
print(f"✅ Matching clips found: {len(results_list)}")
print(f"💾 Frames saved: {saved_count}")
print(f"📁 Output directory: {OUTPUT_DIR}/")
print("\nCheck 'metadata.txt' for timestamps and confidence scores.")