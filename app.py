import streamlit as st
import os
import tempfile
from detector import run_detection
from PIL import Image

# ── PAGE CONFIG ─────────────────────────────────────
st.set_page_config(
    page_title="Smart Video Search",
    page_icon="🔍",
    layout="wide"
)

# ── STYLES ──────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────
st.markdown('<p class="main-title">🔍 Smart Video Search System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload a video and search for anything using natural language</p>', unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    threshold = st.slider(
        "Detection Sensitivity",
        min_value=0.10,
        max_value=0.40,
        value=0.22,
        step=0.01,
        help="Lower = more detections, Higher = stricter matching"
    )

    frame_skip = st.slider(
        "Frame Skip",
        min_value=1,
        max_value=30,
        value=5,
        help="Check every Nth frame. Lower = more accurate but slower"
    )

    st.markdown("---")
    st.markdown("### 💡 Example Prompts")
    st.markdown("""
    - `man in white shirt`
    - `woman with red bag`
    - `black car`
    - `person with backpack`
    - `yellow taxi`
    - `person in blue jacket`
    - `white police van`
    - `dog on street`
    """)

# ── MAIN CONTENT ────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📹 Upload Video")
    video_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if video_file:
        st.video(video_file)

with col2:
    st.subheader("🔍 Search Prompt")
    prompt = st.text_input(
        "What are you looking for?",
        placeholder="e.g. man in white shirt carrying black bag"
    )

    st.markdown("**Search Category:**")
    search_type = st.radio(
        "",
        ["Everything", "Person only", "Vehicle only", "Animal only"],
        horizontal=True
    )

    search_button = st.button("🚀 Start Search", use_container_width=True, type="primary")

# ── RUN DETECTION ───────────────────────────────────
if search_button:
    if not video_file:
        st.error("❌ Please upload a video first!")
    elif not prompt:
        st.error("❌ Please enter a search prompt!")
    else:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        output_dir = "output/smart_search"

        st.markdown("---")
        st.subheader("⏳ Processing...")

        progress_bar  = st.progress(0)
        status_text   = st.empty()

        def update_progress(current, total):
            if total > 0:
                pct = min(int((current / total) * 100), 100)
                progress_bar.progress(pct)
                status_text.text(f"Processing frame {current} of {total}...")

        # Run detection
        saved, saved_frames = run_detection(
            video_path        = tmp_path,
            prompt            = prompt,
            threshold         = threshold,
            frame_skip        = frame_skip,
            output_dir        = output_dir,
            progress_callback = update_progress
        )

        progress_bar.progress(100)
        status_text.text("✅ Done!")

        # ── RESULTS ─────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Results")

        if saved == 0:
            st.warning("⚠️ No matching frames found! Try lowering the sensitivity or changing your prompt.")
        else:
            st.success(f"✅ Found **{saved} matching frames**!")

            st.markdown("### 🖼️ Matched Frames")

            # Show frames in grid
            cols = st.columns(3)
            for i, frame_path in enumerate(saved_frames[:30]):
                with cols[i % 3]:
                    img = Image.open(frame_path)
                    st.image(img, caption=f"Frame {i+1}", use_container_width=True)

            if saved > 30:
                st.info(f"Showing first 30 of {saved} frames. All frames saved to `{output_dir}/`")

        # Cleanup temp file
        os.unlink(tmp_path)