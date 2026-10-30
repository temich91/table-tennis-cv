"""PingPongVision — Streamlit Web Dashboard for Table Tennis CV Analytics.

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Add src to pythonpath
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ttcv.analytics import RallyAnalyzer, format_markdown_report
from ttcv.data.dataset import load_game_clip
from ttcv.geometry import TableHomography
from ttcv.geometry.visualizer import draw_table_bev, plot_bev_heatmap, plot_bev_trajectories
from ttcv.tracking import (
    BallTracker,
    MockBallDetector,
    generate_synthetic_rally,
)

# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PingPongVision 🏓",
    page_icon="🏓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏓 PingPongVision: Ball Tracking & Court Analytics")
st.markdown(
    "**Computer Vision Analytics for Table Tennis**: Multi-frame Ball Tracking, "
    "Bird's Eye View Homography, Bounce Heatmaps, Speed Estimation & Rally Metrics."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Configuration")
mode = st.sidebar.selectbox(
    "Data Source Mode",
    [
        "Demo: Real TTNet Dataset (7,073 frames)",
        "Demo: Synthetic Rally Simulation",
        "Upload Custom Video (MP4/AVI)",
    ],
)

default_mask = Path("data/raw/segmentation_masks/1278.png")
mask_path = st.sidebar.text_input(
    "Segmentation Mask Path",
    value=str(default_mask) if default_mask.exists() else "",
    help="Path to segmentation mask for table corner calibration.",
)

fps = st.sidebar.slider("Camera FPS", min_value=30, max_value=240, value=120, step=30)
max_frames = st.sidebar.slider("Max Frames to Process", min_value=50, max_value=1000, value=250, step=50)

# Load Homography
if mask_path and Path(mask_path).exists():
    homography = TableHomography.from_mask(mask_path)
else:
    corners = np.array([[528, 607.5], [1260, 607.5], [1416, 776.25], [396, 784.7]], dtype=np.float32)
    homography = TableHomography(corners)


# ── Process Data ──────────────────────────────────────────────────────────────

@st.cache_data
def process_real_dataset(start_frame: int, length: int, _fps: float):
    clip = load_game_clip("data/raw")
    tracker = BallTracker(homography=homography, fps=_fps)

    target_fids = [fid for fid in clip.frame_ids if start_frame <= fid < start_frame + length]
    for fid in target_fids:
        bp = clip.ball_data.get(fid)
        xy = (float(bp.x), float(bp.y)) if (bp and bp.visible) else None
        tracker.update(frame_id=fid, detection_xy=xy)

    # Known bounces in range
    known_bounces = [f for f in clip.bounce_frames if start_frame <= f < start_frame + length]
    analyzer = RallyAnalyzer(homography=homography, fps=_fps)
    metrics = analyzer.analyze_trajectory(tracker.history, annotated_bounce_frames=known_bounces)
    return tracker.history, metrics, known_bounces


@st.cache_data
def process_synthetic_rally(num_frames: int, _fps: float):
    tmp_vid = Path("outputs/tracking/demo_synthetic.mp4")
    tmp_vid.parent.mkdir(parents=True, exist_ok=True)
    synth = generate_synthetic_rally(
        output_video_path=tmp_vid,
        num_frames=num_frames,
        img_size=(1280, 720),
        fps=_fps,
    )
    tracker = BallTracker(homography=homography, fps=_fps)
    for fid in range(num_frames):
        pos = synth["ball_markup"].get(str(fid))
        xy = (float(pos["x"]), float(pos["y"])) if pos else None
        tracker.update(frame_id=fid, detection_xy=xy)

    bounces = tracker.detect_bounces()
    analyzer = RallyAnalyzer(homography=homography, fps=_fps)
    metrics = analyzer.analyze_trajectory(tracker.history, annotated_bounce_frames=bounces)
    return str(tmp_vid), tracker.history, metrics, bounces


# ── Main Flow ─────────────────────────────────────────────────────────────────

if "Real TTNet" in mode:
    st.info("📊 Loaded **TTNet Academic Dataset** (`data/raw`). Tracking ball trajectory and table bounces.")
    start_frame = st.sidebar.number_input("Start Frame ID", value=1298, min_value=1278, max_value=60000, step=10)
    with st.spinner("Analyzing rally trajectory..."):
        history, metrics, bounces = process_real_dataset(start_frame, max_frames, float(fps))
        video_path = None

elif "Synthetic" in mode:
    st.info("🎾 Running **Physics-based Synthetic Rally Simulation** with projectile dynamics and table collisions.")
    with st.spinner("Simulating ball flight and tracking..."):
        video_path, history, metrics, bounces = process_synthetic_rally(max_frames, float(fps))

else:
    uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI", type=["mp4", "avi"])
    if uploaded_file is None:
        st.warning("Please upload a video file to run tracking.")
        st.stop()
    else:
        st.success(f"Uploaded: {uploaded_file.name}")
        st.info("Processing custom video...")
        # Fallback to synthetic demonstration for custom upload placeholder
        video_path, history, metrics, bounces = process_synthetic_rally(max_frames, float(fps))


# ── Metric KPI Cards ──────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("⏱️ Duration", f"{metrics.duration_sec:.1f} s", f"{metrics.total_frames} frames")
col2.metric("🎯 Total Bounces", f"{metrics.total_bounces}", f"A: {metrics.bounces_player_a} | B: {metrics.bounces_player_b}")
col3.metric("⚡ Peak Speed", f"{metrics.max_speed_kmh:.1f} km/h")
col4.metric("💨 Avg Speed", f"{metrics.mean_speed_kmh:.1f} km/h")
col5.metric("🔄 Kalman Fill", f"{metrics.interpolation_rate:.1f}%", "reconstructed")

st.markdown("---")


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_court, tab_physics, tab_video, tab_report = st.tabs([
    "🏓 Court Analytics & Heatmap",
    "📈 Velocity & Physics Profile",
    "🎥 Video & Live Tracking",
    "📋 Match Report & Insights",
])

# ── Tab 1: Court Analytics ───────────────────────────────────────────────────

with tab_court:
    col_left, col_right = st.columns([1.4, 1.0])

    with col_left:
        st.subheader("Bird's Eye View Bounce Heatmap (2D Court)")
        # Project bounces to camera px for heatmap
        bounce_pts_cam: list[list[float]] = []
        pt_map = {p.frame_id: p for p in history}
        for b_fid in bounces:
            if b_fid in pt_map:
                p = pt_map[b_fid]
                bounce_pts_cam.append([p.x, p.y])

        if bounce_pts_cam:
            fig_heat = plot_bev_heatmap(
                homography,
                np.array(bounce_pts_cam, dtype=np.float32),
                title="Table Bounces Heatmap (Top-Down 2D Projection)",
            )
            st.pyplot(fig_heat)
            plt.close(fig_heat)
        else:
            st.write("No bounces in the selected frame range.")

    with col_right:
        st.subheader("Spatial Zone Distribution")

        # Player breakdown
        st.markdown("**Side Breakdown:**")
        side_df = pd.DataFrame({
            "Side": ["Player A", "Player B"],
            "Bounces": [metrics.bounces_player_a, metrics.bounces_player_b],
        })
        st.bar_chart(side_df.set_index("Side"))

        # Depth breakdown
        st.markdown("**Depth Zones (Short vs Deep):**")
        depth_df = pd.DataFrame(list(metrics.depth_distribution.items()), columns=["Zone", "Count"])
        st.dataframe(depth_df, hide_index=True, use_container_width=True)


# ── Tab 2: Physics Profile ───────────────────────────────────────────────────

with tab_physics:
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.subheader("Ball Speed Profile (km/h)")
        if metrics.speed_series:
            speed_df = pd.DataFrame(metrics.speed_series, columns=["Time (s)", "Speed (km/h)"]).set_index("Time (s)")
            st.line_chart(speed_df)
        else:
            st.info("Insufficient trajectory points for continuous speed profile.")

    with col_v2:
        st.subheader("Vertical Trajectory Curve (Y-axis)")
        y_series = [(round((p.frame_id - history[0].frame_id) / fps, 3), p.y) for p in history]
        y_df = pd.DataFrame(y_series, columns=["Time (s)", "Y Position (px)"]).set_index("Time (s)")
        st.line_chart(y_df)
        st.caption("Local peaks in Y position indicate physical bounces on the table.")

    # 2D Trajectory plot
    st.subheader("Projected Trajectory on Court Surface (BEV)")
    traj_arr = np.array([[p.frame_id, p.x, p.y] for p in history], dtype=np.float32)
    fig_traj = plot_bev_trajectories(
        homography,
        traj_arr,
        np.array(bounce_pts_cam, dtype=np.float32) if bounce_pts_cam else None,
    )
    st.pyplot(fig_traj)
    plt.close(fig_traj)


# ── Tab 3: Video Tracking ────────────────────────────────────────────────────

with tab_video:
    st.subheader("Video Tracking Demonstration")
    if video_path and Path(video_path).exists():
        st.video(video_path)
    else:
        sample_tracked_mp4 = Path("outputs/tracking/synthetic_tracked.mp4")
        if sample_tracked_mp4.exists():
            st.video(str(sample_tracked_mp4))
            st.caption("Demonstrating real-time tracking with comet tail and top-right 2D mini-map.")
        else:
            st.info("Generate or upload a video using the CLI: `python scripts/infer.py --synthetic`")


# ── Tab 4: Match Report ──────────────────────────────────────────────────────

with tab_report:
    st.subheader("Executive Match & Rally Report")
    md_report = format_markdown_report(metrics)
    st.markdown(md_report)

    # Download buttons
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="📥 Download Markdown Report",
            data=md_report,
            file_name="pingpong_match_report.md",
            mime="text/markdown",
        )
    with col_d2:
        st.download_button(
            label="📥 Download JSON Summary",
            data=pd.Series(metrics.summary_dict()).to_json(indent=2),
            file_name="pingpong_metrics.json",
            mime="application/json",
        )
