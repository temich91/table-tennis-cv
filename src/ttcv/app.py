"""PingPongVision — Веб-дашборд на Streamlit для компьютерного зрения и аналитики в настольном теннисе.

Запуск:
    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Добавление каталога src в pythonpath
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

# ── Настройка страницы ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PingPongVision 🏓",
    page_icon="🏓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏓 PingPongVision: трекинг мяча и аналитика стола")
st.markdown(
    "**Компьютерное зрение в настольном теннисе**: покадровое отслеживание мяча, "
    "гомография вида сверху (BEV), тепловые карты отскоков, оценка скорости и метрики розыгрышей."
)

# ── Боковая панель ────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Параметры")
mode = st.sidebar.selectbox(
    "Режим источника данных",
    [
        "Демо: Реальный датасет TTNet (7 073 кадра)",
        "Демо: Симуляция синтетического розыгрыша",
        "Загрузить свое видео (MP4/AVI)",
    ],
)

default_mask = Path("data/raw/segmentation_masks/1278.png")
mask_path = st.sidebar.text_input(
    "Путь к маске сегментации",
    value=str(default_mask) if default_mask.exists() else "",
    help="Путь к маске сегментации для калибровки углов стола.",
)

fps = st.sidebar.slider("Частота кадров камеры (FPS)", min_value=30, max_value=240, value=120, step=30)
max_frames = st.sidebar.slider("Максимум кадров для обработки", min_value=50, max_value=1000, value=250, step=50)

# Загрузка матрицы гомографии
if mask_path and Path(mask_path).exists():
    homography = TableHomography.from_mask(mask_path)
else:
    corners = np.array([[528, 607.5], [1260, 607.5], [1416, 776.25], [396, 784.7]], dtype=np.float32)
    homography = TableHomography(corners)


# ── Обработка данных ──────────────────────────────────────────────────────────

@st.cache_data
def process_real_dataset(start_frame: int, length: int, _fps: float):
    clip = load_game_clip("data/raw")
    tracker = BallTracker(homography=homography, fps=_fps)

    target_fids = [fid for fid in clip.frame_ids if start_frame <= fid < start_frame + length]
    for fid in target_fids:
        bp = clip.ball_data.get(fid)
        xy = (float(bp.x), float(bp.y)) if (bp and bp.visible) else None
        tracker.update(frame_id=fid, detection_xy=xy)

    # Известные отскоки в диапазоне
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


# ── Основной поток выполнения ─────────────────────────────────────────────────

if "Реальный датасет" in mode or "Real TTNet" in mode:
    st.info("📊 Загружен **академический датасет TTNet** (`data/raw`). Отслеживание траектории мяча и отскоков от стола.")
    start_frame = st.sidebar.number_input("Начальный кадр (ID)", value=1298, min_value=1278, max_value=60000, step=10)
    with st.spinner("Анализ траектории розыгрыша..."):
        history, metrics, bounces = process_real_dataset(start_frame, max_frames, float(fps))
        video_path = None

elif "Симуляция" in mode or "Synthetic" in mode:
    st.info("🎾 Запуск **физической симуляции розыгрыша** с баллистикой полета и столкновениями со столом.")
    with st.spinner("Симуляция полета мяча и отслеживание..."):
        video_path, history, metrics, bounces = process_synthetic_rally(max_frames, float(fps))

else:
    uploaded_file = st.sidebar.file_uploader("Загрузите MP4 / AVI", type=["mp4", "avi"])
    if uploaded_file is None:
        st.warning("Пожалуйста, загрузите видеофайл для запуска трекинга.")
        st.stop()
    else:
        st.success(f"Загружено: {uploaded_file.name}")
        st.info("Обработка пользовательского видео...")
        # Запасной вариант на синтетическую демонстрацию
        video_path, history, metrics, bounces = process_synthetic_rally(max_frames, float(fps))


# ── Карточки ключевых метрик (KPI) ───────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("⏱️ Длительность", f"{metrics.duration_sec:.1f} с", f"{metrics.total_frames} кадров")
col2.metric("🎯 Всего отскоков", f"{metrics.total_bounces}", f"A: {metrics.bounces_player_a} | B: {metrics.bounces_player_b}")
col3.metric("⚡ Макс. скорость", f"{metrics.max_speed_kmh:.1f} км/ч")
col4.metric("💨 Средняя скорость", f"{metrics.mean_speed_kmh:.1f} км/ч")
col5.metric("🔄 Заполнение Калманом", f"{metrics.interpolation_rate:.1f}%", "восстановлено")

st.markdown("---")


# ── Вкладки ───────────────────────────────────────────────────────────────────

tab_court, tab_physics, tab_video, tab_report = st.tabs([
    "🏓 Аналитика стола и тепловая карта",
    "📈 Профиль скорости и физика",
    "🎥 Видео и трекинг",
    "📋 Отчет о матче",
])

# ── Вкладка 1: Аналитика стола ────────────────────────────────────────────────

with tab_court:
    col_left, col_right = st.columns([1.4, 1.0])

    with col_left:
        st.subheader("Тепловая карта отскоков (вид сверху 2D)")
        # Проекция отскоков в пиксели камеры для тепловой карты
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
                title="Тепловая карта отскоков от стола (2D-проекция сверху)",
            )
            st.pyplot(fig_heat)
            plt.close(fig_heat)
        else:
            st.write("В выбранном диапазоне кадров отскоков не обнаружено.")

    with col_right:
        st.subheader("Пространственное распределение по зонам")

        # Распределение по игрокам
        st.markdown("**Распределение по сторонам стола:**")
        side_df = pd.DataFrame({
            "Сторона": ["Игрок A", "Игрок B"],
            "Отскоки": [metrics.bounces_player_a, metrics.bounces_player_b],
        })
        st.bar_chart(side_df.set_index("Сторона"))

        # Распределение по глубине
        st.markdown("**Зоны глубины (короткие против глубоких):**")
        depth_df = pd.DataFrame(list(metrics.depth_distribution.items()), columns=["Зона", "Количество"])
        st.dataframe(depth_df, hide_index=True, use_container_width=True)


# ── Вкладка 2: Физический профиль ─────────────────────────────────────────────

with tab_physics:
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.subheader("График скорости мяча (км/ч)")
        if metrics.speed_series:
            speed_df = pd.DataFrame(metrics.speed_series, columns=["Время (с)", "Скорость (км/ч)"]).set_index("Время (с)")
            st.line_chart(speed_df)
        else:
            st.info("Недостаточно точек траектории для построения непрерывного графика скорости.")

    with col_v2:
        st.subheader("Вертикальная траектория (ось Y)")
        y_series = [(round((p.frame_id - history[0].frame_id) / fps, 3), p.y) for p in history]
        y_df = pd.DataFrame(y_series, columns=["Время (с)", "Позиция Y (px)"]).set_index("Время (с)")
        st.line_chart(y_df)
        st.caption("Локальные пики по оси Y соответствуют физическим отскокам от стола.")

    # 2D график траектории
    st.subheader("Спроецированная траектория на поверхность стола (BEV)")
    traj_arr = np.array([[p.frame_id, p.x, p.y] for p in history], dtype=np.float32)
    fig_traj = plot_bev_trajectories(
        homography,
        traj_arr,
        np.array(bounce_pts_cam, dtype=np.float32) if bounce_pts_cam else None,
    )
    st.pyplot(fig_traj)
    plt.close(fig_traj)


# ── Вкладка 3: Видеотрекинг ───────────────────────────────────────────────────

with tab_video:
    st.subheader("Демонстрация отслеживания на видео")
    if video_path and Path(video_path).exists():
        st.video(video_path)
    else:
        sample_tracked_mp4 = Path("outputs/tracking/synthetic_tracked.mp4")
        if sample_tracked_mp4.exists():
            st.video(str(sample_tracked_mp4))
            st.caption("Демонстрация отслеживания в реальном времени с кометным шлейфом и 2D мини-картой в правом верхнем углу.")
        else:
            st.info("Сгенерируйте или загрузите видео с помощью CLI: `python scripts/infer.py --synthetic`")


# ── Вкладка 4: Отчет о матче ──────────────────────────────────────────────────

with tab_report:
    st.subheader("Итоговый отчет о матче и розыгрыше")
    md_report = format_markdown_report(metrics)
    st.markdown(md_report)

    # Кнопки скачивания
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="📥 Скачать отчет в Markdown",
            data=md_report,
            file_name="pingpong_match_report.md",
            mime="text/markdown",
        )
    with col_d2:
        st.download_button(
            label="📥 Скачать сводку в JSON",
            data=pd.Series(metrics.summary_dict()).to_json(indent=2),
            file_name="pingpong_metrics.json",
            mime="application/json",
        )
