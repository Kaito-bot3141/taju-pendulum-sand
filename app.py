# app.py
# Python 3.12.8
# streamlit run app.py
#
# 多重振り子（連結回転）で先端軌跡を生成し、プレビュー表示＆距離間隔でCSV出力するアプリ
# 最終UI調整:
# - プレビューは円形（半径80mm）・黒背景・白線
# - 軸・ラベル・グラフ感を完全排除
# - 右側UI順：グラフ → 座標間隔 → CSV名 → ダウンロード
# - 内部サンプリング点数は2000固定（表示なし）
# - ★Nを増減しても speeds_pct が不足/超過しても落ちないように自動補正
#
# 追加:
# - すべてのウィジェットに key を付与
# - ★リンク比/速度の±ボタンが確実に効くように修正（同一キー + st.rerun）
# - ★Streamlitの仕様により、ウィジェット生成後に同一keyへ代入しない（APIException回避）
# - ★リンク比はR1=1固定（UI表示しない）

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def generate_path(lengths_mm, base_rev, speed_factors, num_samples, phase_rad=-math.pi / 2):
    t = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    revs = base_rev * speed_factors
    theta = (2.0 * np.pi) * (revs[:, None] * t[None, :]) + phase_rad
    x = np.sum(lengths_mm[:, None] * np.cos(theta), axis=0)
    y = np.sum(lengths_mm[:, None] * np.sin(theta), axis=0)
    return np.stack([x, y], axis=1)


def resample_by_distance(points_xy, step_mm):
    if len(points_xy) < 2:
        return points_xy.copy()
    diffs = np.diff(points_xy, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(s[-1])
    if total <= 1e-12:
        return points_xy[:1].copy()
    n = int(math.floor(total / step_mm))
    targets = np.linspace(0.0, n * step_mm, n + 1)
    x = np.interp(targets, s, points_xy[:, 0])
    y = np.interp(targets, s, points_xy[:, 1])
    return np.stack([x, y], axis=1)


def sanitize_nonneg(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr[np.isnan(arr)] = 0.0
    arr[arr < 0] = 0.0
    return arr


def ratios_to_lengths(ratios, total_mm):
    ratios = sanitize_nonneg(ratios)
    s = ratios.sum()
    if s <= 1e-12:
        return np.full(len(ratios), total_mm / len(ratios))
    return ratios * (total_mm / s)


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="砂絵：多重振り子", layout="wide")
st.title("砂絵：多重振り子トレース生成")

# ---- 初期値 ----
if "N" not in st.session_state:
    st.session_state.N = 2
if "ratios" not in st.session_state:
    st.session_state.ratios = [1.0, 1.3]
if "speeds_pct" not in st.session_state:
    st.session_state.speeds_pct = [500.0, 290.0]
if "base_rev" not in st.session_state:
    st.session_state.base_rev = 10.0
if "total_len" not in st.session_state:
    st.session_state.total_len = 70.0
if "csv_basename" not in st.session_state:
    st.session_state.csv_basename = "pendulum_trace"

left, right = st.columns([1.0, 1.2])

with left:
    st.subheader("パラメータ")

    N = st.number_input(
        "振り子の棒の数",
        1, 12, st.session_state.N, 1,
        key="inp_N"
    )
    st.session_state.N = N

    total_len = st.number_input(
        "棒の長さ合計（mm）",
        1.0, 200.0, st.session_state.total_len, 1.0,
        key="inp_total_len"
    )
    st.session_state.total_len = total_len

    base_rev = st.number_input(
        "L1 の回転数（周）",
        0.0, 2000.0, st.session_state.base_rev, 1.0,
        key="inp_base_rev"
    )
    st.session_state.base_rev = base_rev

    # =========================
    # リンク比（R1=1固定、UIはR2..RN。0〜10、0.2刻み + ±0.2）
    # =========================
    st.markdown("### リンク比")

    # Nに合わせて補正（内部保持用）
    ratios = list(st.session_state.ratios)
    if len(ratios) < N:
        ratios += [1.0] * (N - len(ratios))
    elif len(ratios) > N:
        ratios = ratios[:N]

    # R1は常に1固定
    ratios[0] = 1.0

    # スライダーkeyを先に準備（R2..RNのみ）
    for i in range(1, N):
        k = f"ratio_{i}"
        if k not in st.session_state:
            v0 = float(ratios[i])
            v0 = max(0.0, min(2.0, v0))
            v0 = round(v0 / 0.2) * 0.2
            v0 = round(v0, 1)
            st.session_state[k] = v0

    new_ratios = [1.0]  # R1固定
    for i in range(1, N):
        k = f"ratio_{i}"

        c1, c2, c3, c4 = st.columns([0.9, 0.55, 3.6, 0.55])
        with c1:
            st.markdown(f"R{i+1}")
        with c2:
            if st.button("－", key=f"ratio_minus_{i}"):
                st.session_state[k] = max(0.0, round(st.session_state[k] - 0.2, 1))
                st.rerun()
        with c4:
            if st.button("＋", key=f"ratio_plus_{i}"):
                st.session_state[k] = min(2.0, round(st.session_state[k] + 0.2, 1))
                st.rerun()
        with c3:
            st.slider(
                f"ratio_slider_{i}",
                0.0, 2.0,
                step=0.2,
                key=k
            )

        new_ratios.append(float(st.session_state[k]))

    st.session_state.ratios = new_ratios

    # =========================
    # 速度（%）
    # - R1固定（100%扱い、UIなし）
    # - R2..RN：-500〜+500、5刻み + ±5
    # =========================
    st.markdown("### 速度（%）")

    prev = list(st.session_state.speeds_pct) if isinstance(st.session_state.speeds_pct, (list, tuple, np.ndarray)) else []
    speeds_pct = [100.0] + [100.0] * max(0, N - 1)

    # 旧値の引き継ぎ（R2以降だけ）
    for i in range(1, N):
        if i < len(prev):
            try:
                v = float(prev[i])
            except Exception:
                v = 100.0
            v = max(-500.0, min(500.0, v))
            v = round(v / 5.0) * 5.0
            speeds_pct[i] = v

    # スライダーkey準備（ウィジェット生成前のみ代入OK）
    for i in range(1, N):
        k = f"spd_{i}"
        if k not in st.session_state:
            v0 = int(speeds_pct[i])
            v0 = max(-500, min(500, v0))
            v0 = int(round(v0 / 5.0) * 5)
            st.session_state[k] = v0

    new_speeds = [100.0]  # R1固定

    for i in range(1, N):
        k = f"spd_{i}"

        c1, c2, c3, c4 = st.columns([0.9, 0.55, 3.6, 0.55])
        with c1:
            st.markdown(f"R{i+1}")
        with c2:
            if st.button("－", key=f"spd_minus_{i}"):
                st.session_state[k] = int(max(-500, st.session_state[k] - 5))
                st.rerun()
        with c4:
            if st.button("＋", key=f"spd_plus_{i}"):
                st.session_state[k] = int(min(500, st.session_state[k] + 5))
                st.rerun()
        with c3:
            st.slider(
                f"spd_slider_{i}",
                -500, 500,
                step=5,
                key=k
            )

        new_speeds.append(float(st.session_state[k]))

    st.session_state.speeds_pct = new_speeds

with right:
    st.subheader("プレビュー")

    num_samples = 2000

    lengths_mm = ratios_to_lengths(np.array(st.session_state.ratios), st.session_state.total_len)
    pts = generate_path(
        lengths_mm,
        st.session_state.base_rev,
        np.array(st.session_state.speeds_pct) / 100.0,
        num_samples,
    )

    progress = st.slider(
        "描画進行",
        1, 100, 100,
        key="slider_progress"
    )
    pts_show = pts[: max(2, int(len(pts) * progress / 100))]

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    circle = plt.Circle((0, 0), 80, color="skyblue", zorder=0)
    ax.add_patch(circle)

    ax.plot(
        pts_show[:, 0],
        pts_show[:, 1],
        color="white",
        linewidth=1.2,
        alpha=0.9,
    )

    ax.set_aspect("equal")
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.axis("off")

    st.pyplot(fig)

    step_mm = st.number_input(
        "座標を取る間隔（mm）",
        0.1, 50.0, 5.0, 0.5,
        key="inp_step_mm"
    )
    st.session_state.csv_basename = st.text_input(
        "CSV保存名（.csv不要）",
        st.session_state.csv_basename,
        key="inp_csv_name"
    )

    pts_rs = resample_by_distance(pts, step_mm)
    df_out = pd.DataFrame(pts_rs, columns=["x", "y"]).round(1)
    csv_bytes = df_out.to_csv(index=False, header=False).encode("utf-8")

    fname = st.session_state.csv_basename.strip() or "pendulum_trace"
    st.download_button(
        "CSVをダウンロード",
        csv_bytes,
        file_name=f"{fname}.csv",
        key="btn_download_csv"
    )
