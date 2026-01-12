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
# - ★Nを増減しても speeds_pct が不足/超過しても落ちないように自動補正（今回の修正）
#
# 追加（今回）:
# - すべてのウィジェットに key を付与（挙動は変えず安定化のみ）

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

    st.markdown("### リンク比")
    ratios = st.session_state.ratios[:N] + [1.0] * max(0, N - len(st.session_state.ratios))
    df_ratio = pd.DataFrame({"比率": ratios})
    df_ratio = st.data_editor(
        df_ratio,
        use_container_width=True,
        num_rows="fixed",
        key="editor_ratios"
    )
    st.session_state.ratios = np.round(sanitize_nonneg(df_ratio["比率"]), 1).tolist()

    st.markdown("### 速度（%）")

    # ★今回の修正：Nに合わせて速度配列を自動補正（不足分は100%）
    speeds_pct = list(st.session_state.speeds_pct)
    if len(speeds_pct) < N:
        speeds_pct += [100.0] * (N - len(speeds_pct))
    elif len(speeds_pct) > N:
        speeds_pct = speeds_pct[:N]

    speeds = []
    for i in range(N):
        speeds.append(st.slider(
            f"R{i+1}",
            0, 500,
            int(speeds_pct[i]),
            5,
            key=f"spd_{i}"
        ))
    st.session_state.speeds_pct = speeds

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

    # ===== 砂絵風プレビュー =====
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 円形背景（半径80mm）
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

    # ===== 右側の順序：座標間隔 → CSV名 → ダウンロード =====
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
