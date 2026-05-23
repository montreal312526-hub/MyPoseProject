# python analyze_dtw_3x.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter, resample

#  Windows 環境でグラフの文字化けを防ぐ設定
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

print("V3.2 階梯勾配 (Gradient Incline) DTW 解析エンジンを起動中...")

# ================= 1. データの読み込み ================= clinical_pose_data_0% INCLINE
try:
    df_2 = pd.read_csv('clinical_pose_data_2% INCLINE.csv')
    df_4 = pd.read_csv('clinical_pose_data_4% INCLINE.csv')
    df_6 = pd.read_csv('clinical_pose_data_6% INCLINE.csv')
except FileNotFoundError as e:
    print(f"⚠️ ファイルが見つかりません: {e}")
    exit()

FPS = 30
TARGET_FRAMES = 200  # リサンプリングの目標フレーム数を統一し、病的なアライメントを防止

# ================= 2. コア特徴量の抽出 (片側選択的利用：右下肢) =================
# 右膝関節および右股関節の角度データを抽出
raw_knee_2 = df_2['R_Knee_Angle'].values
raw_hip_2  = df_2['R_Hip_Angle'].values

raw_knee_4 = df_4['R_Knee_Angle'].values
raw_hip_4  = df_4['R_Hip_Angle'].values

raw_knee_6 = df_6['R_Knee_Angle'].values
raw_hip_6  = df_6['R_Hip_Angle'].values

# ================= 3. 信号の平滑化フィルタリング (Savitzky-Golay) =================
smooth_knee_2 = savgol_filter(raw_knee_2, window_length=11, polyorder=2)
smooth_hip_2  = savgol_filter(raw_hip_2, window_length=11, polyorder=2)

smooth_knee_4 = savgol_filter(raw_knee_4, window_length=11, polyorder=2)
smooth_hip_4  = savgol_filter(raw_hip_4, window_length=11, polyorder=2)

smooth_knee_6 = savgol_filter(raw_knee_6, window_length=11, polyorder=2)
smooth_hip_6  = savgol_filter(raw_hip_6, window_length=11, polyorder=2)

# ================= 4. 時間軸の正規化 (スタートラインの強制統一) =================
res_knee_2 = resample(smooth_knee_2, TARGET_FRAMES)
res_hip_2  = resample(smooth_hip_2, TARGET_FRAMES)

res_knee_4 = resample(smooth_knee_4, TARGET_FRAMES)
res_hip_4  = resample(smooth_hip_4, TARGET_FRAMES)

res_knee_6 = resample(smooth_knee_6, TARGET_FRAMES)
res_hip_6  = resample(smooth_hip_6, TARGET_FRAMES)

# ================= 5. 多次元特徴量行列のスタッキング, knee と hip の抽出 =================
features_2 = np.column_stack([res_knee_2, res_hip_2])
features_4 = np.column_stack([res_knee_4, res_hip_4])
features_6 = np.column_stack([res_knee_6, res_hip_6])

# ================= 6. DTW 乖離度の計算 =================
print("🧠 多次元 DTW 計算を実行中...")
# 2% 勾配を Baseline とし、4% と 6% のオフセット量を計算
dist_4, path_4 = fastdtw(features_2, features_4, dist=euclidean)
dist_6, path_6 = fastdtw(features_2, features_6, dist=euclidean)

# 归一化得分
score_4 = dist_4 / len(path_4)
score_6 = dist_6 / len(path_6)

print("-" * 50)
print(f"2% vs 4% 勾配 姿勢代償度 (Score): {score_4:.2f}")
print(f"2% vs 6% 勾配 姿勢代償度 (Score): {score_6:.2f}")
print("-" * 50)

# ================= 7. 学術レベルの可視化 =================
time_axis = np.arange(TARGET_FRAMES) / float(FPS)

plt.figure(figsize=(14, 7))

# 3条件の曲線を比較描画（右膝関節をメイン視点とする）
plt.plot(time_axis, res_knee_2, label='Baseline (2% Incline)', color='#2ca02c', linewidth=2.5, alpha=0.8)
plt.plot(time_axis, res_knee_4, label=f'Medium Load (4% Incline) - DTW Score: {score_4:.2f}', color='#ff7f0e', linewidth=2.5, alpha=0.8)
plt.plot(time_axis, res_knee_6, label=f'Heavy Load (6% Incline) - DTW Score: {score_6:.2f}', color='#d62728', linewidth=2.5, alpha=0.8)

# 論文用の高品位なタイトルとラベル設定
plt.title("Kinematic Shift under Different Incline Loads (Right Knee Angle)", fontsize=16, pad=20, fontweight='bold')
plt.xlabel("Normalized Time (Seconds)", fontsize=13)
plt.ylabel("Right Knee Flexion Angle (°)", fontsize=13)
plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()