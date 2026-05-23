#Python: Create Environment  
#  pip install mediapipe==0.10.9
#  pip install opencv-python （不需要）
# python main3.py 

import cv2
import mediapipe as mp
import time
import csv
import numpy as np
from collections import deque # 追加: 移動平均用の両端キュー (Deque)

print("MediaPipeエンジンの起動...")

# ================= 1. コア数学モジュール (ゼロ除算防止 + 3Dベクトル) =================
def calculate_3d_angle(a, b, c):
    """
    3次元空間における3点のなす角を計算 (bを頂点とする)
    a, b, c: [x, y, z] 形式の3D座標リスト
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    ba = a - b
    bc = c - b
    
    #保護メカニズム: ゼロ除算によるNaNを防止
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-6:
        return 0.0
        
    cosine_angle = np.dot(ba, bc) / denominator
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# ================= 2. データテーブルの初期化 =================
csv_file = open('test3.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)

# ヘッダー: 安定したタイムスタンプ + 8つの主要関節角度 + 33個の特徴点の生XYZ座標
header = ['Stable_Timestamp_sec', 
          'L_Knee_Angle', 'R_Knee_Angle', 'L_Hip_Angle', 'R_Hip_Angle',
          'L_Elbow_Angle', 'R_Elbow_Angle', 'L_Shoulder_Angle', 'R_Shoulder_Angle']
for i in range(33):
    header.extend([f'Pt_{i}_X', f'Pt_{i}_Y', f'Pt_{i}_Z'])
csv_writer.writerow(header)

# ================= 3. フィルタの初期化 (単純移動平均) =================
WINDOW_SIZE = 5
# 辞書(Dictionary)を使用して各関節の履歴キューを一括管理
angle_history = {
    'l_knee': deque(maxlen=WINDOW_SIZE), 'r_knee': deque(maxlen=WINDOW_SIZE),
    'l_hip': deque(maxlen=WINDOW_SIZE),  'r_hip': deque(maxlen=WINDOW_SIZE),
    'l_elbow': deque(maxlen=WINDOW_SIZE),'r_elbow': deque(maxlen=WINDOW_SIZE),
    'l_shoulder': deque(maxlen=WINDOW_SIZE), 'r_shoulder': deque(maxlen=WINDOW_SIZE)
}

# ================= 4. 映像と推論モデルの初期化 =================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,         # 1から2へ変更！最も高精度で重い BlazePose Heavy モデル
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6   
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture("6% INCLINE.mp4") 
#2% INCLINE.mp4
#4% INCLINE .mp4
#6% INCLINE.mp4
if not cap.isOpened(): print("❌ 映像の読み込みに失敗しました")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30
ideal_delay = 1000 / fps  
frame_idx = 0 # 安定したカスタムフレームカウンタ

# ================= 5. メインループ =================
while True:
    start_real_time = time.time()
    ret, frame = cap.read()
    if not ret: break
        
    frame_idx += 1
    stable_timestamp = frame_idx / fps # 絶対的に安定したタイムスタンプ (秒)
    
    # 画面のリサイズと推論
    orig_h, orig_w = frame.shape[:2]
    target_h = 800
    target_w = int(orig_w * (target_h / orig_h))
    frame = cv2.resize(frame, (target_w, target_h))
    
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark
        
       #  全3D生データの抽出
        raw_3d_coords = []
        for i in range(33):
            raw_3d_coords.extend([lm[i].x, lm[i].y, lm[i].z])
            
        # [x, y, z] を素早く取得
        get_3d = lambda idx: [lm[idx].x, lm[idx].y, lm[idx].z]
        
        # 特徴量エンジニアリング: 8つの主要3D角度の計算
        # 下肢
        raw_l_knee = calculate_3d_angle(get_3d(23), get_3d(25), get_3d(27))
        raw_r_knee = calculate_3d_angle(get_3d(24), get_3d(26), get_3d(28))
        raw_l_hip  = calculate_3d_angle(get_3d(11), get_3d(23), get_3d(25))
        raw_r_hip  = calculate_3d_angle(get_3d(12), get_3d(24), get_3d(26))
        # 上肢
        raw_l_elbow = calculate_3d_angle(get_3d(11), get_3d(13), get_3d(15))
        raw_r_elbow = calculate_3d_angle(get_3d(12), get_3d(14), get_3d(16))
        raw_l_shoulder = calculate_3d_angle(get_3d(13), get_3d(11), get_3d(23))
        raw_r_shoulder = calculate_3d_angle(get_3d(14), get_3d(12), get_3d(24))

        # 平滑化フィルタリング (キューに追加して平均値を取得)
        angle_history['l_knee'].append(raw_l_knee)
        angle_history['r_knee'].append(raw_r_knee)
        angle_history['l_hip'].append(raw_l_hip)
        angle_history['r_hip'].append(raw_r_hip)
        angle_history['l_elbow'].append(raw_l_elbow)
        angle_history['r_elbow'].append(raw_r_elbow)
        angle_history['l_shoulder'].append(raw_l_shoulder)
        angle_history['r_shoulder'].append(raw_r_shoulder)
        
        smooth_angles = [
            np.mean(angle_history['l_knee']), np.mean(angle_history['r_knee']),
            np.mean(angle_history['l_hip']),  np.mean(angle_history['r_hip']),
            np.mean(angle_history['l_elbow']),np.mean(angle_history['r_elbow']),
            np.mean(angle_history['l_shoulder']), np.mean(angle_history['r_shoulder'])
        ]
        
        # 現フレームの全データを組み立てて保存
        row_data = [stable_timestamp] + smooth_angles + raw_3d_coords
        csv_writer.writerow(row_data)
        
        # 可視化 (左膝を例に、平滑化後の角度を描画)
        knee_px = tuple(np.multiply([lm[25].x, lm[25].y], [target_w, target_h]).astype(int))
        cv2.putText(frame, f"{int(smooth_angles[0])} deg", knee_px, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    cv2.imshow("Clinical Grade Analysis", frame)
    
    # 速度制御
    wait_time = max(1, int(ideal_delay - (time.time() - start_real_time) * 1000))
    if cv2.waitKey(wait_time) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print("✅ 処理完了！データが保存されました！")