# 必要なライブラリをインポート
import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands, drawing_utils
import numpy as np
import os
import time
from collections import deque

# === グローバル変数 ===
# MediaPipe関連
mp_drawing = drawing_utils
mp_hands = hands
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # カーソル操作は1つの手で行うため1に制限
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 魚の画像とUI関連の変数
fish_img = None
fish_design_paths = []
current_fish_index = 0
ui_thumbnails = []
ui_boxes = []

# 手の追跡と魚の位置関連
fish_positions = {}
hand_positions_histories = {}
hand_landmarks_histories = {}

# FPS計測用変数
fps_counter = 0
fps_start_time = 0
current_fps = 0.0
fps_display_text = "FPS: 0.0"

# 魚の動き分析用変数
fish_position_history = deque(maxlen=30)  # 過去30フレームの位置を保存
movement_jitter_values = deque(maxlen=10)  # 過去10フレームのジッター値
velocity_changes = deque(maxlen=20)  # 過去20フレームの速度変化
current_smoothness = 0.0
current_jitter = 0.0
current_velocity_variance = 0.0


# === 関数定義 ===

def apply_moving_average(history, current_landmarks):
    """ランドマークの履歴に移動平均フィルタを適用し、座標を平滑化する。"""
    if not history:
        return [(lm.x, lm.y, lm.z) for lm in current_landmarks.landmark]

    all_landmarks = list(history) + [current_landmarks]
    num_landmarks = len(current_landmarks.landmark)
    smoothed_coords = []

    for i in range(num_landmarks):
        x_coords = [lm.landmark[i].x for lm in all_landmarks]
        y_coords = [lm.landmark[i].y for lm in all_landmarks]
        z_coords = [lm.landmark[i].z for lm in all_landmarks]
        
        avg_x = sum(x_coords) / len(x_coords)
        avg_y = sum(y_coords) / len(y_coords)
        avg_z = sum(z_coords) / len(z_coords)
        smoothed_coords.append((avg_x, avg_y, avg_z))

    return smoothed_coords


def update_fps():
    """FPS（フレームレート）を計算・更新する関数"""
    global fps_counter, fps_start_time, current_fps, fps_display_text
    
    current_time = time.time()
    fps_counter += 1
    
    # 1秒ごとにFPSを更新
    if fps_counter == 1:
        fps_start_time = current_time
    elif current_time - fps_start_time >= 1.0:
        current_fps = fps_counter / (current_time - fps_start_time)
        fps_display_text = f"Fish Refresh Rate: {current_fps:.1f} FPS"
        fps_counter = 0


def analyze_fish_movement(current_fish_pos):
    """魚の動きを分析し、滑らかさ指標を計算する"""
    global fish_position_history, movement_jitter_values, velocity_changes
    global current_smoothness, current_jitter, current_velocity_variance
    
    if current_fish_pos is None:
        return
    
    # 現在の位置を履歴に追加
    fish_position_history.append(current_fish_pos)
    
    if len(fish_position_history) < 3:
        return
    
    # ジッター計算（連続する3点間の角度変化）
    if len(fish_position_history) >= 3:
        p1, p2, p3 = list(fish_position_history)[-3:]
        
        # ベクトル計算
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # 角度変化を計算（ジッター指標）
        if (v1[0]**2 + v1[1]**2) > 0 and (v2[0]**2 + v2[1]**2) > 0:
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
            cos_angle = np.clip(dot_product / (mag1 * mag2), -1, 1)
            angle_change = np.arccos(cos_angle) * 180 / np.pi
            movement_jitter_values.append(angle_change)
    
    # 速度変化の計算
    if len(fish_position_history) >= 2:
        recent_positions = list(fish_position_history)[-2:]
        p1, p2 = recent_positions
        velocity = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        velocity_changes.append(velocity)
    
    # 指標の更新
    if len(movement_jitter_values) > 0:
        current_jitter = np.mean(movement_jitter_values)
        # 滑らかさは角度変化の逆数（小さいほど滑らか）
        current_smoothness = max(0, 100 - current_jitter)
    
    if len(velocity_changes) > 1:
        current_velocity_variance = np.var(velocity_changes)


def draw_movement_analysis(frame, width, height):
    """魚の動き分析情報を画面に描画する"""
    global current_smoothness, current_jitter, current_velocity_variance
    
    # 左上に分析情報を表示
    analysis_x = 15
    analysis_y = 15
    line_height = 25
    
    # 背景を描画
    bg_width = 350
    bg_height = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (analysis_x - 5, analysis_y - 5), 
                 (analysis_x + bg_width, analysis_y + bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # タイトル（移動平均ありバージョン）
    cv2.putText(frame, "Fish Movement Analysis (With Moving Average)", 
                (analysis_x, analysis_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 滑らかさ指標（高いほど良い）
    smoothness_color = (0, 255, 0) if current_smoothness > 80 else (0, 255, 255) if current_smoothness > 60 else (0, 0, 255)
    cv2.putText(frame, f"Smoothness Index: {current_smoothness:.1f}/100", 
                (analysis_x, analysis_y + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, smoothness_color, 1)
    
    # ジッター値（低いほど良い）
    jitter_color = (0, 255, 0) if current_jitter < 20 else (0, 255, 255) if current_jitter < 40 else (0, 0, 255)
    cv2.putText(frame, f"Position Jitter: {current_jitter:.1f} deg", 
                (analysis_x, analysis_y + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, jitter_color, 1)
    
    # 速度分散（低いほど良い）
    velocity_color = (0, 255, 0) if current_velocity_variance < 100 else (0, 255, 255) if current_velocity_variance < 500 else (0, 0, 255)
    cv2.putText(frame, f"Velocity Variance: {current_velocity_variance:.1f}", 
                (analysis_x, analysis_y + line_height * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, velocity_color, 1)


def draw_fps_info(frame, width, height):
    """フレームにFPS情報を描画する関数"""
    global fps_display_text, current_fps
    
    # FPS表示用の背景を描画
    text_size = cv2.getTextSize(fps_display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    bg_width = text_size[0] + 20
    bg_height = text_size[1] + 20
    
    # 右上角にFPS情報を表示
    bg_x = width - bg_width - 10
    bg_y = 10
    
    # 半透明の背景を描画
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x, bg_y), (bg_x + bg_width, bg_y + bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # FPSテキストを描画（色は性能によって変更）
    if current_fps >= 25:
        text_color = (0, 255, 0)  # 緑色（良好）
    elif current_fps >= 15:
        text_color = (0, 255, 255)  # 黄色（普通）
    else:
        text_color = (0, 0, 255)  # 赤色（低性能）
    
    cv2.putText(frame, fps_display_text, (bg_x + 10, bg_y + text_size[1] + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
    
    # 追加情報（魚の描画回数など）
    fish_count = len([pos for pos in fish_positions.values() if pos is not None])
    additional_info = f"Active Fish: {fish_count}"
    cv2.putText(frame, additional_info, (bg_x + 10, bg_y + text_size[1] + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def load_fish_designs(base_dir):
    """魚のデザインフォルダから利用可能な魚の画像を読み込む"""
    designs_dir = os.path.join(base_dir, "fish_designs")
    designs = {}
    
    if not os.path.exists(designs_dir):
        os.makedirs(designs_dir)
        print(f"作成したフォルダ: {designs_dir}")
        print("このフォルダに魚の画像ファイル（.png）を配置してください。")
        return designs

    image_files = [f for f in os.listdir(designs_dir) if f.lower().endswith('.png')]
    for filename in image_files:
        design_name = os.path.splitext(filename)[0]
        file_path = os.path.join(designs_dir, filename)
        designs[design_name] = file_path
        print(f"魚のデザインを発見: {design_name}")
    
    if not designs:
        print("fish_designs フォルダに魚の画像が見つかりません。")
        
    return designs

def load_image(path, unchanged=True):
    """画像を読み込む共通関数"""
    if path and os.path.exists(path):
        flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
        img = cv2.imread(path, flag)
        if img is not None:
            return img
    print(f"画像の読み込みに失敗しました: {path}")
    return None

def resize_image_to_width(image, target_width):
    """画像の幅を指定してリサイズする"""
    if image is None or image.shape[1] == 0: return None
    scale = target_width / image.shape[1]
    return cv2.resize(image, (target_width, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)

def apply_alpha_overlay(background, overlay, x, y):
    """透明度を持つ画像を背景に合成する"""
    h, w = overlay.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(background.shape[1], x + w), min(background.shape[0], y + h)

    overlay_x1, overlay_y1 = max(0, -x), max(0, -y)
    overlay_x2, overlay_y2 = overlay_x1 + (x2 - x1), overlay_y1 + (y2 - y1)

    if (x2 <= x1 or y2 <= y1 or overlay_x2 <= overlay_x1 or overlay_y2 <= overlay_y1):
        return

    if overlay.shape[2] == 4:
        alpha = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(3):
            background[y1:y2, x1:x2, c] = (alpha * overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c] +
                                          alpha_inv * background[y1:y2, x1:x2, c])
    else:
        background[y1:y2, x1:x2] = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]


def extract_skeleton(frame):
    """フレームから手の骨格情報を抽出する"""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    detected_hands_data = []
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[hand_idx].classification[0].label
            detected_hands_data.append((hand_landmarks, handedness))
    return detected_hands_data


def predict_pose_from_video(source):
    """カメラ映像から手のポーズを推定し、UIと連動させるメイン関数"""
    global fish_img, ui_boxes, current_fish_index

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"エラー: カメラ {source} を開けません。")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    resized_fish_img = resize_image_to_width(fish_img, int(width / 8))

    # === 機能設定 ===
    FISH_LAG_SECONDS = 1.0
    MOVING_AVERAGE_WINDOW = 5
    DWELL_TIME = 2.0  # NEW: 2秒間停止でクリック

    # === Dwell Click 機能の変数 ===
    hover_target_index = None
    hover_start_time = 0
    virtual_cursor_pos = (0, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        # FPS計測を更新
        update_fps()

        # UI描画
        ui_boxes.clear() 
        ui_x_offset, ui_y_offset, ui_padding = 15, 15, 10
        for i, thumb in enumerate(ui_thumbnails):
            thumb_h, thumb_w = thumb.shape[:2]
            box_x, box_y = ui_x_offset, ui_y_offset + i * (thumb_h + ui_padding)
            apply_alpha_overlay(frame, thumb, box_x, box_y)
            border_color = (0, 165, 255) if i == current_fish_index else (255, 255, 255)
            border_thickness = 4 if i == current_fish_index else 2
            cv2.rectangle(frame, (box_x, box_y), (box_x + thumb_w, box_y + thumb_h), border_color, border_thickness)
            ui_boxes.append((box_x, box_y, thumb_w, thumb_h))

        # 手の検出と処理
        raw_detected_hands = extract_skeleton(frame)
        active_hand_ids_this_frame = set()
        current_time = time.time()
        
        if resized_fish_img is None or resized_fish_img.shape != resize_image_to_width(fish_img, int(width / 8)).shape:
             resized_fish_img = resize_image_to_width(fish_img, int(width / 8))

        hand_detected = len(raw_detected_hands) > 0
        if hand_detected:
            # 最初の1つの手だけを処理
            raw_landmarks, handedness = raw_detected_hands[0]
            hand_id = f"{handedness}_0"
            active_hand_ids_this_frame.add(hand_id)
            
            if hand_id not in hand_landmarks_histories:
                hand_landmarks_histories[hand_id] = deque(maxlen=MOVING_AVERAGE_WINDOW)
            
            smoothed_coords = apply_moving_average(hand_landmarks_histories[hand_id], raw_landmarks)
            hand_landmarks_histories[hand_id].append(raw_landmarks)
            
            mp_drawing.draw_landmarks(frame, raw_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # NEW: 仮想カーソルの位置を更新 (中指の先端)
            cursor_lm = smoothed_coords[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            virtual_cursor_pos = (int(cursor_lm[0] * width), int(cursor_lm[1] * height))

            thumb_tip = smoothed_coords[mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = smoothed_coords[mp_hands.HandLandmark.PINKY_TIP]
            distance = np.sqrt((thumb_tip[0] - pinky_tip[0])**2 + (thumb_tip[1] - pinky_tip[1])**2)
            is_paper = distance > 0.15

            if is_paper:
                pos_lm = smoothed_coords[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                current_hand_x, current_hand_y = int(pos_lm[0] * width), int(pos_lm[1] * height)
                if hand_id not in hand_positions_histories: hand_positions_histories[hand_id] = deque()
                hand_positions_histories[hand_id].append((current_time, current_hand_x, current_hand_y))
                while hand_positions_histories[hand_id] and hand_positions_histories[hand_id][0][0] < current_time - FISH_LAG_SECONDS:
                    hand_positions_histories[hand_id].popleft()
                if hand_positions_histories[hand_id]:
                    fish_positions[hand_id] = hand_positions_histories[hand_id][0][1:]
                    # 魚の動き分析を実行
                    analyze_fish_movement(fish_positions[hand_id])
            else:
                fish_positions[hand_id] = None
                if hand_id in hand_positions_histories: hand_positions_histories[hand_id].clear()
        
        # NEW: Dwell to Click ロジック
        currently_hovering_on = None
        if hand_detected:
            for i, box in enumerate(ui_boxes):
                bx, by, bw, bh = box
                if bx <= virtual_cursor_pos[0] < bx + bw and by <= virtual_cursor_pos[1] < by + bh:
                    currently_hovering_on = i
                    break
        
        if currently_hovering_on is not None and currently_hovering_on != hover_target_index:
            # 新しいターゲットに乗り換えた場合
            hover_target_index = currently_hovering_on
            hover_start_time = current_time
        elif currently_hovering_on is None:
            # どこにもホバーしていない場合
            hover_target_index = None
        
        if hover_target_index is not None:
            elapsed_time = current_time - hover_start_time
            # ホバーの進捗を描画
            progress = min(elapsed_time / DWELL_TIME, 1.0)
            box = ui_boxes[hover_target_index]
            progress_radius = int(box[3] / 4) # ボックスの高さの1/4
            center_x = box[0] + box[2] // 2
            center_y = box[1] + box[3] // 2
            cv2.ellipse(frame, (center_x, center_y), (progress_radius, progress_radius), 270, 0, 360 * progress, (0, 255, 255), 5)

            if elapsed_time >= DWELL_TIME:
                print(f"Dwell Click 実行: Index {hover_target_index}")
                current_fish_index = hover_target_index
                full_image = load_image(fish_design_paths[current_fish_index])
                if full_image is not None: fish_img = full_image
                hover_target_index = None # クリック後はリセット

        # 画面から消えた手の情報を削除
        hands_to_remove = set(fish_positions.keys()) - active_hand_ids_this_frame
        for hand_id in hands_to_remove:
            if hand_id in fish_positions: del fish_positions[hand_id]
            if hand_id in hand_positions_histories: del hand_positions_histories[hand_id]
            if hand_id in hand_landmarks_histories: del hand_landmarks_histories[hand_id]

        # 魚を描画
        if resized_fish_img is not None:
            for pos in fish_positions.values():
                if pos:
                    x, y = pos
                    apply_alpha_overlay(frame, resized_fish_img, x - resized_fish_img.shape[1]//2, y - resized_fish_img.shape[0]//2)

        # NEW: 仮想カーソルを描画
        if hand_detected:
            cv2.circle(frame, virtual_cursor_pos, 12, (255, 0, 0), -1)
            cv2.circle(frame, virtual_cursor_pos, 15, (255, 255, 255), 2)
        
        # FPS情報を描画
        draw_fps_info(frame, width, height)
        
        # 魚の動き分析情報を描画
        draw_movement_analysis(frame, width, height)
        
        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """メイン処理"""
    global fish_img, fish_design_paths, current_fish_index, ui_thumbnails

    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"基準ディレクトリ: {base_dir}")

    fish_designs = load_fish_designs(base_dir)
    if not fish_designs:
        print("処理を終了します。")
        return

    fish_design_paths = list(fish_designs.values())
    current_fish_index = 0
    
    # NEW: アイコンサイズを大きくする
    UI_THUMBNAIL_WIDTH = 160 # 80から160に変更
    print("UI用のサムネイルを作成中...")
    for path in fish_design_paths:
        img = load_image(path)
        if img is not None:
            thumb = resize_image_to_width(img, UI_THUMBNAIL_WIDTH)
            ui_thumbnails.append(thumb)
    
    if not ui_thumbnails:
        print("UIに表示できる魚がありません。処理を終了します。")
        return

    fish_img = load_image(fish_design_paths[current_fish_index])
    if fish_img is None:
        print("初期の魚画像の読み込みに失敗しました。")
        return

    window_name = 'Pose Detection'
    cv2.namedWindow(window_name)
    # マウスカーソルを非表示にする (空のコールバックを設定)
    cv2.setMouseCallback(window_name, lambda *args: None)

    predict_pose_from_video(0)

if __name__ == "__main__":
    main()