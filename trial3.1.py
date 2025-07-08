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
    max_num_hands=3,
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
hand_landmarks_histories = {} # <<< 移動平均フィルタのために使用 >>>


# === 関数定義 ===

def apply_moving_average(history, current_landmarks):
    """
    NEW: ランドマークの履歴に移動平均フィルタを適用し、座標を平滑化する。
    """
    if not history:
        # 履歴がない場合は現在のランドマーク座標をそのまま返す
        return [(lm.x, lm.y, lm.z) for lm in current_landmarks.landmark]

    # 現在のフレームも履歴に加えて計算
    all_landmarks = list(history) + [current_landmarks]
    num_landmarks = len(current_landmarks.landmark)
    smoothed_coords = []

    for i in range(num_landmarks):
        # 各ランドマークのx, y, z座標の平均を計算
        x_coords = [lm.landmark[i].x for lm in all_landmarks]
        y_coords = [lm.landmark[i].y for lm in all_landmarks]
        z_coords = [lm.landmark[i].z for lm in all_landmarks]
        
        avg_x = sum(x_coords) / len(x_coords)
        avg_y = sum(y_coords) / len(y_coords)
        avg_z = sum(z_coords) / len(z_coords)
        smoothed_coords.append((avg_x, avg_y, avg_z))

    return smoothed_coords


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

    if overlay.shape[2] == 4: # RGBA画像の場合
        alpha = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
        alpha_inv = 1.0 - alpha

        for c in range(3):
            background[y1:y2, x1:x2, c] = (alpha * overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c] +
                                          alpha_inv * background[y1:y2, x1:x2, c])
    else: # RGB画像の場合
        background[y1:y2, x1:x2] = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]


def handle_mouse_click(event, x, y, flags, param):
    """マウスクリックイベントを処理するコールバック関数"""
    global current_fish_index, fish_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, box in enumerate(ui_boxes):
            bx, by, bw, bh = box
            if bx <= x < bx + bw and by <= y < by + bh:
                if current_fish_index != i:
                    print(f"UIで魚を選択しました: Index {i}")
                    current_fish_index = i
                    full_image = load_image(fish_design_paths[current_fish_index])
                    if full_image is not None:
                       fish_img = full_image
                break


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
    global fish_img, ui_boxes

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
    FISH_LAG_SECONDS = 1.0       # 魚が追従する際の遅延（1秒）
    MOVING_AVERAGE_WINDOW = 5    # 移動平均のフレーム数（大きいほど滑らか）

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # UI描画
        ui_boxes.clear() 
        ui_x_offset, ui_y_offset, ui_padding = 15, 15, 10
        for i, thumb in enumerate(ui_thumbnails):
            thumb_h, thumb_w = thumb.shape[:2]
            box_x, box_y = ui_x_offset, ui_y_offset + i * (thumb_h + ui_padding)
            apply_alpha_overlay(frame, thumb, box_x, box_y)
            border_color = (0, 165, 255) if i == current_fish_index else (255, 255, 255)
            border_thickness = 3 if i == current_fish_index else 1
            cv2.rectangle(frame, (box_x, box_y), (box_x + thumb_w, box_y + thumb_h), border_color, border_thickness)
            ui_boxes.append((box_x, box_y, thumb_w, thumb_h))

        # 手の検出と処理
        raw_detected_hands = extract_skeleton(frame)
        active_hand_ids_this_frame = set()
        current_time = time.time()
        
        if resized_fish_img is None or resized_fish_img.shape != resize_image_to_width(fish_img, int(width / 8)).shape:
             resized_fish_img = resize_image_to_width(fish_img, int(width / 8))

        for idx, (raw_landmarks, handedness) in enumerate(raw_detected_hands):
            hand_id = f"{handedness}_{idx}"
            active_hand_ids_this_frame.add(hand_id)
            
            # === 移動平均フィルタの適用 ===
            if hand_id not in hand_landmarks_histories:
                hand_landmarks_histories[hand_id] = deque(maxlen=MOVING_AVERAGE_WINDOW)
            
            # フィルタを適用して平滑化されたランドマーク座標を取得
            smoothed_coords = apply_moving_average(hand_landmarks_histories[hand_id], raw_landmarks)
            # 今回の生のランドマークを履歴に追加
            hand_landmarks_histories[hand_id].append(raw_landmarks)
            # === フィルタ適用ここまで ===

            mp_drawing.draw_landmarks(frame, raw_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 平滑化された座標を使ってジェスチャー判定
            thumb_tip = smoothed_coords[mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = smoothed_coords[mp_hands.HandLandmark.PINKY_TIP]
            distance = np.sqrt((thumb_tip[0] - pinky_tip[0])**2 + (thumb_tip[1] - pinky_tip[1])**2)
            is_paper = distance > 0.15

            if is_paper:
                # 平滑化された座標を使って手の位置を取得
                pos_lm = smoothed_coords[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                current_hand_x = int(pos_lm[0] * width)
                current_hand_y = int(pos_lm[1] * height)

                if hand_id not in hand_positions_histories:
                    hand_positions_histories[hand_id] = deque()
                hand_positions_histories[hand_id].append((current_time, current_hand_x, current_hand_y))

                while hand_positions_histories[hand_id] and \
                      hand_positions_histories[hand_id][0][0] < current_time - FISH_LAG_SECONDS:
                    hand_positions_histories[hand_id].popleft()
                
                if hand_positions_histories[hand_id]:
                    fish_positions[hand_id] = hand_positions_histories[hand_id][0][1:]
            else:
                fish_positions[hand_id] = None
                if hand_id in hand_positions_histories:
                    hand_positions_histories[hand_id].clear()

        # 画面から消えた手の情報を全履歴から削除
        hands_to_remove = set(fish_positions.keys()) - active_hand_ids_this_frame
        for hand_id in hands_to_remove:
            if hand_id in fish_positions: del fish_positions[hand_id]
            if hand_id in hand_positions_histories: del hand_positions_histories[hand_id]
            if hand_id in hand_landmarks_histories: del hand_landmarks_histories[hand_id]

        # 魚を描画
        if resized_fish_img is not None:
            for hand_id, pos in fish_positions.items():
                if pos:
                    x, y = pos
                    fish_center_x = x - resized_fish_img.shape[1] // 2
                    fish_center_y = y - resized_fish_img.shape[0] // 2
                    apply_alpha_overlay(frame, resized_fish_img, fish_center_x, fish_center_y)

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
    
    print("UI用のサムネイルを作成中...")
    for path in fish_design_paths:
        img = load_image(path)
        if img is not None:
            thumb = resize_image_to_width(img, 80)
            ui_thumbnails.append(thumb)
    
    if not ui_thumbnails:
        print("UIに表示できる魚がありません。処理を終了します。")
        return

    fish_img = load_image(fish_design_paths[current_fish_index])
    if fish_img is None:
        print("初期の魚画像の読み込みに失敗しました。")
        return

    cv2.namedWindow('Pose Detection')
    cv2.setMouseCallback('Pose Detection', handle_mouse_click)

    predict_pose_from_video(0)

if __name__ == "__main__":
    main()