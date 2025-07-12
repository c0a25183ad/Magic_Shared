# 必要なライブラリをインポート
import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands, drawing_utils
import numpy as np
import os
import time
from collections import deque
# 機械学習用ライブラリを追加
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

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

# 機械学習関連変数
templates = {}
model = None
label_encoder = None

# === 機械学習関数 ===

def cosine_similarity(a, b):
    """コサイン類似度を計算する関数"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def weighted_cosine_similarity(a, b, weights=None):
    """重み付きコサイン類似度を計算する関数"""
    if weights is None:
        # デフォルトの重み：指先のランドマークに高い重みを与える
        weights = np.ones(len(a))
        # 指先のランドマーク（4, 8, 12, 16, 20）に高い重み
        finger_tip_indices = [4, 8, 12, 16, 20]
        for idx in finger_tip_indices:
            if idx*3 < len(weights):
                weights[idx*3:(idx+1)*3] = 2.0  # 指先のx,y,zに2倍の重み
    
    # 重み付きコサイン類似度を計算
    weighted_a = a * weights
    weighted_b = b * weights
    return np.dot(weighted_a, weighted_b) / (np.linalg.norm(weighted_a) * np.linalg.norm(weighted_b))

def load_templates():
    """テンプレート（基準となる手のポーズデータ）の読み込み関数"""
    templates = {}
    for gesture in ['rock', 'scissors', 'paper']:
        try:
            templates[gesture] = np.load(f"{gesture}_template.npy")
            print(f"Loaded {gesture}_template.npy")
        except FileNotFoundError:
            print(f"Template for {gesture} not found. Please generate templates first.")
    return templates

def generate_templates():
    """テンプレート生成関数"""
    print("テンプレート生成を開始します...")
    
    # 各ジェスチャーに対して複数のサンプル画像を定義
    image_label_pairs = [
        # グーのサンプル画像
        ("rock.jpg", "rock"),
        ("rock_1.jpg", "rock"),
        ("rock_2.jpg", "rock"),
        ("rock_3.jpg", "rock"),
        ("rock_4.jpg", "rock"),
        
        # チョキのサンプル画像
        ("scissors.jpg", "scissors"),
        ("scissors_1.jpg", "scissors"),
        ("scissors_2.jpg", "scissors"),
        ("scissors_3.jpg", "scissors"),
        ("scissors_4.jpg", "scissors"),
        
        # パーのサンプル画像
        ("paper.jpg", "paper"),
        ("paper_1.jpg", "paper"),
        ("paper_2.jpg", "paper"),
        ("paper_3.jpg", "paper"),
        ("paper_4.jpg", "paper"),
    ]
    
    # 各ジェスチャーのサンプルを格納する辞書
    gesture_samples = {'rock': [], 'scissors': [], 'paper': []}
    
    for image_path, label in image_label_pairs:
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found. Skipping...")
            continue
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to load image at {image_path}")
            continue
        
        # 手のランドマークを抽出
        detected_hands_data = extract_skeleton(frame)
        
        if detected_hands_data:  # 何らかの手が検出された場合
            handpose = []
            for hand_landmarks, handedness in detected_hands_data:
                for landmark in hand_landmarks.landmark:
                    handpose.extend([landmark.x, landmark.y, landmark.z])
                break  # 最初の手のみを使用
            
            if handpose:
                handpose = np.array(handpose)
                if len(handpose) < 63:
                    handpose = np.pad(handpose, (0, 63 - len(handpose)), 'constant')
                elif len(handpose) > 63:
                    handpose = handpose[:63]
                gesture_samples[label].append(handpose)
                print(f"Added sample for {label} from {image_path}")
    
    # 各ジェスチャーのサンプルの平均をテンプレートとして保存
    for gesture, samples in gesture_samples.items():
        if samples:
            template = np.mean(samples, axis=0)
            np.save(f"{gesture}_template.npy", template)
            print(f"Generated {gesture}_template.npy from {len(samples)} samples")
        else:
            print(f"No valid samples found for {gesture}")
    
    print("Templates generation completed.")

def train_svm_model(csv_path):
    """SVM（サポートベクターマシン）モデルをトレーニングする関数"""
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        print("CSV columns:", df.columns.tolist())
        
        # CSVの列名を確認して適切な範囲を設定
        expected_features_start = 'wrist_x'
        expected_features_end = 'pinky_tip_z'
        
        if expected_features_start in df.columns and expected_features_end in df.columns and 'label' in df.columns:
            X = df.loc[:, expected_features_start : expected_features_end].values
            y = df['label'].values
            
            print(f"Feature shape: {X.shape}")
            print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")
            
            # ラベルをエンコード
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # SVMモデルを訓練
            model = SVC(kernel='linear', random_state=42)
            model.fit(X, y_encoded)
            
            # 訓練精度を計算
            y_pred = model.predict(X)
            accuracy = accuracy_score(y_encoded, y_pred)
            print(f"Training accuracy: {accuracy:.2f}")
            
            return model, le
        else:
            print(f"Required columns ('{expected_features_start}' to '{expected_features_end}' and 'label') not found in CSV.")
            return None, None
            
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return None, None

# === 既存の関数定義 ===

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
    """フレームから手の骨格情報を抽出する（機械学習対応版）"""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    detected_hands_data = []
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[hand_idx].classification[0].label
            detected_hands_data.append((hand_landmarks, handedness))
    return detected_hands_data

def predict_pose_from_video(source):
    """カメラ映像から手のポーズを推定し、UIと連動させるメイン関数（機械学習対応版）"""
    global fish_img, ui_boxes, current_fish_index, templates, model, label_encoder

    # 機械学習モデルの初期化
    load_dotenv()
    
    # テンプレートの読み込み
    templates = load_templates()
    if not templates:
        print("テンプレートが見つかりません。生成を試みます...")
        generate_templates()
        templates = load_templates()
        if not templates:
            print("テンプレートの生成に失敗しました。ルールベース判定を使用します。")
    
    # CSVファイルからSVMモデルをトレーニング
    csv_path = "hand_landmarks.csv"
    if os.path.exists(csv_path):
        model, label_encoder = train_svm_model(csv_path)
    else:
        print(f"CSVファイルが見つかりません: {csv_path}")
        print("SVMモデルは使用できません。")
        model, label_encoder = None, None

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
    DWELL_TIME = 2.0

    # === Dwell Click 機能の変数 ===
    hover_target_index = None
    hover_start_time = 0
    virtual_cursor_pos = (0, 0)

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
        predicted_gesture = "No gesture"
        
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
            
            # 仮想カーソルの位置を更新
            cursor_lm = smoothed_coords[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            virtual_cursor_pos = (int(cursor_lm[0] * width), int(cursor_lm[1] * height))

            # 機械学習による手のポーズ認識
            if templates:
                # ランドマークデータを配列に変換
                handpose_array = []
                for landmark in raw_landmarks.landmark:
                    handpose_array.extend([landmark.x, landmark.y, landmark.z])
                
                handpose_array = np.array(handpose_array)
                
                # データサイズを63に統一
                if len(handpose_array) < 63:
                    handpose_array = np.pad(handpose_array, (0, 63 - len(handpose_array)), 'constant')
                elif len(handpose_array) > 63:
                    handpose_array = handpose_array[:63]
                
                # コサイン類似度による認識
                similarities = {
                    gesture: weighted_cosine_similarity(handpose_array, template)
                    for gesture, template in templates.items()
                }
                
                predicted_gesture_cosine = max(similarities, key=similarities.get)
                similarity_score = similarities[predicted_gesture_cosine]
                
                # SVMによる認識
                if model is not None and label_encoder is not None:
                    try:
                        y_pred = model.predict([handpose_array])
                        predicted_gesture_svm = label_encoder.inverse_transform(y_pred)[0]
                        predicted_gesture = f"Cosine: {predicted_gesture_cosine} ({similarity_score:.2f}), SVM: {predicted_gesture_svm}"
                    except Exception as e:
                        predicted_gesture = f"Cosine: {predicted_gesture_cosine} ({similarity_score:.2f}), SVM: Error"
                else:
                    predicted_gesture = f"Cosine: {predicted_gesture_cosine} ({similarity_score:.2f})"
                
                # 魚の表示ロジック（パーの場合のみ）
                is_paper = predicted_gesture_cosine == 'paper' and similarity_score > 0.7
            else:
                # フォールバック：従来のルールベース判定
                thumb_tip = smoothed_coords[mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip = smoothed_coords[mp_hands.HandLandmark.PINKY_TIP]
                distance = np.sqrt((thumb_tip[0] - pinky_tip[0])**2 + (thumb_tip[1] - pinky_tip[1])**2)
                is_paper = distance > 0.15
                predicted_gesture = "Paper (Rule-based)" if is_paper else "Other (Rule-based)"

            # 魚の追従処理
            if is_paper:
                pos_lm = smoothed_coords[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                current_hand_x, current_hand_y = int(pos_lm[0] * width), int(pos_lm[1] * height)
                if hand_id not in hand_positions_histories: hand_positions_histories[hand_id] = deque()
                hand_positions_histories[hand_id].append((current_time, current_hand_x, current_hand_y))
                while hand_positions_histories[hand_id] and hand_positions_histories[hand_id][0][0] < current_time - FISH_LAG_SECONDS:
                    hand_positions_histories[hand_id].popleft()
                if hand_positions_histories[hand_id]:
                    fish_positions[hand_id] = hand_positions_histories[hand_id][0][1:]
            else:
                fish_positions[hand_id] = None
                if hand_id in hand_positions_histories: hand_positions_histories[hand_id].clear()
        
        # 認識結果をテキストで表示
        cv2.putText(frame, predicted_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Dwell Click ロジック
        currently_hovering_on = None
        if hand_detected:
            for i, box in enumerate(ui_boxes):
                bx, by, bw, bh = box
                if bx <= virtual_cursor_pos[0] < bx + bw and by <= virtual_cursor_pos[1] < by + bh:
                    currently_hovering_on = i
                    break
        
        if currently_hovering_on is not None and currently_hovering_on != hover_target_index:
            hover_target_index = currently_hovering_on
            hover_start_time = current_time
        elif currently_hovering_on is None:
            hover_target_index = None
        
        if hover_target_index is not None:
            elapsed_time = current_time - hover_start_time
            progress = min(elapsed_time / DWELL_TIME, 1.0)
            box = ui_boxes[hover_target_index]
            progress_radius = int(box[3] / 4)
            center_x = box[0] + box[2] // 2
            center_y = box[1] + box[3] // 2
            cv2.ellipse(frame, (center_x, center_y), (progress_radius, progress_radius), 270, 0, 360 * progress, (0, 255, 255), 5)

            if elapsed_time >= DWELL_TIME:
                print(f"Dwell Click 実行: Index {hover_target_index}")
                current_fish_index = hover_target_index
                full_image = load_image(fish_design_paths[current_fish_index])
                if full_image is not None: fish_img = full_image
                hover_target_index = None

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

        # 仮想カーソルを描画
        if hand_detected:
            cv2.circle(frame, virtual_cursor_pos, 12, (255, 0, 0), -1)
            cv2.circle(frame, virtual_cursor_pos, 15, (255, 255, 255), 2)
        
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
    
    # UIサムネイルサイズを大きくする
    UI_THUMBNAIL_WIDTH = 160
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
    cv2.setMouseCallback(window_name, lambda *args: None)

    print("=== 機械学習対応版 trial3.3.py を開始 ===")
    print("機能:")
    print("- SVM（サポートベクターマシン）による手のポーズ認識")
    print("- テンプレートベースによる手のポーズ認識")
    print("- 重み付きコサイン類似度による判定")
    print("- CSVファイルからの学習データ読み込み")
    print("- フォールバック：ルールベース判定")
    print("=======================================")

    predict_pose_from_video(0)

if __name__ == "__main__":
    main()