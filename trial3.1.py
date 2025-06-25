# 必要なライブラリをインポート
import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands, drawing_utils
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score
import time
from collections import deque

# MediaPipeの初期化
# mp_hands を使用して手の検出に特化
mp_drawing = drawing_utils
mp_hands = hands

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

hands = mp_hands.Hands( # Handsオブジェクトを初期化
    static_image_mode=False,
    max_num_hands=3, # 最大3つの手を検出（左右+もう1つの手に対応）
    min_detection_confidence=0.5, # 検出の信頼度閾値
    min_tracking_confidence=0.5 # 追跡の信頼度閾値
)

# 魚の画像と位置関連の変数
# 魚の画像を絶対パスで読み込み
# ★★★重要★★★ このパスはあなたの環境に合わせて正確に修正してください
FISH_IMAGES_DIR = "fish_designs"  # 魚のデザインフォルダ
FISH_IMAGE_PATH = "C:/Users/Admin/Downloads/Magic_Shared/Magic_Shared/UploadedImage5.png"  # デフォルトの魚画像
fish_img = None  # 現在選択されている魚の画像

# 魚のデザイン選択機能
def load_fish_designs():
    """
    魚のデザインフォルダから利用可能な魚の画像を読み込む
    """
    fish_designs = {}
    
    # デフォルトの魚画像を追加
    if os.path.exists(FISH_IMAGE_PATH):
        fish_designs["default"] = FISH_IMAGE_PATH
    
    # 魚のデザインフォルダが存在しない場合は作成
    if not os.path.exists(FISH_IMAGES_DIR):
        os.makedirs(FISH_IMAGES_DIR, exist_ok=True)
        print(f"Created fish designs folder: {FISH_IMAGES_DIR}")
        print("魚のデザインフォルダを作成しました。")
        print("このフォルダに魚の画像ファイル（.png, .jpg, .jpeg）を配置してください。")
    
    # 魚のデザインフォルダから画像を読み込み
    if os.path.exists(FISH_IMAGES_DIR):
        image_files = [f for f in os.listdir(FISH_IMAGES_DIR) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            for filename in image_files:
                # ファイル名から拡張子を除いた部分をデザイン名として使用
                design_name = os.path.splitext(filename)[0]
                file_path = os.path.join(FISH_IMAGES_DIR, filename)
                fish_designs[design_name] = file_path
                print(f"Found fish design: {design_name} -> {file_path}")
        else:
            print(f"No image files found in {FISH_IMAGES_DIR}")
            print("魚のデザインフォルダに画像ファイルを追加してください。")
    
    return fish_designs

def select_fish_design(fish_designs):
    """
    利用可能な魚のデザインから選択する
    """
    if not fish_designs:
        print("No fish designs found!")
        print("魚のデザインが見つかりません。")
        print("以下のいずれかの方法で魚の画像を追加してください：")
        print("1. デフォルトの魚画像（UploadedImage5.png）を配置")
        print("2. fish_designsフォルダに魚の画像ファイルを追加")
        return None
    
    print("\n=== 利用可能な魚のデザイン ===")
    design_names = list(fish_designs.keys())
    for i, name in enumerate(design_names):
        print(f"{i+1}. {name}")
    
    # デフォルトの魚画像がある場合は推奨
    if "default" in fish_designs:
        print(f"\n推奨: デフォルトの魚画像を使用する場合は {design_names.index('default') + 1} を選択してください")
    
    while True:
        try:
            choice = input(f"\n魚のデザインを選択してください (1-{len(design_names)}): ")
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(design_names):
                selected_design = design_names[choice_idx]
                selected_path = fish_designs[selected_design]
                print(f"選択されたデザイン: {selected_design}")
                return selected_path
            else:
                print("無効な選択です。もう一度お試しください。")
        except ValueError:
            print("数字を入力してください。")
        except KeyboardInterrupt:
            print("\nデフォルトの魚を使用します。")
            return FISH_IMAGE_PATH if os.path.exists(FISH_IMAGE_PATH) else None

def load_selected_fish_image(fish_path):
    """
    選択された魚の画像を読み込む
    """
    if fish_path and os.path.exists(fish_path):
        img = cv2.imread(fish_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            print(f"魚の画像を読み込みました: {fish_path}")
            return img
        else:
            print(f"魚の画像の読み込みに失敗しました: {fish_path}")
    else:
        print(f"魚の画像ファイルが見つかりません: {fish_path}")
    
    return None

# 複数の魚の位置を管理するための辞書 (key: 手のID, value: (x, y)座標)
fish_positions = {}
# 各手の座標履歴を管理するための辞書 (key: 手のID, value: dequeオブジェクト)
hand_positions_histories = {}

# 魚が手の位置に追従する際の遅延時間（秒）
# この値を調整することで、遅れの度合いが変わります
FISH_LAG_SECONDS = 0.5

# 移動平均フィルタの設定
MOVING_AVERAGE_WINDOW = 5  # 移動平均のウィンドウサイズ

# 各手のランドマーク履歴を管理するための辞書 (key: 手のID, value: dequeオブジェクト)
hand_landmarks_histories = {}

# 移動平均フィルタを適用する関数
def apply_moving_average_filter(landmarks_history, current_landmarks):
    """
    ランドマーク履歴に移動平均フィルタを適用して平滑化する
    座標値のみを平滑化して返す
    """
    try:
        # 履歴がMOVING_AVERAGE_WINDOWに満たない場合は、現在の履歴の平均を使用
        # ここでは、current_landmarks自体もdequeに追加してから平均を計算
        
        # まず、現在のランドマークを履歴に追加（今回は関数内で処理）
        # この関数が呼ばれる前にlandmarks_historyにcurrent_landmarksが追加されている想定
        
        # 履歴から最新のNフレーム分を取得 (現在のランドマークは既に履歴に含まれている)
        # そのため、landmarks_historyの現在の内容全てを使用
        recent_landmarks = list(landmarks_history)
        
        if not recent_landmarks: # 履歴が空の場合は現在の値をそのまま返す（本来は発生しないはずだが安全のため）
            return [(lm.x, lm.y, lm.z) for lm in current_landmarks.landmark]

        smoothed_coordinates = []
        for i in range(len(current_landmarks.landmark)): # ランドマークの数だけループ
            x_values = [lm.landmark[i].x for lm in recent_landmarks]
            y_values = [lm.landmark[i].y for lm in recent_landmarks]
            z_values = [lm.landmark[i].z for lm in recent_landmarks]
            
            # 移動平均を計算
            avg_x = sum(x_values) / len(x_values)
            avg_y = sum(y_values) / len(y_values)
            avg_z = sum(z_values) / len(z_values)
            
            smoothed_coordinates.append((avg_x, avg_y, avg_z))
        
        return smoothed_coordinates
    except Exception as e:
        print(f"Error in moving average filter: {e}")
        # エラーが発生した場合は現在の座標をそのまま返す
        return [(lm.x, lm.y, lm.z) for lm in current_landmarks.landmark]

# コサイン類似度を計算する関数
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 重み付きコサイン類似度を計算する関数
def weighted_cosine_similarity(a, b, weights=None):
    if weights is None:
        # デフォルトの重み：指先のランドマークに高い重みを与える
        weights = np.ones(len(a))
        # 指先のランドマーク（4, 8, 12, 16, 20）に高い重み
        finger_tip_indices = [4, 8, 12, 16, 20]
        for idx in finger_tip_indices:
            weights[idx*3:(idx+1)*3] = 2.0  # 指先のx,y,zに2倍の重み
    
    # 重み付きコサイン類似度を計算
    weighted_a = a * weights
    weighted_b = b * weights
    return np.dot(weighted_a, weighted_b) / (np.linalg.norm(weighted_a) * np.linalg.norm(weighted_b))

# スケルトン（手のランドマーク）抽出関数
# 戻り値: 検出された手のデータのリストと、注釈付きフレーム
# detected_hands_dataの各要素は (handpose_array, raw_landmarks_object, handedness_label) のタプル
def extract_skeleton(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)  # type: ignore
    annotated_frame = frame.copy()
    # ここでのcv2.flipは削除済み（二重反転を避けるため）
    # annotated_frame = cv2.flip(annotated_frame, 1) 
    detected_hands_data = []

    if hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                list(mp_hands.HAND_CONNECTIONS),  # linter対策
                drawing_spec,
                drawing_spec
            )
            handpose_array = np.array([])
            for landmark in hand_landmarks.landmark:
                handpose_array = np.append(handpose_array, [landmark.x, landmark.y, landmark.z])
            handedness = results.multi_handedness[hand_idx].classification[0].label if hasattr(results, 'multi_handedness') else 'Unknown'  # type: ignore
            detected_hands_data.append((handpose_array, hand_landmarks, handedness))
    return detected_hands_data, annotated_frame

# テンプレート（基準となる手のポーズデータ）の読み込み関数
def load_templates():
    templates = {}
    for gesture in ['rock', 'scissors', 'paper']:
        try:
            templates[gesture] = np.load(f"{gesture}_template.npy")
            print(f"Loaded {gesture}_template.npy")
        except FileNotFoundError:
            print(f"Template for {gesture} not found. Please generate templates first.")
            return None
    return templates

# テンプレート生成関数
def generate_templates():
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
        temp_detected_hands_data, _ = extract_skeleton(frame)
        
        if temp_detected_hands_data:  # 何らかの手が検出された場合
            handpose, _, _ = temp_detected_hands_data[0]  # 最初の手のデータを取得
            if handpose.size > 0:
                if len(handpose) < 63:
                    handpose = np.pad(handpose, (0, 63 - len(handpose)), 'constant')
                elif len(handpose) > 63:
                    handpose = handpose[:63]
                gesture_samples[label].append(handpose)
                print(f"Added sample for {label} from {image_path}")
            else:
                print(f"No hand detected for {label} in {image_path}")
        else:
            print(f"No hand detected for {label} in {image_path}")
    
    # 各ジェスチャーのサンプルの平均をテンプレートとして保存
    for gesture, samples in gesture_samples.items():
        if samples:
            # サンプルの平均を計算
            template = np.mean(samples, axis=0)
            np.save(f"{gesture}_template.npy", template)
            print(f"Generated {gesture}_template.npy from {len(samples)} samples")
        else:
            print(f"No valid samples found for {gesture}")
    
    print("Templates generation completed.")

# SVM（サポートベクターマシン）モデルをトレーニングする関数
def train_svm_model(csv_path):
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None, None
    try:
        df = pd.read_csv(csv_path)
        print("CSV columns:", df.columns.tolist())
        expected_features_start = 'wrist_x'
        expected_features_end = 'pinky_tip_z'
        if expected_features_start in df.columns and expected_features_end in df.columns and 'label' in df.columns:
            X = df.loc[:, expected_features_start : expected_features_end].values
            y = df['label'].values
            print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            model = SVC(kernel='linear')
            model.fit(X, y_encoded)
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

# 動画からポーズを推定する関数
def predict_pose_from_video(source, model=None, le=None):
    global fish_img, fish_positions, hand_positions_histories, hand_landmarks_histories # hand_landmarks_histories もグローバル宣言

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Source not found or cannot be opened at {source}")
        print("Possible causes:")
        print("- Camera not connected or in use by another application.")
        print("- Incorrect device number (try 1 or 2 instead of 0).")
        print("- Camera access permissions not granted in Windows settings.")
        return
    
    # カメラの解像度を設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(output_dir, "output_video.mp4")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        print("Possible causes:")
        print("- Incorrect codec (fourcc). Try 'XVID' or 'MJPG'.")
        print("- Insufficient disk space.")
        print("- Output path invalid.")
        cap.release()
        return
    
    templates = load_templates()
    if templates is None:
        print("Attempting to generate templates...")
        generate_templates()
        templates = load_templates()
        if templates is None:
            print("Failed to load or generate templates. Exiting.")
            out.release()
            return

    if fish_img is None:
        print(f"Error: Fish image is not loaded. Please select a design or check default path.")
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        return
    else:
        target_fish_width = int(width / 8) # 画面幅の1/8にリサイズ
        if fish_img.shape[1] > 0:
            resize_scale = target_fish_width / fish_img.shape[1]
            fish_img = cv2.resize(fish_img, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)
        print(f"Fish image resized to: {fish_img.shape[1]}x{fish_img.shape[0]}")
        if fish_img.shape[2] != 4:
            print("Warning: Fish image does not have an alpha channel (RGBA). Transparency might not work.")
    
    frame_count = 1
    
    # 前のフレームで検出された手のIDを追跡し、現在のフレームで検出されなかった魚を削除するため
    active_hand_ids_this_frame = set() 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # フレームを左右反転させる (鏡写し)
        # この反転が、表示を鏡写しにするための唯一の反転
        frame = cv2.flip(frame, 1)

        # 複数の手のデータを取得
        # extract_skeleton 内部には cv2.flip を置かないこと！
        detected_hands_data, annotated_frame = extract_skeleton(frame)
        
        current_time = time.time()

        # 各手について処理
        for idx, hand_data in enumerate(detected_hands_data):
            handpose, landmarks, handedness_label = hand_data
            
            # 手のIDを生成 (例: 'Right Hand 0', 'Left Hand 0', 'Right Hand 1')
            hand_id = f"{handedness_label} Hand {idx}" 
            active_hand_ids_this_frame.add(hand_id) 

            # その手の履歴キューが存在しない場合は初期化
            if hand_id not in hand_positions_histories:
                hand_positions_histories[hand_id] = deque()
            if hand_id not in hand_landmarks_histories:
                hand_landmarks_histories[hand_id] = deque(maxlen=MOVING_AVERAGE_WINDOW) # dequeにmaxlenを設定

            # 平滑化されたランドマークを履歴に追加
            hand_landmarks_histories[hand_id].append(landmarks) # raw_landmarks_object を履歴に追加
            
            # 移動平均フィルタを適用してランドマークを平滑化