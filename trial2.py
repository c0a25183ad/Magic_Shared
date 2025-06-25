# 必要なライブラリをインポート
# cv2: OpenCVライブラリ（画像・動画処理用）
# mediapipe: ポーズや手のランドマーク検出用
# numpy: 数値計算や配列操作用
# os: ファイルやディレクトリ操作用
# sklearn: 機械学習（SVM分類やラベルエンコーディング）用
# pandas: データ処理（CSVファイル操作）用
import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score
import time
from collections import deque # dequeを使用

# MediaPipeの初期化
# mp_holistic: 全身のポーズや手のランドマークを検出するモジュール
# mp_drawing: 検出したランドマークを画像に描画するためのユーティリティ
# drawing_spec: ランドマークの描画スタイル（線の太さや円のサイズ）を指定
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# Holisticオブジェクトを初期化
# static_image_mode=False: 動画処理用に設定
# min_detection_confidence=0.3: 検出の信頼度閾値を0.3に設定（低めにして検出感度を上げる）
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.3)

# 魚の画像と位置関連の変数
# 魚の画像を絶対パスで読み込み
FISH_IMAGE_PATH = "C:/Users/Admin/Downloads/Magic_Shared/Magic_Shared/UploadedImage5.png"
fish_img = cv2.imread(FISH_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

fish_position = None  # 魚の現在位置

# 手の座標履歴と遅延時間の設定
# 各要素は (タイムスタンプ, x座標, y座標) のタプル
hand_positions_history = deque()
# 魚が手の位置に追従する際の遅延時間（秒）
# この値を調整することで、遅れの度合いが変わります
FISH_LAG_SECONDS = 0.5 # 例: 0.5秒遅れで追従

# コサイン類似度を計算する関数
# 2つのベクトルの類似性を測定（値が1に近いほど似ている）
def cosine_similarity(a, b):
    # ベクトルの内積をノルム（長さ）で割ってコサイン類似度を計算
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# スケルトン（手のランドマーク）抽出関数
# 引数: frame（カメラや画像から取得した1フレーム）
# 戻り値: 手のランドマーク座標（handpose）と注釈付きフレーム（annotated_frame）、生の手のランドマークデータ
def extract_skeleton(frame):
    # フレームをBGR（OpenCVのデフォルト）からRGB（MediaPipe用）に変換
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # MediaPipeで手のランドマークを検出
    results = holistic.process(img_rgb)
    
    # フレームのコピーを作成（元のフレームを変更しないため）
    annotated_frame = frame.copy()
    # 右手のランドマークをフレームに描画（線と点で表示）
    mp_drawing.draw_landmarks(annotated_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, drawing_spec, drawing_spec)
    
    # 手のランドマーク座標を格納する空の配列
    handpose = np.array([])
    # 左手のランドマークが検出された場合
    if results.right_hand_landmarks:
        # 各ランドマークのx, y, z座標を配列に追加
        for landmark in results.right_hand_landmarks.landmark:
            handpose = np.append(handpose, [landmark.x, landmark.y, landmark.z])
    return handpose, annotated_frame, results.right_hand_landmarks

# テンプレート（基準となる手のポーズデータ）の読み込み関数
# 各ジェスチャー（rock, scissors, paper）のテンプレートを読み込む
def load_templates():
    templates = {}  # テンプレートを格納する辞書
    for gesture in ['rock', 'scissors', 'paper']:
        try:
            # テンプレートファイルを読み込む（.npy形式）
            templates[gesture] = np.load(f"{gesture}_template.npy")
            print(f"Loaded {gesture}_template.npy")
        except FileNotFoundError:
            # テンプレートファイルが存在しない場合、エラーメッセージを表示
            print(f"Template for {gesture} not found. Please generate templates first.")
            return None
    return templates

# テンプレート生成関数
# 事前に用意した画像（rock.jpg, scissors.jpg, paper.jpg）からテンプレートを作成
def generate_templates():
    # 画像ファイルと対応するラベルのリスト
    image_label_pairs = [
        ("rock.jpg", "rock"),
        ("scissors.jpg", "scissors"),
        ("paper.jpg", "paper"),
    ]
    for image_path, label in image_label_pairs:
        # 画像ファイルが存在しない場合、エラーメッセージを表示
        if not os.path.exists(image_path):
            print(f"Error: {image_path} not found.")
            continue
        # 画像を読み込む
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to load image at {image_path}")
            continue
        # スケルトン抽出関数を使用して手のランドマークを取得
        handpose, _, _ = extract_skeleton(frame)
        if handpose.size > 0:
            # ランドマークデータのサイズを63（21点×3座標）に統一
            if len(handpose) < 63:
                # 足りない場合はゼロでパディング
                handpose = np.pad(handpose, (0, 63 - len(handpose)), 'constant')
            elif len(handpose) > 63:
                # 多い場合は切り捨て
                handpose = handpose[:63]
            # テンプレートを.npyファイルとして保存
            np.save(f"{label}_template.npy", handpose)
            print(f"Generated {label}_template.npy")
        else:
            print(f"No hand detected in {image_path}")
    print("Templates generation completed.")

# SVM（サポートベクターマシン）モデルをトレーニングする関数
# CSVファイルからデータを読み込み、モデルを学習
def train_svm_model(csv_path):
    # CSVファイルが存在しない場合、エラーメッセージを表示
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None, None
    try:
        # CSVファイルを読み込む
        df = pd.read_csv(csv_path)
        print("CSV columns:", df.columns.tolist())
        # 必要な列（'wrist_x'から'pinky_tip_z'と'label'）が存在するか確認
        expected_features_start = 'wrist_x'
        expected_features_end = 'pinky_tip_z'
        if expected_features_start in df.columns and expected_features_end in df.columns and 'label' in df.columns:
            # 特徴量（ランドマーク座標）を抽出
            # 列の範囲で選択
            X = df.loc[:, expected_features_start : expected_features_end].values
            y = df['label'].values
            print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")
            # ラベルを数値に変換（例: 'rock' -> 0, 'scissors' -> 1）
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            # SVMモデル（線形カーネル）を初期化
            model = SVC(kernel='linear')
            # モデルをトレーニング
            model.fit(X, y_encoded)
            # トレーニングデータの精度を計算
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
# カメラや動画ファイルからリアルタイムで手のジェスチャーを認識
def predict_pose_from_video(source, model=None, le=None):
    global fish_position, hand_positions_history # hand_positions_historyもグローバル宣言

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Source not found or cannot be opened at {source}")
        print("Possible causes:")
        print("- Camera not connected or in use by another application.")
        print("- Incorrect device number (try 1 or 2 instead of 0).")
        print("- Camera access permissions not granted in Windows settings.")
        return
    
    # カメラの解像度を設定
    # あなたのカメラがサポートする解像度に合わせて調整してください。
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 例: 幅を1280ピクセルに設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # 例: 高さを720ピクセルに設定

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

    # 魚の画像が読み込まれたかチェックし、サイズ調整
    global fish_img # グローバル変数としてfish_imgを扱う
    if fish_img is None:
        print(f"Error: Could not load fish image from {FISH_IMAGE_PATH}. Please check the path and file existence.")
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        return
    else:
        # 魚の画像をリサイズ (画面の幅の1/8程度のサイズにする)
        target_fish_width = int(width / 4) # 画面幅の1/8
        if fish_img.shape[1] > 0: # ゼロ除算を避ける
            resize_scale = target_fish_width / fish_img.shape[1]
            fish_img = cv2.resize(fish_img, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)
        print(f"Fish image resized to: {fish_img.shape[1]}x{fish_img.shape[0]}")
        if fish_img.shape[2] != 4:
            print("Warning: Fish image does not have an alpha channel (RGBA). Transparency might not work.")
    
    frame_count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
         # ★追加: フレームを左右反転させる (鏡写し)
        frame = cv2.flip(frame, 1) # 1 は左右反転を意味します
        handpose, annotated_frame, landmarks = extract_skeleton(frame)
        predicted_text = "No hand detected"
        
        current_time = time.time() # 現在のタイムスタンプ

        if handpose.size > 0:
            if len(handpose) < 63:
                handpose = np.pad(handpose, (0, 63 - len(handpose)), 'constant')
            elif len(handpose) > 63:
                handpose = handpose[:63]
            
            similarities = {gesture: cosine_similarity(handpose, template) for gesture, template in templates.items()}
            predicted_gesture_cosine = max(similarities, key=similarities.get)
            similarity_score = similarities[predicted_gesture_cosine]
            
            if model is not None and le is not None:
                try:
                    y_pred = model.predict([handpose])
                    predicted_gesture_svm = le.inverse_transform(y_pred)[0]
                    predicted_text = f"Cosine: {predicted_gesture_cosine} (Sim: {similarity_score:.2f})\nSVM: {predicted_gesture_svm}"
                except Exception as e:
                    print(f"Error during SVM prediction: {e}")
                    predicted_text = f"Cosine: {predicted_gesture_cosine} (Sim: {similarity_score:.2f})\nSVM: Error"
            else:
                predicted_text = f"Cosine: {predicted_gesture_cosine} (Sim: {similarity_score:.2f})"
            
            # 魚の追従ロジック
            if predicted_gesture_cosine == 'paper' and landmarks:
                # 中指の付け根のランドマーク (インデックスは9)
                middle_finger_base_landmark = landmarks.landmark[9]
                
                # 現在の手の座標をピクセルに変換
                current_hand_x = int(middle_finger_base_landmark.x * frame.shape[1])
                current_hand_y = int(middle_finger_base_landmark.y * frame.shape[0])

                # 現在の手の座標とタイムスタンプを履歴に追加
                hand_positions_history.append((current_time, current_hand_x, current_hand_y))

                # FISH_LAG_SECONDS秒より古い履歴を削除
                # これにより、dequeには常に最新のFISH_LAG_SECONDS秒間の履歴が保持される
                while hand_positions_history and hand_positions_history[0][0] < current_time - FISH_LAG_SECONDS:
                    hand_positions_history.popleft()
                
                # 魚の位置を決定
                if len(hand_positions_history) > 0:
                    # 履歴の最も古い要素（つまりFISH_LAG_SECONDS秒前の位置）を魚のターゲット位置とする
                    _, fish_target_x, fish_target_y = hand_positions_history[0]
                    fish_position = (fish_target_x, fish_target_y)
                else:
                    # 履歴がまだFISH_LAG_SECONDS秒分に満たない場合（開始直後など）は魚を表示しない
                    fish_position = None
            else:
                # 'paper'以外のジェスチャーの場合、魚の表示を停止し、履歴もクリア
                fish_position = None
                hand_positions_history.clear() # 履歴をクリアして、次の'paper'検出に備える
        else: # 手が検出されない場合
            fish_position = None
            hand_positions_history.clear() # 履歴をクリア
        
        # 魚の描画処理
        if fish_position and fish_img is not None and fish_img.shape[2] == 4:
            fish_h, fish_w = fish_img.shape[:2]
            x, y = fish_position

            x1 = max(0, x - fish_w // 2)
            y1 = max(0, y - fish_h // 2)
            x2 = min(frame.shape[1], x + fish_w // 2)
            y2 = min(frame.shape[0], y + fish_h // 2)

            fish_x1 = max(0, fish_w // 2 - x)
            fish_y1 = max(0, fish_h // 2 - y)
            fish_x2 = fish_x1 + (x2 - x1)
            fish_y2 = fish_y1 + (y2 - y1) # 前回のタイプミスを修正済み

            if (x2 - x1 > 0 and y2 - y1 > 0 and
                fish_x2 - fish_x1 > 0 and fish_y2 - fish_y1 > 0):
                
                alpha_fish = fish_img[fish_y1:fish_y2, fish_x1:fish_x2, 3] / 255.0
                alpha_frame = 1.0 - alpha_fish

                for c in range(3):
                    annotated_frame[y1:y2, x1:x2, c] = (
                        alpha_fish * fish_img[fish_y1:fish_y2, fish_x1:fish_x2, c] +
                        alpha_frame * annotated_frame[y1:y2, x1:x2, c]
                    )
        
        # 推定結果をフレームにテキストとして表示
        y_offset = 30
        for i, line in enumerate(predicted_text.split('\n')):
            cv2.putText(annotated_frame, line, (10, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Pose Detection', annotated_frame)
        
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# メイン処理
def main():
    # 作業ディレクトリを指定
    base_dir = r"C:\Users\Admin\Downloads\Magic_Shared\Magic_Shared"
    os.chdir(base_dir)
    print(f"Current working directory set to: {os.getcwd()}")

    # CSVファイルからSVMモデルをトレーニング
    csv_path = "hand_landmarks.csv"
    model, le = train_svm_model(csv_path) if os.path.exists(csv_path) else (None, None)
    
    # カメラ（0番）を使用して動画処理
    video_source = 0
    predict_pose_from_video(video_source, model, le)

# スクリプトが直接実行された場合にmain()を呼び出す
if __name__ == "__main__":
    main()