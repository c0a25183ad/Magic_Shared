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

# コサイン類似度を計算する関数
# 2つのベクトルの類似性を測定（値が1に近いほど似ている）
def cosine_similarity(a, b):
    # ベクトルの内積をノルム（長さ）で割ってコサイン類似度を計算
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# スケルトン（手のランドマーク）抽出関数
# 引数: frame（カメラや画像から取得した1フレーム）
# 戻り値: 手のランドマーク座標（handpose）と注釈付きフレーム（annotated_frame）
def extract_skeleton(frame):
    # フレームをBGR（OpenCVのデフォルト）からRGB（MediaPipe用）に変換
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # MediaPipeで手のランドマークを検出
    results = holistic.process(img_rgb)
    
    # フレームのコピーを作成（元のフレームを変更しないため）
    annotated_frame = frame.copy()
    # 右手のランドマークをフレームに描画（線と点で表示）
    mp_drawing.draw_landmarks(annotated_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    # 手のランドマーク座標を格納する空の配列
    handpose = np.array([])
    # 右手のランドマークが検出された場合
    if results.right_hand_landmarks:
        # 各ランドマークのx, y, z座標を配列に追加
        for landmark in results.right_hand_landmarks.landmark:
            handpose = np.append(handpose, [landmark.x, landmark.y, landmark.z])
    return handpose, annotated_frame

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
        handpose, _ = extract_skeleton(frame)
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
        if  'label' in df.columns:
            X = df.drop(columns=['label']).values
            y = df['label'].values
            # 特徴量（ランドマーク座標）とラベルを抽出
            X = df.loc[:, 'wrist_x': 'pinky_tip_z'].values
            y = df.loc[:, 'label'].values
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
            print("Required columns ('wrist_x' to 'pinky_tip_z' or 'label') not found in CSV.")
            return None, None
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return None, None

# 動画からポーズを推定する関数
# カメラや動画ファイルからリアルタイムで手のジェスチャーを認識
def predict_pose_from_video(source, model=None, le=None):
    # 動画ソース（カメラの場合は0、動画ファイルの場合はパス）を指定
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Source not found or cannot be opened at {source}")
        print("Possible causes:")
        print("- Camera not connected or in use by another application.")
        print("- Incorrect device number (try 1 or 2 instead of 0).")
        print("- Camera access permissions not granted in Windows settings.")
        return
    
    # 出力ディレクトリを作成（存在しない場合）
    os.makedirs("output", exist_ok=True)
    # テンプレートを読み込む
    templates = load_templates()
    if templates is None:
        # テンプレートが存在しない場合、生成を試みる
        generate_templates()
        templates = load_templates()
        if templates is None:
            return
    
    frame_count = 1 # フレーム番号をカウント
    while cap.isOpened():
        # フレームを1つずつ読み込む
        ret, frame = cap.read()
        if not ret:
            break
        
        # スケルトン抽出
        handpose, annotated_frame = extract_skeleton(frame)
        predicted_text = "No hand detected"  # デフォルトのテキスト
        if handpose.size > 0:
            # ランドマークデータのサイズを63に統一
            if len(handpose) < 63:
                handpose = np.pad(handpose, (0, 63 - len(handpose)), 'constant')
            elif len(handpose) > 63:
                handpose = handpose[:63]
            
            # コサイン類似度によるジェスチャー推定
            similarities = {gesture: cosine_similarity(handpose, template) for gesture, template in templates.items()}
            predicted_gesture_cosine = max(similarities, key=similarities.get)
            similarity_score = similarities[predicted_gesture_cosine]
            print(f"Frame {frame_count}: Cosine Predicted: {predicted_gesture_cosine}, Similarity: {similarity_score:.4f}")
            print(f"Similarities: {similarities}")  # デバッグ用
            
            # SVMによる推定（モデルが利用可能な場合）
            if model is not None and le is not None:
                y_pred = model.predict([handpose])
                predicted_gesture_svm = le.inverse_transform(y_pred)[0]
                print(f"Frame {frame_count}: SVM Predicted: {predicted_gesture_svm}")
                predicted_text = f"Cosine: {predicted_gesture_cosine} (Sim: {similarity_score:.2f})\nSVM: {predicted_gesture_svm}"
            else:
                predicted_text = f"Cosine: {predicted_gesture_cosine} (Sim: {similarity_score:.2f})"
        
        # 推定結果をフレームにテキストとして表示
        cv2.putText(annotated_frame, predicted_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # リアルタイムでフレームを表示
        cv2.imshow('Pose Detection', annotated_frame)
        # 'q'キーを押すと終了
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    # 動画を解放し、ウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()

# メイン処理
def main():
    # 作業ディレクトリを指定
    base_dir = r"C:\Users\Admin\Downloads\Magic_Shared\Magic_Shared"
    os.chdir(base_dir)
    
    # CSVファイルからSVMモデルをトレーニング
    csv_path = "hand_landmarks.csv"
    model, le = train_svm_model(csv_path) if os.path.exists(csv_path) else (None, None)
    
    # カメラ（0番）を使用して動画処理
    video_source = 0
    predict_pose_from_video(video_source, model, le)

# スクリプトが直接実行された場合にmain()を呼び出す
if __name__ == "__main__":
    main()

# 魚の画像を読み込み（背景透過PNG）
fish_img = cv2.imread('UploadedImage5.png', cv2.IMREAD_UNCHANGED)
fish_position = None  # 魚の現在位置

# スケルトン抽出関数の戻り値に landmarks を追加
def extract_skeleton(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)
    annotated_frame = frame.copy()
    mp_drawing.draw_landmarks(annotated_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    handpose = np.array([])
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            handpose = np.append(handpose, [landmark.x, landmark.y, landmark.z])
    return handpose, annotated_frame, results.right_hand_landmarks
# predict_pose_from_video 関数内の while ループに追加
handpose, annotated_frame, landmarks = extract_skeleton(frame)
predicted_text = "No hand detected"

if handpose.size > 0:
    # ...（ジェスチャー推定処理）...

    if predicted_gesture_cosine == 'paper' and landmarks:
        fish_position = (
            int(landmarks.landmark[9].x * frame.shape[1]),
            int(landmarks.landmark[9].y * frame.shape[0])
        )
        time.sleep(1)
    else:
        fish_position = None

# 魚の描画処理
if fish_position:
    fish_h, fish_w = fish_img.shape[:2]
    x, y = fish_position
    x1, y1 = max(0, x - fish_w // 2), max(0, y - fish_h // 2)
    x2, y2 = min(frame.shape[1], x + fish_w // 2), min(frame.shape[0], y + fish_h // 2)
    fish_x1, fish_y1 = max(0, fish_w // 2 - x), max(0, fish_h // 2 - y)
    fish_x2, fish_y2 = fish_x1 + (x2 - x1), fish_y1 + (y2 - y1)

    alpha_fish = fish_img[fish_y1:fish_y2, fish_x1:fish_x2, 3] / 255.0
    alpha_frame = 1.0 - alpha_fish
    for c in range(3):
        annotated_frame[y1:y2, x1:x2, c] = (
            alpha_fish * fish_img[fish_y1:fish_y2, fish_x1:fish_x2, c] +
            alpha_frame * annotated_frame[y1:y2, x1:x2, c]
        )
