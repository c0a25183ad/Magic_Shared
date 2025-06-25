import cv2
import mediapipe as mp
import numpy as np
import os
import time
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score

# MediaPipeの初期化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.3)

# 魚の画像を読み込み（背景透過PNG）
fish_img = cv2.imread('UploadedImage5.png', cv2.IMREAD_UNCHANGED)
fish_position = None  # 魚の現在位置

# コサイン類似度を計算する関数
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# スケルトン抽出関数（landmarksも返す）
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
# テンプレート読み込み関数
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
    image_label_pairs = [
        ("rock.jpg", "rock"),
        ("scissors.jpg", "scissors"),
        ("paper.jpg", "paper"),
    ]
    for image_path, label in image_label_pairs:
        if not os.path.exists(image_path):
            print(f"Error: {image_path} not found.")
            continue
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to load image at {image_path}")
            continue
        handpose, _, _ = extract_skeleton(frame)
        if handpose.size > 0:
            if len(handpose) < 63:
                handpose = np.pad(handpose, (0, 63 - len(handpose)), 'constant')
            elif len(handpose) > 63:
                handpose = handpose[:63]
            np.save(f"{label}_template.npy", handpose)
            print(f"Generated {label}_template.npy")
        else:
            print(f"No hand detected in {image_path}")
    print("Templates generation completed.")

# SVMモデルをトレーニングする関数
def train_svm_model(csv_path):
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None, None
    try:
        df = pd.read_csv(csv_path)
        print("CSV columns:", df.columns.tolist())
        if all(col in df.columns for col in ['wrist_x', 'pinky_tip_z', 'label']):
            X = df.loc[:, 'wrist_x': 'pinky_tip_z'].values
            y = df.loc[:, 'label'].values
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
            print("Required columns ('wrist_x' to 'pinky_tip_z' or 'label') not found in CSV.")
            return None, None
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return None, None
# 動画からポーズを推定する関数
def predict_pose_from_video(source, model=None, le=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Source not found or cannot be opened at {source}")
        return

    os.makedirs("output", exist_ok=True)
    templates = load_templates()
    if templates is None:
        generate_templates()
        templates = load_templates()
    if templates is None:
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        handpose, annotated_frame, landmarks = extract_skeleton(frame)
        predicted_text = "No hand detected"

        if handpose.size > 0:
            if len(handpose) < 63:
                handpose = np.pad(handpose, (0, 63 - len(handpose)), 'constant')
            elif len(handpose) > 63:
                handpose = handpose[:63]

            similarities = {gesture: cosine_similarity(handpose, template) for gesture, template in templates.items()}
            predicted_gesture_cosine = max(similarities, key=similarities.get)
            similarity_score = similarities[predicted_gesture_cosine]

            if predicted_gesture_cosine == 'paper' and landmarks:
                fish_position = (
                    int(landmarks.landmark[9].x * frame.shape[1]),
                    int(landmarks.landmark[9].y * frame.shape[0])
                )
                time.sleep(1)
            else:
                fish_position = None

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

        cv2.putText(annotated_frame, predicted_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Pose Detection', annotated_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# メイン処理
def main():
    base_dir = r"C:\\Users\\ionna\\Downloads\\Magic_Shared\\Magic_Shared"
    os.chdir(base_dir)

    csv_path = "hand_landmarks.csv"
    model, le = train_svm_model(csv_path) if os.path.exists(csv_path) else (None, None)

    video_source = 0
    predict_pose_from_video(video_source, model, le)

if __name__ == "__main__":
    main()
