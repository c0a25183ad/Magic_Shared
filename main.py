import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score

# MediaPipeの初期化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.3)  # 検出閾値を下げて調整

# コサイン類似度計算
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# スケルトン抽出関数
def extract_skeleton(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)
    
    annotated_frame = frame.copy()
    mp_drawing.draw_landmarks(annotated_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    handpose = np.array([])
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            handpose = np.append(handpose, [landmark.x, landmark.y, landmark.z])
    return handpose, annotated_frame

# テンプレートの読み込み
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
        handpose, _ = extract_skeleton(frame)
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

# SVMモデルのトレーニング
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

# 動画処理用のポーズ推定関数
def predict_pose_from_video(source, model=None, le=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Source not found or cannot be opened at {source}")
        print("Possible causes:")
        print("- Camera not connected or in use by another application.")
        print("- Incorrect device number (try 1 or 2 instead of 0).")
        print("- Camera access permissions not granted in Windows settings.")
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
        
        handpose, annotated_frame = extract_skeleton(frame)
        predicted_text = "No hand detected"
        if handpose.size > 0:
            if len(handpose) < 63:
                handpose = np.pad(handpose, (0, 63 - len(handpose)), 'constant')
            elif len(handpose) > 63:
                handpose = handpose[:63]
            
            # コサイン類似度
            similarities = {gesture: cosine_similarity(handpose, template) for gesture, template in templates.items()}
            predicted_gesture_cosine = max(similarities, key=similarities.get)
            similarity_score = similarities[predicted_gesture_cosine]
            print(f"Frame {frame_count}: Cosine Predicted: {predicted_gesture_cosine}, Similarity: {similarity_score:.4f}")
            print(f"Similarities: {similarities}")  # デバッグ用
            
            # SVM予測（モデルが利用可能の場合）
            if model is not None and le is not None:
                y_pred = model.predict([handpose])
                predicted_gesture_svm = le.inverse_transform(y_pred)[0]
                print(f"Frame {frame_count}: SVM Predicted: {predicted_gesture_svm}")
                predicted_text = f"Cosine: {predicted_gesture_cosine} (Sim: {similarity_score:.2f})\nSVM: {predicted_gesture_svm}"
            else:
                predicted_text = f"Cosine: {predicted_gesture_cosine} (Sim: {similarity_score:.2f})"
        
        # ポーズ名を画像にオーバーレイ
        cv2.putText(annotated_frame, predicted_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # リアルタイム表示
        cv2.imshow('Pose Detection', annotated_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

# メイン処理
def main():
    base_dir = r"C:\Users\Takayama Ryuji\Desktop\Magic"
    os.chdir(base_dir)
    
    csv_path = "hand_landmarks.csv"
    model, le = train_svm_model(csv_path) if os.path.exists(csv_path) else (None, None)
    
    video_source = 0
    predict_pose_from_video(video_source, model, le)

if __name__ == "__main__":
    main()