import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# MediaPipeの初期化
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5)

# 画像から右手のランドマークを抽出
def extract_right_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found at {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)
    
    handpose = np.array([])
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            handpose = np.append(handpose, [landmark.x, landmark.y, landmark.z])
    return handpose

# メイン処理
def main():
    base_dir = r"C:\Users\Takayama Ryuji\Desktop\Magic"
    os.chdir(base_dir)
    
    # 画像とラベルのリスト
    image_label_pairs = [
        ("rock.jpg", "rock"),
        ("scissors.jpg", "scissors"),
        ("papar.jpg", "paper"),
    ]
    
    data = []
    for image_path, label in image_label_pairs:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        handpose = extract_right_hand_landmarks(image_path)
        if handpose.size == 0:
            print(f"No right hand detected in {image_path}")
            continue
        
        # 63次元にパディングまたはトリミング
        if len(handpose) < 63:
            handpose = np.pad(handpose, (0, 63 - len(handpose)), 'constant')
        elif len(handpose) > 63:
            handpose = handpose[:63]
        
        # データをリストに追加
        row = [label] + handpose.tolist()
        data.append(row)
    
    # DataFrameに変換して保存
    if data:
        columns = ['label'] + [f'x{i}' for i in range(1, 64)]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv("hand_landmarks.csv", index=False)
        print("Hand landmarks saved to hand_landmarks.csv")
    else:
        print("No data to save.")

if __name__ == "__main__":
    main()