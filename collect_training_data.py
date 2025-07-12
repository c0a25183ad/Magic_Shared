import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from datetime import datetime

class GestureDataCollector:
    def __init__(self):
        # MediaPipe初期化
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # データ収集設定
        self.gestures = ['rock', 'scissors', 'paper']
        self.samples_per_gesture = 50
        self.collected_data = []
        
        # 現在の状態
        self.current_gesture_index = 0
        self.current_samples = 0
        self.is_collecting = False
        
        # ランドマーク名の定義
        self.landmark_names = [
            'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
            'index_mcp', 'index_pip', 'index_dip', 'index_tip',
            'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
            'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
        ]
    
    def extract_landmarks(self, hand_landmarks):
        """手のランドマークを配列に変換"""
        landmarks_array = []
        for landmark in hand_landmarks.landmark:
            landmarks_array.extend([landmark.x, landmark.y, landmark.z])
        return landmarks_array
    
    def create_sample_data(self, landmarks_array, gesture_label):
        """サンプルデータを作成"""
        sample_data = {
            'timestamp': datetime.now().isoformat(),
            'label': gesture_label
        }
        
        # ランドマーク座標を追加
        for i, name in enumerate(self.landmark_names):
            sample_data[f'{name}_x'] = landmarks_array[i*3]
            sample_data[f'{name}_y'] = landmarks_array[i*3+1]
            sample_data[f'{name}_z'] = landmarks_array[i*3+2]
        
        return sample_data
    
    def draw_instructions(self, frame):
        """画面に指示を描画"""
        height, width = frame.shape[:2]
        
        # 背景の半透明オーバーレイ
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 現在のジェスチャー情報
        current_gesture = self.gestures[self.current_gesture_index]
        cv2.putText(frame, f"Current Gesture: {current_gesture.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # プログレス表示
        progress_text = f"Progress: {self.current_samples}/{self.samples_per_gesture}"
        cv2.putText(frame, progress_text, 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 全体の進捗
        total_progress = (self.current_gesture_index * self.samples_per_gesture + self.current_samples)
        total_target = len(self.gestures) * self.samples_per_gesture
        overall_text = f"Overall: {total_progress}/{total_target}"
        cv2.putText(frame, overall_text, 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 操作説明
        if self.is_collecting:
            cv2.putText(frame, "COLLECTING... Press SPACE to save sample", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Press 'S' to start collecting", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, "Press 'Q' to quit, 'N' for next gesture", 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # プログレスバー
        bar_width = 400
        bar_height = 20
        bar_x = 10
        bar_y = height - 40
        
        # 背景
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # プログレス
        progress_ratio = self.current_samples / self.samples_per_gesture
        progress_width = int(bar_width * progress_ratio)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
    
    def draw_gesture_example(self, frame):
        """現在のジェスチャーの例を描画"""
        height, width = frame.shape[:2]
        
        # 右上にジェスチャーの説明を表示
        text_x = width - 300
        text_y = 30
        
        current_gesture = self.gestures[self.current_gesture_index]
        
        if current_gesture == 'rock':
            instructions = [
                "ROCK (グー):",
                "- Close all fingers",
                "- Make a fist",
                "- Keep thumb outside"
            ]
        elif current_gesture == 'scissors':
            instructions = [
                "SCISSORS (チョキ):",
                "- Extend index & middle finger",
                "- Keep other fingers closed",
                "- Form a 'V' shape"
            ]
        elif current_gesture == 'paper':
            instructions = [
                "PAPER (パー):",
                "- Extend all fingers",
                "- Keep fingers spread",
                "- Open palm"
            ]
        
        for i, instruction in enumerate(instructions):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            thickness = 2 if i == 0 else 1
            cv2.putText(frame, instruction, (text_x, text_y + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    
    def collect_data(self):
        """メインのデータ収集処理"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # カメラの解像度設定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("=== Hand Gesture Data Collection ===")
        print("Controls:")
        print("- 'S': Start/Stop collecting")
        print("- 'SPACE': Save current sample")
        print("- 'N': Next gesture")
        print("- 'Q': Quit")
        print("=" * 40)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 画像を反転
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 手の検出
            results = self.hands.process(rgb_frame)
            
            # 手のランドマークを描画
            hand_detected = False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    hand_detected = True
            
            # UI要素を描画
            self.draw_instructions(frame)
            self.draw_gesture_example(frame)
            
            # 手が検出されない場合の警告
            if not hand_detected:
                cv2.putText(frame, "No hand detected!", 
                           (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow('Gesture Data Collection', frame)
            
            # キー入力処理
            key = cv2.waitKey(5) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.is_collecting = not self.is_collecting
                status = "Started" if self.is_collecting else "Stopped"
                print(f"Data collection {status}")
            elif key == ord('n'):
                self.next_gesture()
            elif key == ord(' ') and self.is_collecting and hand_detected:
                self.save_sample(results.multi_hand_landmarks[0])
        
        cap.release()
        cv2.destroyAllWindows()
        
        # データを保存
        self.save_to_csv()
    
    def save_sample(self, hand_landmarks):
        """サンプルを保存"""
        try:
            landmarks_array = self.extract_landmarks(hand_landmarks)
            current_gesture = self.gestures[self.current_gesture_index]
            sample_data = self.create_sample_data(landmarks_array, current_gesture)
            
            self.collected_data.append(sample_data)
            self.current_samples += 1
            
            print(f"Sample saved: {current_gesture} ({self.current_samples}/{self.samples_per_gesture})")
            
            # 現在のジェスチャーのサンプル数が達成されたら次へ
            if self.current_samples >= self.samples_per_gesture:
                self.next_gesture()
        
        except Exception as e:
            print(f"Error saving sample: {e}")
    
    def next_gesture(self):
        """次のジェスチャーに移行"""
        if self.current_gesture_index < len(self.gestures) - 1:
            self.current_gesture_index += 1
            self.current_samples = 0
            current_gesture = self.gestures[self.current_gesture_index]
            print(f"Next gesture: {current_gesture}")
        else:
            print("All gestures completed! Press 'Q' to quit.")
    
    def save_to_csv(self):
        """収集したデータをCSVファイルに保存"""
        if not self.collected_data:
            print("No data to save")
            return
        
        try:
            df = pd.DataFrame(self.collected_data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"hand_landmarks_{timestamp}.csv"
            df.to_csv(filename, index=False)
            
            print(f"\nData saved to: {filename}")
            print(f"Total samples collected: {len(self.collected_data)}")
            
            # データ分布を表示
            print("\nData distribution:")
            print(df['label'].value_counts())
            
        except Exception as e:
            print(f"Error saving CSV: {e}")

def main():
    """メイン処理"""
    collector = GestureDataCollector()
    
    print("Hand Gesture Data Collection Tool")
    print("=" * 50)
    print("This tool will help you collect training data for SVM model.")
    print("You will collect 50 samples for each gesture: rock, scissors, paper")
    print("Total samples to collect: 150")
    print("=" * 50)
    
    input("Press Enter to start data collection...")
    
    try:
        collector.collect_data()
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    except Exception as e:
        print(f"Error during data collection: {e}")
    finally:
        print("Data collection completed!")

if __name__ == "__main__":
    main()
    