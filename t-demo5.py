import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- 設定 ---
WINDOW_SIZE = 10      # 平滑化の強さ (W=10)
DELAY_SECONDS = 1.0   # 遅延させる秒数
FPS_ASSUMPTION = 30   # カメラの想定FPS
BUFFER_SIZE = int(DELAY_SECONDS * FPS_ASSUMPTION) # 遅延用のバッファサイズ

IMAGE_PATH = "../UploadedImage5.png" # 画像ファイルパス
FISH_SCALE = 0.3      # 魚画像の大きさ倍率

# --- 画像を透過合成する関数 ---
def overlay_image_alpha(img, img_overlay, pos, is_facing_right):
    x, y = pos
    if not is_facing_right:
        img_overlay = cv2.flip(img_overlay, 1)

    h_overlay, w_overlay = img_overlay.shape[:2]
    h_bg, w_bg = img.shape[:2]

    x1 = x - w_overlay // 2
    y1 = y - h_overlay // 2
    x2 = x1 + w_overlay
    y2 = y1 + h_overlay

    if x1 < 0:
        img_overlay = img_overlay[:, -x1:]
        x1 = 0
    if y1 < 0:
        img_overlay = img_overlay[-y1:, :]
        y1 = 0
    if x2 > w_bg:
        img_overlay = img_overlay[:, :-(x2 - w_bg)]
        x2 = w_bg
    if y2 > h_bg:
        img_overlay = img_overlay[:-(y2 - h_bg), :]
        y2 = h_bg

    if img_overlay.shape[2] == 4:
        alpha = img_overlay[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha * img_overlay[:, :, c] +
                                    alpha_inv * img[y1:y2, x1:x2, c])
    else:
        img[y1:y2, x1:x2] = img_overlay

# --- 移動平均クラス ---
class MovingAverageFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer_x = deque(maxlen=window_size)
        self.buffer_y = deque(maxlen=window_size)

    def update(self, x, y):
        self.buffer_x.append(x)
        self.buffer_y.append(y)
        if not self.buffer_x: return x, y
        return int(sum(self.buffer_x) / len(self.buffer_x)), int(sum(self.buffer_y) / len(self.buffer_y))
    
    def reset(self):
        """手が検出されなくなった時にバッファをクリアする"""
        self.buffer_x.clear()
        self.buffer_y.clear()

# --- メイン処理 ---
def main():
    try:
        fish_img_orig = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        if fish_img_orig is None:
            raise FileNotFoundError
        new_h, new_w = int(fish_img_orig.shape[0] * FISH_SCALE), int(fish_img_orig.shape[1] * FISH_SCALE)
        fish_img_orig = cv2.resize(fish_img_orig, (new_w, new_h))
    except FileNotFoundError:
        print(f"エラー: 画像ファイル '{IMAGE_PATH}' が見つかりません。")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    
    cap = cv2.VideoCapture(0)
    
    # 平滑化フィルタ
    smoother = MovingAverageFilter(WINDOW_SIZE)
    
    # 遅延用キュー (平滑化済みの座標を保存)
    delay_queue = deque(maxlen=BUFFER_SIZE)

    # 向き判定用の前回座標
    prev_raw_x = 0
    prev_delayed_x = 0
    
    raw_facing_right = True
    delayed_facing_right = True

    print(f"開始します。設定: 移動平均W={WINDOW_SIZE}, 遅延={DELAY_SECONDS}秒")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        top_view = frame.copy()
        bottom_view = frame.copy()

        # テキスト表示
        cv2.putText(top_view, "Real-time (Raw)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(bottom_view, f"Delayed ({DELAY_SECONDS}s) + Smooth (W={WINDOW_SIZE})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 現在の「平滑化済み」座標を計算
        smoothed_pos = None
        raw_pos = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark[9]
                cx, cy = int(lm.x * w), int(lm.y * h)
                raw_pos = (cx, cy)

                # --- 1. 上画面 (Raw リアルタイム) ---
                if abs(cx - prev_raw_x) > 2:
                    raw_facing_right = cx > prev_raw_x
                prev_raw_x = cx
                
                overlay_image_alpha(top_view, fish_img_orig, (cx, cy), raw_facing_right)
                cv2.circle(top_view, (cx, cy), 5, (0, 0, 255), -1)

                # 平滑化を計算
                sx, sy = smoother.update(cx, cy)
                smoothed_pos = (sx, sy)
        else:
            # 手が見つからない時はフィルタをリセット
            smoother.reset()

        # --- 遅延処理 ---
        # 「平滑化された座標 (またはNone)」をキューに追加
        delay_queue.append(smoothed_pos)

        # 1秒前のデータを取り出す
        # queue[0] は常に一番古いデータ
        delayed_smoothed_pos = delay_queue[0]

        # --- 2. 下画面 (遅延 + 平滑化) ---
        if delayed_smoothed_pos is not None:
            dsx, dsy = delayed_smoothed_pos
            
            # 向き判定
            if abs(dsx - prev_delayed_x) > 1: # 平滑化されているので閾値は小さめでOK
                delayed_facing_right = dsx > prev_delayed_x
            prev_delayed_x = dsx
            
            overlay_image_alpha(bottom_view, fish_img_orig, (dsx, dsy), delayed_facing_right)
            cv2.circle(bottom_view, (dsx, dsy), 5, (255, 0, 0), -1)

        # 結合して表示
        combined_display = cv2.vconcat([top_view, bottom_view])
        
        # 画面サイズ調整
        display_h, display_w = combined_display.shape[:2]
        if display_h > 1000:
            scale = 1000 / display_h
            combined_display = cv2.resize(combined_display, (int(display_w * scale), int(display_h * scale)))

        cv2.imshow('Fish System', combined_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()