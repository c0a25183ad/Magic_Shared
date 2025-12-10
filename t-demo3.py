import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- 設定 ---
WINDOW_SIZE = 10      # 平滑化の強さ (W=10)
IMAGE_PATH = "../UploadedImage5.png" 
FISH_SCALE = 0.2      
BG_COLOR = (240, 255, 255) 

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
        self.buffer_x = deque(maxlen=window_size)
        self.buffer_y = deque(maxlen=window_size)

    def update(self, x, y):
        self.buffer_x.append(x)
        self.buffer_y.append(y)
        if not self.buffer_x: return x, y
        return int(sum(self.buffer_x) / len(self.buffer_x)), int(sum(self.buffer_y) / len(self.buffer_y))

# --- 欠損時の発光描画関数 ---
def draw_lost_effect(img, x_start, y_start, width, height, message):
    """
    指定された矩形領域だけを発光させ、警告テキストを表示する
    """
    # 1. 対象領域(ROI)を切り出し
    roi = img[y_start:y_start+height, x_start:x_start+width]
    
    # 2. 白い画像を作成してブレンド（発光表現）
    white_rect = np.zeros_like(roi)
    white_rect[:] = (255, 255, 255)
    # alpha=0.5 で元の色と白を半々に混ぜる
    blended = cv2.addWeighted(roi, 0.5, white_rect, 0.5, 0)
    
    # 3. 元の画像に書き戻す
    img[y_start:y_start+height, x_start:x_start+width] = blended

    # 4. テキスト表示（領域の中央に配置）
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
    
    # テキストの左上座標を計算（領域内での相対位置 + 開始位置）
    text_x = x_start + (width - text_size[0]) // 2
    text_y = y_start + (height + text_size[1]) // 2
    
    cv2.putText(img, message, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)


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
    smoother = MovingAverageFilter(WINDOW_SIZE)
    
    prev_raw_x, prev_smooth_x = 0, 0
    raw_facing_right, smooth_facing_right = True, True

    print(f"開始します。終了は 'q' キー。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)


        # 描画用キャンバス
        display_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
        display_img[:] = BG_COLOR
        
        # 中央の区切り線
        cv2.line(display_img, (w, 0), (w, h), (0, 0, 0), 2)

        # ラベル
        cv2.putText(display_img, "Raw Data", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display_img, f"Proposed (W={WINDOW_SIZE})", (w + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 0), 2)

        if results.multi_hand_landmarks:
            # --- 検出あり ---
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark[9]
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Raw (左側)
                if abs(cx - prev_raw_x) > 2:
                    raw_facing_right = cx > prev_raw_x
                prev_raw_x = cx
                overlay_image_alpha(display_img, fish_img_orig, (cx, cy), raw_facing_right)

                # Smooth (右側)
                sx, sy = smoother.update(cx, cy)
                if abs(sx - prev_smooth_x) > 1:
                    smooth_facing_right = sx > prev_smooth_x
                prev_smooth_x = sx
                overlay_image_alpha(display_img, fish_img_orig, (sx + w, sy), smooth_facing_right)
        
        else:
            # --- 検出なし（欠損） ---
            # 左半分 (Raw) の発光
            draw_lost_effect(display_img, 0, 0, w, h, "LOST (Raw)")
            
            # 右半分 (Smooth) の発光
            draw_lost_effect(display_img, w, 0, w, h, "LOST (Smooth)")

        cv2.imshow('Fish Motion System Demo (Image)', display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()