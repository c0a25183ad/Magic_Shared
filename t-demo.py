import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- 設定 ---
WINDOW_SIZE = 10      # 平滑化の強さ (W=10)
IMAGE_PATH = "../UploadedImage5.png" # 読み込む画像ファイル名
FISH_SCALE = 0.2      # 画像の大きさ倍率 (0.5なら半分、1.0ならそのまま)
BG_COLOR = (240, 255, 255) # 背景色 (薄い水色)

# --- 画像を透過合成する関数 ---
def overlay_image_alpha(img, img_overlay, pos, is_facing_right):
    """
    img: 背景画像 (描画先のキャンバス)
    img_overlay: 重ねる画像 (RGBA, 透明度付き)
    pos: 重ねる位置の中心座標 (x, y)
    is_facing_right: 右向きかどうか
    """
    x, y = pos
    
    # 画像の向き調整（元画像が「左向き」だと仮定した場合の処理例）
    # ※もし元画像が右向きなら、not is_facing_right に変えてください
    if not is_facing_right:
        img_overlay = cv2.flip(img_overlay, 1) # 左右反転

    # 画像サイズ
    h_overlay, w_overlay = img_overlay.shape[:2]
    h_bg, w_bg = img.shape[:2]

    # 左上の座標を計算（中心を合わせる）
    x1 = x - w_overlay // 2
    y1 = y - h_overlay // 2
    x2 = x1 + w_overlay
    y2 = y1 + h_overlay

    # 画面外にはみ出さないようにクリッピング
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

    # 重ね合わせ処理 (アルファブレンド)
    # img_overlayが透明度情報(4チャンネル目)を持っているか確認
    if img_overlay.shape[2] == 4:
        alpha = img_overlay[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        
        # BGRの各チャンネルについて合成
        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha * img_overlay[:, :, c] +
                                    alpha_inv * img[y1:y2, x1:x2, c])
    else:
        # 透明度がない画像の場合はそのまま上書き
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

# --- メイン処理 ---
def main():
    # 画像の読み込み
    try:
        # IMREAD_UNCHANGED でアルファチャンネル(透明度)も含めて読み込む
        fish_img_orig = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        if fish_img_orig is None:
            raise FileNotFoundError
        
        # 指定した倍率にリサイズ
        new_h, new_w = int(fish_img_orig.shape[0] * FISH_SCALE), int(fish_img_orig.shape[1] * FISH_SCALE)
        fish_img_orig = cv2.resize(fish_img_orig, (new_w, new_h))
        
    except FileNotFoundError:
        print(f"エラー: 画像ファイル '{IMAGE_PATH}' が見つかりません。")
        print("同じフォルダに画像を置くか、IMAGE_PATHを修正してください。")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    
    cap = cv2.VideoCapture(0) # 映らない場合は 1 に変更
    smoother = MovingAverageFilter(WINDOW_SIZE)
    
    prev_raw_x, prev_smooth_x = 0, 0
    raw_facing_right, smooth_facing_right = True, True

    print(f"'{IMAGE_PATH}' を使用して開始します。終了は 'q' キー。")

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
        
        cv2.line(display_img, (w, 0), (w, h), (0, 0, 0), 2)
        cv2.putText(display_img, "Raw Data (W=0)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display_img, f"Proposed (W={WINDOW_SIZE})", (w + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark[9] # 中指の付け根
                cx, cy = int(lm.x * w), int(lm.y * h)

                # --- 1. Raw Data (左) ---
                if abs(cx - prev_raw_x) > 2:
                    raw_facing_right = cx > prev_raw_x
                prev_raw_x = cx
                
                # 画像を描画
                overlay_image_alpha(display_img, fish_img_orig, (cx, cy), raw_facing_right)

                # --- 2. Smoothed Data (右) ---
                sx, sy = smoother.update(cx, cy)
                
                if abs(sx - prev_smooth_x) > 1:
                    smooth_facing_right = sx > prev_smooth_x
                prev_smooth_x = sx
                
                # 画像を描画 (右画面用にX座標+w)
                overlay_image_alpha(display_img, fish_img_orig, (sx + w, sy), smooth_facing_right)

        cv2.imshow('Fish Motion System Demo (Image)', display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()