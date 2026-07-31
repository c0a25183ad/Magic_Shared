# -*- coding: utf-8 -*-
"""
オープンキャンパス展示用 手追従フィッシュデモ
=================================================
trial3.2.py をベースに、展示現場で「設定いらず・落ちにくい・フルスクリーン」で
動くように作り直したもの。

■ 遊び方
  - カメラに手をかざして「パー（手を開く）」と魚が出現し、少し遅れて手を追いかけます。
  - 画面左のサムネイルにしばらく手（中指の先端）を重ねると魚の見た目が変わります。

■ 操作キー
  - F : フルスクリーン / ウィンドウ 切り替え
  - H : 手の骨格表示の ON / OFF
  - Q または ESC : 終了

■ trial3.2 からの修正点（既存の弱点対策）
  1. 絶対パスのハードコードを廃止 → このファイルと同じ場所を基準に自動設定
  2. 魚の画像が1枚も無くても動く → 無ければカラフルな魚を自動生成
  3. カメラが開けない / 途切れる場合に落ちない → 複数カメラ番号を試行＆自動再接続
  4. モニターにフルスクリーン表示（アスペクト比を保ったまま拡大）
  5. 日本語表示（Pillowがあれば）。無ければ英語表示に自動で切り替え
"""

import os
import sys
import time
from collections import deque

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    print("mediapipe が見つかりません。`pip install mediapipe` を実行してください。")
    sys.exit(1)

# --- 任意依存：日本語表示用の Pillow（無くても動く）---
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# --- 任意依存：画面解像度取得用の tkinter（無くてもフォールバックする）---
try:
    import tkinter as _tk
    _TK_OK = True
except Exception:
    _TK_OK = False


# ======================================================================
# 基本設定
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FISH_DIR = os.path.join(BASE_DIR, "fish_designs")

CAM_WIDTH, CAM_HEIGHT = 1280, 720   # カメラ取得解像度（この座標系で全処理を行う）
FISH_LAG_SECONDS = 0.7              # 魚が追従する遅延（秒）
MOVING_AVERAGE_WINDOW = 10           # 手座標の移動平均フレーム数（大きいほど滑らか）
DWELL_TIME = 1.5                    # サムネイルにこの秒数重ねると選択
PAPER_THRESHOLD = 0.08            # 親指-小指距離がこれを超えたら「パー」とみなす
MAX_HANDS =5                    # 同時に追従させる手の数（2 なら魚も2匹）

WINDOW_NAME = "Fish Demo"

# MediaPipe Hands 初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ======================================================================
# 日本語テキスト描画（Pillow があれば日本語、無ければ英語にフォールバック）
# ======================================================================
def _find_jp_font():
    """システムから日本語フォントを探す。見つからなければ None。"""
    candidates = [
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\YuGothM.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


_JP_FONT_PATH = _find_jp_font() if _PIL_OK else None
_USE_JP = _PIL_OK and (_JP_FONT_PATH is not None)
_font_cache = {}

# 日本語が使えない環境向けの英語対訳
_EN_FALLBACK = {
    "手を開くと魚が出現します": "Open your hand to summon a fish!",
    "サムネイルに手を重ねて魚を変更": "Hover a thumbnail to change the fish",
    "F:全画面  H:骨格  Q:終了": "F: Fullscreen   H: Skeleton   Q: Quit",
    "カメラを起動できませんでした": "Could not open camera",
}


def _get_font(size):
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(_JP_FONT_PATH, size)
        except Exception:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def draw_text(frame, text, org, size=28, color=(255, 255, 255), outline=True):
    """frame（BGR）に text を描画する。日本語対応。org は左上座標。"""
    if _USE_JP:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = _get_font(size)
        x, y = org
        rgb = (color[2], color[1], color[0])  # BGR -> RGB
        if outline:
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=rgb)
        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        # 英語フォールバック
        text = _EN_FALLBACK.get(text, text)
        scale = size / 30.0
        x, y = org[0], org[1] + size  # putText はベースライン基準なので下げる
        if outline:
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2, cv2.LINE_AA)


# ======================================================================
# 画像ユーティリティ
# ======================================================================
def make_fish_image(body_color, w=240, h=150):
    """透過PNG相当（BGRA）の魚画像をコードで生成する。右向き。"""
    img = np.zeros((h, w, 4), np.uint8)
    b, g, r = body_color
    col = (b, g, r, 255)
    cx, cy = int(w * 0.56), h // 2

    # 尾びれ
    tail = np.array([[int(w * 0.22), cy],
                     [int(w * 0.05), cy - int(h * 0.24)],
                     [int(w * 0.05), cy + int(h * 0.24)]], np.int32)
    cv2.fillPoly(img, [tail], col)
    # 胴体
    cv2.ellipse(img, (cx, cy), (int(w * 0.33), int(h * 0.30)), 0, 0, 360, col, -1)
    # 背びれ
    fin = np.array([[cx - 15, cy - int(h * 0.27)],
                    [cx + 25, cy - int(h * 0.48)],
                    [cx + 35, cy - int(h * 0.25)]], np.int32)
    cv2.fillPoly(img, [fin], col)
    # 目
    ex, ey = int(w * 0.74), cy - int(h * 0.05)
    cv2.circle(img, (ex, ey), int(h * 0.09), (255, 255, 255, 255), -1)
    cv2.circle(img, (ex, ey), int(h * 0.045), (0, 0, 0, 255), -1)
    return img


def load_fish_designs():
    """
    fish_designs フォルダの PNG を読み込む。
    フォルダや画像が無ければ自動生成した魚を使う（＝アセット0本でも動く）。
    戻り値: [(name, BGRA画像), ...]
    """
    designs = []
    if os.path.isdir(FISH_DIR):
        for fn in sorted(os.listdir(FISH_DIR)):
            if fn.lower().endswith(".png"):
                img = cv2.imread(os.path.join(FISH_DIR, fn), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.shape[2] == 3:  # アルファが無ければ付与
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    designs.append((os.path.splitext(fn)[0], img))

    if not designs:
        # 自動生成にフォールバック
        os.makedirs(FISH_DIR, exist_ok=True)
        palette = [
            ("orange", (0, 140, 255)),
            ("blue",   (230, 160, 60)),
            ("green",  (80, 200, 80)),
            ("pink",   (170, 120, 255)),
        ]
        for name, color in palette:
            fish = make_fish_image(color)
            designs.append((name, fish))
            # 次回以降フォルダから読めるよう保存も試みる（失敗しても無視）
            try:
                cv2.imwrite(os.path.join(FISH_DIR, f"auto_{name}.png"), fish)
            except Exception:
                pass
        print("魚の画像が無かったため、自動生成した魚を使用します。")
        print(f"独自の魚を使うには {FISH_DIR} に透過PNGを置いてください。")

    return designs


def resize_to_width(image, target_width):
    if image is None or image.shape[1] == 0:
        return image
    scale = target_width / image.shape[1]
    return cv2.resize(image, (target_width, max(1, int(image.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)


def alpha_overlay(background, overlay, x, y):
    """透過画像 overlay を background の (x, y) を中心に合成する。"""
    h, w = overlay.shape[:2]
    x -= w // 2
    y -= h // 2
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(background.shape[1], x + w), min(background.shape[0], y + h)
    ox1, oy1 = max(0, -x), max(0, -y)
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    if x2 <= x1 or y2 <= y1:
        return
    if overlay.shape[2] == 4:
        alpha = overlay[oy1:oy2, ox1:ox2, 3] / 255.0
        inv = 1.0 - alpha
        for c in range(3):
            background[y1:y2, x1:x2, c] = (
                alpha * overlay[oy1:oy2, ox1:ox2, c] + inv * background[y1:y2, x1:x2, c]
            )
    else:
        background[y1:y2, x1:x2] = overlay[oy1:oy2, ox1:ox2]


# ======================================================================
# 手の検出・平滑化
# ======================================================================
def extract_hands(frame):
    """フレームから手のランドマークを抽出。[(landmarks, handedness), ...] を返す。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)
    out = []
    if results.multi_hand_landmarks:
        for idx, lm in enumerate(results.multi_hand_landmarks):
            handed = "Right"
            if results.multi_handedness and idx < len(results.multi_handedness):
                handed = results.multi_handedness[idx].classification[0].label
            out.append((lm, handed))
    return out


def moving_average(history, current_landmarks):
    """ランドマーク履歴に移動平均を適用し、(x, y, z) のリストを返す。"""
    frames = list(history) + [current_landmarks]
    n = len(current_landmarks.landmark)
    smoothed = []
    for i in range(n):
        xs = [f.landmark[i].x for f in frames]
        ys = [f.landmark[i].y for f in frames]
        zs = [f.landmark[i].z for f in frames]
        smoothed.append((sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)))
    return smoothed


# ======================================================================
# フルスクリーン表示
# ======================================================================
def get_screen_size():
    if _TK_OK:
        try:
            root = _tk.Tk()
            root.withdraw()
            w = root.winfo_screenwidth()
            h = root.winfo_screenheight()
            root.destroy()
            return w, h
        except Exception:
            pass
    return None  # 取得不可 → OpenCV の自動拡大に任せる


def letterbox(frame, screen_w, screen_h):
    """アスペクト比を保ったまま画面いっぱいに配置（余白は黒）。"""
    fh, fw = frame.shape[:2]
    scale = min(screen_w / fw, screen_h / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((screen_h, screen_w, 3), np.uint8)
    ox, oy = (screen_w - nw) // 2, (screen_h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


# ======================================================================
# カメラ（複数番号を試行して開く）
# ======================================================================
def open_camera():
    for index in (0, 1, 2):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            print(f"カメラ {index} を使用します。")
            return cap
        cap.release()
    return None


# ======================================================================
# メイン
# ======================================================================
def main():
    designs = load_fish_designs()
    current_fish_index = 0

    cap = open_camera()
    if cap is None:
        print("カメラを起動できませんでした。接続と権限（プライバシー設定）を確認してください。")
        return

    # ウィンドウ準備（初期はフルスクリーン）
    screen = get_screen_size()
    is_fullscreen = True
    show_skeleton = True
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 手ごとの状態
    landmark_histories = {}   # hand_id -> deque(landmarks)
    position_histories = {}   # hand_id -> deque((t, x, y))
    fish_positions = {}       # hand_id -> (x, y)
    fish_prev_x = {}          # hand_id -> 直前x（向き判定用）

    # Dwell（ホバー選択）用
    hover_index = None
    hover_start = 0.0

    # FPS計測
    prev_t = time.time()
    fps = 0.0
    read_fail = 0

    print("開始しました。ウィンドウ上で F=全画面切替 / H=骨格 / Q=終了")

    while True:
        ret, frame = cap.read()
        if not ret:
            # 読み取り失敗 → 数回まで再接続を試みる
            read_fail += 1
            print(f"フレーム取得に失敗しました（{read_fail}回目）。再接続を試みます...")
            cap.release()
            time.sleep(0.3)
            cap = open_camera()
            if cap is None or read_fail > 10:
                print("カメラを回復できませんでした。終了します。")
                break
            continue
        read_fail = 0

        frame = cv2.flip(frame, 1)  # 鏡表示
        h, w = frame.shape[:2]
        now = time.time()

        # --- 魚サムネイル（左側UI）を描画 ---
        ui_boxes = []
        thumb_w = max(90, w // 12)
        pad, x0, y0 = 12, 15, 15
        for i, (name, img) in enumerate(designs):
            thumb = resize_to_width(img, thumb_w)
            th, tw = thumb.shape[:2]
            bx, by = x0, y0 + i * (thumb_w + pad)  # 正方形前提でwを間隔に使用
            by = y0 + i * (thumb.shape[0] + pad)
            alpha_overlay(frame, thumb, bx + tw // 2, by + th // 2)
            selected = (i == current_fish_index)
            cv2.rectangle(frame, (bx, by), (bx + tw, by + th),
                          (0, 165, 255) if selected else (255, 255, 255),
                          4 if selected else 2)
            ui_boxes.append((bx, by, tw, th))

        # --- 手の検出 ---
        hands_data = extract_hands(frame)
        active_ids = set()
        cursor_pos = None

        for idx, (raw_lm, handed) in enumerate(hands_data):
            hand_id = f"{handed}_{idx}"
            active_ids.add(hand_id)

            if hand_id not in landmark_histories:
                landmark_histories[hand_id] = deque(maxlen=MOVING_AVERAGE_WINDOW)
            smoothed = moving_average(landmark_histories[hand_id], raw_lm)
            landmark_histories[hand_id].append(raw_lm)

            if show_skeleton:
                mp_drawing.draw_landmarks(frame, raw_lm, mp_hands.HAND_CONNECTIONS)

            # 先頭の手をカーソルとして使う（中指先端）
            if idx == 0:
                cx, cy = smoothed[mp_hands.HandLandmark.MIDDLE_FINGER_TIP][:2]
                cursor_pos = (int(cx * w), int(cy * h))

            # パー判定（親指先端 - 小指先端の距離）
            thumb = smoothed[mp_hands.HandLandmark.THUMB_TIP]
            pinky = smoothed[mp_hands.HandLandmark.PINKY_TIP]
            dist = np.hypot(thumb[0] - pinky[0], thumb[1] - pinky[1])
            is_paper = dist > PAPER_THRESHOLD

            if is_paper:
                pos = smoothed[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                px, py = int(pos[0] * w), int(pos[1] * h)
                if hand_id not in position_histories:
                    position_histories[hand_id] = deque()
                position_histories[hand_id].append((now, px, py))
                # 遅延ぶんより古い履歴を捨てる
                while (position_histories[hand_id] and
                       position_histories[hand_id][0][0] < now - FISH_LAG_SECONDS):
                    position_histories[hand_id].popleft()
                if position_histories[hand_id]:
                    _, fx, fy = position_histories[hand_id][0]
                    fish_positions[hand_id] = (fx, fy)
            else:
                fish_positions[hand_id] = None
                if hand_id in position_histories:
                    position_histories[hand_id].clear()

        # --- Dwell（ホバー2秒で魚を選択）---
        hovering = None
        if cursor_pos is not None:
            for i, (bx, by, bw, bh) in enumerate(ui_boxes):
                if bx <= cursor_pos[0] < bx + bw and by <= cursor_pos[1] < by + bh:
                    hovering = i
                    break
        if hovering is not None and hovering != hover_index:
            hover_index, hover_start = hovering, now
        elif hovering is None:
            hover_index = None
        if hover_index is not None:
            progress = min((now - hover_start) / DWELL_TIME, 1.0)
            bx, by, bw, bh = ui_boxes[hover_index]
            center = (bx + bw // 2, by + bh // 2)
            radius = int(bh / 3)
            cv2.ellipse(frame, center, (radius, radius), 270, 0, int(360 * progress),
                        (0, 255, 255), 5)
            if progress >= 1.0:
                current_fish_index = hover_index
                hover_index = None

        # --- 魚の描画 ---
        fish_img = designs[current_fish_index][1]
        drawn = resize_to_width(fish_img, w // 6)
        for hand_id, pos in fish_positions.items():
            if pos is None:
                continue
            fx, fy = pos
            # 進行方向で左右反転（右向き素材なので、左へ動くとき反転）
            prev = fish_prev_x.get(hand_id, fx)
            facing_left = fx < prev - 2
            fish_prev_x[hand_id] = fx
            sprite = cv2.flip(drawn, 1) if facing_left else drawn
            alpha_overlay(frame, sprite, fx, fy)

        # 消えた手の状態を掃除
        for hid in list(fish_positions.keys()):
            if hid not in active_ids:
                fish_positions[hid] = None

        # --- 説明テキスト・タイトル ---
        draw_text(frame, "手を開くと魚が出現します", (x0 + thumb_w + 30, 20),
                  size=34, color=(0, 255, 255))
        draw_text(frame, "サムネイルに手を重ねて魚を変更", (x0 + thumb_w + 30, 64),
                  size=24, color=(255, 255, 255))
        draw_text(frame, "F:全画面  H:骨格  Q:終了", (20, h - 44),
                  size=22, color=(200, 200, 200))

        # FPS
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, now - prev_t))
        prev_t = now
        cv2.putText(frame, f"FPS: {fps:4.1f}", (w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # --- 表示（フルスクリーンならレターボックス）---
        if is_fullscreen and screen is not None:
            display = letterbox(frame, screen[0], screen[1])
        else:
            display = frame
        cv2.imshow(WINDOW_NAME, display)

        # --- キー入力 ---
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord('h'):
            show_skeleton = not show_skeleton

        # ウィンドウの×ボタンで閉じられた場合の対策
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("終了しました。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断しました。")
    except Exception as e:
        print(f"予期しないエラー: {e}")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
