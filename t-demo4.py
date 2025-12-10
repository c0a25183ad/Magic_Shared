import cv2
import numpy as np
from collections import deque

def main():
    # --- 設定 ---
    # 移動平均の対象となるフレーム数（ウィンドウサイズ）
    # この値を大きくすると、より滑らかになりますが、残像感も強くなります。
    N_FRAMES = 10
    
    # 表示する各画面の幅（処理負荷軽減のためリサイズ）
    RESIZE_WIDTH = 640
    
    # カメラの初期化（引数はカメラのID。通常は0）
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")
        return

    # 直近のNフレームを保持するためのバッファ（キュー）
    frame_buffer = deque(maxlen=N_FRAMES)
    
    print("プログラムを開始します。終了するには映像ウィンドウで 'q' キーを押してください。")

    while True:
        # カメラから1フレーム読み込む
        ret, frame = cap.read()
        if not ret:
            print("フレームの読み込みに失敗しました。")
            break

        # 処理負荷軽減と表示サイズ統一のためにリサイズ
        height, width = frame.shape[:2]
        scale = RESIZE_WIDTH / width
        frame_resized = cv2.resize(frame, (int(width * scale), int(height * scale)))

        # --- 上段用画像（移動平均なし）の作成 ---
        top_display = frame_resized.copy()
        # テキストの描画
        cv2.putText(top_display, "Not moving average", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # --- 下段用画像（移動平均あり）の計算と作成 ---
        # 計算精度を保つため、float32型に変換してバッファに追加
        frame_float = frame_resized.astype(np.float32)
        frame_buffer.append(frame_float)

        # バッファ内の全フレームの平均を計算
        # axis=0 は時間方向（フレーム方向）の平均を意味します
        avg_frame_float = np.mean(frame_buffer, axis=0)
        
        # 表示用にuint8型（0-255の整数）に戻す
        bottom_display = avg_frame_float.astype(np.uint8)
        
        # テキストの描画
        cv2.putText(bottom_display, "moving average (N={})".format(N_FRAMES), (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # --- 上下2画面の結合と表示 ---
        # 垂直方向(vertical)に連結
        combined_display = cv2.vconcat([top_display, bottom_display])
        
        cv2.imshow('Comparison: Top=Raw, Bottom=Moving Average', combined_display)

        # 'q' キーが押されたらループを終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 後処理
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()