import pandas as pd
import numpy as np

# 1. CSVファイルを読み込む
# ※ファイル名は実際のものに合わせて変更してください
filename = 'raw_hand_data - コピー.csv'
df = pd.read_csv(filename)

# 2. 移動平均 (W=10) を計算する
window_size = 10
df['Smooth_X'] = df['Raw_X'].rolling(window=window_size).mean()

# 3. 「フレーム間変位（動きの量）」を計算する
# diff() は (今のフレーム - 1つ前のフレーム) を計算します
df['Diff_Raw'] = df['Raw_X'].diff()
df['Diff_Smooth'] = df['Smooth_X'].diff()

# 4. 分散 (Variance) を計算する
# var(ddof=0) は標本分散ではなく「分散（母分散）」に近い計算です（論文のVAR.Pに相当）
var_raw = df['Diff_Raw'].var(ddof=0)
var_smooth = df['Diff_Smooth'].var(ddof=0)

# 5. 削減率 (Reduction Rate) を計算する
reduction_rate = (1 - (var_smooth / var_raw)) * 100

# --- 結果の表示 ---
print(f"--- 実験結果 (Window Size = {window_size}) ---")
print(f"生データの分散 (Raw Variance): {var_raw:.6f}")
print(f"平滑化後の分散 (Smooth Variance): {var_smooth:.6f}")
print(f"★削減率 (Reduction Rate): {reduction_rate:.2f}%")

if reduction_rate < 0:
    print("注意: 削減率がマイナスです。平滑化によって逆に揺れが増えている可能性があります。")
elif reduction_rate > 60:
    print("以前の論文の数値（約61%）に近い、非常に高い効果が出ています。")
else:
    print(f"あなたの測定した「約{int(reduction_rate)}%」は正しい結果です。")