import pandas as pd
import numpy as np
import os
from datetime import datetime

def analyze_data_quality():
    """データ品質を詳細分析"""
    print("=== データ品質詳細分析 ===")
    
    # 有効なCSVファイルを特定
    valid_files = []
    
    # hand_landmarks.csv
    if os.path.exists("hand_landmarks.csv"):
        try:
            df = pd.read_csv("hand_landmarks.csv")
            if 'label' in df.columns:
                valid_files.append(("hand_landmarks.csv", df))
                print(f"✓ hand_landmarks.csv: {len(df)} サンプル")
            else:
                print("⚠️  hand_landmarks.csv: label列なし")
        except Exception as e:
            print(f"✗ hand_landmarks.csv: {e}")
    
    # hand_landmarks_20250712_184356.csv
    if os.path.exists("hand_landmarks_20250712_184356.csv"):
        try:
            df = pd.read_csv("hand_landmarks_20250712_184356.csv")
            if 'label' in df.columns:
                valid_files.append(("hand_landmarks_20250712_184356.csv", df))
                print(f"✓ hand_landmarks_20250712_184356.csv: {len(df)} サンプル")
        except Exception as e:
            print(f"✗ hand_landmarks_20250712_184356.csv: {e}")
    
    return valid_files

def create_balanced_dataset():
    """バランスの取れたデータセットを作成"""
    print("\n=== バランス調整済みデータセット作成 ===")
    
    valid_files = analyze_data_quality()
    
    if not valid_files:
        print("有効なCSVファイルが見つかりません")
        return
    
    # 全データを統合
    all_data = []
    for filename, df in valid_files:
        all_data.append(df)
        print(f"統合: {filename}")
    
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # 重複を除去
    original_size = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    print(f"重複除去: {original_size} → {len(merged_df)} サンプル")
    
    # 現在の分布を確認
    print("\n現在の分布:")
    current_distribution = merged_df['label'].value_counts()
    print(current_distribution)
    
    # 各ラベルを50サンプルずつに調整
    target_samples = 50
    balanced_data = []
    
    for label in ['rock', 'scissors', 'paper']:
        label_data = merged_df[merged_df['label'] == label]
        
        if len(label_data) >= target_samples:
            # 十分なデータがある場合はランダムサンプリング
            sampled_data = label_data.sample(n=target_samples, random_state=42)
            balanced_data.append(sampled_data)
            print(f"{label}: {len(label_data)} → {target_samples} サンプル")
        else:
            # データが不足している場合は全て使用
            balanced_data.append(label_data)
            print(f"{label}: {len(label_data)} サンプル (不足: {target_samples - len(label_data)})")
    
    # バランス調整済みデータを作成
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    
    # シャッフル
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 元のファイルをバックアップ
    if os.path.exists("hand_landmarks.csv"):
        backup_name = f"hand_landmarks_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.rename("hand_landmarks.csv", backup_name)
        print(f"バックアップ作成: {backup_name}")
    
    # 新しいファイルを保存
    balanced_df.to_csv("hand_landmarks.csv", index=False)
    
    print(f"\n=== 作成完了 ===")
    print(f"出力ファイル: hand_landmarks.csv")
    print(f"総サンプル数: {len(balanced_df)}")
    print("最終分布:")
    print(balanced_df['label'].value_counts())
    
    return balanced_df

def improve_existing_data():
    """既存データの品質を改善"""
    print("\n=== 既存データ品質改善 ===")
    
    # 新しいバランス調整済みデータでトレーニング用データを作成
    try:
        df = create_balanced_dataset()
        
        print("\n✅ データ品質改善完了")
        print("次のコマンドを実行してください:")
        print("python trial3.3.py")
        
        return True
        
    except Exception as e:
        print(f"エラー: {e}")
        return False

def main():
    """メイン処理"""
    print("データ品質向上ツール")
    print("=" * 50)
    
    success = improve_existing_data()
    
    if success:
        print("\n🎉 データ品質向上が完了しました！")
        print("trial3.3.py を実行して、改善された機械学習機能をテストしてください。")
    else:
        print("\n❌ データ品質向上に失敗しました。")

if __name__ == "__main__":
    main()