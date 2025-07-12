import os

# 現在のフォルダの内容を確認
current_dir = r"c:\Users\ionna\Downloads\Magic_Shared"
print(f"フォルダ: {current_dir}")
print("=" * 50)

# ファイル一覧を取得
try:
    files = os.listdir(current_dir)
    
    # 機械学習関連のファイルを分類
    ml_files = {
        'テンプレートファイル': [],
        'CSVデータファイル': [],
        'サンプル画像': [],
        'その他のファイル': []
    }
    
    for file in files:
        if file.endswith('_template.npy'):
            ml_files['テンプレートファイル'].append(file)
        elif file.endswith('.csv'):
            ml_files['CSVデータファイル'].append(file)
        elif file.endswith(('.jpg', '.jpeg', '.png')) and any(gesture in file.lower() for gesture in ['rock', 'scissors', 'paper']):
            ml_files['サンプル画像'].append(file)
        else:
            ml_files['その他のファイル'].append(file)
    
    # 結果を表示
    for category, file_list in ml_files.items():
        print(f"\n【{category}】")
        if file_list:
            for file in file_list:
                print(f"  ✓ {file}")
        else:
            print(f"  ✗ 見つかりませんでした")
    
except FileNotFoundError:
    print("フォルダが見つかりません")
except PermissionError:
    print("フォルダへのアクセス権限がありません")