import os
import pandas as pd

def check_current_data():
    """現在の学習データの状況を確認"""
    print("=== 現在の学習データ状況 ===")
    
    # CSVファイルの確認
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    print(f"\n【CSVファイル】")
    
    total_samples = 0
    csv_data_details = {}
    
    if csv_files:
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                samples_count = len(df)
                total_samples += samples_count
                
                print(f"  ✓ {csv_file}: {samples_count} サンプル")
                
                if 'label' in df.columns:
                    label_distribution = df['label'].value_counts().to_dict()
                    print(f"    - データ分布: {label_distribution}")
                    csv_data_details[csv_file] = {
                        'samples': samples_count,
                        'distribution': label_distribution
                    }
                else:
                    print(f"    - ⚠️  'label'列が見つかりません")
                    
            except Exception as e:
                print(f"  ✗ {csv_file}: 読み込みエラー - {e}")
    else:
        print("  ✗ CSVファイルが見つかりません")
    
    # テンプレートファイルの確認
    templates = ['rock_template.npy', 'scissors_template.npy', 'paper_template.npy']
    print(f"\n【テンプレートファイル】")
    
    template_status = {}
    for template in templates:
        if os.path.exists(template):
            print(f"  ✓ {template}")
            template_status[template] = True
        else:
            print(f"  ✗ {template}")
            template_status[template] = False
    
    # サンプル画像の確認
    sample_images = []
    for gesture in ['rock', 'scissors', 'paper']:
        for i in range(5):  # 基本 + 4つの追加サンプル
            if i == 0:
                filename = f"{gesture}.jpg"
            else:
                filename = f"{gesture}_{i}.jpg"
            
            if os.path.exists(filename):
                sample_images.append(filename)
    
    print(f"\n【サンプル画像】")
    if sample_images:
        for img in sample_images:
            print(f"  ✓ {img}")
    else:
        print("  ✗ サンプル画像が見つかりません")
    
    # 学習データの品質分析
    print(f"\n=== 学習データ品質分析 ===")
    
    if total_samples > 0:
        print(f"総サンプル数: {total_samples}")
        
        # データバランスの確認
        if csv_data_details:
            all_labels = {}
            for file_data in csv_data_details.values():
                for label, count in file_data['distribution'].items():
                    all_labels[label] = all_labels.get(label, 0) + count
            
            print(f"全体のラベル分布: {all_labels}")
            
            # バランスチェック
            if all_labels:
                label_counts = list(all_labels.values())
                min_count = min(label_counts)
                max_count = max(label_counts)
                balance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                print(f"データバランス比: {balance_ratio:.2f}")
                if balance_ratio > 1.5:
                    print("⚠️  データの不均衡が検出されました")
                    min_label = min(all_labels, key=all_labels.get)
                    max_label = max(all_labels, key=all_labels.get)
                    print(f"   最少: {min_label} ({all_labels[min_label]})")
                    print(f"   最多: {max_label} ({all_labels[max_label]})")
                else:
                    print("✅ データバランスは良好です")
    
    # 推奨アクション
    print(f"\n=== 推奨アクション ===")
    
    if total_samples == 0:
        print("📊 学習データ収集を開始してください")
        print("   → python collect_training_data.py を実行")
        print("   → 目標: 各ジェスチャー50サンプル（合計150サンプル）")
        
    elif total_samples < 150:
        print(f"📈 より多くのデータが必要です（現在: {total_samples}, 目標: 150+）")
        print("   → python collect_training_data.py で追加データを収集")
        
        # 不足分の詳細
        if csv_data_details:
            all_labels = {}
            for file_data in csv_data_details.values():
                for label, count in file_data['distribution'].items():
                    all_labels[label] = all_labels.get(label, 0) + count
            
            target_per_class = 50
            print("   → 各ジェスチャーの不足分:")
            for gesture in ['rock', 'scissors', 'paper']:
                current_count = all_labels.get(gesture, 0)
                needed = max(0, target_per_class - current_count)
                if needed > 0:
                    print(f"     {gesture}: {needed} サンプル不足")
                else:
                    print(f"     {gesture}: ✅ 十分")
    
    elif total_samples < 300:
        print(f"📈 データは十分ですが、さらなる精度向上が可能です（現在: {total_samples}）")
        print("   → 追加データ収集で精度向上を図る")
        print("   → 異なる角度・照明条件でのデータ収集を推奨")
        
    else:
        print(f"✅ 十分なデータがあります（{total_samples} サンプル）")
        print("   → データ品質の向上や多様性の向上を検討")
    
    # 機械学習機能の利用可能性
    print(f"\n=== 機械学習機能の利用可能性 ===")
    
    svm_available = total_samples >= 30  # 最低限のSVM学習データ
    template_available = any(template_status.values())
    
    print(f"SVM学習: {'✅ 利用可能' if svm_available else '❌ データ不足'}")
    print(f"テンプレート学習: {'✅ 利用可能' if template_available else '❌ テンプレート不足'}")
    
    if not svm_available and not template_available:
        print("⚠️  機械学習機能が利用できません - ルールベース判定のみ")
    elif svm_available and template_available:
        print("🎯 全ての機械学習機能が利用可能です")
    
    # 次に実行すべきコマンド
    print(f"\n=== 次に実行すべきコマンド ===")
    
    if total_samples == 0:
        print("1. python collect_training_data.py  # データ収集開始")
        print("2. python trial3.3.py              # 動作確認")
    elif total_samples < 150:
        print("1. python collect_training_data.py  # 追加データ収集")
        print("2. python improve_data_quality.py  # データ品質向上")
        print("3. python trial3.3.py              # 改善後のテスト")
    else:
        print("1. python improve_data_quality.py  # データ品質向上")
        print("2. python trial3.3.py              # 高精度モードでテスト")
    
    return {
        'total_samples': total_samples,
        'csv_files': csv_data_details,
        'templates': template_status,
        'sample_images': sample_images,
        'svm_available': svm_available,
        'template_available': template_available
    }

def main():
    """メイン実行関数"""
    print("機械学習データ状況チェックツール")
    print("=" * 60)
    
    try:
        result = check_current_data()
        
        # 簡潔なサマリー
        print(f"\n" + "=" * 60)
        print("📊 サマリー")
        print("=" * 60)
        print(f"総学習データ: {result['total_samples']} サンプル")
        print(f"機械学習機能: {'フル機能' if result['svm_available'] and result['template_available'] else '制限あり'}")
        
        if result['total_samples'] < 150:
            print(f"🎯 推奨: データ収集を継続してください")
        else:
            print(f"🎯 推奨: 高精度モードでテストしてください")
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()