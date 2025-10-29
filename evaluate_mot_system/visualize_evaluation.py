"""
MOT評価結果を可視化するスクリプト
HTMLレポートとグラフ画像を生成
"""

import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 日本語フォントの設定（macOSの場合）
try:
    plt.rcParams['font.family'] = 'Hiragino Sans'  # macOS用
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'  # フォールバック

def load_evaluation_results(json_path):
    """評価結果JSONファイルを読み込む"""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_visualization_report(results, output_dir=None):
    """
    評価結果を可視化してHTMLレポートとグラフ画像を生成
    
    Args:
        results: 評価結果の辞書
        output_dir: 出力ディレクトリ（Noneの場合はJSONファイルと同じディレクトリ）
    """
    if output_dir is None:
        # JSONファイルと同じディレクトリを使用
        json_path = "video/evaluation_results.json"
        output_dir = os.path.dirname(json_path) if os.path.dirname(json_path) else "."
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. グラフ画像を生成
    plot_path = os.path.join(output_dir, f"evaluation_report_{timestamp}_plots.png")
    create_plots(results, plot_path)
    
    # 2. HTMLレポートを生成
    html_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.html")
    create_html_report(results, html_path, plot_path)
    
    print(f"\n✅ 評価レポートを生成しました:")
    print(f"   📊 グラフ画像: {plot_path}")
    print(f"   📄 HTMLレポート: {html_path}")
    
    return html_path, plot_path

def create_plots(results, output_path):
    """評価結果のグラフを作成"""
    fig = plt.figure(figsize=(16, 10))
    
    # カラーパレット
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'success': '#06A77D',
        'warning': '#F18F01',
        'error': '#C73E1D',
        'info': '#6C757D'
    }
    
    # 1. コアメトリクスのバーグラフ
    ax1 = plt.subplot(2, 3, 1)
    core_metrics = {
        'MOTA': results['MOTA'],
        'MOTP': results['MOTP'] / 100,  # ピクセル値を0-1スケールに変換
        'IDF1': results['IDF1']
    }
    bars1 = ax1.bar(range(len(core_metrics)), list(core_metrics.values()), 
                    color=[colors['primary'], colors['success'], colors['secondary']])
    ax1.set_xticks(range(len(core_metrics)))
    ax1.set_xticklabels(list(core_metrics.keys()))
    ax1.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax1.set_title('Core Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis='y', alpha=0.3)
    
    # 値のラベルを追加
    for i, (key, val) in enumerate(core_metrics.items()):
        label = f"{val:.4f}" if key != 'MOTP' else f"{results['MOTP']:.2f}px"
        ax1.text(i, val + 0.02, label, ha='center', va='bottom', fontweight='bold')
    
    # 2. 検出精度のパイチャート
    ax2 = plt.subplot(2, 3, 2)
    tp = results['True_Positives']
    fp = results['False_Positives']
    fn = results['False_Negatives']
    total_detections = tp + fp + fn
    
    if total_detections > 0:
        sizes = [tp, fp, fn]
        labels = ['True Positives', 'False Positives', 'False Negatives']
        colors_pie = [colors['success'], colors['warning'], colors['error']]
        explode = (0.05, 0.05, 0.05)
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie,
                                           autopct='%1.1f%%', startangle=90, explode=explode,
                                           textprops={'fontsize': 9})
        ax2.set_title('Detection Accuracy', fontsize=12, fontweight='bold')
        
        # 数値も表示
        for i, (wedge, size) in enumerate(zip(wedges, sizes)):
            percentage = (size / total_detections) * 100
            autotexts[i].set_text(f'{size}\n({percentage:.1f}%)')
    
    # 3. トラッキング品質のバーグラフ
    ax3 = plt.subplot(2, 3, 3)
    tracking_quality = {
        'ID Switches': results['ID_Switches'],
        'Fragmentations': results['Fragmentations']
    }
    bars3 = ax3.bar(range(len(tracking_quality)), list(tracking_quality.values()),
                    color=[colors['warning'], colors['error']])
    ax3.set_xticks(range(len(tracking_quality)))
    ax3.set_xticklabels(list(tracking_quality.keys()), rotation=15, ha='right')
    ax3.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax3.set_title('Tracking Quality Issues', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 値のラベルを追加
    for i, (key, val) in enumerate(tracking_quality.items()):
        ax3.text(i, val + max(tracking_quality.values()) * 0.02, f"{int(val)}", 
                ha='center', va='bottom', fontweight='bold')
    
    # 4. 統計情報のテキスト表示
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    stats_text = f"""
    📊 Evaluation Statistics
    
    Total Frames: {results['Total_Frames']}
    Average Track Length: {results['Avg_Track_Length']:.1f} frames
    
    Detection Counts:
    • True Positives: {results['True_Positives']:,}
    • False Positives: {results['False_Positives']:,}
    • False Negatives: {results['False_Negatives']:,}
    
    Tracking Issues:
    • ID Switches: {results['ID_Switches']}
    • Fragmentations: {results['Fragmentations']}
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold')
    
    # 5. メトリクスのレーダーチャート風表示
    ax5 = plt.subplot(2, 3, 5)
    metrics_normalized = {
        'MOTA': results['MOTA'],
        'IDF1': results['IDF1'],
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'F1': 2 * (tp / (tp + fp) * tp / (tp + fn)) / (tp / (tp + fp) + tp / (tp + fn)) 
              if (tp + fp) > 0 and (tp + fn) > 0 else 0
    }
    
    categories = list(metrics_normalized.keys())
    values = list(metrics_normalized.values())
    
    # バーグラフとして表示
    bars5 = ax5.barh(categories, values, color=colors['primary'])
    ax5.set_xlim([0, 1.0])
    ax5.set_xlabel('Score (0-1)', fontsize=10, fontweight='bold')
    ax5.set_title('All Metrics Overview', fontsize=12, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    
    # 値のラベルを追加
    for i, (cat, val) in enumerate(zip(categories, values)):
        ax5.text(val + 0.02, i, f"{val:.4f}", va='center', fontweight='bold')
    
    # 6. 総合評価スコアのゲージ表示
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 総合スコアを計算（重み付け平均）
    overall_score = (results['MOTA'] * 0.4 + results['IDF1'] * 0.3 + 
                    (tp / (tp + fp) if (tp + fp) > 0 else 0) * 0.3)
    
    # 円形ゲージ風の表示
    theta = np.linspace(0, 2 * np.pi, 100)
    r = 0.8
    
    # 背景円
    ax6.plot(theta, [r] * len(theta), 'k-', linewidth=20, alpha=0.2)
    
    # スコア円
    score_theta = np.linspace(0, 2 * np.pi * overall_score, int(100 * overall_score))
    color = colors['success'] if overall_score > 0.8 else colors['warning'] if overall_score > 0.6 else colors['error']
    ax6.plot(score_theta, [r] * len(score_theta), color=color, linewidth=20)
    
    # スコアテキスト
    ax6.text(0, 0, f"{overall_score:.2%}", ha='center', va='center',
            fontsize=24, fontweight='bold')
    ax6.text(0, -1.2, 'Overall Score', ha='center', va='center',
            fontsize=14, fontweight='bold')
    ax6.set_xlim([-1.5, 1.5])
    ax6.set_ylim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 グラフ画像を保存: {output_path}")

def create_html_report(results, output_path, plot_path):
    """HTMLレポートを作成"""
    plot_filename = os.path.basename(plot_path)
    
    # 計算値
    tp = results['True_Positives']
    fp = results['False_Positives']
    fn = results['False_Negatives']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    overall_score = (results['MOTA'] * 0.4 + results['IDF1'] * 0.3 + precision * 0.3)
    
    # スコアの評価
    def get_score_grade(score):
        if score >= 0.9:
            return ("優秀", "#28a745", "🟢")
        elif score >= 0.7:
            return ("良好", "#ffc107", "🟡")
        elif score >= 0.5:
            return ("普通", "#fd7e14", "🟠")
        else:
            return ("要改善", "#dc3545", "🔴")
    
    mota_grade, mota_color, mota_icon = get_score_grade(results['MOTA'])
    idf1_grade, idf1_color, idf1_icon = get_score_grade(results['IDF1'])
    overall_grade, overall_color, overall_icon = get_score_grade(overall_score)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MOT評価レポート</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Hiragino Sans', 'Meiryo', 'MS PGothic', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 40px;
        }}
        .score-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .score-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .score-card:hover {{
            transform: translateY(-5px);
        }}
        .score-card h3 {{
            font-size: 1.1em;
            color: #333;
            margin-bottom: 15px;
        }}
        .score-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }}
        .score-grade {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .metric-box {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid #667eea;
        }}
        .metric-box h4 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .plot-section {{
            text-align: center;
            margin: 40px 0;
            background: #f8f9fa;
            padding: 30px;
            border-radius: 10px;
        }}
        .plot-section img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
        }}
        .summary-table th,
        .summary-table td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        .summary-table th {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        .summary-table tr:hover {{
            background: #f5f5f5;
        }}
        .footer {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 MOT評価レポート</h1>
            <p>Multiple Object Tracking Evaluation Report</p>
            <p style="margin-top: 10px; font-size: 0.9em;">生成日時: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}</p>
        </div>
        
        <div class="content">
            <h2 style="margin-bottom: 20px; color: #2c3e50;">📊 総合評価スコア</h2>
            <div class="score-cards">
                <div class="score-card">
                    <h3>MOTA</h3>
                    <div class="score-value">{results['MOTA']:.4f}</div>
                    <span class="score-grade" style="background: {mota_color}; color: white;">
                        {mota_icon} {mota_grade}
                    </span>
                    <p class="metric-label">Multiple Object Tracking Accuracy</p>
                </div>
                <div class="score-card">
                    <h3>IDF1</h3>
                    <div class="score-value">{results['IDF1']:.4f}</div>
                    <span class="score-grade" style="background: {idf1_color}; color: white;">
                        {idf1_icon} {idf1_grade}
                    </span>
                    <p class="metric-label">ID F1 Score</p>
                </div>
                <div class="score-card">
                    <h3>総合スコア</h3>
                    <div class="score-value">{overall_score:.2%}</div>
                    <span class="score-grade" style="background: {overall_color}; color: white;">
                        {overall_icon} {overall_grade}
                    </span>
                    <p class="metric-label">Weighted Average Score</p>
                </div>
            </div>
            
            <h2 style="margin-bottom: 20px; color: #2c3e50; margin-top: 40px;">📈 詳細メトリクス</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <h4>MOTP (Position Error)</h4>
                    <div class="metric-value">{results['MOTP']:.2f} pixels</div>
                    <p class="metric-label">平均位置誤差（低いほど良い）</p>
                </div>
                <div class="metric-box">
                    <h4>Precision</h4>
                    <div class="metric-value">{precision:.4f}</div>
                    <p class="metric-label">検出精度: {tp:,} / ({tp:,} + {fp:,})</p>
                </div>
                <div class="metric-box">
                    <h4>Recall</h4>
                    <div class="metric-value">{recall:.4f}</div>
                    <p class="metric-label">検出再現率: {tp:,} / ({tp:,} + {fn:,})</p>
                </div>
                <div class="metric-box">
                    <h4>F1 Score</h4>
                    <div class="metric-value">{f1:.4f}</div>
                    <p class="metric-label">PrecisionとRecallの調和平均</p>
                </div>
                <div class="metric-box">
                    <h4>ID Switches</h4>
                    <div class="metric-value">{results['ID_Switches']}</div>
                    <p class="metric-label">ID切り替わり回数（低いほど良い）</p>
                </div>
                <div class="metric-box">
                    <h4>Fragmentations</h4>
                    <div class="metric-value">{results['Fragmentations']}</div>
                    <p class="metric-label">トラック断片化回数（低いほど良い）</p>
                </div>
            </div>
            
            <div class="plot-section">
                <h2 style="margin-bottom: 20px; color: #2c3e50;">📊 可視化グラフ</h2>
                <img src="{plot_filename}" alt="Evaluation Plots">
            </div>
            
            <h2 style="margin-bottom: 20px; color: #2c3e50; margin-top: 40px;">📋 統計サマリー</h2>
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>項目</th>
                        <th>値</th>
                        <th>説明</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Total Frames</strong></td>
                        <td>{results['Total_Frames']:,}</td>
                        <td>処理された総フレーム数</td>
                    </tr>
                    <tr>
                        <td><strong>True Positives</strong></td>
                        <td>{tp:,}</td>
                        <td>正しく検出・追跡できた数</td>
                    </tr>
                    <tr>
                        <td><strong>False Positives</strong></td>
                        <td>{fp:,}</td>
                        <td>誤検出数</td>
                    </tr>
                    <tr>
                        <td><strong>False Negatives</strong></td>
                        <td>{fn:,}</td>
                        <td>見逃し数</td>
                    </tr>
                    <tr>
                        <td><strong>Average Track Length</strong></td>
                        <td>{results['Avg_Track_Length']:.1f} frames</td>
                        <td>平均トラック長</td>
                    </tr>
                    <tr>
                        <td><strong>ID Switches Rate</strong></td>
                        <td>{(results['ID_Switches'] / results['Total_Frames'] * 100):.2f}%</td>
                        <td>フレームあたりのID切り替わり率</td>
                    </tr>
                    <tr>
                        <td><strong>Fragmentation Rate</strong></td>
                        <td>{(results['Fragmentations'] / results['Total_Frames'] * 100):.2f}%</td>
                        <td>フレームあたりの断片化率</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Generated by MOT Evaluation Visualization Tool</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 HTMLレポートを保存: {output_path}")

if __name__ == "__main__":
    import sys
    
    # JSONファイルのパスを指定
    json_path = "video/evaluation_results.json"
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    if not os.path.exists(json_path):
        print(f"❌ エラー: ファイルが見つかりません: {json_path}")
        sys.exit(1)
    
    print(f"📖 評価結果を読み込み中: {json_path}")
    results = load_evaluation_results(json_path)
    
    print(f"✅ 評価結果を読み込みました")
    print(f"   MOTA: {results['MOTA']:.4f}")
    print(f"   IDF1: {results['IDF1']:.4f}")
    print(f"   Total Frames: {results['Total_Frames']:,}")
    
    # 可視化レポートを生成
    html_path, plot_path = create_visualization_report(results)
    
    print(f"\n🎉 完了！ブラウザで以下を開いてください:")
    print(f"   {os.path.abspath(html_path)}")

