# Elliott Wave Sentiment Checker

エリオット波動理論に基づく高度な市場センチメント分析ツール

## 概要

このツールは、エリオット波動理論と複数のテクニカル指標を組み合わせて、市場の現在のステージを判定し、適切な投資アクションを提案します。S&P 500、NASDAQ、個別株などの分析に対応しています。

## 主な機能

- 🌊 **エリオット波動ステージ分析**: 9つの市場ステージを自動判定
- 📊 **複合指標分析**: STOCH RSI、HLT、RSI、Fear & Greed、VIXなど
- 📈 **視覚化**: 価格チャート、指標、出来高を統合表示
- 📝 **詳細レポート**: 現在の市場状況と推奨アクションを生成
- 🎯 **リアルタイムデータ**: Yahoo Financeから最新データを取得

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/elliott-wave-sentiment.git
cd elliott-wave-sentiment

# 必要なパッケージのインストール
pip install -r requirements.txt
```

## 使用方法

### 基本的な使用例

```python
from bot import AdvancedElliottSentimentChecker

# チェッカーのインスタンス作成
checker = AdvancedElliottSentimentChecker()

# S&P 500の分析
data = checker.fetch_market_data("^GSPC", period="6mo")
data_with_stages = checker.analyze_stage_history(data)
current_analysis = checker.analyze_stage(data_with_stages)

# レポート生成
report = checker.generate_detailed_report(data_with_stages, current_analysis)
print(report)

# チャート表示
checker.visualize_analysis(data_with_stages, "S&P 500")
```

### コマンドラインから実行

```bash
python bot.py
```

## エリオット波動ステージ詳細

### 📈 上昇相場のステージ

| ステージ | 名称 | 特徴 | リスクレベル |
|---------|------|------|-------------|
| **A** | 初動上昇（1波） | 底値圏での仕込みゾーン。STOCH RSI < 30 | 低 |
| **B** | 加速上昇（3波） | 最も収益性の高いトレンドゾーン | 低 |
| **C** | 調整（4波） | 一時的な押し目形成 | 中 |
| **D** | 過熱上昇（5波） | 高値圏での最終上昇 | 高 |
| **D-BC** | バイイングクライマックス | 天井形成、出来高急増 | 非常に高 |

### 📉 下落相場のステージ

| ステージ | 名称 | 特徴 | リスクレベル |
|---------|------|------|-------------|
| **E** | 調整A波 | 急落開始、トレンド転換 | 高 |
| **F** | 戻りB波 | ブルトラップ、一時的な反発 | 中 |
| **G** | 本格下落C波 | 主要な下落トレンド | 高 |
| **G-SC** | セリングクライマックス | 底値圏、反転の機会 | 機会 |

## 使用する指標

### 1. STOCH RSI（ストキャスティクスRSI）
- RSIのストキャスティクス版
- 買われ過ぎ・売られ過ぎを判定
- K線とD線のクロスでトレンド転換を検出

### 2. HLT（ハイローターゲット）
- 一定期間の高値・安値に対する現在価格の位置
- 0-100の値で相対的な価格位置を表示

### 3. Fear & Greed Index
- 市場センチメントの総合指標
- 価格モメンタム、VIX、出来高から計算

### 4. 出来高スパイク検出
- 20日移動平均の2倍以上で検出
- クライマックスの判定に使用

## チャートの見方

生成されるチャートは4つのパネルで構成：

1. **価格チャート**: 背景色でステージを表示
2. **STOCH RSI**: K線（青）とD線（赤）、過熱ゾーンを表示
3. **Fear & Greed + VIX**: 市場センチメントとボラティリティ
4. **出来高**: スパイク検出（赤色バー）

## 推奨アクション例

### ステージAの場合
- ✅ 段階的な買い増し検討
- 📊 週足STOCH RSIの底打ち確認
- 💡 リスク管理: ポジションサイズは控えめに

### ステージD-BCの場合
- 🚨 即座に大部分を利益確定
- 📊 上ヒゲ・出来高急増を確認
- 💡 逆張りショートの検討も可

## カスタマイズ

### 分析期間の変更
```python
# 1年間のデータで分析
data = checker.fetch_market_data("AAPL", period="1y")
```

### 指標パラメータの調整
```python
# STOCH RSI期間を変更
k, d = checker.calculate_stoch_rsi(prices, period=21, smooth_k=5, smooth_d=5)
```

## 注意事項

- このツールは投資判断の参考情報であり、投資助言ではありません
- 過去のパフォーマンスは将来の結果を保証しません
- 実際の投資判断は自己責任で行ってください
- VIXデータは米国市場のみ利用可能です

## トラブルシューティング

### よくある問題

1. **データ取得エラー**
   - インターネット接続を確認
   - Yahoo Financeのティッカーシンボルが正しいか確認

2. **日本語表示の問題**
   - DejaVu Sansフォントがインストールされているか確認
   - 必要に応じて`plt.rcParams['font.sans-serif']`を調整

3. **警告メッセージ**
   - warningsは意図的に無視しています（問題ありません）

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

プルリクエストを歓迎します！大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 作者

[Your Name]

## 更新履歴

- v1.0.0 (2024-01) - 初回リリース
  - エリオット波動9ステージの実装
  - 複合指標による高精度判定
  - 視覚化機能の追加