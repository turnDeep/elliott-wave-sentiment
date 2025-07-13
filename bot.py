import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Tuple, Optional, List
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントの設定

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

class AdvancedElliottSentimentChecker:
    """エリオット波動理論に基づく高度なセンチメントチェッカー"""
    def __init__(self):
        self.stages = {
            'A': {'name': '初動上昇（1波）', 'color': 'lightblue', 'risk': 'low'},
            'B': {'name': '加速上昇（3波）', 'color': 'green', 'risk': 'low'},
            'C': {'name': '調整（4波）', 'color': 'yellow', 'risk': 'medium'},
            'D': {'name': '過熱上昇（5波）', 'color': 'orange', 'risk': 'high'},
            'D-BC': {'name': 'バイイングクライマックス', 'color': 'red', 'risk': 'very_high'},
            'E': {'name': '調整A波', 'color': 'darkred', 'risk': 'high'},
            'F': {'name': '戻りB波', 'color': 'lightyellow', 'risk': 'medium'},
            'G': {'name': '本格下落C波', 'color': 'darkred', 'risk': 'high'},
            'G-SC': {'name': 'セリングクライマックス', 'color': 'darkblue', 'risk': 'opportunity'}
        }

    def fetch_market_data(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        """Yahoo Financeから市場データを取得"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            # VIXデータの取得（^VIX）
            if symbol.upper() != '^VIX':
                vix = yf.Ticker('^VIX').history(period=period)['Close']
                data['vix'] = vix.reindex(data.index, method='ffill')
            else:
                data['vix'] = data['Close']

            # Fear & Greedインデックスの簡易計算
            # （実際のF&Gは複雑な計算ですが、ここでは簡易版）
            data['fear_greed'] = self._calculate_fear_greed(data)

            return data
        except Exception as e:
            print(f"データ取得エラー: {e}")
            return None

    def _calculate_fear_greed(self, data: pd.DataFrame) -> pd.Series:
        """Fear & Greedインデックスの簡易計算"""
        # 複数の要素を組み合わせて計算
        # 1. 価格モメンタム（20日リターン）
        momentum = data['Close'].pct_change(20).fillna(0)
        momentum_score = (momentum + 0.1) / 0.2 * 100  # -10%～+10%を0～100にマッピング

        # 2. ボラティリティ（VIXの逆数）
        if 'vix' in data.columns:
            vix_score = 100 - (data['vix'] - 10) / 30 * 100  # VIX 10-40を100-0にマッピング
        else:
            vix_score = 50

        # 3. 出来高
        vol_ma = data['Volume'].rolling(20).mean()
        vol_score = (data['Volume'] / vol_ma - 0.5) / 1 * 100  # 0.5x-1.5xを0-100にマッピング

        # 総合スコア
        fg = (momentum_score * 0.4 + vix_score * 0.4 + vol_score * 0.2).clip(0, 100)

        return fg

    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """全ての指標を計算してデータフレームに追加"""
        # STOCH RSI
        k, d = self.calculate_stoch_rsi(data['Close'])
        data['stoch_rsi_k'] = k
        data['stoch_rsi_d'] = d

        # HLT
        data['hlt'] = self.calculate_hlt(data['High'], data['Low'], data['Close'])

        # 出来高スパイク
        data['volume_spike'] = self.detect_volume_spike(data['Volume'])

        # 移動平均
        data['sma_20'] = data['Close'].rolling(20).mean()
        data['sma_50'] = data['Close'].rolling(50).mean()

        # RSI
        data['rsi'] = self.calculate_rsi(data['Close'])

        return data

    def calculate_stoch_rsi(self, prices: pd.Series, period: int = 14,
                           smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """ストキャスティクスRSIを計算"""
        # RSI計算
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # ストキャスティクス計算
        rsi_min = rsi.rolling(window=period).min()
        rsi_max = rsi.rolling(window=period).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100

        # スムージング
        k = stoch_rsi.rolling(window=smooth_k).mean()
        d = k.rolling(window=smooth_d).mean()

        return k, d

    def calculate_hlt(self, high: pd.Series, low: pd.Series,
                     close: pd.Series, period: int = 20) -> pd.Series:
        """ハイローターゲット（HLT）を計算"""
        hh = high.rolling(window=period).max()
        ll = low.rolling(window=period).min()
        hlt = (close - ll) / (hh - ll) * 100
        return hlt

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSIを計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def detect_volume_spike(self, volume: pd.Series, threshold: float = 2.0) -> pd.Series:
        """出来高スパイクを検出"""
        vol_ma = volume.rolling(window=20).mean()
        vol_spike = volume > (vol_ma * threshold)
        return vol_spike

    def analyze_stage_history(self, data: pd.DataFrame) -> pd.DataFrame:
        """履歴データ全体のステージを分析"""
        # 全指標を計算
        data = self.calculate_all_indicators(data)

        # 各日のステージを判定
        stages = []
        confidences = []

        for i in range(len(data)):
            if i < 50:  # 最初の50日はデータ不足のためスキップ
                stages.append(None)
                confidences.append(0)
                continue

            # その日までのデータで分析
            current_data = data.iloc[:i+1]
            result = self.analyze_stage(current_data)
            stages.append(result['current_stage'])
            confidences.append(result['confidence'])

        data['stage'] = stages
        data['stage_confidence'] = confidences

        return data

    def analyze_stage(self, data: pd.DataFrame) -> Dict[str, any]:
        """現在のステージを分析（基本実装と同じ）"""
        results = {
            'current_stage': None,
            'stage_description': None,
            'indicators': {},
            'confidence': 0.0,
            'warnings': []
        }

        # 指標の計算（既に計算済みの場合はそれを使用）
        if 'stoch_rsi_k' not in data.columns:
            data = self.calculate_all_indicators(data)

        # 最新値を取得
        latest_stoch_k = data['stoch_rsi_k'].iloc[-1]
        latest_stoch_d = data['stoch_rsi_d'].iloc[-1]
        latest_hlt = data['hlt'].iloc[-1]
        latest_vol_spike = data['volume_spike'].iloc[-1]
        latest_rsi = data['rsi'].iloc[-1]

        # 週足STOCH RSI（簡易的に5日分を使用）
        weekly_stoch_k = data['stoch_rsi_k'].iloc[-5:].mean() if len(data) >= 5 else latest_stoch_k

        # FGとVIXの処理
        latest_fg = data['fear_greed'].iloc[-1] if 'fear_greed' in data else 50
        latest_vix = data['vix'].iloc[-1] if 'vix' in data else 15

        results['indicators'] = {
            'stoch_rsi_k': latest_stoch_k,
            'stoch_rsi_d': latest_stoch_d,
            'hlt': latest_hlt,
            'volume_spike': latest_vol_spike,
            'fear_greed': latest_fg,
            'vix': latest_vix,
            'weekly_stoch_rsi': weekly_stoch_k,
            'rsi': latest_rsi
        }

        # ステージ判定ロジック（基本実装と同じ）
        confidence_scores = {}

        # ステージA: 初動上昇（仕込みゾーン）
        if weekly_stoch_k < 30 and not latest_vol_spike:
            confidence_scores['A'] = 0.8
            if latest_hlt < 30:
                confidence_scores['A'] += 0.2

        # ステージB: 加速上昇（トレンド発生）
        if (latest_stoch_k > 50 and latest_stoch_k > latest_stoch_d and
            latest_hlt > 50 and latest_hlt < 80):
            confidence_scores['B'] = 0.7
            if data['stoch_rsi_k'].iloc[-5:].mean() < data['stoch_rsi_k'].iloc[-1]:
                confidence_scores['B'] += 0.3

        # ステージC: 調整（押し目形成）
        if (latest_stoch_k < latest_stoch_d and
            latest_hlt > 30 and latest_hlt < 70):
            confidence_scores['C'] = 0.7
            if not latest_vol_spike:
                confidence_scores['C'] += 0.2

        # ステージD: 過熱上昇（高値圏）
        if (latest_stoch_k > 80 and latest_fg > 70 and
            latest_vix < 15 and latest_vol_spike):
            confidence_scores['D'] = 0.8

        # ステージD-BC: バイイングクライマックス
        if (latest_fg >= 85 and latest_vol_spike and
            latest_stoch_k > 90):
            confidence_scores['D-BC'] = 0.9
            if data['Close'].iloc[-1] < data['High'].iloc[-1] * 0.98:
                confidence_scores['D-BC'] += 0.1

        # ステージE: 調整A波（急落開始）
        if (latest_stoch_k < 50 and latest_stoch_d > latest_stoch_k and
            latest_fg < 50):
            confidence_scores['E'] = 0.7
            if data['Close'].pct_change(5).iloc[-1] < -0.05:
                confidence_scores['E'] += 0.3

        # ステージF: 戻りB波（ブルトラップ）
        if (40 < latest_fg < 60 and latest_vol_spike == False and
            30 < latest_stoch_k < 70):
            confidence_scores['F'] = 0.6
            if data['Close'].pct_change(3).iloc[-1] > 0.03:
                confidence_scores['F'] += 0.2

        # ステージG: 本格下落C波
        if (latest_fg < 30 and latest_stoch_k < 30 and
            latest_vix > 20):
            confidence_scores['G'] = 0.8

        # ステージG-SC: セリングクライマックス
        if (latest_fg <= 10 and latest_vix > 30 and
            latest_stoch_k < 20):
            confidence_scores['G-SC'] = 0.9
            if latest_vol_spike:
                confidence_scores['G-SC'] += 0.1

        # 最も可能性の高いステージを選択
        if confidence_scores:
            best_stage = max(confidence_scores.items(), key=lambda x: x[1])
            results['current_stage'] = best_stage[0]
            results['confidence'] = best_stage[1]
            results['stage_description'] = self.stages[best_stage[0]]['name']
        else:
            # デフォルトのステージを設定
            results['current_stage'] = 'C'  # 例えば「調整」をデフォルトに
            results['confidence'] = 0.5
            results['stage_description'] = self.stages['C']['name']
            results['warnings'].append('明確なステージを判断できませんでした。デフォルトのステージを表示しています。')

        # 警告メッセージの追加
        if results['current_stage'] in ['D', 'D-BC']:
            results['warnings'].append('⚠️ 高値圏注意: トレンド転換リスクあり')
        elif results['current_stage'] in ['G-SC']:
            results['warnings'].append('✅ セリクラ可能性: 反発準備を検討')
        elif results['current_stage'] == 'B':
            results['warnings'].append('🚀 トレンド発生: 最も収益性が高いゾーン')

        return results

    def visualize_analysis(self, data: pd.DataFrame, symbol: str = ""):
        """分析結果を視覚化"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        # 1. 価格チャート with ステージ背景色
        ax1 = axes[0]
        ax1.plot(data.index, data['Close'], 'k-', linewidth=1.5, label='Close')

        # ステージごとに背景色を設定
        for stage in self.stages.keys():
            stage_data = data[data['stage'] == stage]
            if len(stage_data) > 0:
                for i in range(len(stage_data)):
                    ax1.axvspan(stage_data.index[i], stage_data.index[i] + pd.Timedelta(days=1),
                              alpha=0.3, color=self.stages[stage]['color'])

        ax1.set_ylabel('Price', fontsize=10)
        ax1.set_title(f'{symbol} Elliott Wave Sentiment Analysis', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. STOCH RSI
        ax2 = axes[1]
        ax2.plot(data.index, data['stoch_rsi_k'], 'b-', label='STOCH RSI K', linewidth=1)
        ax2.plot(data.index, data['stoch_rsi_d'], 'r--', label='STOCH RSI D', linewidth=1)
        ax2.axhline(y=80, color='r', linestyle=':', alpha=0.5)
        ax2.axhline(y=20, color='g', linestyle=':', alpha=0.5)
        ax2.fill_between(data.index, 80, 100, alpha=0.1, color='red')
        ax2.fill_between(data.index, 0, 20, alpha=0.1, color='green')
        ax2.set_ylabel('STOCH RSI', fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Fear & Greed + VIX
        ax3 = axes[2]
        ax3_twin = ax3.twinx()

        line1 = ax3.plot(data.index, data['fear_greed'], 'g-', label='Fear & Greed', linewidth=1.5)
        line2 = ax3_twin.plot(data.index, data['vix'], 'r-', label='VIX', linewidth=1.5, alpha=0.7)

        ax3.axhline(y=80, color='r', linestyle=':', alpha=0.5)
        ax3.axhline(y=20, color='g', linestyle=':', alpha=0.5)
        ax3.set_ylabel('Fear & Greed', fontsize=10, color='g')
        ax3_twin.set_ylabel('VIX', fontsize=10, color='r')
        ax3.set_ylim(0, 100)
        ax3.tick_params(axis='y', labelcolor='g')
        ax3_twin.tick_params(axis='y', labelcolor='r')
        ax3.grid(True, alpha=0.3)

        # 4. Volume with spikes
        ax4 = axes[3]
        colors = ['red' if spike else 'gray' for spike in data['volume_spike']]
        ax4.bar(data.index, data['Volume'], color=colors, alpha=0.7, width=0.8)
        ax4.set_ylabel('Volume', fontsize=10)
        ax4.set_xlabel('Date', fontsize=10)
        ax4.grid(True, alpha=0.3)

        # 凡例を作成
        legend_elements = []
        for stage, info in self.stages.items():
            legend_elements.append(mpatches.Patch(color=info['color'],
                                                 label=f"{stage}: {info['name']}",
                                                 alpha=0.3))

        ax1.legend(handles=legend_elements, loc='upper left',
                  bbox_to_anchor=(1.01, 1), fontsize=8)

        plt.tight_layout()
        plt.show()

    def generate_detailed_report(self, data: pd.DataFrame, analysis_result: Dict) -> str:
        """詳細な分析レポートを生成"""
        report = []
        report.append("=" * 70)
        report.append("エリオット波動センチメント分析 詳細レポート")
        report.append("=" * 70)
        report.append(f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 現在のステージ
        report.append(f"【現在のステージ】")
        stage = analysis_result['current_stage']
        if stage:
            stage_info = self.stages[stage]
            report.append(f"  ステージ: {stage} - {stage_info['name']}")
            report.append(f"  信頼度: {analysis_result['confidence']*100:.1f}%")
            report.append(f"  リスクレベル: {stage_info['risk']}")
        report.append("")

        # 主要指標
        report.append(f"【主要指標の現在値】")
        indicators = analysis_result['indicators']
        report.append(f"  STOCH RSI (K/D): {indicators['stoch_rsi_k']:.1f} / {indicators['stoch_rsi_d']:.1f}")
        report.append(f"  HLT: {indicators['hlt']:.1f}")
        report.append(f"  RSI: {indicators['rsi']:.1f}")
        report.append(f"  Fear & Greed: {indicators['fear_greed']:.0f}")
        report.append(f"  VIX: {indicators['vix']:.1f}")
        report.append(f"  出来高スパイク: {'検出' if indicators['volume_spike'] else '通常'}")
        report.append("")

        # トレンド分析
        report.append(f"【トレンド分析】")
        price_change_5d = data['Close'].pct_change(5).iloc[-1] * 100
        price_change_20d = data['Close'].pct_change(20).iloc[-1] * 100
        report.append(f"  5日間変化率: {price_change_5d:+.2f}%")
        report.append(f"  20日間変化率: {price_change_20d:+.2f}%")

        if 'sma_20' in data.columns and 'sma_50' in data.columns:
            sma_trend = "上昇" if data['sma_20'].iloc[-1] > data['sma_50'].iloc[-1] else "下降"
            report.append(f"  移動平均トレンド: {sma_trend}")
        report.append("")

        # 警告・アドバイス
        if analysis_result['warnings']:
            report.append(f"【注意事項】")
            for warning in analysis_result['warnings']:
                report.append(f"  {warning}")
            report.append("")

        # 推奨アクション
        report.append(f"【推奨アクション】")
        action_map = {
            'A': [
                '✅ 段階的な買い増し検討',
                '📊 週足STOCH RSIの底打ち確認',
                '💡 リスク管理: ポジションサイズは控えめに'
            ],
            'B': [
                '🚀 トレンドフォローで積極買い',
                '📈 押し目での買い増し',
                '💡 利益確定ラインの設定'
            ],
            'C': [
                '⏸️ 新規買いは控えめに',
                '📊 HLT 30-50での押し目買い検討',
                '💡 既存ポジションは維持'
            ],
            'D': [
                '⚠️ 段階的な利益確定開始',
                '📊 出来高とFGを注視',
                '💡 トレーリングストップの活用'
            ],
            'D-BC': [
                '🚨 即座に大部分を利益確定',
                '📊 上ヒゲ・出来高急増を確認',
                '💡 逆張りショートの検討も可'
            ],
            'E': [
                '🔻 ロングポジション手仕舞い',
                '📊 戻り高値でのショート検討',
                '💡 現金比率を高める'
            ],
            'F': [
                '⚠️ 戻り売りのチャンス',
                '📊 上値の重さを確認',
                '💡 ブルトラップに注意'
            ],
            'G': [
                '🔻 ショートまたは現金保有',
                '📊 セリクラサインを待つ',
                '💡 逆張り買いは時期尚早'
            ],
            'G-SC': [
                '✅ 段階的な買い開始',
                '📊 出来高急増・VIX急騰を確認',
                '💡 中長期投資のチャンス'
            ]
        }

        if stage in action_map:
            for action in action_map[stage]:
                report.append(f"  {action}")

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)

# 使用例とデモ

if __name__ == "__main__":
    # センチメントチェッカーのインスタンス作成
    checker = AdvancedElliottSentimentChecker()

    print("エリオット波動センチメントチェッカー（拡張版）")
    print("-" * 50)

    # 実際の市場データで分析（例: S&P 500）
    symbols = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC"}
    for name, symbol in symbols.items():
        print(f"\n{name} ({symbol})のデータを取得中...")

        # データ取得
        data = checker.fetch_market_data(symbol, period="6mo")

        if data is not None:
            # 全期間のステージ分析
            print("ステージ分析を実行中...")
            data_with_stages = checker.analyze_stage_history(data)

            # 現在のステージ分析
            current_analysis = checker.analyze_stage(data_with_stages)

            # 詳細レポート生成
            report = checker.generate_detailed_report(data_with_stages, current_analysis)
            print("\n" + report)

            # 視覚化
            print("\nチャートを生成中...")
            checker.visualize_analysis(data_with_stages, f"{name} ({symbol})")

            # ステージ遷移の統計
            print("\n【ステージ遷移統計】")
            stage_counts = data_with_stages['stage'].value_counts()
            for stage, count in stage_counts.items():
                if stage:
                    percentage = (count / len(data_with_stages)) * 100
                    print(f"  {stage}: {count}日 ({percentage:.1f}%)")
        else:
            print("データの取得に失敗しました。")
