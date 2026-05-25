# 🎓 MyAIStudy 進階版 v4.0 - 深層学習原理と LLM エンジニアリング

> 統合版の第 13-24 週に対応する日本語版ドキュメントです。線形代数、逆伝播、最適化器から、LLM アーキテクチャ、Prompt Engineering、LoRA 微調整、推論最適化までを一つの実装体系として学べます。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## ✨ 特徴

### 📚 理論から実装まで一貫
- **ゼロからの実装**: コアアルゴリズムを手で実装し、内部原理を理解する
- **数理的な厳密さ**: 勾配チェック誤差 < 1e-10 を目標に検証
- **PyTorch 比較検証**: 公式実装と照合して正しさを確認
- **可視化重視**: 複雑な概念を図・曲線・注意マップで確認

### 🚀 エンジニアリング観点の強化
- **モジュール化設計**: 拡張しやすい構成
- **性能最適化**: 分散訓練、混合精度、データパイプライン最適化
- **実践重視**: CIFAR-10 級の学習設定と現実的な訓練フロー
- **監視対応**: 訓練過程、性能指標、圧縮効果を追跡

## 🎯 コア機能モジュール

### 1️⃣ 深層学習基礎（ゼロから実装）
- ✅ 線形代数と自動微分
- ✅ 逆伝播の逐次導出
- ✅ SGD / Adam / RMSProp
- ✅ CNN / Transformer 基礎実装

### 2️⃣ LLM アーキテクチャ（LLaMA 実装）
- ✅ Multi-Head Attention
- ✅ RoPE 位置エンコーディング
- ✅ RMSNorm
- ✅ SwiGLU FFN

### 3️⃣ 深層学習訓練システム
- ✅ DDP 多 GPU 並列訓練
- ✅ FP16 混合精度
- ✅ INT8 量子化
- ✅ モデル剪定（Magnitude / Structured / Global / Iterative）
- ✅ LMDB ベースのデータパイプライン
- ✅ EarlyStopping と Checkpoint 管理

### 4️⃣ Prompt Engineering
- ✅ Prompt 品質分析
- ✅ Few-shot サンプル管理
- ✅ バッチ Prompt テスト
- ✅ 出力品質の自動評価

### 5️⃣ 大規模モデル微調整（LoRA / QLoRA / PEFT）
- ✅ LoRA 低ランク適応
- ✅ QLoRA 4-bit 量子化微調整
- ✅ PEFT 統一インターフェース
- ✅ 小規模 ChatGPT 風モデルのデモ

### 6️⃣ 推論最適化（Batched Inference / KV Cache）
- ✅ KV Cache による生成高速化
- ✅ バッチ推論エンジン
- ✅ 遅延 / スループット / メモリ追跡
- ✅ 推論ベンチマーク

### 7️⃣ 可視化と分析
- ✅ インタラクティブダッシュボード
- ✅ 学習曲線と学習率モニタリング
- ✅ モデル性能比較可視化
- ✅ Attention 可視化

## 💡 想定読者

- 🎓 深層学習初学者で、数理原理まで踏み込みたい人
- 👨‍💻 アルゴリズムエンジニアで、実験・検証用の土台が欲しい人
- 👩‍🏫 教学・社内研修用に、可視化付きのサンプルが欲しい人
- 🔬 研究用に、モジュール化された実験ベースコードが必要な人

## 📁 プロジェクト構成

```text
MyAIStudy/
├── run_intermediate_examples.py             # 統一入口
├── intermediate/
│   ├── __init__.py
│   ├── linear_algebra.py                    # 線形代数と自動微分
│   ├── backpropagation.py                   # 逆伝播の詳細実装
│   ├── optimizer_comparison.py              # 最適化器比較
│   ├── cnn_transformer.py                   # CNN と Transformer 基礎
│   ├── layers.py                            # NN レイヤ
│   ├── models.py                            # 簡易ネットワーク
│   ├── optimizers.py                        # 最適化器実装
│   ├── performance.py                       # 性能分析補助
│   ├── models_torch.py                      # PyTorch モデル
│   ├── training.py                          # Trainer / EarlyStopping / Checkpoint
│   ├── checkpointing.py                     # チェックポイント補助
│   ├── evaluation.py                        # モデル評価と圧縮比較
│   ├── pruning.py                           # モデル剪定
│   ├── monitoring.py                        # 性能監視
│   ├── data.py                              # データローダ
│   ├── kaggle_data.py                       # Kaggle データパイプライン
│   ├── kaggle_models.py                     # 競技向けモデル
│   ├── llm_architecture.py                  # LLaMA 実装
│   ├── llm_visualization.py                 # LLM 可視化
│   ├── finetuning.py                        # LoRA / QLoRA / PEFT
│   ├── inference_optimization.py            # KV Cache / バッチ推論
│   ├── visualization.py                     # ダッシュボード
│   └── training_monitor.py                  # 訓練モニタ
├── AI学习12周实战计划表（进阶版）.md
└── requirements.txt
```

## 🚀 実行方法

### 🎮 対話モード
```bash
python run_intermediate_examples.py
```

### ⚡ クイックデモ
```bash
python run_intermediate_examples.py quick
```

### 📖 個別機能

**基礎知識デモ**
```bash
python run_intermediate_examples.py fundamentals
```

**LLM アーキテクチャデモ**
```bash
python run_intermediate_examples.py llm
```

**性能テスト**
```bash
python run_intermediate_examples.py snn
```

**ダッシュボード**
```bash
python run_intermediate_examples.py dashboard
```

**深層学習訓練**
```bash
python run_intermediate_examples.py train
```

**Prompt Engineering**
```bash
python run_intermediate_examples.py prompt
```

**モデル剪定**
```bash
python run_intermediate_examples.py pruning
```

**大規模モデル微調整**
```bash
python run_intermediate_examples.py finetuning
```

**推論最適化**
```bash
python run_intermediate_examples.py inference
```

**ヘルプ表示**
```bash
python run_intermediate_examples.py help
```

## 💡 モジュール別説明

### 1. 🧮 線形代数と自動微分
**対象ファイル:** `linear_algebra.py`

- 基礎行列演算、ベクトル演算、Jacobian 計算
- 解析勾配と数値勾配の比較
- Xavier 初期化付きの手書き線形層
- PyTorch autograd との対照検証

### 2. 🔄 逆伝播の詳細
**対象ファイル:** `backpropagation.py`

- 2 層ネットワークの手書き実装
- 逆伝播公式の逐次展開
- 数値法による勾配検証
- 損失曲線の可視化

### 3. ⚙️ 最適化アルゴリズム
**対象ファイル:** `optimizer_comparison.py`

- SGD、Momentum、Adam、RMSProp の比較
- 収束曲線と学習率影響の分析
- 2D 軌跡やベンチマーク出力

### 4. 🧠 CNN と Transformer 基礎
**対象ファイル:** `cnn_transformer.py`

- im2col ベースの畳み込み
- Max Pooling 実装
- Self-Attention と Multi-Head Attention
- 計算量分析と可視化

### 5. 🚀 LLM アーキテクチャ
**対象ファイル:** `llm_architecture.py`, `llm_visualization.py`

- LLaMA 風の残差ブロック
- RoPE、RMSNorm、SwiGLU の分解理解
- Attention ウェイトとテキスト生成デモ

### 6. 🎯 訓練システム
**対象ファイル:** `training.py`, `models_torch.py`

- CIFAR-10 向け訓練器
- 分散学習、混合精度、EarlyStopping、Checkpoint
- 量子化・圧縮・性能比較の基礎土台

### 7. 📊 データパイプライン最適化
**対象ファイル:** `data.py`, `kaggle_data.py`

- LMDB キャッシュ
- Albumentations 増強
- 分散 DataLoader
- IO 性能最適化

### 8. 🧠 Prompt Engineering と Few-shot
**統合先:** `run_intermediate_examples.py`

- PromptDebugger による品質分析
- FewShotManager による例管理
- バッチ評価と自動最適化

### 9. 🔪 モデル剪定と圧縮
**対象ファイル:** `pruning.py`, `evaluation.py`

- Magnitude / Structured / Global / Iterative 剪定
- スパース率の統計
- モデルサイズ・推論速度・精度の比較

### 10. 🎨 大規模モデル微調整
**対象ファイル:** `finetuning.py`

- LoRA / QLoRA / PEFT の統一デモ
- SimpleLLM による軽量検証
- 可学習パラメータ比率の可視化

### 11. ⚡ 推論最適化
**対象ファイル:** `inference_optimization.py`

- KVCache
- AttentionWithKVCache
- BatchedInferenceEngine
- レイテンシとスループットの比較

## 🔬 コード例

### Prompt Engineering

```python
from run_intermediate_examples import PromptDebugger

debugger = PromptDebugger()
result = debugger.test_prompt("Calculate 7+6")
optimized = debugger.optimize_prompt("Calculate 7+6", target="13")
```

### シンプルなニューラルネットワーク

```python
from intermediate.models import SimpleNN
from intermediate.optimizers import Adam
import numpy as np

model = SimpleNN([784, 128, 64, 10])
optimizer = Adam(lr=0.001)

X = np.random.randn(784, 100)
Y = np.eye(10)[:, np.random.randint(0, 10, 100)]
loss = model.train_step(X, Y, optimizer)
```

### PyTorch Trainer

```python
from intermediate.training import Trainer, TrainerConfig
from intermediate.models_torch import CIFAR10Net

config = TrainerConfig(
    batch_size=64,
    learning_rate=0.001,
    mixed_precision=True,
    use_early_stopping=True,
)

model = CIFAR10Net()
trainer = Trainer(model, train_loader, val_loader, config)
results = trainer.train()
```

### LoRA / QLoRA

```python
from intermediate.finetuning import PEFTModel, SimpleLLM

model = SimpleLLM(vocab_size=50000, d_model=512, n_layers=6)
peft_model = PEFTModel(model, method='lora', rank=8, alpha=16)
```

### KV Cache 推論

```python
from intermediate.inference_optimization import BatchedInferenceEngine

engine = BatchedInferenceEngine(model=your_model, max_batch_size=32, max_seq_len=512)
generated, metrics = engine.generate(input_ids, max_new_tokens=50, use_cache=True)
```

## 📊 性能指標

| 機能モジュール | 改善効果 | 検証指標 |
|---------|---------|---------|
| データパイプライン最適化 | IO 3-5 倍 | スループット計測 |
| 混合精度訓練 | 訓練速度 2-3 倍 | FP16 vs FP32 |
| モデル量子化 | モデルサイズ 75% 削減 | 精度損失 < ±1% |
| モデル剪定 | パラメータ 30-50% 削減 | 精度損失 < ±1% |
| LoRA 微調整 | 学習パラメータ 99% 削減 | 0.5-1% のみ学習 |
| QLoRA | VRAM 75% 節約 | 4-bit + LoRA |
| KV Cache | 生成速度 2-10 倍 | 遅延低減 |
| バッチ推論 | スループット 3-8 倍 | batch_size=4-16 |
| 勾配チェック | 誤差 < 1e-10 | 解析 vs 数値勾配 |

## 🎯 検証結果

- ✅ 数値誤差の基準を満たす勾配チェック
- ✅ PyTorch 実装と整合する主要モジュール
- ✅ 訓練速度、IO、推論速度の改善を確認
- ✅ 統一入口 `run_intermediate_examples.py` から全機能へアクセス可能

## 📚 ドキュメントリソース

- 📖 [使用指南](docs/USAGE_GUIDE.md)
- 📊 [学習計画](AI学习12周实战计划表（进阶版）.md)
- 🧭 [日本語メイン README](README_ja.md)

## 🤝 コントリビューション

Issue、改善提案、Pull Request を歓迎します。

## 📄 ライセンス

本プロジェクトは MIT ライセンスで公開されています。詳細は [LICENSE](LICENSE) を参照してください。

## 🙏 謝辞

- PyTorch チームの深層学習フレームワーク
- オープンソースコミュニティの貢献

## 📮 連絡先

- プロジェクトページ: [GitHub Repository](https://github.com/robert0921/MyAIStudy)
- 問題報告: [Issues](https://github.com/robert0921/MyAIStudy/issues)

## 📈 バージョン履歴

### v4.0 (2026-05-25) 🎉
- ✅ タイトル、入口説明、コマンド例を v4.0 に統一
- ✅ 実行例を `run_intermediate_examples.py` に統一
- ✅ `python run_example.py intermediate` からの導線を整理
- ✅ 12 週間の進階版学習計画リンクを現行資料に合わせて更新

### v2.3 (2025-11-06)
- ✅ `finetuning.py` による LoRA / QLoRA / PEFT 実装
- ✅ `inference_optimization.py` による KV Cache / バッチ推論
- ✅ ベンチマークと詳細ガイドの追加

### v2.2 (2025-11-05)
- ✅ `pruning.py` による 4 種類の剪定戦略
- ✅ 圧縮効果比較の強化

### v2.1 (2025-11-04)
- ✅ EarlyStopping と CheckpointManager の導入

### v2.0 (2025-11-04)
- ✅ コード統合と入口整理
- ✅ 依存管理の整理

---

<div align="center">
  <p>⭐ このプロジェクトが役に立ったら Star をお願いします。</p>
  <p>Made with ❤️ by robert0921</p>
</div>