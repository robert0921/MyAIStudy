# 🎓 MyAIStudy - AIエンジニア向けフルスタック学習システム v4.0

> Python 入門からエンタープライズ AI アプリケーションの実運用までを一貫して学べる完全ロードマップ | 48 週間の統合メインライン + アプリケーション開発短期集中

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-latest-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

---

⭐ **このプロジェクトが役に立った場合は、ぜひ Star をお願いします。** ⭐

## 📖 プロジェクト概要

MyAIStudy は、プログラミング基礎からエンタープライズ AI アプリケーションの実装・運用までを体系的に学ぶための総合トレーニング体系です。現在のリポジトリでは、**4 本の段階別トラック**と、**重複を除去した 1 本の統合学習プラン**を提供しています。主な使い方は次の 2 通りです。

- 段階別に進める: 入門版 → 進階版 → 高級版
- 統合版の主線で学ぶ: AI学习48周实战计划表（整合版）.md を参照

| トラック | 期間 | 位置付け | 主な内容 | 想定読者 |
|------|------|------|----------|----------|
| 🌱 **[入門版](#-入門版第1-12週)** | 12週 | 基礎主線 | Python、NumPy、Pandas、Scikit-Learn、PyTorch の基礎 | プログラミング経験者、AI 初学者 |
| 🎓 **[進階版](#-進階版第13-24週)** | 12週 | 原理主線 | ゼロからの実装、LLM アーキテクチャ、最適化アルゴリズム、エンジニアリング実践 | 原理を深く理解したい開発者 |
| 🚀 **[高級版](#-高級版第25-36週)** | 12週 | システム化主線 | RAG システム、AI Agent、研究志向のアウトプット、キャリア準備 | 体系的なエンジニアリング力を身につけたい人 |
| 🏢 **[アプリケーション開発版](#-アプリケーション開発版平行実践トラック)** | 12週 | 企業導入短期集中 | RAG エンジニアリング、Agent アーキテクチャ、LLM 微調整、高並列デプロイ、AI Coding | AI 基礎があり、企業導入に集中したい人 |
| 🧭 **統合版総合プラン** | 48週 | 重複除去済み主線 | 重複内容を整理した完全な学習スケジュール | 一つの主線で継続的に成長したい学習者 |

**最初に確認することを推奨:** AI学习48周实战计划表（整合版）.md  
**アプリケーション開発短期集中:** README_application_ja.md + AI学习12周实战计划表（应用开发版）.md  
**統一エントリ:** run_example.py はトラック直行と引数透過をサポートします。例: `python run_example.py application quick`

---

## 🎯 全体学習ロードマップ

```
┌────────────────────────────────────────────────────────────────────┐
│  フェーズ1 (第1-4週) - Python とデータサイエンス基礎                │
├────────────────────────────────────────────────────────────────────┤
│  Python / NumPy / Pandas / Matplotlib                              │
└────────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────────┐
│  フェーズ2 (第5-12週) - 機械学習と深層学習入門                      │
├────────────────────────────────────────────────────────────────────┤
│  Scikit-Learn / PyTorch / CNN / RNN / 総合プロジェクト             │
└────────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────────┐
│  フェーズ3 (第13-24週) - 深層学習原理と LLM エンジニアリング基礎    │
├────────────────────────────────────────────────────────────────────┤
│  逆伝播 / 最適化器 / Transformer / 微調整 / 推論最適化 / 訓練系      │
└────────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────────┐
│  フェーズ4 (第25-36週) - 企業向け RAG、Agent、サービス化            │
├────────────────────────────────────────────────────────────────────┤
│  Prompt / Embedding / RAG / MCP / Agent / LangGraph / FastAPI      │
└────────────────────────────────────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────────┐
│  フェーズ5 (第37-48週) - 実運用、研究志向アウトプット、職業力強化    │
├────────────────────────────────────────────────────────────────────┤
│  vLLM / SGLang / AI Coding / 論文再現 / ナレッジ管理 / 発表          │
└────────────────────────────────────────────────────────────────────┘
```

**統合版タイムテーブル:** AI学习48周实战计划表（整合版）.md  
**企業導入短期集中:** README_application_ja.md

---

## 🌱 入門版（第1-12週）

### 学習目標
ゼロからスタートし、Python プログラミング、データサイエンスツール、機械学習、深層学習の基礎を身につけます。

### 主な特徴
- ✅ **体系的な基礎訓練** - Python → NumPy → Pandas → Scikit-Learn → PyTorch
- ✅ **実践プロジェクト駆動** - 学生成績管理、EC 分析、住宅価格予測、画像分類
- ✅ **段階的な構成** - 毎週 1 テーマで着実に知識体系を構築
- ✅ **ツールチェーン完備** - データ処理、可視化、モデリング、評価を一通り体験

### 学習ルート

**第1-4週: Python とデータサイエンス基礎**
- 🐍 Week 1: Python 基礎とオブジェクト指向
- 🔢 Week 2: NumPy 配列操作と行列演算
- 📊 Week 3: Pandas データ処理と分析
- 📈 Week 4: Matplotlib / Seaborn 可視化

**第5-8週: 機械学習基礎**
- 🤖 Week 5: 機械学習概念と分類タスク（Iris）
- 📉 Week 6: 線形回帰と決定木（住宅価格予測）
- 🔍 Week 7: クラスタリング（ユーザーセグメンテーション）
- ⚙️ Week 8: モデル改善と交差検証

**第9-12週: 深層学習入門**
- 🧠 Week 9: PyTorch 基礎と MNIST 手書き数字認識
- 🖼️ Week 10: CNN（CIFAR-10）
- 📝 Week 11: RNN / LSTM シーケンスモデル（テキスト分類）
- 🎯 Week 12: 総合プロジェクト実践

### クイックスタート

```bash
# 方法1: 対話メニュー
python run_beginner_examples.py

# 方法2: 特定週を実行
python run_beginner_examples.py week1    # Python 基礎
python run_beginner_examples.py week5-8  # 機械学習
python run_beginner_examples.py week9-12 # 深層学習

# 方法3: ステージ単位で実行
python run_beginner_examples.py stage1   # 第1-4週
python run_beginner_examples.py all      # 12週すべて
```

### 学習成果
- 📊 12 個の実践プロジェクト
- 📈 6 本のデータ分析レポート
- 🤖 3 種類の深層学習モデル（MLP、CNN、LSTM）
- 📝 1000 行超の実践コード

**📖 詳細ドキュメント:** [README_beginner_ja.md](./README_beginner_ja.md)

---

## 🎓 進階版（第13-24週）

### 学習目標
深層学習の数理原理を深く理解し、大規模モデルの訓練・最適化技術を体系的に身につけます。

### 主な特徴
- 🧮 **コアアルゴリズムをゼロから実装** - 逆伝播、最適化器、畳み込み、注意機構を手書き実装
- 🔬 **数理的に厳密** - 勾配チェック誤差 < 1e-10、PyTorch と比較して正しさを検証
- 🚀 **エンジニアリング最適化** - 分散訓練、混合精度、モデル圧縮、推論最適化
- 🤖 **LLM 全スタック** - アーキテクチャ実装、Prompt Engineering、LoRA 微調整、KV Cache

### 学習ルート

**第13-16週: 深層学習の数理コア**
- 📐 Week 13: 線形代数と自動微分
- 🔄 Week 14: 逆伝播の詳細
- ⚙️ Week 15: 最適化アルゴリズム比較（SGD / Adam / RMSProp）
- 🧠 Week 16: CNN と Transformer 基礎

**第17-20週: エンジニアリング実践**
- 🎯 Week 17: PyTorch 訓練システム（分散 / 混合精度）
- 🔪 Week 18: モデル剪定と圧縮
- 🎨 Week 19: 大規模モデル微調整（LoRA / QLoRA / PEFT）
- ⚡ Week 20: 推論最適化（KV Cache / バッチ推論）

**第21-24週: LLM 専門テーマ**
- 🤖 Week 21: LLaMA アーキテクチャ完全実装
- 💬 Week 22: Prompt Engineering と Few-shot 学習
- 📊 Week 23: データ可視化とダッシュボード
- 🔧 Week 24: 性能監視と訓練パイプライン

### クイックスタート

```bash
# 方法1: 対話メニュー
python run_intermediate_examples.py

# 方法2: 機能モジュール
python run_intermediate_examples.py fundamentals  # 基礎知識デモ
python run_intermediate_examples.py llm           # LLM アーキテクチャ
python run_intermediate_examples.py train         # 訓練システム
python run_intermediate_examples.py finetuning    # 大規模モデル微調整
python run_intermediate_examples.py inference     # 推論最適化

# 方法3: デモモード
python run_intermediate_examples.py quick         # 短時間デモ（5分）
python run_intermediate_examples.py all           # 完全デモ（30分）
```

### 学習成果
- 📐 4 個の数理コアモジュール（線形代数、逆伝播、最適化器、Transformer）
- 🎯 5 個のエンジニアリング実践モジュール（訓練、剪定、微調整、推論、監視）
- 🧠 2 個の LLM 専門モジュール（LLaMA、Prompt Engineering）
- 📊 13,000 行超の完全実装コード

**📖 詳細ドキュメント:** [README_intermediate_ja.md](./README_intermediate_ja.md)

---

## 🚀 高級版（第25-36週）

### 学習目標
本番運用レベルの RAG システム、AI Agent、ナレッジ管理、プロジェクト発表力を身につけます。

### 主な特徴
- 📚 **RAG 完全実装** - 文書処理、ベクトル検索、Pipeline 最適化、ハイブリッド検索
- 🤖 **AI Agent システム** - Memory、Tool-Use、Planning、Multi-Agent 協調
- 🌐 **サービス化デプロイ** - FastAPI、WebSocket、セッション管理、性能最適化
- 📊 **システム監視** - Prometheus、ELK、異常検知、自動復旧
- 📝 **研究志向のアウトプット** - 論文管理、実験追跡、ナレッジグラフ、技術文書
- 💼 **キャリア準備** - プロジェクト展示、ホワイトペーパー、面接問題集

### 学習ルート

**第25-30週: RAG システムとエージェント**
- 📚 Week 25: LangChain と RAG の原理
- 🔍 Week 26: ベクトル DB インデックス機構（Flat / IVF / HNSW）
- 🎯 Week 27: RAG Pipeline 最適化（Chunking / Re-ranking）
- 🤖 Week 28: AI Agent アーキテクチャ設計
- 🌐 Week 29: FastAPI によるサービス化
- 📊 Week 30: システム監視と異常復旧

**第31-36週: 研究志向とキャリア形成**
- 📄 Week 31-32: 論文管理と実験追跡
- 💰 Week 33: GPU 性能最適化とコスト評価
- 📝 Week 34-35: ナレッジ管理と文書生成
- 💼 Week 36: プロジェクト発表と面接準備

### クイックスタート

```bash
# 方法1: 対話メニュー
python run_advanced_examples.py

# 方法2: 特定週を実行（フェーズ4）
python run_advanced_examples.py week13      # LangChain と RAG
python run_advanced_examples.py week14      # ベクトルデータベース
python run_advanced_examples.py week15-18   # RAG 最適化 / Agent / サービス化

# 方法3: 特定週を実行（フェーズ5）
python run_advanced_examples.py week19-20   # 論文管理 / 実験追跡
python run_advanced_examples.py week21      # GPU 最適化 / コスト評価
python run_advanced_examples.py week22-23   # ナレッジ管理 / 文書生成
python run_advanced_examples.py week24      # プロジェクト発表 / 面接準備

# 方法4: デモモード
python run_advanced_examples.py quick       # 短時間デモ（10分）
python run_advanced_examples.py all         # 完全デモ（40分）
```

### 学習成果
- 🔍 6 個の RAG / Agent 系モジュール（基礎、最適化、Agent、サービス化、監視）
- 📚 4 個の研究・知識管理ツール（論文管理、実験追跡、知識管理、プロジェクト展示）
- 📊 2,800 行超の上級機能コード
- 💼 完整な就職用ポートフォリオ

**📖 詳細ドキュメント:** [README_advanced_ja.md](./README_advanced_ja.md)

---

## 🏢 アプリケーション開発版（平行実践トラック）

### 学習目標
エンタープライズ向け大規模モデル活用を想定し、12 週間で動作する・説明できる・拡張できる業務プロトタイプを完成させます。

### 主な特徴
- 🗂️ **RAG エンジニアリング全工程** - Prompt、Embedding、Chunking、検索、評価までを一気通貫で理解
- 🤖 **Agent アーキテクチャ実践** - Function Calling、MCP、Memory、ReAct の最小実行例
- 🔧 **フレームワーク選定力** - LangChain、LlamaIndex、AutoGen、Coze、Dify の位置付け比較
- ⚡ **デプロイと微調整の判断軸** - LoRA / QLoRA、vLLM、SGLang、Ollama の選択基準
- 💼 **エンジニアリング閉ループ** - Spec Coding、受け入れ条件、Text-to-SQL、総合プロジェクト演示

### 学習ルート

**第1-4週: 大規模モデル基礎と RAG エンジニアリング**
- 💬 Week 1: Prompt Engineering & Context Engineering
- 🔢 Week 2: Embedding 原理とベクトル DB 選定
- 📚 Week 3: RAG の中核フローとローカル知識ベース構築
- 🎯 Week 4: ハイブリッド検索 + Reranking + RAG 効果評価

**第5-8週: Agent アーキテクチャとフレームワーク実践**
- 🔧 Week 5: Function Calling と MCP プロトコル
- 🧠 Week 6: Agent の計画、記憶、ReAct / LangGraph 実践
- 🛠️ Week 7: LangChain / LlamaIndex / AutoGen を実務観点で理解
- 🖱️ Week 8: Coze / Dify と企業システム統合

**第9-12週: 微調整、デプロイ、エンジニアリング効率**
- ⚙️ Week 9: LoRA / QLoRA 微調整と VRAM 最適化
- 🚀 Week 10: vLLM / SGLang / Ollama による高並列推論デプロイ
- 💻 Week 11: AI Coding 実践と ChatBI プロジェクト
- 🏆 Week 12: 総合プロジェクト · 企業 RAG + Agent + デプロイ

### 学習成果
- 📦 企業向け知識 QA Agent の原型（軽量コード版）
- 📊 微調整戦略比較表 + 高並列デプロイ比較表
- 📄 実際のフレームワークへ置き換え可能な工程骨格

### クイックスタート

```bash
# 統一入口
python run_example.py application
python run_example.py application quick

# アプリケーション開発版単独入口
python run_application_examples.py
python run_application_examples.py quick
python run_application_examples.py week1-4
python run_application_examples.py week12
```

**📖 詳細ドキュメント:** [README_application_ja.md](./README_application_ja.md)  
**📖 詳細計画:** [AI学习12周实战计划表（应用开发版）.md](./AI学习12周实战计划表（应用开发版）.md)

---

## 🚀 はじめ方

### 統一入口（推奨）

```bash
# 対話形式でトラックを選択
python run_example.py

# メニュー項目:
# [1] 入門版 - Python と AI 基礎
# [2] 進階版 - 深層学習原理とエンジニアリング
# [3] 高級版 - RAG と研究志向アウトプット
# [4] アプリケーション開発版 - 企業向け大規模モデル実践
# [5] プロジェクト情報を表示
# [6] 終了
```

### コマンドラインで直接起動

```bash
# 入門版を起動
python run_example.py beginner

# 進階版を起動
python run_example.py intermediate

# 高級版を起動
python run_example.py advanced

# アプリケーション開発版を起動
python run_example.py application
python run_example.py application quick

# ヘルプを表示
python run_example.py --help
```

---

## 📁 プロジェクト構成

```
MyAIStudy/
├── 📌 統一入口
│   ├── run_example.py                 # メイン入口（トラック選択）
│   ├── run_beginner_examples.py       # 入門版専用入口
│   ├── run_intermediate_examples.py   # 進階版専用入口
│   ├── run_advanced_examples.py       # 高級版専用入口
│   └── run_application_examples.py    # アプリケーション開発版専用入口
│
├── 🌱 入門版コード (beginner/)
│   ├── week1_python_basics.py
│   ├── week2_numpy_operations.py
│   ├── week3_pandas_analysis.py
│   ├── week4_visualization.py
│   ├── week5_8_machine_learning.py
│   └── week9_12_deep_learning.py
│
├── 🎓 進階版コード (intermediate/)
│   ├── 📐 数理コア
│   │   ├── linear_algebra.py
│   │   ├── backpropagation.py
│   │   ├── optimizer_comparison.py
│   │   └── cnn_transformer.py
│   │
│   ├── 🎯 訓練システム
│   │   ├── models_torch.py
│   │   ├── training.py
│   │   ├── pruning.py
│   │   ├── finetuning.py
│   │   └── inference_optimization.py
│   │
│   └── 🤖 LLM アーキテクチャ
│       ├── llm_architecture.py
│       └── llm_visualization.py
│
├── 🚀 高級版コード (advanced/)
│   ├── 📚 RAG システム (Week 13-18)
│   │   ├── week13_langchain_rag.py
│   │   ├── week14_vector_database.py
│   │   ├── week15_rag_optimization.py
│   │   ├── week16_ai_agent.py
│   │   ├── week17_fastapi_service.py
│   │   └── week18_monitoring.py
│   │
│   └── 🔬 研究・知識管理ツール (Week 19-24)
│       ├── week19_20_research_tools.py
│       ├── week21_optimization.py
│       ├── week22_23_knowledge_management.py
│       └── week24_presentation.py
│
├── 🏢 アプリケーション開発版コード (application/)
│   ├── week1_4_rag_engineering.py
│   ├── week5_8_agent_workflows.py
│   └── week9_12_delivery.py
│
├── 📚 ドキュメント
│   ├── README.md                      # 中国語メイン README
│   ├── README_ja.md                   # 日本語メイン README
│   ├── README_beginner.md             # 中国語 入門版 README
│   ├── README_beginner_ja.md          # 日本語 入門版 README
│   ├── README_intermediate.md         # 中国語 進階版 README
│   ├── README_intermediate_ja.md      # 日本語 進階版 README
│   ├── README_advanced.md             # 中国語 高級版 README
│   ├── README_advanced_ja.md          # 日本語 高級版 README
│   ├── README_application.md          # 中国語 アプリケーション開発版 README
│   ├── README_application_ja.md       # 日本語 アプリケーション開発版 README
│   └── AI学习48周实战计划表（整合版）.md
│
└── 💾 データとモデル
    ├── checkpoints/                  # モデルチェックポイント
    └── data/                         # データセット（自動ダウンロード）
```

---

## 🛠️ 環境設定

### Python バージョン
Python 3.8+

### 依存関係のインストール

**入門版依存:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision
```

**進階版追加依存:**
```bash
pip install numba plotly dash
```

**高級版追加依存（任意）:**
```bash
pip install langchain faiss-cpu chromadb fastapi uvicorn redis prometheus-client
```

**アプリケーション開発版の現在のサンプル:**

アプリケーション開発版の軽量デモは標準では Python 標準ライブラリのみを使用します。実際のフレームワーク実装に置き換える場合は、README_application_ja.md の案内に従って段階的に依存関係を追加してください。

**一括インストール:**
```bash
pip install -r requirements.txt
```

---

## 💡 想定読者

| 学習段階 | 前提条件 | 想定読者 | 学習目標 |
|---------|---------|---------|---------|
| 🌱 **入門版** | 基本的なプログラミング知識 | プログラミング経験者、AI 初学者 | Python と機械学習の基礎を習得 |
| 🎓 **進階版** | 入門版修了または深層学習の基礎知識 | 原理を深く理解したい開発者 | 深層学習の数理とエンジニアリング実践を習得 |
| 🚀 **高級版** | 進階版修了または深層学習の実務理解 | 企業向け AI アプリを構築したい人 | RAG、Agent、研究志向アウトプット、キャリア力を習得 |
| 🏢 **アプリケーション開発版** | Python と AI の基礎概念 | エンタープライズ向け大規模モデル実装に集中したい開発者 | RAG、Agent、サービス構築を単独で形にする |
| 🧭 **統合版主線** | 長期的に継続して学ぶ意思 | 重複のない一本のルートで学びたい人 | 48 週で基礎から実運用までの成長を完成 |

---

## 🏆 学習成果

48 週間の統合主線を完走する、または前三段階の後にアプリケーション開発版の重要モジュールを補完すると、次の力が身につきます。

✅ **堅実な理論基盤** - 数理原理からエンジニアリング実践まで体系的に理解  
✅ **豊富なプロジェクト経験** - 30 件以上の実践プロジェクトを通じてデータ分析、訓練、デプロイを経験  
✅ **完全な技術スタック** - Python、PyTorch、RAG、Agent、API、監視までを横断  
✅ **研究志向の思考** - 論文再現、実験管理、技術文書、知識の蓄積  
✅ **職業的な実践力** - プロジェクト発表、技術講演、面接準備、履歴書改善

---

## 📊 性能指標

### 進階版の性能指標

| 機能モジュール | 改善効果 | 検証指標 |
|---------|---------|---------|
| データパイプライン最適化 | IO 性能 3-5 倍 | スループット計測 |
| 混合精度訓練 | 訓練速度 2-3 倍 | FP16 vs FP32 |
| モデル量子化 | モデルサイズ 75% 削減 | 精度損失 < ±1% |
| モデル剪定 | パラメータ 30-50% 削減 | 精度損失 < ±1% |
| LoRA 微調整 | 学習対象パラメータ 99% 削減 | 0.5-1% のみ学習 |
| KV Cache | 生成速度 2-10 倍 | 遅延低減 |
| バッチ推論 | スループット 3-8 倍 | batch_size=4-16 |

### 高級版の性能指標

| 機能 | 指標 | 説明 |
|------|------|------|
| ベクトル検索（IVF） | 10-20 倍高速化 | Flat Index 比 |
| RAG QA | Top-3 精度 | 類似度検索 |
| 推奨次元 | 384-768 | 性能と精度のバランス |

---

## 📚 推奨学習パス

### 推奨メインライン
1. まず AI学习48周实战计划表（整合版）.md を確認する。
2. フェーズ1 → フェーズ5 の順に進め、高級版とアプリケーション開発版を行き来しすぎない。
3. 各フェーズ終了時に、少なくとも 1 つのプロジェクト、1 本の文書、1 つの指標表を残す。

### AI 完全初学者
1. 入門版の第 1 週から開始する。
2. 各週のコード練習と週次振り返りを完了する。
3. 第 12 週終了後に進階版へ進み、その後統合版の第 25 週以降へ進む。

### 深層学習の基礎がある人
1. 進階版のコアモジュール、または統合版第 13 週から始める。
2. LoRA、推論最適化、訓練システムなどの実務基盤を補完する。
3. その後、第 25 週以降の RAG / Agent / デプロイに進む。

### 企業向けアプリを早く形にしたい人
1. アプリケーション開発版の 12 週間集中ルートから始める。
2. README_application_ja.md と AI学习12周实战计划表（应用开发版）.md を併読する。
3. 集中ルート終了後、統合版第 41-48 週の研究志向・キャリア形成部分を補う。

---

## 📝 学習アドバイス

### 時間配分
- **週あたりの学習時間:** 10-15 時間（1 日 1.5-2 時間目安）
- **総学習期間:** 48 週間主線、または 12 週間のアプリケーション開発短期集中
- **推奨ペース:** 平日 1.5 時間、週末 5 時間

### 学習方法
1. **手を動かすことを優先** - すべての知識点をコードで検証する
2. **ノートを残す** - 「学んだこと + つまずいた点 + 解決方法」を記録する
3. **比較しながら学ぶ** - アルゴリズム比較、性能比較を行う
4. **進捗を可視化する** - Excel や Notion で学習進度を管理する

### デバッグのコツ
- `print()` で途中結果を確認する
- `shape` と `dtype` でテンソルの次元を確認する
- VS Code Debugger でブレークポイントを使う
- 公式ドキュメントや Stack Overflow を参照する

---

## 🤝 コントリビューション

改善提案や Issue、Pull Request を歓迎します。

1. 本プロジェクトを Fork する
2. 機能ブランチを作成する (`git checkout -b feature/AmazingFeature`)
3. 変更をコミットする (`git commit -m 'Add some AmazingFeature'`)
4. リモートへ push する (`git push origin feature/AmazingFeature`)
5. Pull Request を作成する

---

## 📄 ライセンス

本プロジェクトは MIT ライセンスで公開されています。詳細は [LICENSE](LICENSE) を参照してください。

---

## 🙏 謝辞

- PyTorch チームによる優れた深層学習フレームワーク
- LangChain コミュニティによる RAG 技術への貢献
- オープンソースコミュニティの支援と知見

---

## 📮 連絡先

- プロジェクトページ: [GitHub Repository](https://github.com/robert0921/MyAIStudy)
- 問題報告: [Issues](https://github.com/robert0921/MyAIStudy/issues)

---

## 📈 バージョン履歴

### v4.0 (2026-05-25) 🎉
- ✅ **統一入口の刷新**: `run_example.py` を 4 トラック構成に再編し、引数透過をサポート
- ✅ **アプリケーション開発版の補完**: 12 週 README、サンプルコード、単独入口、統合ルート説明を追加
- ✅ **48 週主線の成立**: 重複除去済みの統合学習計画を追加し、推奨学習パスを明確化
- ✅ **文書の口径統一**: メイン README と各段階 README を v4.0 と実際のスクリプト名へ統一
- ✅ **高級版説明の更新**: Week 15-17 を計画占位から実装済みモジュール説明へ更新

### v3.0 (2025-11-13) 🎉
- ✅ **高級版 Week 19-24 完成**: 研究志向アウトプットとキャリア準備を追加
- ✅ **論文管理ツール追加**: Paper / PaperLibrary / Experiment / ExperimentTracker
- ✅ **GPU 最適化モジュール追加**: コスト計算、性能分析、モデル圧縮
- ✅ **知識管理システム追加**: 文書生成、知識グラフ、ノート管理
- ✅ **キャリアモジュール追加**: プロジェクト展示、ホワイトペーパー、面接問題集

### v2.3 (2025-11-06)
- ✅ **進階版微調整機能**: LoRA / QLoRA / PEFT 実装
- ✅ **進階版推論最適化**: KV Cache / バッチ推論
- ✅ **性能ベンチマーク**: 遅延、スループット、メモリ分析を整備

### v2.2 (2025-11-05)
- ✅ **進階版モデル剪定**: 4 種類の剪定戦略（Magnitude / Structured / Global / Iterative）
- ✅ **圧縮評価強化**: 精度、モデルサイズ、推論速度の比較を追加

### v2.1 (2025-11-04)
- ✅ **早期終了**: EarlyStopping クラス追加
- ✅ **チェックポイント管理**: CheckpointManager クラス追加

### v2.0 (2025-11-04)
- ✅ **統一入口**: run_example.py で三つのトラックを統合
- ✅ **依存関係整理**: オプション依存の管理を改善

### v1.0 (2025-10)
- ✅ **入門版完成**: 12 週の Python と AI 基礎
- ✅ **進階版基盤**: 深層学習原理の実装
- ✅ **高級版 Week 13-14**: RAG 基礎

---

<div align="center">

**Made with ❤️ by [robert0921](https://github.com/robert0921)**

[⬆️ ページ先頭へ戻る](#-myaistudy---aiエンジニア向けフルスタック学習システム-v40)

</div>