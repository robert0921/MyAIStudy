# 🚀 MyAIStudy 高級版 v4.0 - RAG、Agent、研究志向アウトプット

> モジュール内部では Week 13-24 として管理され、統合版では第 25-36 週に対応します。RAG の原理から AI エンジニアとしての研究・発表・面接準備までを、日本語で俯瞰できる上級トラックです。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-latest-orange.svg)](https://numpy.org/)
[![Status](https://img.shields.io/badge/status-complete-success.svg)]()

## 📋 学習目標

高級版を完了すると、次の力を獲得できます。

✅ **RAG システム開発** - Retrieval-Augmented Generation を理解し、企業向け知識 QA を構築する  
✅ **ベクトル検索最適化** - Flat / IVF / HNSW の特性を把握し、高効率検索を設計する  
✅ **AI Agent 設計** - Memory、Tool-Use、Planning を組み合わせたエージェントを構築する  
✅ **サービス化デプロイ** - LLM API をサービス化し、監視や異常復旧まで見通す  
✅ **研究志向の思考** - 論文管理、実験追跡、ベンチマーク、最適化評価を行う  
✅ **ナレッジ管理** - 知識グラフ、技術文書、ノート体系化を進める  
✅ **キャリア実践力** - プロジェクト展示、技術白書、面接準備をまとめる  

---

## 🎯 学習ロードマップ（Week 13-24）

### フェーズ4: RAG とエージェントシステム（Week 13-18）

| 週 | 学習重点 | コア内容 | 実践成果 | 状態 |
|------|----------|----------|----------|------|
| **第13週** | LangChain と RAG の原理 | 文書読み込み、分割、ベクトル化、検索 | シンプルな知識 QA システム | ✅ 完了 |
| **第14週** | ベクトル DB インデックス機構 | Flat / IVF / HNSW、性能比較 | 次元別検索性能分析 | ✅ 完了 |
| **第15週** | RAG Pipeline 最適化 | Embedding 選定、Chunking 戦略 | 分割戦略比較実験 | ✅ 完了 |
| **第16週** | AI Agent アーキテクチャ | Memory / Tool-Use / Planning | 多段階タスクエージェント | ✅ 完了 |
| **第17週** | FastAPI サービス化 | RESTful API、セッション管理 | 本番志向 LLM API 設計 | ✅ 完了 |
| **第18週** | 監視と異常復旧 | ログ、性能監視、復旧戦略 | 監視・復旧メカニズム | ✅ 完了 |

### フェーズ5: 研究志向アウトプットと職業化（Week 19-24）

| 週 | 学習重点 | コア内容 | 実践成果 | 状態 |
|------|----------|----------|----------|------|
| **第19-20週** | 論文再現と実験管理 | 論文管理、実験追跡、ベンチマーク | 論文ライブラリ + 実験基盤 | ✅ 完了 |
| **第21週** | GPU 性能最適化とコスト評価 | GPU 選定、コスト計算、性能分析 | コスト最適化案 | ✅ 完了 |
| **第22-23週** | ナレッジ管理と文書生成 | 技術文書、知識グラフ、学習ノート | 個人知識ベース | ✅ 完了 |
| **第24週** | プロジェクト展示と面接準備 | 発表資料、ホワイトペーパー、面接問題集 | 就職用ポートフォリオ | ✅ 完了 |

**現在の進捗:** 100% (10 / 10 モジュール完了) 🎉

---

## 🚀 クイックスタート

### 方法1: 統一入口から起動

```bash
# 主入口から高級版を起動
python run_example.py advanced

# またはメニューの [3] を選択
python run_example.py
```

### 方法2: 高級版を直接起動

```bash
# 対話メニュー
python run_advanced_examples.py

# フェーズ4を実行
python run_advanced_examples.py week13
python run_advanced_examples.py week14
python run_advanced_examples.py week15-18

# フェーズ5を実行
python run_advanced_examples.py week19-20
python run_advanced_examples.py week21
python run_advanced_examples.py week22-23
python run_advanced_examples.py week24

# デモモード
python run_advanced_examples.py quick
python run_advanced_examples.py all
```

---

## 📚 詳細内容

### 第13週: LangChain と RAG の原理

**主要知識点:**
- 文書ロードと前処理
- テキスト分割戦略
- Embedding によるベクトル化
- ベクトルストアと類似検索
- RAG QA チェーン構築

**実装例:**

```python
splitter = SimpleTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(document)

embedder = SimpleEmbedding(embedding_dim=384)
vectors = embedder.embed_documents(chunks)

vector_store = SimpleVectorStore(embedder)
vector_store.add_documents(chunks)

rag_chain = SimpleRAGChain(vector_store, k=3)
result = rag_chain.run("RAG とは何か？")
```

**確認できること:**
- Top-K 類似文書の取得
- コンテキスト組み立てと回答生成
- 検索性能の簡易統計

---

### 第14週: ベクトル DB インデックス機構

**主要知識点:**
- Flat Index
- IVF Index
- HNSW Index
- 性能ベンチマーク
- 次元数と検索性能の関係

**実装例:**

```python
index_flat = FlatIndex(dimension=384)
index_ivf = IVFIndex(dimension=384, n_clusters=100)
index_hnsw = HNSWIndex(dimension=384, M=16)
```

**主な発見:**
- 次元数が増えるほど距離計算コストが上がる
- IVF は 10-20 倍の高速化余地がある
- 384-768 次元がバランスのよい選択肢

---

### 第15週: RAG Pipeline 最適化 ✅

**現在の実装:**
- ✂️ Chunking 戦略
  - FixedSizeChunking
  - SentenceChunking
  - SemanticChunking
  - RecursiveChunking

- 🔄 Embedding モデル比較
  - TFIDFEmbedding
  - Word2VecEmbedding
  - TransformerEmbedding

- 🎯 検索強化と再ランキング
  - BM25Retriever
  - CrossEncoderReranker
  - HybridRetriever

---

### 第16週: AI Agent アーキテクチャ設計 ✅

**現在の実装:**
- 💾 Memory
  - ShortTermMemory
  - LongTermMemory

- 🛠️ Tool-Use
  - CalculatorTool
  - SearchTool
  - WeatherTool
  - ToolRegistry

- 🧠 Planning
  - ReActAgent
  - PlanAndExecuteAgent

- 👥 Multi-Agent 協調
  - MultiAgentSystem

---

### 第17週: FastAPI サービス化 ✅

**現在の実装:**
- 🌐 API 設計
  - GET /health
  - POST /chat/completions
  - POST /chat/completions/stream

- 🗃️ セッション管理
  - Session
  - SessionManager

- ⚡ サービス制御
  - RateLimiter
  - LLMService

- 🧪 実装スタイル
  - FastAPI 風の擬似コードで設計を学ぶ
  - Web サーバー自体は起動しない

---

### 第18週: システム監視と異常復旧

**実装内容:**
- 📊 Prometheus 指標収集
- 📝 ELK ログ分析
- 🚨 異常検知と通知
- 🔄 自動復旧、フォールバック、サーキットブレーカ

---

### 第19-20週: 論文再現と実験管理 ✅

**主要知識点:**
- 論文メタデータ管理
- 実験追跡と比較
- ベンチマーク実行
- 結果可視化

**実装コンポーネント:**
- Paper / PaperLibrary
- Experiment / ExperimentTracker
- BenchmarkSuite

---

### 第21週: GPU 性能最適化とコスト評価 ✅

**主要知識点:**
- GPU コスト試算
- レイテンシ、スループット、メモリ分析
- 量子化、蒸留、剪定などの圧縮手法
- P50 / P95 / P99 指標の評価

**実装コンポーネント:**
- GPUCostCalculator
- PerformanceProfiler
- ModelCompressor

---

### 第22-23週: ナレッジ管理と文書生成 ✅

**主要知識点:**
- 技術文書自動生成
- ナレッジグラフ構築
- 学習ノート整理
- Mermaid 可視化

**実装コンポーネント:**
- TechDocument / DocumentGenerator
- KnowledgeGraph / KnowledgeNode
- NoteManager / LearningNote

---

### 第24週: プロジェクト展示と面接準備 ✅

**主要知識点:**
- プロジェクト紹介資料
- 技術白書
- 面接問題集
- 模擬面接

**実装コンポーネント:**
- ProjectShowcase
- TechnicalWhitepaper / WhitepaperGenerator
- InterviewQuestionBank

---

## 📊 プロジェクト構成

```text
MyAIStudy/
├── advanced/
│   ├── week13_langchain_rag.py
│   ├── week14_vector_database.py
│   ├── week15_rag_optimization.py
│   ├── week16_ai_agent.py
│   ├── week17_fastapi_service.py
│   ├── week18_monitoring.py
│   ├── week19_20_research_tools.py
│   ├── week21_optimization.py
│   ├── week22_23_knowledge_management.py
│   └── week24_presentation.py
├── run_advanced_examples.py        # 高級版入口
├── run_example.py                  # 統一入口
├── README_advanced_ja.md           # 本文書
└── README_ja.md                    # メイン README 日本語版
```

## 🛠️ 環境要件

### 基本依存
```bash
python >= 3.8
numpy >= 1.21.0
```

### 任意依存（実フレームワークへ置き換える場合）
```bash
# RAG 関連
sentence-transformers
faiss-cpu / faiss-gpu
chromadb
langchain

# API サービス
fastapi
uvicorn
redis

# 監視
prometheus-client
elasticsearch
```

---

## 📈 学習アドバイス

### 想定読者
- ✅ MyAIStudy 進階版を終えた人
- ✅ Python、PyTorch、深層学習基礎に慣れている人
- ✅ エンタープライズ AI アプリを構築したい人
- ✅ RAG や Agent に強い関心がある人

### 学習パス
1. **Step 1**: Week 13-14 で RAG 基礎を固める
2. **Step 2**: Week 15-18 で RAG 最適化からサービス化まで一気につなぐ
3. **Step 3**: Week 19-24 で研究志向の整理とキャリア資料化まで行う

### 時間配分
- **週あたり**: 10-15 時間
- **総期間**: 12 週間
- **1 日の目安**: 1.5-2 時間

---

## 💡 実践プロジェクト例

### プロジェクト1: 企業向け知識 QA システム
- 多様な文書形式の取り込み
- 高効率なベクトル検索
- 回答根拠の提示

### プロジェクト2: AI カスタマーサポート Agent
- 多輪会話管理
- ツール呼び出し
- ユーザー履歴の保持

### プロジェクト3: コードアシスタント
- コードベース理解と検索
- 自動補完とバグ診断
- API 文書参照

---

## 🔗 関連リソース

### 公式ドキュメント
- [LangChain ドキュメント](https://python.langchain.com/)
- [FAISS ドキュメント](https://faiss.ai/)
- [FastAPI ドキュメント](https://fastapi.tiangolo.com/)

### 学習資料
- [RAG 論文集](https://github.com/Tongji-KGLLM/RAG-Survey)
- [Agent 論文集](https://github.com/WooooDyy/LLM-Agent-Paper-List)

### 関連プロジェクト
- [LangChain](https://github.com/langchain-ai/langchain)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)

---

## 📝 学習ノートテンプレート

````markdown
# Week XX 学習ノート

## 学習目標
- [ ] 目標1
- [ ] 目標2

## コア概念
1. 概念A: ...
2. 概念B: ...

## コード実践
```python
# 重要なコード断片
```

## 問題と解決
1. 問題: ...
   解決: ...

## 学びの振り返り
- 重要な収穫 ...
- さらに深掘りしたい点 ...

## 次週の計画
- [ ] タスク1
- [ ] タスク2
````

---

## 📄 ライセンス

本プロジェクトは MIT ライセンスで公開されています。

## 📮 連絡先

- プロジェクトページ: [GitHub Repository](https://github.com/robert0921/MyAIStudy)
- Issue: [GitHub Issues](https://github.com/robert0921/MyAIStudy/issues)
