# 🏢 MyAIStudy アプリケーション開発版 v4.0 - エンタープライズ大規模モデルアプリケーション 12 週間実践

> 主入口 run_example.py v4.0 と整合した、日本語版のアプリケーション開発トラックです。12 週間で、説明可能・再利用可能・拡張可能な企業向け AI アプリケーションの最小実装を作り上げます。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Track](https://img.shields.io/badge/track-application--engineering-orange.svg)]()

## 📋 プロジェクト概要

このプロジェクトは、AI学习12周实战计划表（应用开发版）.md に対応する軽量コード実装です。目的は、本番用フレームワークをそのまま置き換えることではなく、**依存を抑えつつ、動かして説明できる学習サンプル**を提供することにあります。これにより、RAG、Agent、デプロイ、エンジニアリングの主要概念を短期間で一つの流れとして理解できます。

前三段階と比べて、このトラックには次の 2 つの明確な狙いがあります。

- ✅ 純粋な Python サンプルで、企業向け AI アプリケーションの主線を最後まで動かす
- ✅ LangChain、FastAPI、vLLM、Dify などの実運用フレームワークへ段階的に置き換えやすい構造で整理する

---

## 🎯 学習目標

このトラックを完了すると、次のことができるようになります。

✅ **RAG エンジニアリングの閉ループ構築** - Prompt、Embedding、Chunking、検索評価までを通して最小の知識 QA を作る  
✅ **Agent 実行メカニズムの理解** - Function Calling、MCP、Memory、ReAct の中核構造を把握する  
✅ **フレームワーク選定判断** - LangChain、LlamaIndex、AutoGen、Coze、Dify の役割差を整理する  
✅ **実運用導入の視点獲得** - LoRA / QLoRA、サービス化、監視指標、品質ゲートを理解する  
✅ **業務プロトタイプの組み立て** - 「知識ベース + Agent + サービス層」の総合サンプルを作成し、デモや提案に使う  

---

## 🎯 学習ロードマップ（第1-12週）

### フェーズ1: 大規模モデル基礎と RAG エンジニアリング（Week 1-4）

| 週 | 学習重点 | コア内容 | 実践成果 | 状態 |
|------|----------|----------|----------|------|
| **第1週** | Prompt / Context Engineering | プロンプト構造、出力制約、コンテキスト整理 | Prompt 実験比較 | ✅ 実行可能 |
| **第2週** | Embedding とベクトル検索 | 疎ベクトル、類似度、ベクトル DB 基礎能力 | Top-K 検索デモ | ✅ 実行可能 |
| **第3週** | RAG Pipeline | 文書分割、知識取り込み、検索生成閉ループ | 簡易知識 QA | ✅ 実行可能 |
| **第4週** | RAG 評価 | hit rate、MRR、再現率の定量化 | 検索評価レポート | ✅ 実行可能 |

### フェーズ2: Agent アーキテクチャと開発フレームワーク（Week 5-8）

| 週 | 学習重点 | コア内容 | 実践成果 | 状態 |
|------|----------|----------|----------|------|
| **第5週** | Function Calling と MCP | ツール登録、引数受け渡し、プロトコル化呼び出し | MCP ツール呼び出しデモ | ✅ 実行可能 |
| **第6週** | Agent の計画、記憶、ReAct | Memory、Tool-Use、タスク分解 | 多段階 Agent サンプル | ✅ 実行可能 |
| **第7週** | フレームワーク選定 | LangChain / LlamaIndex / AutoGen の位置付け | 適用シーン比較表 | ✅ 実行可能 |
| **第8週** | ローコードワークフロー | Coze / Dify の発想、意図認識、フロー編成 | ローコード客服ワークフロー | ✅ 実行可能 |

### フェーズ3: 微調整、デプロイ、エンジニアリング効率（Week 9-12）

| 週 | 学習重点 | コア内容 | 実践成果 | 状態 |
|------|----------|----------|----------|------|
| **第9週** | LoRA / QLoRA 戦略 | 微調整方式、資源コスト、デプロイ適合性 | 微調整戦略比較表 | ✅ 実行可能 |
| **第10週** | 高並列デプロイ | vLLM / SGLang / Ollama の選定 | フレームワーク比較ベンチ | ✅ 実行可能 |
| **第11週** | エンジニアリング効率 | Spec Coding、受入条件、Text-to-SQL | 工程チェックリスト + SQL Copilot | ✅ 実行可能 |
| **第12週** | 総合プロジェクト | RAG + Agent + デプロイ要素の統合 | 企業向け AI アプリ原型 | ✅ 実行可能 |

---

## 🚀 クイックスタート

### 方法1: 統一入口から起動

```bash
# 主入口からアプリケーション開発版を起動
python run_example.py application
python run_example.py application quick
python run_example.py application week9-12

# または主メニューから選択
python run_example.py
```

### 方法2: アプリケーション開発版を直接実行

```bash
# 対話メニュー（推奨）
python run_application_examples.py

# 特定週を実行
python run_application_examples.py week1
python run_application_examples.py week6
python run_application_examples.py week12

# ステージ単位で実行
python run_application_examples.py week1-4
python run_application_examples.py week5-8
python run_application_examples.py week9-12

# 短時間デモ
python run_application_examples.py quick

# 全 12 週の完全デモ
python run_application_examples.py all
```

---

## 📚 詳細内容

### 第1-4週: RAG エンジニアリングの主線

このフェーズでは、いきなり第三者フレームワークに依存するのではなく、まず Prompt の構造化、ベクトル表現、Chunking、検索、評価の基本ロジックを動かします。コードは application/week1_4_rag_engineering.py にあります。

**主要コンポーネント:**

```python
from application.week1_4_rag_engineering import (
    PromptWorkbench,
    FixedChunker,
    SimpleVectorStore,
    SimpleRAGPipeline,
)

workbench = PromptWorkbench()
results = workbench.run_experiment("企業向け知識 QA システムをどう構築するか？")

chunker = FixedChunker(chunk_size=45, overlap=8)
vector_store = SimpleVectorStore()
pipeline = SimpleRAGPipeline(vector_store=vector_store, chunker=chunker)
```

**確認できる内容:**

- ✅ Prompt 構造の違いが出力制御性に与える影響
- ✅ 疎ベクトルで Embedding と類似検索を模擬する方法
- ✅ 最小限で動く RAG フロー
- ✅ hit rate@3 と MRR@3 による基礎評価

---

### 第5-8週: Agent とワークフロー編成

このフェーズの重点は「多 Agent を増やすこと」ではなく、ツール登録、プロトコル呼び出し、状態保存、タスク分解を実際に理解することです。コードは application/week5_8_agent_workflows.py にあります。

**主要コンポーネント:**

```python
from application.week5_8_agent_workflows import (
    ToolRegistry,
    SimpleMCPServer,
    ConversationMemory,
    SimpleReActAgent,
    build_default_registry,
)

registry = build_default_registry()
server = SimpleMCPServer(registry)
memory = ConversationMemory()
agent = SimpleReActAgent(registry, memory)
```

**確認できる内容:**

- ✅ MCP 風の tools/list と tools/call
- ✅ Agent がタスクキーワードに応じてツールを選択する流れ
- ✅ Memory が直近のコンテキストを保存する方法
- ✅ LangChain / LlamaIndex / AutoGen の差分整理
- ✅ ローコード客服ワークフローの最小閉ループ

---

### 第9-12週: 微調整、デプロイ、総合プロジェクト

このフェーズでは、「概念を説明できる」段階から、「どう落とし込むかを説明できる」段階へ進みます。コードは application/week9_12_delivery.py にあります。

**主要コンポーネント:**

```python
from application.week9_12_delivery import (
    FinetuningPlanner,
    DeploymentBenchmarker,
    SpecCodingAssistant,
    EnterpriseAIAssistant,
)

planner = FinetuningPlanner()
benchmarker = DeploymentBenchmarker()
assistant = EnterpriseAIAssistant()
```

**確認できる内容:**

- ✅ Full Fine-Tuning / LoRA / QLoRA の資源差分
- ✅ vLLM / SGLang / Ollama のデプロイ比較視点
- ✅ Spec Coding で受入条件とテスト項目を生成する流れ
- ✅ RAG と Agent を統合した業務プロトタイプの応答フロー

---

## 📊 プロジェクト構成

```
MyAIStudy/
├── application/
│   ├── __init__.py
│   ├── week1_4_rag_engineering.py   # Prompt / Embedding / RAG / 評価
│   ├── week5_8_agent_workflows.py   # Function Calling / MCP / Agent / ワークフロー
│   └── week9_12_delivery.py         # 微調整 / デプロイ / 工程効率 / 総合プロジェクト
│
├── run_application_examples.py      # アプリケーション開発版統一入口
├── README_application_ja.md         # 本文書
├── AI学习12周实战计划表（应用开发版）.md
└── AI学习48周实战计划表（整合版）.md
```

---

## 🛠️ 環境要件

### 基本依存

```bash
python >= 3.8
```

現在の学習サンプルは Python 標準ライブラリのみで動きます。実際の業務プロトタイプへ拡張する場合は、段階ごとに次の依存を導入してください。

```bash
# 実運用 RAG 実装
pip install langchain faiss-cpu chromadb sentence-transformers

# API サービスと監視
pip install fastapi uvicorn redis prometheus-client

# 微調整とデプロイ
pip install transformers peft trl unsloth
```

---

## 🔗 他トラックとの関係

- 🌱 **入門版** は Python、データ処理、機械学習、深層学習の導入を扱います。
- 🎓 **進階版** は数理原理、モデル訓練、微調整、推論最適化を扱います。
- 🚀 **高級版** はシステム化されたアウトプット、研究志向能力、キャリア準備に重心があります。
- 🏢 **アプリケーション開発版** は、企業導入の流れを 12 週間に圧縮して再構成したトラックです。

重複の少ない推奨主線で学ぶ場合は、AI学习48周实战计划表（整合版）.md を優先して参照してください。

---

## 💡 学習アドバイス

### 想定読者

- ✅ Python の基礎がある
- ✅ 基本的な深層学習の概念を理解している
- ✅ RAG、Agent、デプロイを一つの実装ストーリーとしてつなげたい
- ✅ デモ、面接用プロジェクト、社内研修サンプルを作りたい

### 推奨する学び方

1. 各週のデモをまず動かし、その後コード構造を逆にたどる。
2. 各フェーズで最低 1 枚は「人に説明できる図や表」を残す。
3. サンプル中の静的実装を、FAISS や実際の API などの実フレームワークへ置き換えてみる。
4. 各フェーズ完了時に、入力・出力・リスク・検証方法を整理した受入チェックリストを作る。

---

## 📄 関連ドキュメント

- README_ja.md
- README_advanced_ja.md
- AI学习12周实战计划表（应用开发版）.md
- AI学习48周实战计划表（整合版）.md

---

## 📮 フィードバックとサポート

- GitHub Issues: [問題を登録する](https://github.com/robert0921/MyAIStudy/issues)
- このトラックが役に立った場合は、実際の業務 API や知識ベースへ置き換えて拡張してみてください

---

## 📄 ライセンス

本プロジェクトは MIT ライセンスで公開されています。詳細は LICENSE を参照してください。

---

<div align="center">

**MyAIStudy アプリケーション開発版**  
*最小実装から、企業向け AI アプリの落とし込み方を理解する*

</div>