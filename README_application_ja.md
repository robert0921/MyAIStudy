# 🏢 MyAIStudy アプリケーション開発版 v4.0 - エンタープライズ大規模モデルアプリケーション 12 週間実践

> 主入口 run_example.py v4.0 と整合し、知乎知学堂 2026 年新版課綱を踏まえて再構成したアプリケーショントラックです。12 週間で、動かせる・説明できる・拡張できる企業向け AI アプリケーションの主線を作ります。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Track](https://img.shields.io/badge/track-application--engineering-orange.svg)]()

[中文](./README_application.md) | [日本語](./README_application_ja.md)

## 📋 プロジェクト概要

このプロジェクトは、AI学习12周实战计划表（应用开发版）.md に対応する軽量コード実装であり、同時に《知乎知学堂「AI大模型应用开发实战训练营」26年新版课纲》を参考に再設計した学習トラックです。目的は、本番用フレームワークをそのまま複製することではなく、**依存を抑えつつ、動かして説明できる学習サンプル**を提供することにあります。これにより、RAG、Agent、デプロイ、工程化の主要概念を短期間で一つの流れとして理解できます。

この版は 2 層構造で設計しています。

- ✅ **第 1 層: コア実行主線** - 純粋な Python サンプルで企業向け AI アプリケーションの主線を最後まで動かす
- ✅ **第 2 層: 拡張トピック** - 多モーダル文書処理、Agent の可制御性 / 自己反省、Harness Engineering、ローコード基盤、AI Testing、ChatBI を新版課綱に沿って吸収する

そのため、この文書では「すでにリポジトリ内で実行できる内容」と、「実フレームワークへ差し替えて学ぶべき内容」を明確に分けて説明します。

---

## 🎯 学習目標

このトラックを完了すると、次のことができるようになります。

✅ **RAG エンジニアリングの閉ループ構築** - Prompt、Embedding、Chunking から Query Rewrite、Hybrid Search、評価までを通して知識 QA 主線を作る  
✅ **可制御 Agent の理解** - Function Calling、MCP、JSON 制約、ReAct、Self-Reflection、Human-in-the-Loop などの中核構造を把握する  
✅ **Harness と長期記憶の理解** - Orchestration / Memory / Execution / Feedback の 4 層閉ループを説明し、タスク編成案を設計できる  
✅ **実運用導入と提效の視点獲得** - LoRA / QLoRA / 蒸留、高並列デプロイ、AI Testing、Text-to-SQL、ChatBI の関係を理解する  
✅ **業務プロトタイプと受入資料の組み立て** - 「知識ベース + Agent + サービス層」の総合サンプルを作成し、図表・指標・受入条件まで残す  

---

## 🎯 学習ロードマップ（第1-12週）

### フェーズ1: 大規模モデル基礎、RAG、マルチモーダル知識処理（Week 1-4）

| 週 | 学習重点 | コア内容 | 実践成果 | 対応方式 |
|------|----------|----------|----------|----------|
| **第1週** | Prompt / Context Engineering | プロンプト構造、出力制約、長コンテキスト、可制御生成 | Prompt 実験比較 + 出力制約テンプレート | ✅ コード主線 |
| **第2週** | Embedding とベクトル検索 | 疎ベクトル、類似度、Embedding 選定、ベクトル DB 基礎能力 | Top-K 検索デモ + モデル選定メモ | ✅ コード主線 |
| **第3週** | 文書取り込みと Native RAG | 文書分割、知識取り込み、検索生成閉ループ、PDF / Word / Web 解析の考え方 | 簡易知識 QA + 取り込みフロー図 | ✅ 主線 + 🔶 文書解析拡張 |
| **第4週** | RAG 最適化と評価 | Query Rewrite、Hybrid Search、Rerank、hit rate / MRR、知識庫運用の観点 | 検索評価レポート + 調優チェックリスト | ✅ 主線 + 🔶 評価拡張 |

### フェーズ2: 可制御 Agent、Harness、開発フレームワーク（Week 5-8）

| 週 | 学習重点 | コア内容 | 実践成果 | 対応方式 |
|------|----------|----------|----------|----------|
| **第5週** | Function Calling / MCP / A2A | ツール登録、引数受け渡し、プロトコル化呼び出し、A2A の関係理解 | MCP ツール呼び出しデモ | ✅ コード主線 |
| **第6週** | 可制御 Agent 設計 | Planning、Memory、ReAct、Self-Reflection、Human-in-the-Loop | 多段階 Agent サンプル + 承認ノード草図 | ✅ 主線 + 🔶 可制御性拡張 |
| **第7週** | Harness と長期記憶 | Orchestration / Memory / Execution / Feedback、長期記憶、タスク閉ループ、多 Agent 調度 | Harness 層設計図 + 記憶階層草案 | 🔶 拡張トピック |
| **第8週** | フレームワークとローコード統合 | LangChain / LlamaIndex / AutoGen / Coze / Dify / OpenManus 比較 | 選定表 + ワークフロー設計案 | ✅ 主線 + 🔶 統合拡張 |

### フェーズ3: 微調整、デプロイ、工程提效（Week 9-12）

| 週 | 学習重点 | コア内容 | 実践成果 | 対応方式 |
|------|----------|----------|----------|----------|
| **第9週** | LoRA / QLoRA / データ工程 | 微調整方式、資源コスト、データ準備、VRAM 見積もり、蒸留の認識 | 微調整戦略比較表 | ✅ 主線 + 🔶 データ拡張 |
| **第10週** | 高並列デプロイ | vLLM / SGLang / Ollama、PagedAttention、Continuous Batching、監視 | フレームワーク比較ベンチ | ✅ コード主線 |
| **第11週** | AI Coding と工程提效 | Spec Coding、AI Testing、受入条件、Text-to-SQL、ChatBI | 工程チェックリスト + SQL Copilot / ChatBI 草案 | ✅ 主線 + 🔶 提效拡張 |
| **第12週** | 総合プロジェクトと多モーダル拡張 | RAG + Agent + デプロイ要素の統合、質検 / 動画理解 / 多モーダル拡張の入口設計 | 企業向け AI アプリ原型 + 路演素材 | ✅ コード主線 |

### 2026 版課綱の強化項目をどう取り込むか

- **すでにコード化されている主線**: application/week1_4_rag_engineering.py、application/week5_8_agent_workflows.py、application/week9_12_delivery.py で、知識庫、Agent、デプロイ、工程効率の閉ループを一通り体験できます。
- **実フレームワークで学ぶべき拡張トピック**: PDF / Word / Web 解析、多モーダル RAG、OpenManus / Hermes 風 Harness、Coze / Dify API、AI Testing、ChatBI。
- **推奨順序**: まず実行可能な主線を走らせ、その後同じ業務要件を LangChain / LangGraph / FastAPI / vLLM / Coze / Dify などへ差し替えるのが最も理解しやすい進め方です。

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

このフェーズでは、いきなり第三者フレームワークに依存するのではなく、まず Prompt の構造化、ベクトル表現、Chunking、検索、評価の基本ロジックを動かします。新版課綱に合わせるなら、この主線に Query Rewrite、多モーダル文書解析、Hybrid Search を段階的に足していくのが自然です。コードは application/week1_4_rag_engineering.py にあります。

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
- 🔶 PDF / Word / Web 文書解析を知識取り込みチェーンのどこに置くべきか
- 🔶 Query Rewrite / Hybrid Search / Rerank / RAGAS をどこで接続するか

---

### 第5-8週: Agent とワークフロー編成

このフェーズの重点は「多 Agent を増やすこと」ではなく、ツール登録、プロトコル呼び出し、状態保存、タスク分解を実際に理解することです。新版課綱に合わせるなら、ここに Agent の可制御性、自己反省、Harness 分層、ローコード統合までつなげて理解するのが重要です。コードは application/week5_8_agent_workflows.py にあります。

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
- 🔶 JSON モード、承認ノード、反省プロンプトで Agent をどのように制御するか
- 🔶 Harness / 長期記憶 / 多 Agent 調度を OpenManus / Hermes 型システムでどう設計するか

---

### 第9-12週: 微調整、デプロイ、総合プロジェクト

このフェーズでは、「概念を説明できる」段階から、「どう落とし込むかを説明できる」段階へ進みます。新版課綱に合わせるなら、LoRA と推論フレームワーク比較だけでなく、データ工程、蒸留、高並列調優、AI Testing、ChatBI を一つの交付チェーンとしてつなぐことが重要です。コードは application/week9_12_delivery.py にあります。

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
- 🔶 データ工程、蒸留、性能監視、Continuous Batching / RadixAttention へ拡張する方向性
- 🔶 AI Testing、ChatBI、質検 / 多モーダル業務を現在の骨格へ接続する方法

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

多モーダル解析、長期記憶、ローコード基盤、AI Testing などの拡張は、学習テーマに応じて個別に依存を追加する方が整理しやすく、一括で詰め込まない方が運用しやすいです。

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
- openspec/README.md
- 培训课程产品化推荐案.md

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