# 🌱 MyAIStudy 入門版 v4.0 - 12 週間 AI 基礎実践

> 統合版の第 1-12 週に対応し、プログラミング基礎から AI 実践までを体系的に学ぶ日本語版ガイドです。

[中文](./README_beginner.md) | [日本語](./README_beginner_ja.md)

## 📚 プロジェクト概要

本プロジェクトは、**AI学习12周实战计划表（入门版）** に対応する完全なコード実装です。Python や AI に不慣れな初学者でも、12 週間を通じて次の力を身につけられるように設計されています。

- ✅ Python 文法とオブジェクト指向を理解する
- ✅ NumPy、Pandas、Matplotlib などのデータサイエンス基礎ツールを使いこなす
- ✅ 機械学習の基本原理を理解し、実践プロジェクトを完成させる
- ✅ PyTorch を使って深層学習モデルを構築する
- ✅ 画像分類やテキスト分類などの典型タスクを体験する

## 🗂️ プロジェクト構成

```text
beginner/
├── __init__.py                      # モジュール初期化
├── week1_python_basics.py           # 第1週: Python 基礎と OOP
├── week2_numpy_operations.py        # 第2週: NumPy 配列操作と行列演算
├── week3_pandas_analysis.py         # 第3週: Pandas データ処理と分析
├── week4_visualization.py           # 第4週: Matplotlib / Seaborn 可視化
├── week5_8_machine_learning.py      # 第5-8週: 機械学習基礎
└── week9_12_deep_learning.py        # 第9-12週: 深層学習入門

run_beginner_examples.py             # 統一入口スクリプト
README_beginner_ja.md                # 本文書
```

## 🎯 学習ロードマップ

### フェーズ1: Python とデータサイエンス基礎（第1-4週）

| 週 | 学習テーマ | コアスキル | 実践プロジェクト |
|------|---------|---------|----------|
| 第1週 | Python 基本文法、関数、クラス、モジュール | オブジェクト指向、デコレータ、例外処理 | 学生成績管理システム |
| 第2週 | NumPy 配列操作、ブロードキャスト、行列演算 | 配列処理、線形代数、固有値分解 | 画像グレースケール化、行列演算 |
| 第3週 | Pandas による読み込み、整形、集計、結合 | データクレンジング、groupby、ピボット | EC ユーザー行動分析 |
| 第4週 | Matplotlib / Seaborn 可視化 | 各種チャート、ヒートマップ、サブプロット | 売上データ可視化 |

**このフェーズでの成果:**
- 完整な学生成績管理システム（OOP 実装）
- EC データ分析レポート（リピート率、RFM 分析など）
- 売上推移可視化ダッシュボード

---

### フェーズ2: 機械学習基礎と Scikit-Learn（第5-8週）

| 週 | 学習テーマ | コアスキル | 実践プロジェクト |
|------|---------|---------|----------|
| 第5週 | 教師あり / 教師なし学習、評価指標 | 分類モデル、交差検証、混同行列 | Iris 分類 |
| 第6週 | 線形回帰、ロジスティック回帰、決定木 | 回帰分析、特徴量設計、モデル比較 | 住宅価格予測 |
| 第7週 | クラスタリング: K-Means、DBSCAN | 教師なし学習、シルエット係数、エルボー法 | ユーザーセグメンテーション |
| 第8週 | モデル改善: 交差検証、グリッドサーチ、特徴量工夫 | ハイパーパラメータ調整、重要度分析、学習曲線 | Titanic 最適化 |

**このフェーズでの成果:**
- Iris 分類器（精度 > 95%）
- 住宅価格予測モデル（R² > 0.85）
- ユーザークラスタリングレポート（K-Means + DBSCAN）
- モデル改善の実験ノート

---

### フェーズ3: 深層学習入門と PyTorch（第9-12週）

| 週 | 学習テーマ | コアスキル | 実践プロジェクト |
|------|---------|---------|----------|
| 第9週 | PyTorch 基礎: Tensor、autograd、Dataset | テンソル操作、勾配計算、データローダ | MNIST 手書き数字認識 |
| 第10週 | CNN の原理と実装 | 畳み込み層、プーリング、特徴マップ可視化 | CIFAR-10 画像分類 |
| 第11週 | RNN / LSTM 基礎 | シーケンスモデル、Embedding、LSTM ユニット | テキスト感情分類 |
| 第12週 | 総合プロジェクト | 完全な Pipeline、モデル保存、推論最適化 | エンドツーエンド実践 |

**このフェーズでの成果:**
- MNIST 分類器（精度 > 98%）
- CIFAR-10 CNN モデル
- LSTM ベースの感情分類器
- 完整な深層学習 Pipeline

---

## 🚀 クイックスタート

### 1. 環境準備

**Python バージョン:** Python 3.8+

**依存関係のインストール:**

```bash
# 基礎依存（フェーズ1-2）
pip install numpy pandas matplotlib seaborn scikit-learn

# 深層学習依存（フェーズ3）
pip install torch torchvision

# 一括インストール
pip install -r requirements.txt
```

### 2. 実行方法

**対話メニュー:**
```bash
python run_beginner_examples.py
```

**コマンドラインモード:**
```bash
# 単独週を実行
python run_beginner_examples.py week1
python run_beginner_examples.py week2
python run_beginner_examples.py week5-8

# ステージ単位で実行
python run_beginner_examples.py stage1
python run_beginner_examples.py stage2
python run_beginner_examples.py stage3

# 12 週すべてを実行
python run_beginner_examples.py all

# モジュール状況を確認
python run_beginner_examples.py status

# ヘルプを表示
python run_beginner_examples.py help
```

### 3. 個別モジュールの実行

```bash
python -m beginner.week1_python_basics
python -m beginner.week2_numpy_operations
python -m beginner.week3_pandas_analysis
python -m beginner.week4_visualization
python -m beginner.week5_8_machine_learning
python -m beginner.week9_12_deep_learning
```

---

## 📖 詳細内容

### 第1週: Python 基礎文法とオブジェクト指向

**学習内容:**
- Python 基礎文法（リスト内包表記、辞書内包表記、Lambda）
- 関数型プログラミング（デコレータ、クロージャ）
- オブジェクト指向（クラス、継承、カプセル化）
- 例外処理とファイル操作

**実践プロジェクト: 学生成績管理システム**

```python
from beginner.week1_python_basics import demonstrate_python_basics

manager = demonstrate_python_basics()
```

**ポイント:**
- 学生情報管理
- 成績統計
- ランキング計算

### 第2週: NumPy 配列操作と行列演算

**学習内容:**
- NumPy 配列生成と添字操作
- ブロードキャスト
- ufunc と配列演算
- 線形代数（行列積、固有値、SVD）

**実践プロジェクト: 画像グレースケール化**

```python
from beginner.week2_numpy_operations import demonstrate_image_processing

gray_image = demonstrate_image_processing()
```

**比較する 3 つの方法:**
- 平均値法: Gray = (R + G + B) / 3
- 加重法: Gray = 0.299*R + 0.587*G + 0.114*B
- 最大値法: Gray = max(R, G, B)

### 第3週: Pandas データ処理と分析

**学習内容:**
- DataFrame の作成と操作
- 欠損値、外れ値、重複値の処理
- groupby 集計
- データ結合とピボットテーブル

**実践プロジェクト: EC ユーザー行動分析**

```python
from beginner.week3_pandas_analysis import analyze_ecommerce_data

customers, orders, analysis = analyze_ecommerce_data()
```

**分析の見どころ:**
- リピート率の算出
- 時系列による売上傾向分析
- 年齢層、都市、購買行動によるセグメンテーション

### 第4週: Matplotlib / Seaborn 可視化

**学習内容:**
- 折れ線、散布図、棒グラフ、ヒストグラム
- サブプロット配置とスタイル調整
- Seaborn の統計可視化
- 相関関係の可視化

**実践プロジェクト: 売上データ可視化**

```python
from beginner.week4_visualization import demonstrate_sales_visualization

df = demonstrate_sales_visualization()
```

**作成するチャート例:**
- 日次売上トレンド + 移動平均
- 月次売上棒グラフ
- 商品カテゴリ別円グラフ
- 地域 × 月次ヒートマップ

### 第5-8週: 機械学習基礎

**第5週: 分類タスク（Iris）**

```python
from beginner.week5_8_machine_learning import week5_classification

model, X_test, y_test = week5_classification()
```

**第6週: 回帰タスク（住宅価格予測）**

```python
from beginner.week5_8_machine_learning import week6_regression

lr_model, dt_model = week6_regression()
```

**第7週: クラスタリング（ユーザー分群）**

```python
from beginner.week5_8_machine_learning import week7_clustering

kmeans, dbscan, data = week7_clustering()
```

**第8週: モデル改善**

```python
from beginner.week5_8_machine_learning import week8_model_tuning

best_model = week8_model_tuning()
```

### 第9-12週: 深層学習入門

**第9週: MNIST 手書き数字認識**

```python
from beginner.week9_12_deep_learning import week9_mnist

model = week9_mnist()
```

**第10週: CIFAR-10 画像分類（CNN）**

```python
from beginner.week9_12_deep_learning import week10_cifar10

cnn_model = week10_cifar10()
```

**第11週: テキスト感情分類（RNN / LSTM）**

```python
from beginner.week9_12_deep_learning import week11_text_classification

rnn_model = week11_text_classification()
```

**第12週: 総合プロジェクト**

```python
from beginner.week9_12_deep_learning import week12_comprehensive_project

final_model = week12_comprehensive_project()
```

---

## 🛠️ 技術スタック

### データサイエンス系ツール
- **NumPy**: 数値計算と配列処理
- **Pandas**: データ処理と分析
- **Matplotlib**: 基本可視化
- **Seaborn**: 統計可視化

### 機械学習フレームワーク
- **Scikit-Learn**: 代表的な機械学習アルゴリズム
  - 分類: ロジスティック回帰、決定木、ランダムフォレスト
  - 回帰: 線形回帰、決定木回帰
  - クラスタリング: K-Means、DBSCAN
  - ツール: グリッドサーチ、交差検証、特徴量工学

### 深層学習フレームワーク
- **PyTorch**: 深層学習モデリング
  - Tensor と自動微分
  - `nn.Module`
  - `DataLoader`
  - Adam / SGD などの最適化器

---

## 📊 学習成果例

### フェーズ1（第1-4週）
- ✅ 学生成績管理システム（300 行超）
- ✅ 画像グレースケール化アルゴリズム（3 手法比較）
- ✅ EC データ分析レポート（10 指標以上）
- ✅ 売上可視化ダッシュボード（15 種類以上のグラフ）

### フェーズ2（第5-8週）
- ✅ Iris 分類器（精度 > 95%）
- ✅ 住宅価格予測モデル（R² = 0.85+）
- ✅ ユーザーセグメンテーション分析（K-Means + DBSCAN）
- ✅ 改善済みランダムフォレスト（精度 5% 以上向上）

### フェーズ3（第9-12週）
- ✅ MNIST 手書き数字認識（MLP）
- ✅ CIFAR-10 画像分類（CNN）
- ✅ テキスト感情分類（LSTM）
- ✅ 完整な深層学習 Pipeline

---

## 💡 学習アドバイス

### 学習ペース
- 毎週 **8-10 時間**（1 日 1.5 時間程度）
- 学習内容 + 実装 + ノート整理をセットで進める
- 週末に振り返り時間を確保する

### 実践アドバイス
1. **実装を最優先** - すべての知識点をコードで確かめる
2. **ノートを残す** - 学んだこと、詰まった点、解決策を記録する
3. **比較しながら学ぶ** - 文法差分やアルゴリズム差分を意識する
4. **進捗を可視化する** - Excel や Notion で学習進度を管理する

### デバッグのコツ
- `print()` で途中結果を確認する
- `shape` と `dtype` でテンソル次元を確認する
- VS Code Debugger を使う
- 公式ドキュメントと Stack Overflow を活用する

### 次に進むルート
12 週間完了後は、次のテーマへ進めます。
- Transformer アーキテクチャ（BERT、GPT）
- コンピュータビジョン（検出、セグメンテーション）
- 自然言語処理（NER、翻訳）
- 強化学習（Q-Learning、DQN）
- Kaggle コンペでの実践蓄積

---

## 🔧 よくある質問

### Q1: Python の経験がなくても学べますか？
**A:** はい。第1週で Python 基礎を丁寧に扱います。C/C++ など他言語経験があれば、比較しながら習得しやすい構成です。

### Q2: GPU がなくても深層学習を学べますか？
**A:** 可能です。本プロジェクトのサンプルは CPU でも動作します。必要に応じて Google Colab や Kaggle Notebook の無料 GPU を活用してください。

### Q3: 毎週の時間を確保しにくい場合は？
**A:** 16 週や 24 週へ伸ばしても構いません。まずはコア概念を押さえ、2 周目で細部を補う進め方も有効です。

### Q4: 学習効果はどう確認すればよいですか？
**A:** 各週のプロジェクトを完了し、GitHub にコードを蓄積し、技術ブログや週報で要点を言語化するのが効果的です。

### Q5: 学習後に目指せることは？
**A:** 初級データアナリスト、機械学習エンジニアのインターン、AI アルゴリズム実習、または次段階の学習に十分つながります。

---

## 📚 おすすめリソース

### オンライン教材
- [Kaggle Learn](https://www.kaggle.com/learn)
- [PyTorch 公式チュートリアル](https://pytorch.org/tutorials/)
- [Scikit-Learn ドキュメント](https://scikit-learn.org/)

### 実践プラットフォーム
- [Kaggle](https://www.kaggle.com/)
- [天池](https://tianchi.aliyun.com/)
- [和鲸社区](https://www.heywhale.com/)

### 書籍
- 『Python Programming』
- 『Python for Data Analysis』
- 『機械学習実戦』
- 『深層学習入門』

---

## 🤝 貢献とフィードバック

Issue と Pull Request を歓迎します。

- **Bug 修正**: コード問題を見つけたら Issue を作成
- **機能強化**: 新規サンプル追加や既存コード改善
- **ドキュメント改善**: README や注釈の補強

### フィードバック窓口
- GitHub Issues: 技術的な質問やバグ報告
- Discussions: 学習交流や経験共有

---

## 📄 ライセンス

本プロジェクトは MIT ライセンスです。詳細は [LICENSE](LICENSE) を参照してください。

---

## 🎉 謝辞

- NumPy、Pandas、Matplotlib、Seaborn
- Scikit-Learn
- PyTorch
- Jupyter / VS Code

---

## 📞 連絡先

- **プロジェクト URL**: [GitHub仓库链接]
- **作者**: AI Learning Team
- **バージョン**: v4.0
- **更新日**: 2026-05-25

---

**🚀 さっそく AI 学習を始めましょう。**

```bash
python run_beginner_examples.py
```

---

*「AI を学ぶのに最適な時期は今です。ゼロから一緒に、AI の中核スキルを体系的に身につけましょう。」*