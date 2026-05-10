# RAG (Retrieval-Augmented Generation) 解説

> 本ドキュメントは `sample_rag_application.ipynb` の補足資料です。
> ノートブックで実装したシンプルな RAG の背景知識・設計上の考え方・改善方針を整理します。

## 目次

1. [RAG とは何か](#1-rag-とは何か)
2. [RAG の全体アーキテクチャ](#2-rag-の全体アーキテクチャ)
3. [各コンポーネントの役割（詳細）](#3-各コンポーネントの役割詳細)
4. [検索方式の詳細](#4-検索方式の詳細)
5. [チャンク戦略の深掘り](#5-チャンク戦略の深掘り)
6. [評価とモニタリング](#6-評価とモニタリング)
7. [元コードからの改善点](#7-元コードからの改善点)
8. [発展: Agentic RAG / マルチモーダル](#8-発展-agentic-rag--マルチモーダル)
9. [参考: ミニマル RAG の完全コード](#9-参考-ミニマル-rag-の完全コード)
10. [まとめ](#10-まとめ)

---

## 1. RAG とは何か

**RAG (Retrieval-Augmented Generation / 検索拡張生成)** は、
LLM が回答を生成する際に、**外部の知識ソースから関連情報を検索して文脈 (context) として与える** 手法です。

### 従来の LLM と RAG の比較

| | 従来の LLM のみ | RAG |
|---|---|---|
| 知識の出典 | 学習時のデータに固定 | 外部ドキュメントから動的に取得 |
| 幻覚 (Hallucination) | 起きやすい | 文脈に基づくため抑制できる |
| 最新情報への対応 | 再学習が必要 | ドキュメントを差し替えるだけ |
| 根拠の提示 | 困難 | 検索元のドキュメントを明示できる |
| 社内・個別ドメイン知識 | 対応できない | 自社ドキュメントを登録して対応 |
| コスト | ファインチューニングは高コスト | 追加学習不要、インデックス更新のみ |
| セキュリティ | モデルに埋め込まれると削除困難 | ドキュメント単位で削除・権限管理可能 |

### RAG の基本的な動作原理

1. **事前処理 (Indexing)**: ドキュメントをチャンクに分割し、Embedding モデルでベクトル化して Vector Store に保存
2. **検索 (Retrieval)**: ユーザーのクエリもベクトル化し、Vector Store から意味的に近いチャンクを取得
3. **拡張 (Augmentation)**: 取得したチャンクを LLM のプロンプトに「文脈」として埋め込む
4. **生成 (Generation)**: LLM は与えられた文脈を踏まえて回答を生成

### いつ RAG を使うか

| 用途 | 具体例 |
|---|---|
| 社内文書への Q&A | 就業規則、設計書、議事録 |
| 技術ドキュメントの検索 | API リファレンス、運用手順書 |
| FAQ 自動化 | カスタマーサポート、ヘルプデスク |
| ITSM | ServiceNow インシデントの解決支援、過去事例検索 |
| コンプライアンス | 法令・規程に基づく判断支援 |
| 根拠付き回答が必須の領域 | 医療、法務、金融 |

### RAG が向かない場面
- 純粋な推論タスク（数学問題、コーディング）— LLM 単体の方が良い
- 検索対象がない創作タスク — LLM 単体で十分
- リアルタイム性が極端に要求される用途 — Embedding 計算のレイテンシがボトルネックになることがある

---

## 2. RAG の全体アーキテクチャ

RAG は大きく **Indexing (事前処理)** と **Retrieval & Generation (実行時)** の 2 フェーズに分かれます。

### Indexing フェーズ (事前処理: 1 回 or 定期的に)

```
 ドキュメント     Loader        Splitter      Embeddings    Vector Store
 ─────────  ─────────────   ───────────   ──────────── ───────────────
  .md / .pdf    GitLoader     Recursive     OpenAI       Chroma
  /Web / DB → PyPDFLoader → CharSplitter → Embeddings → (index 保存)
  /Notion     WebLoader    (chunk化)      (ベクトル化)
```

**ポイント**:
- このフェーズは **時間がかかる処理** (Embedding の API 呼び出しなど)
- **データ更新時に再実行** する必要がある
- 差分更新の仕組みを入れることで効率化可能

### Retrieval & Generation フェーズ (実行時: 1 リクエストごと)

```
 ユーザー質問
    ↓
 [1] Embedding でベクトル化
    ↓
 [2] Vector Store で類似検索 → 関連チャンクを取得 (Retrieval)
    ↓
 [3] プロンプトに {context} として埋め込む (Augmentation)
    ↓
 [4] LLM が回答生成 (Generation)
    ↓
 回答 (+ 参照元)
```

**ポイント**:
- このフェーズは **レイテンシが重要**
- 各ステップを並列化・キャッシュすることで高速化できる
- 検索段階 (2) の品質が最終的な回答品質に最も大きく影響する

---

## 3. 各コンポーネントの役割（詳細）

### 3.1 Document Loader

ドキュメントを読み込んで `Document` オブジェクト (`page_content` + `metadata`) に変換する。

#### 代表的なローダー

| ローダー | 用途 | 備考 |
|---|---|---|
| `GitLoader` | GitHub / GitLab リポジトリ | `file_filter` で拡張子フィルタ可 |
| `PyPDFLoader` / `PyMuPDFLoader` | PDF | ページ単位で分割。画像抽出は別途 |
| `UnstructuredPDFLoader` | PDF | テーブル・レイアウト保持。精度高いが遅い |
| `WebBaseLoader` | Web ページ | BeautifulSoup ベース |
| `RecursiveUrlLoader` | Web サイト丸ごと | 再帰的にクロール |
| `NotionDBLoader` / `NotionDirectoryLoader` | Notion | DB 単位 or エクスポート |
| `ConfluenceLoader` | Confluence | スペース単位で取得 |
| `SlackDirectoryLoader` | Slack | エクスポートファイル |
| `GoogleDriveLoader` | Google Drive | OAuth 認証必要 |
| `S3DirectoryLoader` | AWS S3 | バケット単位 |
| `UnstructuredFileLoader` | 汎用 | Office 文書・HTML・メール等を統一 API で処理 |
| `DirectoryLoader` | ローカルディレクトリ | 他の Loader をラップ |

#### Document オブジェクトの構造

```python
Document(
    page_content="LangChain は LLM アプリ開発のための...",  # 本文
    metadata={
        "source": "README.md",     # ファイルパス
        "file_type": ".md",
        "page": 1,                  # PDF なら ページ番号
        "row": 42,                  # CSV なら 行番号
        # カスタムメタデータも自由に追加可能
        "department": "engineering",
        "updated_at": "2026-01-15",
    }
)
```

#### メタデータ設計のコツ
メタデータは **検索時のフィルタリング** に使えるため、以下のような情報を意識的に付与する:
- **source**: ファイルパス・URL（引用元表示に必須）
- **部署・チーム**: 権限制御に使う
- **日付**: 古い情報を除外する / 新しい情報を優先する
- **カテゴリ・タグ**: 検索スコープの絞り込み
- **言語**: 多言語ドキュメントの場合

---

### 3.2 Text Splitter

ドキュメントを検索・LLM 処理に適したサイズの **チャンク** に分割する。

#### 主要な Splitter の比較

| Splitter | 特徴 | 向いている用途 |
|---|---|---|
| `CharacterTextSplitter` | 単一区切り文字で分割 | 構造が単純なテキスト |
| `RecursiveCharacterTextSplitter` ⭐ | 段落 → 行 → 文 の順で再帰的に分割 | 一般的な文書 (RAG の標準) |
| `MarkdownHeaderTextSplitter` | Markdown の見出し単位で分割 | 構造化された Markdown |
| `MarkdownTextSplitter` | Markdown 構文を認識して分割 | Markdown ドキュメント |
| `HTMLHeaderTextSplitter` | HTML の見出しタグで分割 | Web ページ |
| `PythonCodeTextSplitter` | Python の構文を認識 | ソースコード |
| `TokenTextSplitter` | トークン数ベース | LLM のコンテキスト長を厳密に管理 |
| `SemanticChunker` | 意味の切れ目で分割 (Embedding ベース) | 文脈保持が最重要な場合 |

#### `RecursiveCharacterTextSplitter` が標準的な理由

`separators` パラメータに優先順位付きの区切り文字リストを渡し、以下のように再帰的に試します:

```
1. まず "\n\n" (段落) で分割を試みる
2. 分割後のチャンクが chunk_size より大きければ、"\n" (行) で再分割
3. まだ大きければ "。" (文末) で再分割
4. それでも大きければ " " (単語) で再分割
5. 最後は文字単位
```

これにより **意味の境界を尊重しながら** サイズ制約を守れます。

#### `SemanticChunker` (新しい選択肢)

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
)
```

- 文ごとに Embedding を計算し、**意味が大きく変わる箇所で分割**
- 精度は高いが Embedding 計算コストがかかる
- 長文・複数トピックが混在するドキュメントに特に有効

詳細は [第5章](#5-チャンク戦略の深掘り) を参照。

---

### 3.3 Embeddings

テキストを **固定長の数値ベクトル** に変換するモデル。意味が近いテキストはベクトル空間上でも近い位置に配置される。

#### Embedding モデルの選択肢

| モデル | 次元数 | 特徴 | コスト |
|---|---|---|---|
| `text-embedding-3-small` ⭐ | 1536 | OpenAI。軽量・高速・標準 | 低 |
| `text-embedding-3-large` | 3072 | OpenAI。高精度 | 中 |
| `text-embedding-ada-002` | 1536 | OpenAI 旧モデル (非推奨) | 低 |
| `voyage-3` | 1024 | Voyage AI。多言語高精度 | 中 |
| `cohere-embed-multilingual-v3` | 1024 | Cohere。100言語対応 | 中 |
| `bge-m3` | 1024 | OSS (BAAI)。高精度・日本語対応 | 無料 (自前ホスト) |
| `multilingual-e5-large` | 1024 | OSS。多言語対応 | 無料 (自前ホスト) |
| `nomic-embed-text` | 768 | OSS。Ollama で動く | 無料 (自前ホスト) |

#### 日本語タスクでの推奨
- **プロトタイプ・一般用途**: `text-embedding-3-small` で十分
- **精度重視**: `text-embedding-3-large` または `voyage-3`
- **オンプレ要件あり**: `bge-m3`, `multilingual-e5-large`
- **Ollama 等ローカル LLM と組み合わせ**: `nomic-embed-text`, `bge-m3`

#### 次元数と精度のトレードオフ
- 次元数が大きい = 情報量が多い = 精度が上がる可能性
- ただし検索速度・ストレージコストも増える
- `text-embedding-3-large` は dimensions パラメータで縮約可能 (Matryoshka Embedding)

```python
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,  # 3072 次元から 1024 次元に縮約
)
```

#### 重要な原則
- **Indexing と Retrieval で同じ Embedding モデルを使う** (必須)
- **モデルを変えたら Vector Store を作り直す** (次元数が変わる場合は特に)

---

### 3.4 Vector Store

ベクトル化したチャンクを保存し、**類似度検索** を可能にするデータベース。

#### 主要な Vector Store の比較

| Vector Store | 分類 | 特徴 | 向いている規模 |
|---|---|---|---|
| **Chroma** ⭐ | OSS / ローカル | 軽量・SQLite ベース | ~100万件 |
| **FAISS** | OSS / ライブラリ | Meta 製。インメモリで高速 | ~1000万件 |
| **Pinecone** | マネージド SaaS | スケール容易、メタデータフィルタ強力 | 無制限 |
| **Weaviate** | OSS / マネージド | GraphQL API、多機能 | 大規模 |
| **Qdrant** | OSS / マネージド | Rust 実装、高速、フィルタが柔軟 | 大規模 |
| **Milvus** | OSS / マネージド | 大規模向け、クラウドネイティブ | 超大規模 |
| **pgvector** | OSS (PostgreSQL拡張) | 既存 Postgres を使い回せる | 中規模 |
| **OpenSearch** | OSS | 全文検索 + ベクトル検索。ハイブリッド容易 | 大規模 |
| **Elasticsearch** | OSS / 商用 | OpenSearch と同様 | 大規模 |
| **Azure AI Search** | マネージド | Azure 環境で統合容易 | 大規模 |
| **AWS OpenSearch Service** | マネージド | AWS 環境向け | 大規模 |

#### 選定の指針

| 要件 | 推奨 |
|---|---|
| 学習・プロトタイプ | Chroma |
| ~100 万件、ローカル完結 | Chroma / FAISS |
| 既存 Postgres あり | pgvector |
| 既存 OpenSearch / Elasticsearch あり | 同じものを使う (ハイブリッド検索が楽) |
| クラウドで楽にスケールしたい | Pinecone |
| OSS でスケールさせたい | Qdrant / Weaviate / Milvus |
| Azure 環境 | Azure AI Search |
| AWS 環境 | OpenSearch Service または pgvector (RDS) |

#### 永続化 (Chroma の例)

```python
# 保存
db = Chroma.from_documents(
    docs, embeddings,
    persist_directory="./chroma_db",
)

# 再読み込み
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
```

---

### 3.5 Retriever

Vector Store を「検索インターフェース」として抽象化したもの。
LangChain のチェーンに組み込めるよう `Runnable` 互換の API を提供する。

```python
retriever = db.as_retriever(
    search_type="similarity",      # "similarity" / "mmr" / "similarity_score_threshold"
    search_kwargs={
        "k": 4,                    # 返す件数
        "score_threshold": 0.7,    # similarity_score_threshold 時のしきい値
        "filter": {"source": "foo.md"},  # メタデータフィルタ
    },
)
```

検索方式のバリエーションは [第4章](#4-検索方式の詳細) で詳述します。

---

### 3.6 LLM + Prompt

検索された context を埋め込んだプロンプトを LLM に与えて回答生成。

#### プロンプトの基本形

```python
prompt = ChatPromptTemplate.from_template("""
以下の文脈だけを踏まえて質問に解答してください。
文脈に答えが含まれていない場合は「情報が見つかりません」と回答してください。

文脈:
{context}

質問: {question}
""")
```

#### プロンプト設計のコツ

| コツ | 理由 |
|---|---|
| 「文脈だけに基づいて」と明示 | LLM が訓練データから勝手に回答するのを抑制 |
| 「情報がない場合は…」と明示 | 無理な回答を防ぐ (幻覚抑制) |
| 回答形式を指定 (箇条書き等) | 出力の一貫性 |
| 引用を求める (`[1]` など) | 透明性の向上 |
| 言語を指定 (「日本語で」) | 文脈が英語でも回答を日本語にできる |

#### より堅牢なプロンプト例

```python
prompt = ChatPromptTemplate.from_template("""
あなたは正確で信頼できる回答を提供する専門アシスタントです。

# 指示
- 以下の【文脈】だけを根拠に質問に答えてください
- 【文脈】に答えが含まれない場合は「提供された情報からは回答できません」と述べ、推測で答えないでください
- 回答の根拠として使った部分のソース ([1], [2] 等) を文末に明記してください
- 回答は日本語で、簡潔かつ丁寧に

# 文脈
{context}

# 質問
{question}

# 回答
""")
```

#### LLM の選択

| モデル | 特徴 | RAG での用途 |
|---|---|---|
| `gpt-4o-mini` ⭐ | 高速・低コスト | 標準的な RAG |
| `gpt-4o` | 高精度 | 複雑な推論が必要な RAG |
| `claude-opus-4-6` | 最高精度 | 重要な業務用途 |
| `claude-haiku-4-5` | 高速・低コスト | 大量トラフィック |
| Gemini 2.5 | 長コンテキスト | 大量の context を処理 |
| ローカル LLM (Llama, Qwen) | オンプレ・無料 | セキュリティ要件が厳しい場合 |

#### `temperature` の設定
- RAG では **`temperature=0`** が基本 (回答を決定的・一貫性のあるものにする)
- 創造性が必要な用途 (要約の言い換え等) では `0.3〜0.7`

---

### 3.7 LCEL (LangChain Expression Language)

`|` (パイプ) 演算子で Runnable を繋げて宣言的にパイプラインを書く方法。

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

#### LCEL の利点
- **`|` の意味**: 左側の出力が右側の入力になる (Unix パイプと同じ)
- **`RunnablePassthrough()`**: 入力をそのまま通す Runnable
- **型安全**: 各ステップの入出力型が接続時にチェックされる
- **統一 API**: 全てのチェーンで以下が使える

| メソッド | 用途 |
|---|---|
| `.invoke(input)` | 同期実行 |
| `.ainvoke(input)` | 非同期実行 |
| `.stream(input)` | ストリーミング (トークン単位で返す) |
| `.astream(input)` | 非同期ストリーミング |
| `.batch([inputs])` | バッチ実行 (並列化) |
| `.abatch([inputs])` | 非同期バッチ |

#### 主要な Runnable プリミティブ

| プリミティブ | 役割 |
|---|---|
| `RunnablePassthrough` | 入力をそのまま通す |
| `RunnableLambda` | 任意の関数を Runnable に変換 |
| `RunnableParallel` | 複数の Runnable を並列実行して dict を返す |
| `RunnableBranch` | 条件分岐 |
| `RunnableWithFallbacks` | 失敗時のフォールバック |
| `RunnablePassthrough.assign(...)` | 既存の dict に追加項目を計算 |

#### RunnableParallel の活用例 (引用元表示)

```python
from langchain_core.runnables import RunnableParallel

rag_chain_with_sources = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    .assign(answer=prompt | model | StrOutputParser())
)
# 返り値: {"context": [Document, ...], "question": "...", "answer": "..."}
```

---

## 4. 検索方式の詳細

ここが **RAG の精度を左右する最も重要なパート** です。以下、レベルを上げながら順に説明します。

### 4.1 ベクトル類似度の計算方式 (Vector Store 内部の話)

ベクトル化したチャンクとクエリの「近さ」をどう測るかの話。Vector Store の内部でどう距離計算するかのレベル。

| 方式 | 計算内容 | 特徴 |
|---|---|---|
| **コサイン類似度** ⭐ | ベクトル間の角度の cos | ベクトルの長さに影響されない。**最も一般的** |
| 内積 (dot product) | ベクトルの内積 | 正規化済みベクトルならコサインと等価。計算が速い |
| L2 距離 (ユークリッド距離) | ベクトル間の直線距離 | 長さの違いも考慮する。画像系で使われることも |

**実務上はほぼコサイン類似度一択**。OpenAI の Embedding は正規化されているため、コサインでも内積でも結果は同じです。Chroma / FAISS / Pinecone などはデフォルトでコサイン類似度になっています。

### 4.2 検索戦略 (Retrieval Strategy)

こちらが実務上の関心事で、「どのチャンクを LLM に渡すか」を決める戦略レベルの話です。

#### 4.2.1 Similarity Search (純粋なベクトル検索) ⭐ 最も基本

```python
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)
```

- クエリと最も類似度が高い k 件を返す
- **最も広く使われる**。まずこれから始めるのが定石
- 弱点: **似たような内容のチャンクが上位を独占する**ことがある (例: 同じ段落から分割された隣接チャンクが全部ヒット)

#### 4.2.2 MMR (Maximum Marginal Relevance) ⭐ 多様性重視

```python
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
)
```

- 「クエリとの類似度」と「既に選んだ結果との非類似度」を両立させて選ぶ
- `fetch_k=20` で候補を広めに取り、そこから多様な `k=5` 件を選択
- `lambda_mult`: 1.0 に近いほど類似度重視、0.0 に近いほど多様性重視 (通常 0.5〜0.7)
- **重複チャンクで LLM のコンテキストを無駄にしたくないとき**に有効
- 使いどころ: 要約系タスク、複数観点からの回答が欲しい場合

**MMR アルゴリズムの直感**:
```
1. クエリに最も類似したチャンクを1件選ぶ
2. 次の候補を選ぶ時、「クエリとの類似度 × λ - 既選択との類似度 × (1-λ)」が最大のものを選ぶ
3. k件になるまで2を繰り返す
```

#### 4.2.3 Similarity Score Threshold (しきい値検索)

```python
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.75},
)
```

- スコアが一定以上のチャンクだけを返す
- 関連性の低い結果を LLM に渡さないことで、幻覚を抑制
- **「該当なし」というケースがある用途**で有効 (FAQ で該当なしの時に無理に答えさせたくない等)

#### 4.2.4 ハイブリッド検索 (ベクトル + キーワード) ⭐⭐ 精度重視の定番

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(docs)
bm25.k = 5
vector_retriever = db.as_retriever(search_kwargs={"k": 5})

retriever = EnsembleRetriever(
    retrievers=[bm25, vector_retriever],
    weights=[0.5, 0.5],
)
```

- **BM25 (キーワード検索) とベクトル検索の両方を走らせて結果を統合**
- ベクトル検索の弱点を補う:
  - **固有名詞・型番・エラーコード・API 名**のような「単語そのもの」が重要な情報に強い
  - 例: `"KMS-1234"` というエラーコード検索、`"ec2:DescribeInstances"` のような IAM アクション名
- 統合方式は **RRF (Reciprocal Rank Fusion)** が内部で使われる
- **実務では非常に効果的**。技術ドキュメント・社内システムマニュアル・ServiceNow のナレッジ記事などに特に向く

**BM25 とは**:
- 古典的なキーワード検索アルゴリズム (TF-IDF の発展形)
- 「文書中の単語頻度」と「その単語がどれだけ珍しいか」でスコア計算
- 単語の **完全一致** に強い

**なぜ組み合わせると強いか**:
- ベクトル検索: 意味的な類似性に強い (「車」で検索して「自動車」がヒット)
- キーワード検索: 固有名詞・記号列に強い (「EC2」「gpt-4o-mini」等)
- → **両方の良いとこ取り**

#### 4.2.5 リランキング (Re-ranking) ⭐⭐⭐ 精度最重視ならこれ

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

base_retriever = db.as_retriever(search_kwargs={"k": 20})  # 広めに取る
reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)

retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever,
)
```

- **2 段構えの検索**:
  1. 1 段目: Retriever が TOP-20 など広めに候補を取る (**Recall 重視**)
  2. 2 段目: Cross-Encoder や専用 API が各候補とクエリのペアを再評価して上位 5 件を選ぶ (**Precision 重視**)
- Cross-Encoder は「クエリとチャンクを両方同時に読んで」スコアを出すため、Embedding の類似度より **本質的に精度が高い**

**なぜ Cross-Encoder の方が精度が高いか**:
- Embedding (Bi-Encoder): クエリとドキュメントを別々にベクトル化 → 類似度計算
- Cross-Encoder: クエリとドキュメントを **連結して** モデルに入力 → スコアを直接出力
- Cross-Encoder はクエリとドキュメントの **相互作用** を考慮できるため精度が高い
- ただし全ドキュメントに適用するのは計算量的に無理 → Retriever で絞った後の再評価に使う

**代表的な選択肢**:

| Reranker | 種類 | 特徴 |
|---|---|---|
| **Cohere Rerank** ⭐ | API | 日本語含む多言語対応、業界標準 |
| **BGE-reranker** | OSS | ローカル実行可。`bge-reranker-v2-m3` が多言語対応 |
| **Voyage AI rerank** | API | 高精度で注目されている |
| **Jina Reranker** | API / OSS | `jina-reranker-v2-base-multilingual` |
| **Cross-Encoder (sentence-transformers)** | OSS | 自前でファインチューニング可能 |

**デメリット**: レイテンシとコストが増える。ただし精度向上が非常に大きいので、**本番 RAG の定番構成**

ServiceNow のインシデント検索や CMDB 検索のような「正確性が重要」な用途に特に有効。

#### 4.2.6 Multi-Query Retrieval (クエリ展開)

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(),
    llm=ChatOpenAI(model="gpt-4o-mini"),
)
```

- LLM にユーザーのクエリを **複数の言い換え** に展開させ、それぞれで検索して結果を統合
- 例: 「LangChain を使う理由は?」 → 「LangChain の利点」「LangChain のメリット」「なぜ LangChain を選ぶのか」
- **ユーザーのクエリが曖昧・短い**ときに有効
- コスト増 (LLM 呼び出しが増える) に注意

#### 4.2.7 HyDE (Hypothetical Document Embeddings)

```python
from langchain.chains import HypotheticalDocumentEmbedder
```

- LLM に「この質問への仮想的な回答」を生成させ、**その回答のベクトル**で検索する
- 質問と回答では表現が違うため、質問ベクトル同士より「仮想回答ベクトル」の方が本物の回答ドキュメントと近いことがある
- 効果はユースケースによる。**技術的な質問 + 解説ドキュメントの組み合わせ**で特に効く
- 欠点: LLM 呼び出しのコストとレイテンシ

#### 4.2.8 Parent Document Retriever (親子チャンク戦略)

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=db,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(raw_docs)
```

- **小さいチャンクで検索精度を出し、親の大きなチャンクを LLM に渡す**
- 「検索の粒度」と「文脈の粒度」を分離する
- 例: 400 文字で検索 → ヒットしたチャンクの親 (2000 文字) を LLM に渡す
- **長文の文脈が必要な Q&A に有効**

#### 4.2.9 メタデータフィルタ (ハード制約)

```python
retriever = db.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source": {"$eq": "aws_docs/cloudwatch.md"}},
    }
)
```

- ベクトル検索の前段で「特定の部署のドキュメントだけ」「特定の日付以降」などで絞り込む
- **権限管理、テナント分離、スコープ限定** に必須
- マルチアカウント・マルチテナントの社内 RAG では事実上必須の機能

**フィルタ演算子の例** (Vector Store により記法は異なる):
```python
# 特定ソース
{"source": "manual.md"}

# OR 条件
{"$or": [{"source": "a.md"}, {"source": "b.md"}]}

# 比較演算子
{"year": {"$gte": 2024}}

# 複合条件
{"$and": [{"dept": "eng"}, {"year": {"$gte": 2024}}]}
```

#### 4.2.10 Self-Query Retriever (自然言語からフィルタを生成)

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever

retriever = SelfQueryRetriever.from_llm(
    llm,
    db,
    document_content_description="技術ドキュメント",
    metadata_field_info=[
        AttributeInfo(name="year", description="発行年", type="integer"),
        AttributeInfo(name="dept", description="部署", type="string"),
    ],
)
# "2024年以降の技術ドキュメントから…" → 自動的に filter={"year": {"$gte": 2024}} を生成
```

- ユーザーの自然言語から **フィルタ条件を自動抽出**
- 「2024年以降の」「エンジニアリング部の」などの制約を LLM が理解してメタデータ検索に変換
- ユーザーが SQL ライクなクエリを書かなくて良くなる

#### 4.2.11 Contextual Compression (文脈圧縮)

```python
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=db.as_retriever(),
)
```

- 検索結果の各チャンクから、**クエリに関連する部分だけを LLM で抽出**
- LLM のコンテキストを節約 + ノイズ削減
- コスト増 (各チャンクで LLM 呼び出し) と引き換え

### 4.3 実務での推奨構成

用途別に現実的な選択肢を整理します。

#### レベル 1: プロトタイピング / 学習用途
```
Similarity Search (k=4)
```
- LangChain の `db.as_retriever()` のデフォルト
- ノートブックで動かしている今の構成

#### レベル 2: 社内 RAG の実用最小構成 ⭐ 多くの案件はここ
```
Hybrid (BM25 + Vector) + メタデータフィルタ
```
- 固有名詞に強い + 権限制御できる
- 追加コストがほぼゼロ (BM25 はローカル計算)

#### レベル 3: 本番プロダクション品質 ⭐⭐ 実務の定番
```
Hybrid Retrieval (TOP-20) → Cohere Rerank (TOP-5) → LLM
```
- 現在の商用 RAG プロダクトの標準的な構成
- 精度向上が明確に体感できる
- Cohere Rerank の API コストは許容範囲 (1000 クエリあたり数ドル程度)

#### レベル 4: 精度を極限まで追求
```
Multi-Query or HyDE でクエリ拡張
  ↓
Hybrid Retrieval (TOP-30)
  ↓
Cross-Encoder Reranker (TOP-5)
  ↓
Contextual Compression (無関係部分を除去)
  ↓
LLM
```
- 法務・医療・金融など正確性が致命的に重要な領域向け
- レイテンシとコストが大きいので用途を選ぶ

### 4.4 使い分けの判断基準

| ユースケース | 推奨構成 |
|---|---|
| 学習・デモ | Similarity Search |
| 要約・複数観点の回答 | MMR |
| 技術ドキュメント・マニュアル (型番や API 名が重要) | **Hybrid (BM25 + Vector)** |
| 社内 FAQ・ヘルプデスク | **Hybrid + Rerank** |
| ITSM (ServiceNow インシデント検索など) | **Hybrid + Rerank + メタデータフィルタ** |
| 長文ドキュメント (法令・契約書) | Parent Document Retriever + Rerank |
| 曖昧な自然言語クエリが多い | Multi-Query + Hybrid |
| 権限管理が必要なマルチテナント | メタデータフィルタ (必須) + Hybrid |
| 日付・カテゴリで絞りたいことが多い | Self-Query Retriever |

### 4.5 最も使われる / 最も精度が高い方式

**最もよく使われる方式**
1. **Similarity Search (コサイン類似度ベース)** — チュートリアル・プロトタイプでは圧倒的
2. **Hybrid Search (BM25 + Vector)** — 実務プロダクトの第一選択
3. **MMR** — 多様性が必要な場面で広く採用

**最も精度が高い方式**
1. **Hybrid + Cross-Encoder Reranker** — 現在の RAG のデファクトスタンダード
   - 「広く取って、賢く絞る」の 2 段構成が効く
   - Cohere Rerank や BGE-reranker が定番
2. **上記にクエリ拡張 (Multi-Query / HyDE) を追加** — さらに Recall を上げたい場合
3. **Parent Document Retriever** — 長文の文脈保持が必要な場合の決定打

**覚えておくべき原則**:
検索精度を上げる最もコスパの良い 1 手は **「Reranker の導入」** です。Embedding モデルを高価なものに変えるより、Reranker を入れる方が効果が大きいケースが多いです。社内 RAG やエージェント用途の検索層を作るなら、**「Hybrid + Rerank」を基本形**として押さえておくと応用が効きます。

---

## 5. チャンク戦略の深掘り

検索精度は Retriever の工夫と並んで **チャンク戦略** に大きく依存します。

### 5.1 チャンクサイズの選び方

| サイズ | 文字数 | トークン数 (日本語目安) | 用途 |
|---|---|---|---|
| 小 | 200〜500 | 100〜250 | FAQ・定義型 Q&A。精密な検索 |
| 中 ⭐ | 500〜1500 | 250〜750 | 技術文書・ブログ。**RAG の標準** |
| 大 | 1500〜3000 | 750〜1500 | 長い文脈保持が必要な要約系 |
| 超大 | 3000〜 | 1500〜 | コンテキスト長に余裕がある場合のみ |

**トレードオフ**:
- 小さい: 検索が正確になるが、文脈が不足して LLM の回答が浅くなる
- 大きい: 文脈は豊富だが、1 チャンクに複数トピックが混在して検索精度が下がる

### 5.2 `chunk_overlap` の設計

- 一般的に `chunk_size` の **10〜20%**
- 目的: チャンク境界で重要な情報が分断されるのを防ぐ

```
チャンク1: "...モニタリングツールとしては CloudWatch が..."
チャンク2: "Agent をインストールして使うのが一般的です。..."
     ↑ ここで「CloudWatch Agent」という重要な用語が分断される
```

オーバーラップを設けると:
```
チャンク1: "...モニタリングツールとしては CloudWatch Agent を..."
チャンク2: "CloudWatch Agent をインストールして使うのが一般的..."
     ↑ どちらのチャンクでも「CloudWatch Agent」が読める
```

### 5.3 構造を尊重する分割

**Markdown 見出し分割**:
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
```

- 見出し情報がメタデータに保存される → 検索結果で「どの章か」が分かる
- 章単位で意味の境界を保てる

**コード分割**:
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=200,
)
```

- 関数・クラスの境界を尊重
- `Language.PYTHON`, `Language.JS`, `Language.JAVA` など多数対応

### 5.4 Semantic Chunking

```python
from langchain_experimental.text_splitter import SemanticChunker

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)
```

アルゴリズム:
1. 文ごとに分割
2. 隣接する文の Embedding 類似度を計算
3. 類似度が急落する箇所 (意味の変わり目) でチャンクを区切る

- **意味の境界で分割できる** ため、チャンク内の一貫性が高い
- 欠点: Embedding 計算のコストがかかる

### 5.5 メタデータの活用

チャンク分割時に以下のメタデータを埋めておくと、後の検索・フィルタで有用:

```python
for doc in docs:
    doc.metadata.update({
        "section": extract_section(doc.page_content),
        "created_at": get_file_mtime(doc.metadata["source"]),
        "language": detect_language(doc.page_content),
        "doc_type": classify_type(doc.metadata["source"]),  # "manual", "faq", "incident"
    })
```

---

## 6. 評価とモニタリング

RAG は「動いた」で終わらせず、**継続的な品質計測** が重要です。

### 6.1 評価の観点

RAG の品質は以下の 2 つに分解できます:

| 観点 | 指標 | 意味 |
|---|---|---|
| **検索品質** | Context Precision | 検索結果のうち本当に関連あるものの割合 |
| | Context Recall | 必要な情報が検索されている割合 |
| **生成品質** | Faithfulness | 回答が context に忠実か (幻覚していないか) |
| | Answer Relevancy | 回答が質問に答えているか |
| | Answer Correctness | 正解と一致しているか (正解データがある場合) |

### 6.2 Ragas による自動評価

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

dataset = Dataset.from_dict({
    "question": [...],
    "answer": [...],           # RAG の回答
    "contexts": [...],         # 検索結果
    "ground_truth": [...],     # 正解 (あれば)
})

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)
```

- 評価自体にも LLM を使う (LLM-as-a-Judge)
- 小規模な評価セット (50〜200 件) から始めて、改善のベースラインを作る

### 6.3 LangSmith によるトレースとモニタリング

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "..."
os.environ["LANGCHAIN_PROJECT"] = "my-rag-project"

# 以降、全てのチェーン実行が自動的にトレースされる
chain.invoke("質問")
```

- 各ステップの入出力、レイテンシ、トークン数を可視化
- 失敗ケースの根本原因を追える
- 本番運用の監視・改善に必須

### 6.4 評価データセットの作り方

1. **実ユーザーのクエリを収集** (最優先)
2. **LLM で自動生成** (初期ブートストラップ)
   ```python
   # ドキュメントから「質問と回答のペア」を LLM に生成させる
   ```
3. **ドメインエキスパートに作ってもらう** (最高品質だが高コスト)

評価セットは **継続的に拡充** する。特に「間違えたケース」を蓄積することで、回帰テスト化できる。

---

## 7. 元コードからの改善点

ノートブック (`sample_rag_application.ipynb`) の再作成にあたって行った主な改善:

### 7.1 Splitter を `RecursiveCharacterTextSplitter` に変更
- **理由**: 段落・文の境界を尊重する分割の方が、文脈が途切れにくく検索精度が上がる
- Before: `CharacterTextSplitter(chunk_size=1455, chunk_overlap=0)`
- After: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=[...])`

### 7.2 `chunk_overlap` の追加
- **理由**: 0 だとチャンク境界で情報が分断される。200 文字の重複で文脈連続性を確保。

### 7.3 pip install の整理
- **理由**: 元コードでは `opentelemetry` の細かなバージョン固定があったが、現行の `langchain-chroma` では不要
- 全依存ライブラリを冒頭の 1 セルにまとめて見通しを改善

### 7.4 API キー未設定時のエラーチェック
- `userdata.get()` が `None` を返した場合に明示的なエラーメッセージを出す

### 7.5 ベクトル全出力の廃止
- Before: `print(vector)` で 1536 次元すべて出力
- After: 次元数と先頭 5 要素のみ表示

### 7.6 `ChatOpenAI` の引数を `model=` に統一
- `model_name=` は古い書き方 (alias として残っているが、現行ドキュメントでは `model=` を使う)

### 7.7 引用元表示セルの追加
- `RunnableParallel` を使って「回答 + 参照ドキュメント」を両方返すパターンを追加
- RAG の大きな利点である **根拠の追跡可能性** を実装例として示した

### 7.8 全体像を最初に提示
- ノートブック冒頭に RAG の全体フロー図と、各ステップでやることを明示
- 各セクションに表形式の比較や指針を追加

---

## 8. 発展: Agentic RAG / マルチモーダル

### 8.1 Agentic RAG

シンプルな RAG は「1 回の検索 → 1 回の生成」ですが、複雑なクエリでは不十分。
**LangGraph** の `StateGraph` で以下のような制御フローを実装できます。

```
[質問]
  ↓
[Query Analyzer] — 質問を分解・書き換え
  ↓
[Retrieve] — 検索実行
  ↓
[Grade Documents] — 検索結果の関連性を LLM で評価
  ↓     ↓
  OK    NG → [Rewrite Query] → Retrieve に戻る
  ↓
[Generate]
  ↓
[Check Hallucination] — 回答が context に基づいているか検証
  ↓     ↓
  OK    NG → Generate に戻る or 検索しなおし
  ↓
[回答]
```

#### Agentic RAG の代表的なパターン

| パターン | 概要 |
|---|---|
| **Self-RAG** | LLM 自身が「検索が必要か」「検索結果が十分か」を判断 |
| **Corrective RAG (CRAG)** | 検索結果の質を評価し、不十分なら Web 検索で補完 |
| **Adaptive RAG** | クエリの複雑さに応じて検索戦略を切り替え |
| **GraphRAG (Microsoft)** | ドキュメントをナレッジグラフ化して検索 |

### 8.2 マルチモーダル RAG

画像・表・チャートを含むドキュメントへの対応:

```python
# 画像を LLM (GPT-4o, Claude) で説明させてテキスト化
# → そのテキストを Embedding して通常の RAG と同じフロー
```

または Multimodal Embedding (CLIP 系) を使って画像とテキストを同じベクトル空間に配置する方法も。

### 8.3 Structured Output

回答を JSON などの構造化形式で返す:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

structured_llm = model.with_structured_output(Answer)
```

API 連携・ワークフロー自動化で重要。

---

## 9. 参考: ミニマル RAG の完全コード

```python
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=lambda p: p.endswith(".md"),
)
raw_docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(raw_docs)

# 3. Embed + Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever()

# 4. Chain
prompt = ChatPromptTemplate.from_template(
    "以下の文脈だけを踏まえて質問に解答してください。\n\n"
    "文脈:\n{context}\n\n"
    "質問: {question}"
)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 5. Run
print(chain.invoke("LangChain を利用する理由は何ですか?"))
```

### 参考: ハイブリッド検索 + リランキングの完全コード (プロダクション品質)

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# ベクトル検索
vector_retriever = db.as_retriever(search_kwargs={"k": 10})

# キーワード検索
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 10

# ハイブリッド
ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],
)

# リランキング
reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)
retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble,
)

# あとは同じチェーン構成
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

---

## 10. まとめ

RAG は **「LLM に最新かつドメイン特化の知識を与える」** ための事実上の標準アプローチです。
本ノートブックでは最小構成で全体像を示しましたが、実プロジェクトでは以下を組み合わせて継続的に改善していくことが重要です。

### 改善の優先順位 (ROI が高い順)

1. **チャンク戦略の見直し** — サイズ・オーバーラップ・Splitter の選択
2. **Hybrid Search (BM25 + Vector) の導入** — 固有名詞・記号列に強くなる
3. **Reranker の導入** — 最もコスパの良い精度向上策
4. **引用元の明示** — ユーザーの信頼性向上
5. **メタデータフィルタ** — 権限管理・スコープ限定
6. **評価の自動化** (Ragas + LangSmith) — 継続的改善の基盤
7. **Agentic RAG への拡張** — 複雑なクエリへの対応

### 実務で押さえるべき「型」

| 段階 | 構成 |
|---|---|
| プロト | `Similarity Search (k=4)` |
| 実用最小 | `Hybrid (BM25 + Vector) + メタデータフィルタ` |
| **本番標準** ⭐ | `Hybrid → Rerank → LLM` |
| 精度最重視 | `Multi-Query → Hybrid → Rerank → Compression → LLM` |

最小構成から始め、評価をしながら必要に応じて段階的にレベルを上げていくのが実務的なアプローチです。
