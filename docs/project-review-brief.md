# Perplexity Lite 项目说明

## 1. 项目定位

本项目是一个面向搜索型问答场景的 AI 系统原型，目标不是做“纯聊天机器人”，而是实现一个更接近 Perplexity Lite 的 grounded QA 系统。

核心目标有四个：

1. 能从公开语料中检索到相关证据。
2. 能对候选结果进行重排，提高证据相关性。
3. 能基于证据生成带引用的答案，降低幻觉。
4. 能记录和评估系统在检索质量、延迟和引用有效性上的表现。

项目当前使用 HotpotQA 作为主要数据来源：

- `train` 用于构建检索语料库
- `dev_distractor` 用于评估和回归测试

整体上，这是一个“检索系统 + rerank + LLM 生成 + citation grounding”的组合式系统，而不是只调用一个大模型接口。

## 2. 系统架构

当前主链路如下：

`Query -> Query Normalization -> Dual Retrieval -> Fusion -> Dedup -> Rerank -> Top-k Evidence -> Answer Generation -> Used Citation Filtering`

其中关键模块包括：

- `src/data/`
  - 负责 HotpotQA 数据读取、chunk 导出、语料组织
- `src/indexing/`
  - 负责离线构建 BM25、dense vector、metadata 等索引
- `src/retrieval/`
  - 负责在线检索，包括：
    - BM25 稀疏检索
    - FAISS 向量检索
    - RRF 融合
    - title-level 去重
    - fast path / slow path 分流
- `src/rerank/`
  - 对融合后的候选结果做启发式重排
- `src/generation/`
  - 负责基于证据生成答案，并只保留答案实际使用到的 citations
- `src/pipeline/`
  - 负责把检索、重排、生成串成完整端到端流程
- `src/api/` 与 `app/`
  - 分别提供 FastAPI 接口和 Streamlit 演示页面

## 3. 核心实现原理

### 3.1 数据与语料构建

项目先将 HotpotQA 样本中的 `context` 展平成 paragraph-level chunks，形成统一语料。每个 chunk 包含：

- `chunk_id`
- `doc_id`
- `title`
- `text`
- `metadata`

这样做的原因是：

- 段落级 chunk 比句子级 chunk 更适合检索，语义更完整
- 后续仍然可以通过 citation 把答案映射回更细粒度证据

### 3.2 双路检索

在线检索采用双路召回：

1. `BM25` 稀疏检索
2. `FAISS` 向量检索

BM25 负责 lexical recall，适合精确词项命中；dense retrieval 负责 semantic recall，适合召回概念相近但词面不完全一致的内容。

向量检索当前已经接入真实 `FAISS`，dense backend 默认使用 transformer embedding：

- `sentence-transformers/all-MiniLM-L6-v2`

这意味着系统不再在查询时构建 corpus embedding，而是采用“离线建索引、在线只编码 query”的标准结构。

### 3.3 检索融合与去重

双路召回之后，不直接把分数相加，而是使用 `RRF (Reciprocal Rank Fusion)` 融合排序。

这样做的优点是：

- 不依赖 BM25 分数和向量相似度的量纲统一
- 在混合检索里通常比硬加权更稳

融合后再做 `title-level dedup`，限制同一 title 的重复 chunk 数量，避免 evidence 被同一文档反复占满。

### 3.4 Rerank

候选结果进入 reranker 后，会基于以下特征做二次排序：

- entity overlap
- title overlap
- attribute overlap
- comparison query coverage
- lexical mismatch penalty
- duplicate / near-duplicate penalty

这一步的目标不是训练一个复杂模型，而是在工程上用较轻量的规则，把明显噪声和重复候选压下去。

### 3.5 Answer Generation 与 Citation Grounding

生成层支持两种模式：

1. `fast_path`
   - 对常见结构化实体问题直接本地生成
   - 如 nationality、profession、older/younger、same nationality 等
2. `LLM / offline fallback`
   - 对更一般的问题，使用 DeepSeek 或离线 fallback 生成 grounded answer

生成时只允许基于检索证据回答，并通过 `[1] [2]` 这样的编号给出引用。

之后再做一次 `used citation filtering`：

- 只保留答案里真正引用到的 chunk
- 不把 top-k 全量 evidence 原样暴露给用户

## 4. 为什么要这样设计

这个架构的设计目标主要是解决普通 RAG demo 常见的三个问题：

### 4.1 只检索，不会排序

很多 RAG demo 能找一些相关段落，但排不准。项目引入 rerank 和 dedup，就是为了让 top evidence 更干净。

### 4.2 会回答，但不引用来源

如果没有 citation，系统就更像聊天，而不是搜索产品。本项目强调 citation grounding，是为了增强可解释性和可信度。

### 4.3 能跑，但没法评估

项目除了回答功能外，还实现了：

- retrieval / citation evaluation
- 固定 regression suite
- benchmark 脚本

这样每一轮优化都能验证：

- 质量是否变好
- 延迟是否下降
- 是否引入了回归

## 5. 主要优化历程

项目从空仓库逐步演化，大致经历了这些阶段：

### Phase 1: 数据与骨架

- 下载并验证 HotpotQA
- 建立 `train -> corpus / dev -> eval` 数据分工
- 搭建 `src/` 模块结构

### Phase 2: 端到端链路打通

- 实现 `retrieve -> rerank -> generate -> cite`
- 接入 FastAPI 与 Streamlit

### Phase 3: 接入真实 LLM

- 使用 OpenAI-compatible client 接入 DeepSeek
- 支持 `.env` 配置和 provider fallback

### Phase 4: 质量优化

- 加 entity-aware retrieval
- 加 title-level dedup
- 加 comparison-aware rerank
- citation 只保留实际引用证据

### Phase 5: 性能优化

- 增加阶段耗时日志
- 持久化 BM25 和 dense index
- 引入 fast path
- 引入 FAISS
- 升级 dense backend 到 transformer embeddings

### Phase 6: 稳定性与回归

- 固定 regression suite
- benchmark 输出 mean / p50 / p95
- 改 Streamlit 首屏逻辑，避免自动查询阻塞页面

## 6. 当前功能状态

目前项目已经支持：

- 离线构建索引
- BM25 + FAISS 双路检索
- RRF 融合
- title-level 去重
- heuristic rerank
- DeepSeek / offline fallback 生成
- citation grounding
- Streamlit UI 与 FastAPI API
- regression 与 benchmark

当前较稳定的问题类型包括：

- nationality
- profession
- older / younger
- same nationality / same profession
- 一部分 experiment / concept 类 slow path 问题

## 7. 当前存在的问题

虽然系统已经可以运行并回答，但还没有完全达到“产品级可用”。

主要问题包括：

1. 启动仍然较重  
   当前 runtime 在启动时会预热 BM25、FAISS 和 transformer query encoder，因此启动时间仍然偏长。

2. slow path 质量仍有提升空间  
   对 experiment / concept 类问题，top-1 往往正确，但 top-2 以后仍可能混入噪声。

3. fallback answer 仍然偏保守  
   在证据不够直接时，系统会倾向于保守回答，这有利于减少幻觉，但也会牺牲部分覆盖率。

## 8. 适合专家点评的几个重点

如果让专家评审，这个项目比较值得点评的点包括：

1. 双路检索 + RRF 的设计是否合理
2. rerank 的规则特征是否足够，是否值得升级为 cross-encoder
3. chunk 粒度是否还需要进一步调整
4. citation grounding 是否已经足够产品化
5. startup / preload / lazy load 的工程权衡是否合理
6. 是否应继续保留 fast path 规则系统，还是统一交给更强的 rerank + generator

## 9. 当前项目的一句话总结

这是一个以检索和排序为核心、LLM 负责证据整合与自然语言生成的搜索增强问答系统；它已经从普通 RAG demo 演进成一个具备离线索引、双路召回、FAISS 检索、citation grounding、回归测试和延迟基准能力的 production-style QA 原型。
