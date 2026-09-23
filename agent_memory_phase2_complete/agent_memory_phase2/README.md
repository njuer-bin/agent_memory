# AML Phase 2 Memory Engine

这是一个可直接运行的 Agent Memory Engine 原型，目标是把普通 RAG：

Markdown → Chunk → Embedding → Vector DB → Reranker

升级为：

Raw Memory → Atomic Fact → Relation → Timeline → Profile/Rule
→ Memory Governance → Hybrid Retrieval → RRF → Structured Filter
→ Selective Multi-hop → Reranker → Evidence Fusion

## 1. 当前实现

### 写入

`POST /add`

- request_id 幂等
- user_id 严格隔离
- raw memory 保存
- 原子事实抽取
- relation / event / rule / profile 抽取
- 当前状态与历史状态治理
- superseded 链
- embedding 索引
- 20 条消息 / 2000 词确定性切分

### 检索

`POST /search`

- query rewrite
- BM25
- dense vector
- RRF
- structured filtering
- current/history 状态
- profile / relation / event evidence
- selective multi-hop
- reranking
- evidence reconstruction

## 2. 安装

```bash
G:\aconda\python.exe -m pip install -r requirements.txt
```

## 3. 启动

```bash
G:\aconda\python.exe app.py
```

默认：

`http://127.0.0.1:8000`

## 4. Smoke Test

另开终端：

```bash
G:\aconda\python.exe tests\smoke_test.py
```

## 5. Ollama Embedding

默认尝试：

```text
qwen3-embedding:4b
http://127.0.0.1:11434/api/embeddings
```

如果 Ollama 不可用，会自动 fallback 到稳定 hash embedding，所以项目仍然可以启动和测试。

如果不想访问 Ollama：

```text
USE_OLLAMA_EMBEDDING=0
```

## 6. 为什么不是所有 Query 都 Multi-hop

普通问题直接：

```text
Query
→ Hybrid
→ Rerank
```

只有检测到关系、多跳、指代等复杂查询时：

```text
Query
→ First Search
→ Graph Expansion
→ Evidence Reconstruction
→ Rerank
```

这样控制额外延迟。

## 7. 当前版本定位

这是 Phase 2 的工程基线，不是最终榜单优化版本。

下一步建议在真实评测集上增加：

1. LLM Memory Analyzer
2. 更强中文 BM25 / tokenizer
3. BGE reranker
4. 时间表达式归一化
5. 更强 conflict resolution
6. 语义 fingerprint
7. query difficulty classifier
8. iterative search budget
9. benchmark / ablation
10. metrics dashboard

最终应该比较：

- Vector only
- BM25 only
- BM25 + Vector
- BM25 + Vector + RRF
- + Governance
- + Temporal
- + Multi-hop
- + Reranker

这样才能证明每一个模块带来的实际收益。
