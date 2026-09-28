# AML Phase 2 Memory Engine

面向 Agent Memory Challenge 第二期的可运行 Memory Engine。当前版本以**稳定性、用户隔离、幂等写入、可复现测试和参赛部署**为优先，不再依赖继续调参才能工作的实验性路径。

## 1. Architecture

### Add

```text
POST /add
  ↓
atomic request_id claim
  ↓
20 messages / ~2000 words deterministic batching
  ↓
Raw Memory
  ↓
Fact / Relation / Event / Rule / Profile
  ↓
Governance
  ↓
Embedding
  ↓
SQLite + process-level vector index
```

### Search

```text
POST /search
  ↓
Query Analyzer
  ↓
BM25 + Dense
  ↓
RRF + structured signals
  ↓
selective graph expansion
  ↓
lightweight reranker
  ↓
evidence reconstruction
  ↓
Top-K
```

核心设计包括：

- Raw memory 与结构化 memory 并存
- Atomic Fact 的 current/history 治理
- Relation / Event / Rule / Profile 分层
- BM25 + Dense hybrid retrieval
- RRF 融合
- 时间查询与历史查询
- 选择性 multi-hop graph expansion
- user_id 严格隔离
- Add → Search 立即可见
- request_id 原子幂等 claim

## 2. API

### Health

```http
GET /health
```

无认证，返回：

```json
{"status":"ok"}
```

### Add

```http
POST /add
```

请求必须包含：

- `request_id`
- `messages`
- `user_id`
- `session_id`

重复 `request_id` 会被幂等处理，不会重复写入。

### Search

```http
POST /search
```

支持：

- `query` / `question`
- `user_id`
- `top_k`
- `session_id`
- `include_history`
- `start_time` / `end_time`
- `memory_types`
- `multi_hop`

Search 只返回记忆证据，不负责生成最终自然语言答案。

## 3. Authentication

设置环境变量：

```text
MEMORY_API_KEY=your-secret
```

之后 `/add` 和 `/search` 支持：

```http
Authorization: Bearer your-secret
```

或：

```http
X-API-Key: your-secret
```

`/health` 保持无认证。

## 4. Installation

Windows：

```bat
G:\\aconda\\python.exe -m pip install -r requirements.txt
```

启动：

```bat
G:\\aconda\\python.exe app.py
```

默认监听：

```text
http://127.0.0.1:8000
```

> 当前 vector index 是进程内内存索引。参赛部署保持单 worker，不要直接使用多 worker 启动方式。

## 5. Embedding

默认配置：

```text
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:4b
OLLAMA_URL=http://127.0.0.1:11434/api/embeddings
USE_OLLAMA_EMBEDDING=1
```

Ollama 不可用时，代码会 fallback 到 deterministic hash embedding，保证服务仍可启动。

**参赛评测时建议固定 embedding 环境和 `USE_OLLAMA_EMBEDDING`，否则不同环境可能得到不同的检索指标。**

## 6. Tests

Smoke：

```bat
G:\\aconda\\python.exe tests\\smoke_test.py
```

Benchmark：

```bat
G:\\aconda\\python.exe tests\\benchmark.py
```

当前已验证的本地稳定基线：

| Metric | Result |
|---|---:|
| Benchmark cases | 31/31 PASS |
| Recall@1 | 0.742 |
| Recall@3 | 0.935 |
| Recall@5 | 0.935 |
| Recall@10 | 1.000 |
| MRR@10 | 0.835 |
| Search p50 | 61.52 ms |
| Search p95 | 88.98 ms |
| Search mean | 63.91 ms |
| User isolation | PASS |

以上是当前已运行基线，不代表任何外部比赛榜单结果。

## 7. Release Hardening

本版本额外处理了几个参赛环境风险：

1. **request_id 并发竞态**
   - 旧逻辑是 `request_seen → 写入 → register_request`
   - 两个并发 Add 可能同时通过检查
   - 现在改为 SQLite `INSERT OR IGNORE` 原子 claim
   - Add 失败会释放 request_id，允许安全重试

2. **运行时数据库污染 Git**
   - `data/*.db`、WAL/SHM 文件加入 `.gitignore`
   - 已从参赛代码树移除历史运行数据库

3. **单 worker 部署约束**
   - SQLite 是 durable source of truth
   - vector index 是 process-local cache
   - 因此参赛部署不启用多 worker

4. **实验算法冻结**
   - 当前 release 不继续修改 hybrid / reranker 权重
   - 后续优化必须通过 benchmark / ablation 证明收益后再合入

## 8. Data Isolation

所有主要 memory 表和 embedding 查询都以 `user_id` 为边界。

因此：

```text
user A Add
    ↓
A Search → A memories

user B Search
    ↓
不会读取 A 的 memory
```

这是比赛接口的核心正确性要求之一。

## 9. Repository Hygiene

运行时数据库不应提交到 Git。推荐本地目录结构：

```text
agent_memory_phase2/
├─ app.py
├─ api/
├─ memory_engine/
├─ tests/
├─ requirements.txt
├─ .gitignore
└─ data/
   └─ # runtime database generated locally
```

如果从 GitHub clone 后首次运行，SQLite 数据库会由程序自动创建。

## 10. Current Release Position

当前版本的重点是：

```text
正确性
→ 隔离
→ 幂等
→ 时间 / 历史
→ Multi-hop
→ 检索排序
→ 延迟
→ 参赛可复现性
```

不把尚未经过 benchmark 验证的优化写成已取得的收益。
