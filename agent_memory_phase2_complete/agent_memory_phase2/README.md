# AML Phase 2 Memory Engine

面向 **Agent Memory Challenge 第二期** 的可运行 Memory Engine。

当前版本以：

* 稳定性
* 用户隔离
* 幂等写入
* Add → Search 一致性
* 时间 / 历史记忆
* Multi-hop 检索
* 可复现测试
* 参赛部署

为主要目标。

当前 release 已完成基础 Competition Test、Smoke Test、Benchmark、重启恢复和认证验证。算法部分暂时冻结，不依赖继续调参才能工作。

---

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

* Raw memory 与结构化 memory 并存
* Atomic Fact 的 current/history 治理
* Relation / Event / Rule / Profile 分层
* BM25 + Dense hybrid retrieval
* RRF 融合
* 时间查询与历史查询
* 选择性 multi-hop graph expansion
* lightweight reranking
* evidence reconstruction
* `user_id` 严格隔离
* Add → Search 立即可见
* `request_id` 原子幂等 claim

---

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

* `request_id`
* `messages`
* `user_id`
* `session_id`

示例：

```json
{
  "request_id": "example-001",
  "user_id": "user-001",
  "session_id": "session-001",
  "messages": [
    {
      "role": "user",
      "content": "我现在住在杭州，我喜欢咖啡。",
      "timestamp": 1760000000000
    }
  ]
}
```

重复的 `request_id` 会被幂等处理，不会重复写入。

### Search

```http
POST /search
```

支持：

* `query` / `question`
* `user_id`
* `top_k`
* `session_id`
* `include_history`
* `start_time`
* `end_time`
* `memory_types`
* `multi_hop`

示例：

```json
{
  "user_id": "user-001",
  "query": "我现在住在哪里？",
  "top_k": 5
}
```

Search 只负责返回记忆证据，不负责生成最终自然语言答案。

---

## 3. Authentication

默认情况下，如果没有设置 `MEMORY_API_KEY`，API 不强制认证。

设置环境变量：

```bat
set MEMORY_API_KEY=your-secret
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

当前已实际验证：

```text
正确 X-API-Key       → 200
错误 X-API-Key       → 401
缺少认证             → 401
正确 Bearer Token    → 200
```

---

## 4. Installation

当前项目以 Windows + Python 环境为主要开发 / 测试环境。

安装依赖：

```bat
G:\aconda\python.exe -m pip install -r requirements.txt
```

启动：

```bat
G:\aconda\python.exe app.py
```

默认监听：

```text
http://127.0.0.1:8000
```

健康检查：

```text
GET http://127.0.0.1:8000/health
```

返回：

```json
{"status":"ok"}
```

### Deployment note

当前 vector index 是**进程内内存索引**。

SQLite 是 durable source of truth，内存 vector index 是 process-local cache。

因此当前参赛版本建议：

```text
single worker
```

不要直接使用多 worker 启动方式，否则不同 worker 会拥有各自独立的 RAM vector index。

---

## 5. Embedding

默认配置：

```text
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:4b
OLLAMA_URL=http://127.0.0.1:11434/api/embeddings
USE_OLLAMA_EMBEDDING=1
```

Embedding 默认通过本地 Ollama 服务生成。

当 Ollama 不可用时，代码会 fallback 到 deterministic hash embedding，使服务仍能够启动和运行。

但是，**参赛评测时应固定 embedding 环境**。

推荐固定：

```text
OLLAMA_EMBEDDING_MODEL
OLLAMA_URL
USE_OLLAMA_EMBEDDING
```

原因是不同 embedding 环境可能导致向量空间不同，从而影响检索指标。

因此 README 中的 benchmark 结果仅代表当前固定环境下的本地测试结果。

---

## 6. Tests

项目提供三类主要测试：

### Smoke Test

验证基础 API 和幂等行为：

```bat
G:\aconda\python.exe tests\smoke_test.py
```

当前已通过：

```text
/health       200
/add          200
/add retry    200
/search       200
```

并验证了重复 `request_id` 不会造成重复写入。

---

### Competition Test

用于检查比赛接口要求：

```bat
G:\aconda\python.exe tests\competition_test.py
```

当前：

```text
COMPETITION TEST PASSED: 16/16
COMPETITION READY
```

覆盖：

* health
* basic add/search
* request_id idempotency
* immediate consistency
* user isolation
* current/history retrieval
* relation / multi-hop
* event / rule / temporal retrieval
* memory filters
* session filter
* 20-message boundary
* over-20-message input
* large message
* malformed payloads
* latency
* authentication

---

### Benchmark

```bat
G:\aconda\python.exe tests\benchmark.py
```

当前 `master` 实际运行结果：

| Metric          |     Result |
| --------------- | ---------: |
| Benchmark cases | 31/31 PASS |
| Recall@1        |      0.677 |
| Recall@3        |      0.968 |
| Recall@5        |      0.968 |
| Recall@10       |      1.000 |
| MRR@10          |      0.810 |
| Search p50      |   75.94 ms |
| Search p95      |   98.93 ms |
| Search mean     |   79.41 ms |
| User isolation  |       PASS |

说明：

这些是当前本地测试环境的实际结果，**不代表任何外部比赛榜单结果**。

由于本地 embedding、硬件和运行状态都会影响延迟及排序指标，单次 benchmark 不应被解释为稳定的统计性能估计。

---

## 7. Persistence / Restart Recovery

SQLite 是持久化 source of truth。

Memory Vector Index 在进程内运行，因此服务重启后需要从 SQLite 恢复 embedding 数据。

已实际验证：

```text
Add memory
    ↓
Search
    ↓
Stop API
    ↓
Restart API
    ↓
Search again
```

重启前后的关键 memory ID 和内容保持一致。

因此：

```text
SQLite persistence        PASS
Restart recovery          PASS
```

---

## 8. Release Hardening

本版本额外处理了几个参赛环境风险。

### 8.1 request_id 并发竞态

旧逻辑：

```text
request_seen
    ↓
write memory
    ↓
register_request
```

理论上存在：

```text
Thread A: request_seen = False
Thread B: request_seen = False

A → write
B → write
```

导致相同 `request_id` 存在并发重复写入风险。

当前改为 SQLite 原子 claim：

```text
INSERT OR IGNORE
        ↓
claim success
        ↓
perform Add
```

只有成功 claim 的请求才能继续执行。

如果 Add 过程中发生异常，会释放 request claim，允许后续安全重试。

---

### 8.2 运行时数据库污染 Git

运行时数据库不应该进入代码仓库。

`.gitignore` 已包含：

```text
data/*.db
data/*.db-shm
data/*.db-wal
```

历史运行数据库也已经从 Git tracking 中移除。

---

### 8.3 Single Worker

当前架构：

```text
SQLite
  ↓
durable source of truth

RAM Vector Index
  ↓
process-local cache
```

因此当前 release 保持：

```text
single worker
```

避免多个 worker 之间出现独立 RAM index。

如果未来需要多 worker，需要进一步设计共享向量索引或跨 worker cache invalidation，而不是简单增加 worker 数量。

---

### 8.4 实验算法冻结

当前 release 不继续修改：

* hybrid retrieval 权重
* RRF 参数
* reranker 权重
* query-aware ranking

除非新的改动能够通过 benchmark / ablation 实验证明收益，否则不进入稳定 release。

目标是避免为了局部指标优化破坏：

```text
correctness
isolation
latency
stability
reproducibility
```

---

## 9. Data Isolation

所有主要 memory 数据和 embedding 查询都以 `user_id` 为边界。

逻辑：

```text
user A Add
    ↓
A memories

user A Search
    ↓
只返回 A 的 memories
```

而：

```text
user B Search
    ↓
不会读取 A 的 memory
```

这是 Memory Engine 的核心正确性要求之一。

当前 Competition Test 和 Benchmark 均包含 user isolation 验证。

结果：

```text
Isolation: PASS
```

---

## 10. Memory Governance

对于结构化 memory，系统不仅保存原始文本，还进行一定程度的状态治理。

例如当前事实：

```text
我现在住在杭州
```

如果之后出现新的冲突事实：

```text
我现在住在上海
```

系统可以将旧 current fact 标记为：

```text
superseded
```

并建立：

```text
new fact
    ↓
supersedes
    ↓
old fact
```

这样可以同时支持：

```text
current memory
```

和：

```text
historical memory
```

查询历史时可以通过：

```text
include_history
```

以及时间条件获取历史证据。

---

## 11. Query Understanding

Search 并不是简单执行一次向量相似度搜索。

Query Analyzer 会识别：

* current / history
* temporal expressions
* memory type
* relation intent
* predicate
* multi-hop intent
* deterministic query rewrite

例如：

```text
我现在住哪里？
```

会转化为更加结构化的检索意图：

```text
用户
当前
居住地
```

类似地：

```text
以前住哪里？
```

会偏向：

```text
用户
历史
居住地
```

该模块采用确定性规则，不依赖 LLM，因此结果具有较好的可重复性。

---

## 12. Hybrid Retrieval

Search 使用：

```text
BM25
  +
Dense Retrieval
  ↓
RRF
  ↓
structured signals
  ↓
graph expansion
  ↓
reranker
  ↓
evidence
```

### BM25

用于精确关键词和文本匹配。

适合：

* 人名
* 地名
* 专有名词
* 明确关键词

### Dense Retrieval

用于语义相似度召回。

适合：

* 同义表达
* 自然语言改写
* 语义相近但关键词不同的问题

### RRF

将多个检索结果进行 rank fusion，降低单一检索器失败的影响。

---

## 13. Multi-hop Retrieval

系统支持轻量级 graph expansion。

例如：

```text
A 是 B 的朋友
B 在上海工作
```

查询：

```text
A 的朋友在哪里工作？
```

系统可以：

```text
A
 ↓
friend
 ↓
B
 ↓
workplace
 ↓
上海
```

当前实现是选择性 graph expansion，而不是完整图数据库。

这样可以在增加检索能力的同时控制额外延迟。

---

## 14. Repository Hygiene

推荐目录结构：

```text
agent_memory_phase2/
├─ app.py
├─ api/
├─ memory_engine/
├─ tests/
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ data/
   └─ # runtime database generated locally
```

`data/` 中的 SQLite 数据属于运行时状态，不应该提交到 Git。

从 GitHub clone 后首次运行时，数据库会由程序自动创建。

---

## 15. Competition Readiness

当前 release 已完成以下验证：

```text
Health
    PASS

Add / Search
    PASS

request_id idempotency
    PASS

Immediate consistency
    PASS

User isolation
    PASS

Current / History
    PASS

Temporal retrieval
    PASS

Relation / Multi-hop
    PASS

Rule / Event
    PASS

Memory filters
    PASS

Session filter
    PASS

20-message boundary
    PASS

Large message
    PASS

Malformed payload
    PASS

Restart recovery
    PASS

Authentication
    PASS

Competition Test
    16/16 PASS

Benchmark
    31/31 PASS
```

---

## 16. Current Release Position

当前版本的工程优先级：

```text
正确性
  ↓
用户隔离
  ↓
幂等写入
  ↓
Add → Search 一致性
  ↓
时间 / 历史
  ↓
Multi-hop
  ↓
Hybrid Retrieval
  ↓
排序
  ↓
延迟
  ↓
参赛可复现性
```

当前策略不是无限调参，而是：

```text
问题
 ↓
定位
 ↓
修改
 ↓
Benchmark
 ↓
验证收益
 ↓
合入 Release
```

没有经过 benchmark / ablation 验证的优化，不作为已经取得的性能收益进行描述。

---

## 17. Known Constraints

当前 release 有几个明确的工程约束：

### 1. 单 worker

因为 vector index 是 process-local，所以当前参赛部署保持 single worker。

### 2. Embedding 环境需要固定

不同 embedding model 或 embedding backend 可能改变检索结果。

### 3. SQLite

SQLite 适合当前参赛规模和本地部署。

如果未来需要高并发、多实例部署，需要进一步替换或扩展：

```text
shared database
+
shared vector store
+
distributed cache / index
```

### 4. Graph Retrieval

当前 multi-hop 是轻量 graph expansion，不是完整知识图谱系统。

### 5. Benchmark

Benchmark 是本地固定测试集，不代表外部比赛真实流量或最终排行榜结果。

---

## 18. Final Status

当前 `master` release 已完成：

```text
Git clean
      ↓
Release Hardening merged
      ↓
Competition Test 16/16
      ↓
Smoke Test PASS
      ↓
Benchmark 31/31
      ↓
Recall@10 = 1.000
      ↓
Isolation PASS
      ↓
Restart Recovery PASS
      ↓
Authentication PASS
```

当前版本可以作为 AML Phase 2 的稳定参赛版本继续使用。

后续如果继续优化，应建立在：

```text
benchmark
+
ablation
+
regression test
```

之上，而不是仅依据单次运行的 Recall、MRR 或延迟波动判断改动是否有效。
