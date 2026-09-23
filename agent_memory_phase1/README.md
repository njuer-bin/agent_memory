# Agent Memory Challenge — Phase 1

这是第一阶段的稳定接口骨架，目标是先把 Add / Search / Health 和最重要的 user_id 隔离、request_id 幂等跑通。

## 运行

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Smoke

另开终端：

```bash
python tests/smoke_test.py
```

如果设置了 API_KEY：

```bash
set API_KEY=your_key
set BASE_URL=http://127.0.0.1:8000
python tests/smoke_test.py
```

## 当前阶段的明确边界

当前检索是轻量 lexical baseline，不是最终比赛算法。

下一阶段会替换为：

1. 稀疏检索
2. 稠密向量检索
3. RRF
4. 结构化过滤
5. 时间检索
6. 原子事实/关系/规则/画像分层
7. rerank
8. evidence fusion

不要在这个版本上直接 Full；先通过 Smoke，再进入算法迭代。

## 协议注意

本实现遵循手册中的核心约束：
- Add: request_id / messages / user_id / session_id
- Search: 只返回证据，不生成答案
- Health: 无鉴权 GET
- user_id 严格隔离
- request_id 幂等
- 不静默截断
