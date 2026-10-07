# LoCoMo-Refined 多跳推理（Multi-hop）测试数据集

## 来源

- 原始仓库：https://github.com/mem-eval-suite/LoCoMo_refined
- 上游基准：LoCoMo（Snap Research, ACL 2024），每段对话约 300 轮 / 约 9,000 tokens，分布在多个会话中
- 筛选规则：从 1,382 道评测问题中筛出 `category == "1"` 的**多跳推理（multi-hop）**问题，共 **213 条**

## 多跳问题的判定依据

类别映射经数据交叉验证（类别 1 平均引用 2.8 条证据、跨 2.42 个会话，明显为跨片段关联推理）：

| category | 含义 | 数量 |
|---|---|---|
| **1** | **多跳推理 Multi-hop**（本数据集） | **213** |
| 2 | 时序推理 Temporal | 299 |
| 3 | 开放域推理 Open-domain | 68 |
| 4 | 单跳检索 Single-hop | 802 |

## 文件清单

| 文件 | 内容 | 条数 |
|---|---|---|
| `locomo_refined_multihop_testset.json` | **完整测试包**：按对话组织，含每段对话的完整历史文本（`conversation_history_text`）、按会话组织的 `sessions`、以及该对话对应的多跳问题。**推荐用于端到端测试：先灌入对话记忆，再用 questions 提问** | 10 段对话 / 213 题 |
| `locomo_refined_multihop_questions.jsonl` | 全部 213 条多跳问题（每行一个 JSON 对象），不含对话历史 | 213 |
| `locomo_refined_multihop_textonly.jsonl` | 纯文本子集（剔除含图片的多模态问题），适合不支持多模态的系统 | 77 |
| `locomo_refined_multihop_questions.csv` | 问题概览表格，便于人工浏览 | 213 |

## 问题字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `qa_id` | string | 唯一 ID，如 `conv-26#q0003` |
| `sample_id` | string | 所属对话 ID，如 `conv-26` |
| `question` | string | 问题文本（英文） |
| `answers` | string[] | 参考答案列表（可多个可接受答案） |
| `category` / `category_name` | string | 固定为 `"1"` / `"multi-hop"` |
| `is_multi_modality` | bool | 是否涉及图片内容 |
| `evidence` | string[] | 证据消息 ID，如 `["D2:8", "D5:3"]`（Dn:m = 第 n 会话第 m 条消息） |
| `evidence_messages` | object[] | 证据消息全文（含 speaker/text/images） |
| `speaker_a` / `speaker_b` | string | 对话双方 |

## 使用建议

1. 把 `testset.json` 中某段对话的 `conversation_history_text`（或 `sessions`）灌入你的记忆系统
2. 用该对话下的 `questions` 逐条提问并记录模型回答
3. 与 `answers` 比对评分。可配合原仓库的评测脚本（`src/evaluate.py`、`src/llm_judge.py`）做 BLEU/F1 与 LLM-as-judge 评分
4. 若系统不支持多模态，请使用 `textonly` 子集（77 条）；多模态问题的 `evidence_messages` 中含图片引用与 BLIP 描述

## 注意事项

- 所有 213 条多跳问题的 `question`、`answers`、`evidence`（证据 ID）均非空，已逐条校验
- 其中 1 条记录的 `evidence_messages` 为空（源数据固有，其 `evidence` ID 仍在），评测时可仅用 `evidence` 定位
- 136 条为多模态问题（涉及图片），77 条为纯文本；不支持多模态时直接用 `textonly` 子集

## 各对话问题分布

conv-26: 24, conv-30: 9, conv-41: 23, conv-42: 25, conv-43: 23, conv-44: 27, conv-47: 16, conv-48: 14, conv-49: 26, conv-50: 26（合计 213）
