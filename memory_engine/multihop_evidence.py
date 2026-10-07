from __future__ import annotations

class EvidenceAnchorExtractor:
    """Extract conservative search-time anchors from structured and raw memory."""

    STOP = {
        "user", "用户", "我的", "朋友", "同事", "老板", "问题", "什么",
        "哪里", "哪个", "哪些", "怎么", "如何", "现在", "目前", "之前",
        "以前", "推荐", "介绍", "工作", "上班", "这个", "那个",
    }

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def extract(self, item):
        anchors = []
        metadata = item.get("metadata") or {}
        for key in ("subject", "object", "value", "key"):
            value = str(metadata.get(key) or "").strip()
            if value and value not in self.STOP:
                anchors.append(value)

        content = str(item.get("content") or "").strip()
        if content:
            try:
                analyzed = self.analyzer.analyze(
                    item.get("user_id", ""),
                    content,
                    int(item.get("timestamp") or 0),
                )
                for rel in analyzed.get("relations", []):
                    for value in (rel.subject, rel.object):
                        value = str(value).strip()
                        if value and value not in self.STOP:
                            anchors.append(value)
            except Exception:
                pass

            token = ""
            for char in content:
                if char.isalnum() or "一" <= char <= "鿿" or char in "_-":
                    token += char
                else:
                    if len(token) >= 2 and token not in self.STOP:
                        anchors.append(token)
                    token = ""
            if len(token) >= 2 and token not in self.STOP:
                anchors.append(token)

        result = []
        for value in anchors:
            if value and value not in result:
                result.append(value)
        return result[:8]

    @staticmethod
    def annotate(item, anchors):
        result = dict(item)
        metadata = dict(result.get("metadata") or {})
        content = str(result.get("content") or "")
        matched = [a for a in anchors if a and a in content]
        if matched:
            metadata["evidence_anchors"] = matched[:4]
        result["metadata"] = metadata
        return result
