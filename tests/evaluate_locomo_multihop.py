"""Offline LoCoMo-Refined multi-hop retrieval evaluator.

Usage:
    python tests/evaluate_locomo_multihop.py

Or point it at a local copy of the complete testset:
    python tests/evaluate_locomo_multihop.py --dataset path/to/locomo_refined_multihop_testset.json

The evaluator measures retrieval evidence coverage, not final-answer quality.
It deliberately checks whether the original evidence messages are present in
the Top-K retrieval context. Window/session views are counted as hits when
they contain the corresponding evidence message text.

The complete testset is kept separately from this code because it is a
benchmark artifact. See locomo_multihop_dataset/README.md on the master
branch for the dataset files and schema.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Allow running this file directly from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_engine.engine import MemoryEngine
from memory_engine.models import AddMessage, AddRequest, SearchRequest


def norm(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    return []


def extract_samples(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in ("samples", "data", "testset", "conversations"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        # A single conversation/sample is also accepted.
        return [payload]

    raise ValueError("Unsupported LoCoMo testset JSON structure")


def extract_sessions(sample: dict) -> list[list[dict]]:
    """Normalize the common LoCoMo session representations."""
    raw = sample.get("sessions")
    if raw is None:
        raw = sample.get("session")

    sessions: list[list[dict]] = []

    if isinstance(raw, dict):
        # e.g. {"D1": [...], "D2": [...]}
        raw = list(raw.values())

    if isinstance(raw, list):
        for session in raw:
            if isinstance(session, dict):
                # Sometimes a session is wrapped in a messages field.
                messages = session.get("messages") or session.get("dialogue") or session.get("turns")
                if messages is None:
                    messages = [session]
            else:
                messages = session

            if not isinstance(messages, list):
                continue

            rows = [m for m in messages if isinstance(m, dict)]
            if rows:
                sessions.append(rows)

    return sessions


def extract_questions(sample: dict) -> list[dict]:
    for key in ("questions", "qas", "qa"):
        value = sample.get(key)
        if isinstance(value, list):
            return [q for q in value if isinstance(q, dict)]
    return []


def message_text(message: dict) -> str:
    return str(
        message.get("text")
        or message.get("content")
        or message.get("message")
        or ""
    )


def message_id(message: dict, session_index: int, message_index: int) -> str:
    return str(
        message.get("dia_id")
        or message.get("id")
        or f"D{session_index + 1}:{message_index + 1}"
    )


def flatten_messages(sample: dict) -> list[tuple[str, dict]]:
    """Return (LoCoMo evidence id, message) pairs in session order."""
    rows = []
    sessions = extract_sessions(sample)

    for si, session in enumerate(sessions):
        for mi, msg in enumerate(session):
            mid = message_id(msg, si, mi)
            rows.append((mid, msg))

    return rows


def evidence_messages_for_question(question: dict) -> list[dict]:
    rows = question.get("evidence_messages") or []
    return [x for x in rows if isinstance(x, dict) and message_text(x).strip()]


def ingest_sample(engine: MemoryEngine, sample: dict, user_id: str) -> int:
    messages = flatten_messages(sample)
    if not messages:
        raise ValueError(
            "No structured sessions/messages found in testset sample. "
            "Use locomo_refined_multihop_testset.json, not questions.jsonl."
        )

    # Preserve session boundaries. The engine uses session_id to build
    # adjacent Window View and Session View.
    by_session: dict[str, list[dict]] = defaultdict(list)
    for mid, msg in messages:
        si = msg.get("session_index")
        if si is None:
            match = re.match(r"D(\\d+):", mid)
            si = int(match.group(1)) if match else 0

        timestamp = msg.get("timestamp")
        if timestamp is None:
            # Stable synthetic time is enough for retrieval; the benchmark
            # evidence relation itself is identified by session/message id.
            mi = msg.get("message_index", 0)
            timestamp = (int(si) * 100000 + int(mi or 0)) * 1000

        by_session[str(si)].append(
            {
                "role": "user",
                "content": message_text(msg),
                "timestamp": int(timestamp),
            }
        )

    total = 0
    for session_key, rows in by_session.items():
        for start in range(0, len(rows), 20):
            batch = rows[start : start + 20]
            request = AddRequest(
                request_id=f"locomo-{user_id}-{session_key}-{start}-{uuid.uuid4().hex}",
                user_id=user_id,
                session_id=f"D{session_key}",
                messages=[AddMessage(**m) for m in batch],
            )
            engine.add(request)
            total += len(batch)

    return total


def evidence_texts(question: dict) -> list[str]:
    rows = evidence_messages_for_question(question)
    if rows:
        return [norm(message_text(x)) for x in rows if norm(message_text(x))]

    # Some records contain evidence IDs but no evidence_messages. In that
    # case the evaluator cannot safely infer the missing text.
    return []


def _semantic_match(
    target: str,
    normalized_rows: list[tuple[int, dict, str]],
    embedder,
    threshold: float,
) -> tuple[float, int | None, dict | None]:
    """Find the strongest semantic candidate for one gold evidence message.

    This is diagnostic-only. It never changes the official exact-match metric
    or the retrieval result ordering. Reusing the engine's embedding provider
    also keeps the diagnostic in the same embedding space as retrieval.
    """
    if not target or embedder is None:
        return 0.0, None, None

    target_vec = embedder.embed(target)
    best_score = -1.0
    best_rank = None
    best_row = None

    for rank, row, content in normalized_rows:
        if not content:
            continue
        score = float(embedder.cosine(target_vec, embedder.embed(content))) if hasattr(embedder, "cosine") else 0.0
        if score > best_score:
            best_score = score
            best_rank = rank
            best_row = row

    return best_score, best_rank, best_row


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _debug_question(
    question: dict,
    evidence: list[str],
    rows: list[dict],
    debug_top_n: int,
    embedder=None,
    semantic_threshold: float = 0.65,
) -> dict:
    """Print exact/semantic evidence diagnostics without changing retrieval."""
    qa_id = question.get("qa_id") or question.get("id") or "<unknown>"
    query = str(question.get("question") or question.get("query") or "").strip()
    print()
    print("-" * 72)
    print(f"DEBUG {qa_id}")
    print(f"Question: {query}")
    print(f"Returned rows: {len(rows)}")
    normalized_rows = [(i + 1, row, norm(row.get("content"))) for i, row in enumerate(rows)]
    semantic_hits = 0
    true_misses = 0

    print("Evidence status:")
    for ev_idx, target in enumerate(evidence, 1):
        matches = [
            (rank, row)
            for rank, row, content in normalized_rows
            if target in content or content in target
        ]
        if matches:
            rank, row = matches[0]
            print(
                f"  [{ev_idx}] EXACT HIT rank={rank} id={row.get('id')} "
                f"source={row.get('source')} type={row.get('memory_type')}"
            )
            continue

        score, rank, row = _semantic_match(
            target,
            normalized_rows,
            embedder,
            semantic_threshold,
        )
        if row is not None and score >= semantic_threshold:
            semantic_hits += 1
            print(
                f"  [{ev_idx}] SEMANTIC HIT score={score:.4f} rank={rank} "
                f"id={row.get('id')} source={row.get('source')} "
                f"type={row.get('memory_type')} target={target[:160]}"
            )
        else:
            true_misses += 1
            if row is not None:
                print(
                    f"  [{ev_idx}] TRUE MISS best_score={score:.4f} rank={rank} "
                    f"id={row.get('id')} target={target[:160]}"
                )
            else:
                print(f"  [{ev_idx}] TRUE MISS target={target[:180]}")

    print(
        f"Diagnostic summary: exact={sum(1 for target in evidence if any("
        f"target in content or content in target for _, _, content in normalized_rows))} "
        f"semantic={semantic_hits} true_miss={true_misses}"
    )
    print(f"Top {min(debug_top_n, len(rows))} final candidates:")
    for rank, row, _content in normalized_rows[:debug_top_n]:
        metadata = row.get("metadata") or {}
        reqs = row.get("_evidence_requirements") or metadata.get("_evidence_requirements") or []
        content = str(row.get("content") or "").replace("\\n", " ").strip()
        print(
            f"  {rank:3d}. score={float(row.get('score', 0.0)):.4f} "
            f"id={row.get('id')} source={row.get('source')} "
            f"type={row.get('memory_type')} reqs={reqs} :: {content[:220]}"
        )

    return {"semantic_hits": semantic_hits, "true_misses": true_misses}

def evaluate_sample(
    engine: MemoryEngine,
    sample: dict,
    user_id: str,
    top_k: int,
    limit: int | None,
    debug: bool = False,
    debug_top_n: int = 20,
    semantic_threshold: float = 0.65,
) -> list[dict]:
    questions = extract_questions(sample)
    if limit is not None:
        questions = questions[:limit]

    results = []

    for question in questions:
        evidence = evidence_texts(question)
        if not evidence:
            continue

        query = str(question.get("question") or question.get("query") or "").strip()
        if not query:
            continue

        rows = engine.search(
            SearchRequest(
                query=query,
                user_id=user_id,
                top_k=top_k,
                multi_hop=True,
            )
        )

        returned = [norm(row.get("content")) for row in rows]

        diagnostic = {"semantic_hits": 0, "true_misses": 0}
        if debug:
            diagnostic = _debug_question(
                question,
                evidence,
                rows,
                max(1, debug_top_n),
                embedder=engine.embedder,
                semantic_threshold=semantic_threshold,
            )

        hits = 0
        for target in evidence:
            if any(target in candidate or candidate in target for candidate in returned):
                hits += 1

        coverage = hits / len(evidence)
        results.append(
            {
                "qa_id": question.get("qa_id") or question.get("id") or query,
                "question": query,
                "evidence_count": len(evidence),
                "evidence_hits": hits,
                "coverage": coverage,
                "complete": hits == len(evidence),
                "top_k": top_k,
                "hop_count": len(evidence),
                "semantic_hits": diagnostic["semantic_hits"],
                "true_misses": diagnostic["true_misses"],
            }
        )

    return results


def summarize(results: list[dict]) -> None:
    if not results:
        print("No evaluable questions found.")
        return

    total = len(results)
    avg_coverage = sum(r["coverage"] for r in results) / total
    complete = sum(r["complete"] for r in results)
    any_hit = sum(r["evidence_hits"] > 0 for r in results)

    print("=" * 72)
    print("LoCoMo-Refined Multi-hop Evidence Evaluation")
    print("=" * 72)
    print(f"Questions evaluated : {total}")
    print(f"Evidence Coverage   : {avg_coverage * 100:.2f}%")
    print(f"Any Evidence Hit    : {any_hit / total * 100:.2f}%")
    print(f"Complete Evidence   : {complete / total * 100:.2f}%")
    print()

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in results:
        grouped[row["hop_count"]].append(row)

    print("By evidence count:")
    print("  hops/evidence | questions | coverage | complete")
    for hop_count in sorted(grouped):
        rows = grouped[hop_count]
        cov = sum(r["coverage"] for r in rows) / len(rows)
        comp = sum(r["complete"] for r in rows) / len(rows)
        print(f"  {hop_count:13d} | {len(rows):9d} | {cov * 100:7.2f}% | {comp * 100:7.2f}%")

    misses = [r for r in results if r["coverage"] < 1.0]
    print()
    print(f"Incomplete cases: {len(misses)}")
    for row in misses[:20]:
        print(
            f"  {row['qa_id']}: "
            f"{row['evidence_hits']}/{row['evidence_count']}  "
            f"{row['question']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="locomo_multihop_dataset/locomo_refined_multihop_testset.json",
        help="Path to the complete LoCoMo-Refined testset JSON",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="Print per-question evidence hit/miss diagnostics")
    parser.add_argument("--debug-top-n", type=int, default=20, help="Final candidates to print per question in debug mode")
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.65,
        help="Diagnostic-only embedding similarity threshold for SEMANTIC HIT",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(
            f"Dataset not found: {dataset}\\n"
            "Use the complete testset JSON from locomo_multihop_dataset on master."
        )
        return 2

    payload = load_json(dataset)
    samples = extract_samples(payload)

    with tempfile.TemporaryDirectory(prefix="locomo_eval_") as tmp:
        for index, sample in enumerate(samples):
            user_id = f"locomo-eval-{index}"
            engine = MemoryEngine(os.path.join(tmp, f"memory_{index}.db"))

            count = ingest_sample(engine, sample, user_id)
            results = evaluate_sample(
                engine,
                sample,
                user_id,
                top_k=args.top_k,
                limit=args.limit,
                debug=args.debug,
                debug_top_n=args.debug_top_n,
                semantic_threshold=args.semantic_threshold,
            )
            print(f"\\nSample {index + 1}/{len(samples)}: ingested {count} messages")
            summarize(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
