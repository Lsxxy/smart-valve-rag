"""Opt-in snapshots from the request that actually produced the answer."""
from qa_core.retrieval.results import RetrievalResult


def record_lane(context, lane, result, *, executed):
    if not getattr(context, "evaluation_capture", False):
        return
    initial = result.pre_rerank_hits
    context.evaluation_trace.setdefault("lanes", {})[lane] = {
        "executed": bool(executed),
        "rerank_applied": result.rerank_applied,
        "query": result.query,
        "candidate_count": len(initial) if initial is not None else None,
        "kb_version": getattr(context, "active_kb_version", None),
        "initial": (RetrievalResult(hits=initial, source_type=lane).source_payloads(limit=len(initial), full_content=True) if initial is not None else None),
        "reranked": result.source_payloads(limit=len(result.hits), full_content=True),
    }


def record_context(context, documents):
    if not getattr(context, "evaluation_capture", False):
        return
    context.evaluation_trace["context"] = [
        {"content": d.page_content, "metadata": dict(d.metadata),
         "source_type": "faq" if d.metadata.get("standard_question") else "doc"}
        for d in documents
    ]
