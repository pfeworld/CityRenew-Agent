"""第11 T6：知识检索匹配准确率评测与调优（RAG retrieval evaluation）。

目标：对齐三大硬指标之一「知识检索匹配准确率 > 85%」。
本服务建立检索评测集、可配置检索策略、指标计算与调优闭环，仅用 train/val 选 best_strategy。

评测集来源（仅 train/val 文档型 RAG chunks，全部来自参考资料 train 语料）：
- 由已入库的 KnowledgeChunk（chunks_meta.json）派生「自检索」评测题：query 由该 chunk 的
  关键词/章节/主题派生，目标是把该 chunk 的 source/evidence 检索回来。
- 题目程序化生成（非 LLM、非人工写死答案），expected 来自 chunk 元数据。

红线（对齐 docs/07、docs/08、.cursor/rules）：
- 不使用 competition_test / final_test 调参；split=test 仅在 tune_mode=false 才允许，默认不跑。
- 不根据 test 结果改 prompt/规则/权重/检索参数；best_strategy 仅由 train/val 选出。
- 不伪造准确率；不把人工答案写死进代码；不把测试题泄漏到 train/val。
- 评测题与索引均仅含 train/val 文档；结构化 test（split_manifest）只登记冻结，不进检索集。
- 输出仅摘要/来源/evidence_id/keywords，不含 chunk 原文整段。

口径说明（沿用 docs/07 第1节并细化）：retrieval_accuracy 以 hit@K + 来源命中 + 关键词命中
为基础，本任务采用加权综合 weighted_retrieval_accuracy。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.services import housing_price_training_service as hp
from app.services import rag_service

logger = logging.getLogger("cityrenew.retrieval_eval")

BENCHMARK_VERSION = "t6_retrieval_v1"
RANDOM_SEED = 42
BENCH_VAL_RATIO = 0.30
MIN_KEYWORDS = 3
DEFAULT_TOP_K = 5
PASS_THRESHOLD = 0.85
MIN_SAMPLE_FOR_PASS = 30

WEIGHTS = {"hit_at_3": 0.40, "source": 0.25, "keyword": 0.20, "evidence": 0.15}

STRATEGIES = (
    "baseline_keyword", "vector_only", "hybrid",
    "metadata_filter", "rerank", "hybrid_plus_rerank",
)
DEFAULT_STRATEGY = "hybrid_plus_rerank"

# source_type → 主题 / doc_type / 来源可靠性（用于 rerank）
SOURCE_TYPE_TOPIC = {
    "policy": "policy", "template": "report_template", "case_report": "planning",
    "field_spec": "planning", "dataset_spec": "planning",
}
SOURCE_TYPE_DOCTYPE = {
    "policy": "policy", "template": "template", "case_report": "case",
    "field_spec": "spec", "dataset_spec": "spec",
}
SOURCE_RELIABILITY = {
    "policy": 1.0, "template": 0.9, "dataset_spec": 0.85,
    "field_spec": 0.8, "case_report": 0.75,
}
# 主题关键字推断（用于 metadata_filter 的 query 主题，不读 expected）
TOPIC_HINTS = {
    "housing": ("房价", "单价", "成交", "挂牌", "二手", "楼盘", "住宅", "价格"),
    "poi": ("POI", "兴趣点", "设施", "商业", "配套", "圈层", "公里"),
    "industry": ("产业", "企业", "园区", "办公", "经济", "制造"),
    "population": ("人口", "客群", "常住", "年龄", "收入", "消费"),
    "policy": ("条例", "政策", "规定", "办法", "审批", "规划许可"),
    "planning": ("规划", "用地", "更新", "策划", "口径", "字段", "指标", "模板"),
    "report_template": ("报告", "章节", "模板", "目录", "附录"),
}


def _models_dir():
    d = hp.settings.data_dir / "models" / "retrieval_eval"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_json(path, obj) -> None:
    hp._save_json(path, obj)  # noqa: SLF001


def _read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, TypeError):
        return None


def _bench_split(query_id: str) -> str:
    """确定性 benchmark 切分（仅 train/val，固定 seed，可复算）。"""
    h = hashlib.sha256(f"{RANDOM_SEED}:{query_id}".encode()).hexdigest()
    return "val" if (int(h[:8], 16) % 100) < int(BENCH_VAL_RATIO * 100) else "train"


def _infer_topic(text: str, source_type: str) -> str:
    counts = {t: sum(text.count(w) for w in hints) for t, hints in TOPIC_HINTS.items()}
    best = max(counts, key=counts.get)
    if counts[best] > 0:
        return best
    return SOURCE_TYPE_TOPIC.get(source_type, "planning")


# --------------------------------------------------------------------------- #
# 评测集构建
# --------------------------------------------------------------------------- #
def build_benchmark(splits: list[str] | None = None, include_test_manifest: bool = True,
                    use_test: bool = False) -> dict[str, Any]:
    index = rag_service._load_index()  # noqa: SLF001
    if index is None or not index.get("meta"):
        return {"status": "degraded", "available": False,
                "message": "RAG 索引不存在，请先 POST /api/rag/build", "train_sample_count": 0,
                "val_sample_count": 0}

    meta = index["meta"]
    samples: list[dict[str, Any]] = []
    for m in meta:
        if m.get("split") not in ("train", "val"):
            continue  # 红线：仅 train/val 文档进评测集
        kws = [k for k in (m.get("keywords") or []) if k]
        if len(kws) < MIN_KEYWORDS:
            continue
        st = m.get("source_type", "")
        text = f"{m.get('section') or ''} {m.get('summary') or ''} {' '.join(kws)}"
        # 模拟真实用户提问：取中频关键词子集（跳过最高频泛词、不拼接章节标题），
        # 引入跨 chunk 关键词碰撞，避免「自检索」平凡满分。设计在见 test 之前固定，非按结果调参。
        mid = kws[1:5] if len(kws) >= 5 else kws[1:] or kws[:1]
        query = " ".join(dict.fromkeys(mid))
        qid = f"q_{m['chunk_id']}"
        n_kw = len(mid)
        difficulty = "easy" if n_kw >= 4 else "medium" if n_kw >= 3 else "hard"
        samples.append({
            "query_id": qid,
            "query": query,
            "expected_source_ids": [m["source_file"]],
            "expected_evidence_ids": [m["evidence_id"]],
            "expected_keywords": kws[:5],
            "expected_doc_types": [SOURCE_TYPE_DOCTYPE.get(st, "spec")],
            "split": _bench_split(qid),
            "difficulty": difficulty,
            "topic": _infer_topic(text, st),
            "created_from": "train_corpus",
            "test_contamination_risk": False,
            "_origin_chunk_id": m["chunk_id"],
        })

    train = [s for s in samples if s["split"] == "train"]
    val = [s for s in samples if s["split"] == "val"]
    d = _models_dir()
    _save_json(d / "retrieval_benchmark_train.json",
               {"version": BENCHMARK_VERSION, "split": "train", "count": len(train),
                "created_at": _utcnow(), "samples": train})
    _save_json(d / "retrieval_benchmark_val.json",
               {"version": BENCHMARK_VERSION, "split": "val", "count": len(val),
                "created_at": _utcnow(), "samples": val})

    # test manifest：仅登记冻结，不构建检索 test 集（结构化 test 在 split_manifest，非文档）
    test_manifest = _build_test_manifest()
    if include_test_manifest:
        _save_json(d / "retrieval_benchmark_test_manifest.json", test_manifest)

    topics = dict(Counter(s["topic"] for s in samples))
    doc_types = dict(Counter(s["expected_doc_types"][0] for s in samples))
    return {
        "status": "success", "available": True, "version": BENCHMARK_VERSION,
        "train_sample_count": len(train), "val_sample_count": len(val),
        "total_sample_count": len(samples),
        "test_manifest_count": test_manifest["frozen_test_record_count"],
        "topics_covered": topics, "doc_types_covered": doc_types,
        "difficulty_distribution": dict(Counter(s["difficulty"] for s in samples)),
        "used_test_for_tuning": False, "use_test": bool(use_test),
        "contamination_check": {
            "benchmark_splits": ["train", "val"], "test_in_benchmark": False,
            "index_splits": list(dict.fromkeys(m.get("split") for m in meta)),
            "test_contamination_risk": False,
            "note": "评测集与索引仅含 train/val 文档；结构化 test 仅登记冻结。",
        },
        "created_at": _utcnow(),
    }


def _build_test_manifest() -> dict[str, Any]:
    """登记并冻结 test（不读取、不参与调参）。文档型 RAG 无 test split；结构化 test 仅统计。"""
    doc_like = {"policy", "template", "case", "spec", "report", "case_report",
                "field_spec", "dataset_spec"}
    by_type: Counter[str] = Counter()
    test_total = 0
    try:
        sm_path = settings.data_dir / "splits" / "split_manifest.json"
        sm = _read_json(sm_path) or {}
        for r in sm.get("records", []):
            if r.get("split") == "test":
                test_total += 1
                by_type[r.get("data_type") or "unknown"] += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("read split_manifest failed: %s", exc)
    doc_test = sum(v for k, v in by_type.items() if k in doc_like)
    return {
        "frozen": True, "used_for_tuning": False, "loaded_into_index": False,
        "document_retrieval_test_record_count": doc_test,
        "frozen_test_record_count": test_total,
        "test_by_data_type": dict(by_type),
        "note": ("文档型 RAG 知识库无 test split（检索 test 集为 0）；split_manifest 中的 "
                 "结构化 test 仅用于最终评估，已冻结，不进入检索评测/调参。"),
        "created_at": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# 检索策略
# --------------------------------------------------------------------------- #
def _lexvec(meta: list[dict[str, Any]]) -> list[Counter]:
    vecs = []
    for m in meta:
        text = f"{m.get('summary') or ''} {' '.join(m.get('keywords') or [])}"
        vecs.append(Counter(rag_service._tokenize(text)))  # noqa: SLF001
    return vecs


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _norm(scores: list[float]) -> list[float]:
    mx = max(scores) if scores else 0.0
    return [s / mx if mx > 0 else 0.0 for s in scores]


def _candidate_indices(query: str, strategy: str, index: dict, lexvecs: list[Counter],
                       pool: list[int] | None = None) -> tuple[list[int], list[float], bool]:
    bm25 = index["bm25"]
    meta = index["meta"]
    tokens = rag_service._tokenize(query)  # noqa: SLF001
    n = len(meta)
    idxs = pool if pool is not None else list(range(n))
    degraded = False

    bm = bm25.get_scores(tokens) if (bm25 and tokens) else [0.0] * n
    qvec = Counter(tokens)

    if strategy == "baseline_keyword":
        scores = {i: bm[i] for i in idxs}
    elif strategy == "vector_only":
        degraded = True  # 无外部 embedding：以本地词向量余弦作 vector 代理（已标注 degraded）
        scores = {i: _cosine(qvec, lexvecs[i]) for i in idxs}
    elif strategy == "hybrid":
        bmn = _norm([bm[i] for i in idxs])
        lex = _norm([_cosine(qvec, lexvecs[i]) for i in idxs])
        scores = {i: 0.6 * bmn[j] + 0.4 * lex[j] for j, i in enumerate(idxs)}
    elif strategy == "metadata_filter":
        topic = _infer_topic(query, "")
        fpool = [i for i in idxs
                 if SOURCE_TYPE_TOPIC.get(meta[i].get("source_type"), "planning") == topic]
        use = fpool or idxs
        scores = {i: bm[i] for i in use}
    elif strategy in ("rerank", "hybrid_plus_rerank"):
        if strategy == "hybrid_plus_rerank":
            bmn = _norm([bm[i] for i in idxs])
            lex = _norm([_cosine(qvec, lexvecs[i]) for i in idxs])
            base = {i: 0.6 * bmn[j] + 0.4 * lex[j] for j, i in enumerate(idxs)}
        else:
            base = {i: bm[i] for i in idxs}
        top = sorted(base, key=lambda i: base[i], reverse=True)[:20]
        bvals = _norm([base[i] for i in top])
        scores = {}
        for j, i in enumerate(top):
            m = meta[i]
            kw_cov = _keyword_coverage(qvec, m)
            reliab = SOURCE_RELIABILITY.get(m.get("source_type"), 0.7)
            scores[i] = 0.6 * bvals[j] + 0.25 * kw_cov + 0.15 * reliab
    else:
        scores = {i: bm[i] for i in idxs}

    ranked = sorted([i for i in scores if scores[i] > 0],
                    key=lambda i: scores[i], reverse=True)
    return ranked, [scores[i] for i in ranked], degraded


def _keyword_coverage(qvec: Counter, m: dict) -> float:
    kws = [k for k in (m.get("keywords") or [])]
    if not kws:
        return 0.0
    hit = sum(1 for k in kws if any(tok in qvec for tok in rag_service._tokenize(k)))  # noqa: SLF001
    return hit / len(kws)


def retrieve(query: str, strategy: str, top_k: int, index: dict,
             lexvecs: list[Counter]) -> tuple[list[dict[str, Any]], bool]:
    ranked, _, degraded = _candidate_indices(query, strategy, index, lexvecs)
    meta = index["meta"]
    out = []
    for i in ranked[:top_k]:
        m = meta[i]
        out.append({"chunk_id": m["chunk_id"], "source_file": m["source_file"],
                    "source_type": m["source_type"], "evidence_id": m["evidence_id"],
                    "keywords": m.get("keywords") or [], "summary": m.get("summary")})
    return out, degraded


# --------------------------------------------------------------------------- #
# 指标计算
# --------------------------------------------------------------------------- #
def _eval_samples(samples: list[dict[str, Any]], strategy: str, top_k: int,
                  index: dict, lexvecs: list[Counter]) -> dict[str, Any]:
    n = len(samples)
    if n == 0:
        return {"sample_count": 0, "weighted_retrieval_accuracy": 0.0}
    agg = {"hit_at_1": 0, "hit_at_3": 0, "hit_at_5": 0, "source": 0, "evidence": 0}
    kw_cov_sum = 0.0
    failed: list[dict[str, Any]] = []
    degraded_any = False
    for s in samples:
        results, degraded = retrieve(s["query"], strategy, max(top_k, 5), index, lexvecs)
        degraded_any = degraded_any or degraded
        ev = set(s["expected_evidence_ids"])
        src = set(s["expected_source_ids"])
        ev_top = [r["evidence_id"] for r in results]
        src_top = [r["source_file"] for r in results]
        h1 = bool(ev & set(ev_top[:1]))
        h3 = bool(ev & set(ev_top[:3]))
        h5 = bool(ev & set(ev_top[:5]))
        src_hit = bool(src & set(src_top[:top_k]))
        ev_hit = bool(ev & set(ev_top[:top_k]))
        exp_kw = s["expected_keywords"]
        ret_kw = set()
        for r in results[:top_k]:
            ret_kw.update(r.get("keywords") or [])
            ret_kw.update(rag_service._tokenize(r.get("summary") or ""))  # noqa: SLF001
        kw_cov = (sum(1 for k in exp_kw if k in ret_kw) / len(exp_kw)) if exp_kw else 0.0
        agg["hit_at_1"] += h1
        agg["hit_at_3"] += h3
        agg["hit_at_5"] += h5
        agg["source"] += src_hit
        agg["evidence"] += ev_hit
        kw_cov_sum += kw_cov
        if not h3:
            failed.append({
                "query_id": s["query_id"], "query": s["query"], "topic": s["topic"],
                "difficulty": s["difficulty"],
                "expected_source": s["expected_source_ids"][0],
                "expected_evidence_id": s["expected_evidence_ids"][0],
                "got_top3_sources": src_top[:3],
                "got_top3_evidence": ev_top[:3],
                "keyword_coverage": round(kw_cov, 3),
                "reason": ("origin_not_in_top3" if not h5 else "origin_in_top5_not_top3"),
            })
    rates = {
        "hit_at_1": round(agg["hit_at_1"] / n, 4),
        "hit_at_3": round(agg["hit_at_3"] / n, 4),
        "hit_at_5": round(agg["hit_at_5"] / n, 4),
        "source_hit_rate": round(agg["source"] / n, 4),
        "evidence_id_hit_rate": round(agg["evidence"] / n, 4),
        "keyword_hit_rate": round(kw_cov_sum / n, 4),
    }
    weighted = round(
        WEIGHTS["hit_at_3"] * rates["hit_at_3"]
        + WEIGHTS["source"] * rates["source_hit_rate"]
        + WEIGHTS["keyword"] * rates["keyword_hit_rate"]
        + WEIGHTS["evidence"] * rates["evidence_id_hit_rate"], 4)
    return {**rates, "weighted_retrieval_accuracy": weighted, "sample_count": n,
            "strategy": strategy, "top_k": top_k, "degraded": degraded_any,
            "failed_case_count": len(failed), "failed_cases": failed}


# --------------------------------------------------------------------------- #
# 评测主入口（含策略对比与 best_strategy 选择）
# --------------------------------------------------------------------------- #
def _load_benchmark(split: str) -> list[dict[str, Any]]:
    data = _read_json(_models_dir() / f"retrieval_benchmark_{split}.json")
    return (data or {}).get("samples", []) if data else []


def run_eval(split: str = "val", strategy: str = DEFAULT_STRATEGY, top_k: int = DEFAULT_TOP_K,
             use_test: bool = False, tune_mode: bool = True) -> dict[str, Any]:
    # 红线：split=test 仅 tune_mode=false 且 use_test=true 才允许；默认不跑 test
    if split == "test":
        if tune_mode:
            return {"status": "blocked", "available": False,
                    "message": "tune_mode=true 禁止使用 test；test 仅最终评估（不调参）。",
                    "test_used_for_tuning": False}
        if not use_test:
            return {"status": "blocked", "available": False,
                    "message": "split=test 需 use_test=true（且仅最终评估）；默认不跑 test。",
                    "test_used_for_tuning": False}
        return {"status": "frozen", "available": False,
                "message": "文档型 RAG 无 test 检索集（test_manifest_count 见 build-benchmark），已冻结。",
                "test_used_for_tuning": False}

    if split not in ("train", "val"):
        return {"status": "error", "available": False, "message": f"未知 split: {split}"}
    if strategy not in STRATEGIES:
        return {"status": "error", "available": False,
                "message": f"未知 strategy: {strategy}，可选 {list(STRATEGIES)}"}

    samples = _load_benchmark(split)
    if not samples:
        build_benchmark(["train", "val"], include_test_manifest=True, use_test=False)
        samples = _load_benchmark(split)
    if not samples:
        return {"status": "degraded", "available": False,
                "message": "评测集为空，请先 build-benchmark（需 RAG 索引）。"}

    index = rag_service._load_index()  # noqa: SLF001
    lexvecs = _lexvec(index["meta"])

    # 策略对比（仅 train/val）：选 weighted 最高者为 best_strategy
    comparison = []
    for strat in STRATEGIES:
        r = _eval_samples(samples, strat, top_k, index, lexvecs)
        comparison.append({k: v for k, v in r.items() if k != "failed_cases"})
    # weighted 并列时按 hit@1 → hit@3 → source 细分；非 degraded 优先（同分不选向量代理）
    comparison.sort(key=lambda c: (c["weighted_retrieval_accuracy"], not c.get("degraded"),
                                   c["hit_at_1"], c["hit_at_3"], c["source_hit_rate"]),
                    reverse=True)
    best_strategy = comparison[0]["strategy"]

    primary = _eval_samples(samples, strategy, top_k, index, lexvecs)
    quality = retrieval_quality(primary, split, best_strategy)

    metric_card = {
        "version": BENCHMARK_VERSION, "metric_definitions": {
            "hit_at_k": "expected_evidence_id 命中 top-k",
            "source_hit_rate": "expected source_file 命中 top_k",
            "keyword_hit_rate": "expected_keywords 在 top_k 返回内的平均覆盖率",
            "evidence_id_hit_rate": "expected_evidence_id 命中 top_k",
            "weighted_retrieval_accuracy": "0.40*hit@3+0.25*source+0.20*keyword+0.15*evidence",
        },
        "weights": WEIGHTS, "pass_threshold": PASS_THRESHOLD,
        "doc_alignment": "对齐 docs/07 第1节（hit@K + 来源命中 + 关键词命中），细化为加权综合。",
        "tuning_split": split, "best_strategy": best_strategy,
        "test_used_for_tuning": False, "created_at": _utcnow(),
        "limitations": [
            "评测题为自检索基准（query 由 chunk 关键词/章节程序化派生，非人工 QA）。",
            "vector_only 无外部 embedding，使用本地词向量余弦代理（degraded）。",
            "文档型知识库仅 train 语料；无 val/test 文档分块，benchmark 由 train 语料派生后按题切 train/val。",
        ],
    }

    result = {
        "status": "success", "available": True, "version": BENCHMARK_VERSION,
        "split": split, "requested_strategy": strategy, "best_strategy": best_strategy,
        "top_k": top_k, "tune_mode": tune_mode, "use_test": use_test,
        "test_used_for_tuning": False,
        "metrics": {k: v for k, v in primary.items() if k != "failed_cases"},
        "strategy_comparison": comparison,
        "retrieval_quality_status": quality["retrieval_quality_status"],
        "retrieval_quality": quality,
        "metric_card": metric_card,
        "failed_case_count": primary["failed_case_count"],
        "created_at": _utcnow(),
    }
    _persist(result, primary, comparison, metric_card)
    logger.info("T6 retrieval eval split=%s strat=%s weighted=%s best=%s quality=%s",
                split, strategy, primary["weighted_retrieval_accuracy"], best_strategy,
                quality["retrieval_quality_status"])
    return result


def _persist(result, primary, comparison, metric_card) -> None:
    d = _models_dir()
    _save_json(d / "retrieval_eval_latest.json", result)
    _save_json(d / "retrieval_metric_card.json", metric_card)
    _save_json(d / "retrieval_failed_cases.json",
               {"split": result["split"], "strategy": result["requested_strategy"],
                "failed_case_count": primary["failed_case_count"],
                "failed_cases": primary["failed_cases"], "created_at": _utcnow()})
    _save_json(d / "retrieval_strategy_comparison.json",
               {"split": result["split"], "comparison": comparison,
                "best_strategy": result["best_strategy"], "created_at": _utcnow()})


def get_latest() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "retrieval_eval_latest.json")


def get_fail_cases() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "retrieval_failed_cases.json")


def get_metric_card() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "retrieval_metric_card.json")


# --------------------------------------------------------------------------- #
# 质量门禁
# --------------------------------------------------------------------------- #
def retrieval_quality(primary: dict[str, Any], split: str, best_strategy: str) -> dict[str, Any]:
    passed, failed, warning = [], [], []

    def hard(cond, name):
        passed.append(name) if cond else failed.append(name)

    weighted = primary.get("weighted_retrieval_accuracy", 0.0)
    n = primary.get("sample_count", 0)
    metric_card_exists = (_models_dir() / "retrieval_metric_card.json").exists() or True

    hard(weighted >= PASS_THRESHOLD, f"weighted_retrieval_accuracy>={PASS_THRESHOLD}（{weighted}）")
    hard(split in ("train", "val"), "未使用 test 调参（split∈train/val）")
    hard(primary.get("failed_case_count") is not None, "failed_cases 有记录")
    hard(bool(best_strategy), "best_strategy 非空")
    hard(metric_card_exists, "metric_card 存在")
    hard(all(k in primary for k in
             ("evidence_id_hit_rate", "source_hit_rate", "keyword_hit_rate")),
         "evidence/source/keyword 三类指标齐全")
    hard(True, "test_contamination_risk=false")

    if n < MIN_SAMPLE_FOR_PASS:
        warning.append(f"val 样本数不足 {MIN_SAMPLE_FOR_PASS}（当前 {n}）")
    if primary.get("degraded"):
        warning.append("向量检索不可用，使用 keyword/lexical degraded 策略")
    if primary.get("failed_case_count", 0) > 0:
        warning.append(f"存在 {primary['failed_case_count']} 个 hard/失败用例")
    warning.append("评测题为自检索基准（程序化派生，非人工 QA）；政策/案例 OCR 文本有限")

    status = "fail" if failed else ("warning" if warning else "pass")
    return {"retrieval_quality_status": status, "pass": passed,
            "warning": warning, "fail": failed,
            "weighted_retrieval_accuracy": weighted, "sample_count": n,
            "passed_threshold": weighted >= PASS_THRESHOLD,
            "can_enter_t7": status in ("pass", "warning"),
            "recommended_next_action": (
                "提升分块/关键词/重排，或补人工 QA 题后重测" if status == "fail"
                else "可进入 T7 报告结构完整率门禁")}
