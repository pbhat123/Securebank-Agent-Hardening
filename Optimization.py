"""
Production-Grade Cost Optimization
-------------------------------------------------------
Builds a production-grade cost measurement and optimization pipeline.

Production features:
  - Structured JSON logging with trace IDs
  - Per-intent cost and latency breakdown
  - Semantic caching with hit-rate measurement
  - Cost threshold alerting and budget tracking
  - Audit logging (JSONL) for fintech compliance
  - Quality regression testing after optimization
"""
import os
import sys
import json
import time
import uuid
import logging
import math
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

import tiktoken
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import OpenAIEmbeddings

# Test queries covering all agent paths
TEST_QUERIES = [
    "What is the overdraft fee?",
    "What credit score do I need for a personal loan?",
    "How much does a domestic wire transfer cost?",
    "How long do I have to report unauthorized transactions?",
    "What is the monthly fee for a Premium Checking account?",
    "What is the balance on ACC-12345?",
    "Show me recent transactions for ACC-67890.",
    "This is terrible service! I want to speak to a manager!",
]

# Quality regression: expected terms in responses for key queries
QUALITY_CHECKS = {
    "What is the overdraft fee?": ["overdraft", "fee"],
    "What credit score do I need for a personal loan?": ["credit", "loan"],
    "What is the balance on ACC-12345?": ["balance", "12450", "12,450"],
    "This is terrible service! I want to speak to a manager!": [
        "support", "specialist", "follow",
    ],
}

# ===================================================================
# Structured JSON Logging
# ===================================================================
class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured observability pipelines."""
    def format(self, record):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            entry.update(record.extra)
        return json.dumps(entry)

app_logger = logging.getLogger("fintech.cost")
app_logger.setLevel(logging.INFO)
app_logger.propagate = False
_console = logging.StreamHandler()
_console.setFormatter(JSONFormatter())
app_logger.addHandler(_console)

AUDIT_LOG_PATH = Path(__file__).parent / "audit_log.jsonl"
audit_logger = logging.getLogger("fintech.audit")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False
_file = logging.FileHandler(AUDIT_LOG_PATH, mode="w", encoding="utf-8")
_file.setFormatter(JSONFormatter())
audit_logger.addHandler(_file)

# ===================================================================
# Semantic Cache
# ===================================================================

class SemanticCache:
    """
    In-memory semantic cache using embedding cosine similarity.
    Production replacement: Redis + vector index, or GPTCache.
    """

    def __init__(self, embedding_model, threshold=0.95):
        self.embeddings = embedding_model
        self.threshold = threshold
        self._store: list[tuple[list[float], str, dict]] = []
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    def get(self, query: str) -> dict | None:
        if not self._store:
            self.misses += 1
            return None
        qvec = self.embeddings.embed_query(query)
        for cvec, _cq, cresult in self._store:
            if self._cosine_sim(qvec, cvec) >= self.threshold:
                self.hits += 1
                return cresult
        self.misses += 1
        return None

    def put(self, query: str, result: dict):
        vec = self.embeddings.embed_query(query)
        self._store.append((vec, query, result))

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

# ===================================================================
# Cost Tracker with Alerting
# ===================================================================

class CostTracker:
    """Tracks token usage, cost, and latency per query and per intent."""

    def __init__(self, daily_budget: float = 1.0, per_query_alert: float = 0.01):
        self.daily_budget = daily_budget
        self.per_query_alert = per_query_alert
        self.records: list[dict] = []
        self.alerts: list[str] = []

    def record(self, *, trace_id: str, query: str, intent: str,
               prompt_tokens: int, completion_tokens: int,
               cost: float, latency_ms: float, cache_hit: bool = False):
        entry = {
            "trace_id": trace_id, "query": query, "intent": intent,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost, "latency_ms": latency_ms, "cache_hit": cache_hit,
        }
        self.records.append(entry)
        if cost > self.per_query_alert:
            self.alerts.append(
                f"HIGH COST: ${cost:.6f} > ${self.per_query_alert} "
                f"(trace={trace_id}, intent={intent})"
            )
        cumulative = sum(r["cost"] for r in self.records)
        if cumulative > self.daily_budget * 0.8:
            self.alerts.append(
                f"BUDGET WARNING: ${cumulative:.4f} > 80% of "
                f"${self.daily_budget} daily budget"
            )

    @property
    def total_cost(self):
        return sum(r["cost"] for r in self.records)

    @property
    def avg_cost(self):
        return self.total_cost / len(self.records) if self.records else 0

    @property
    def avg_prompt(self):
        return (sum(r["prompt_tokens"] for r in self.records)
                / len(self.records)) if self.records else 0

    @property
    def avg_completion(self):
        return (sum(r["completion_tokens"] for r in self.records)
                / len(self.records)) if self.records else 0

    @property
    def avg_latency(self):
        return (sum(r["latency_ms"] for r in self.records)
                / len(self.records)) if self.records else 0

    def intent_summary(self) -> dict:
        by_intent: dict[str, list[dict]] = defaultdict(list)
        for r in self.records:
            by_intent[r["intent"]].append(r)
        summary = {}
        for intent, recs in by_intent.items():
            n = len(recs)
            summary[intent] = {
                "count": n,
                "avg_cost": sum(r["cost"] for r in recs) / n,
                "avg_prompt": sum(r["prompt_tokens"] for r in recs) / n,
                "avg_completion": sum(r["completion_tokens"] for r in recs) / n,
                "avg_latency_ms": sum(r["latency_ms"] for r in recs) / n,
            }
        return summary

def measure(app, tracker, label, cache=None):
    """Run TEST_QUERIES through app; record cost/latency per query. Returns [(query, result)]."""
    results = []
    print(f"{'Intent':<15} {'Prompt':>7} {'Compl':>6} {'Cost':>10} {'ms':>7}  Query")
    for query in TEST_QUERIES:
        trace_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        cached = cache.get(query) if cache is not None else None
        if cached is not None:
            result, p_tok, c_tok, cost, hit = cached, 0, 0, 0.0, True
        else:
            with get_openai_callback() as cb:
                result = ask(app, query)
            p_tok, c_tok, cost, hit = cb.prompt_tokens, cb.completion_tokens, cb.total_cost, False
        latency_ms = (time.perf_counter() - start) * 1000
        tracker.record(trace_id=trace_id, query=query, intent=result["intent"],
                       prompt_tokens=p_tok, completion_tokens=c_tok,
                       cost=cost, latency_ms=latency_ms, cache_hit=hit)
        audit_logger.info("query_processed", extra={"extra": {
            "trace_id": trace_id, "config": label, "query": query, "intent": result["intent"],
            "prompt_tokens": p_tok, "completion_tokens": c_tok, "cost": cost, "cache_hit": hit}})
        app_logger.info("query_cost", extra={"extra": {
            "trace_id": trace_id, "config": label, "cost": cost, "latency_ms": round(latency_ms, 1)}})
        print(f"{result['intent']:<15} {p_tok:>7} {c_tok:>6} ${cost:>9.6f} {latency_ms:>7.0f}  {query[:40]}"
              f"{' [CACHE HIT]' if hit else ''}")
        results.append((query, result))
    for alert in tracker.alerts:
        print(f"  ALERT: {alert}")
    return results

# ===================================================================
# TOKEN COUNTING
# ===================================================================

print("=" * 70)
print("SEGMENT 1: TOKEN COUNTING")
print("=" * 70)

supervisor_prompt = (
    "Classify the customer query into exactly one category:\n"
    "- \"policy\" — general questions about account fees, loans, transfers, "
    "fraud policies, or banking products\n"
    "- \"account_status\" — requests to check balance, view transactions, "
    "or look up a SPECIFIC account (usually contains an account number like ACC-XXXXX)\n"
    "- \"escalation\" — complaints, frustration, requests for a manager, "
    "fraud reports, or complex issues needing human attention\n\n"
    "Respond with ONLY the category name."
)

encoder = tiktoken.encoding_for_model("gpt-4o-mini")
prompt_tokens = len(encoder.encode(supervisor_prompt))
print(f"Supervisor prompt: {prompt_tokens} tokens")
print(f"Hidden cost: {prompt_tokens} x 1,000 queries/day = {prompt_tokens * 1000:,} tokens/day\n")

# ===================================================================
# BEFORE / AFTER COMPARISON
# ===================================================================

baseline_agent = build_support_agent(
    collection_name="exercise_baseline", chunk_size=1000, chunk_overlap=100, top_k=5
)

baseline_tracker = CostTracker(daily_budget=1.0, per_query_alert=0.005)
baseline_results = []

if baseline_agent is not None:
    baseline_app = baseline_agent["app"]
    print(f"\n{'=' * 70}")
    print(f"MEASURING: BEFORE: Baseline (chunk=1000, k=5)")
    print(f"{'=' * 70}")

    baseline_results = measure(baseline_app, baseline_tracker, "baseline")

else:
    print("Complete step 3 first.")

# ---------------------------------------------------------------------------
# Build the OPTIMIZED pipeline
#
# Build_support_agent() with:
#   - collection_name="exercise_optimized"
#   - chunk_size=400, chunk_overlap=50
#   - top_k=3
# ---------------------------------------------------------------------------
optimized_agent = build_support_agent(
    collection_name="exercise_optimized", chunk_size=400, chunk_overlap=50, top_k=3
)

optimized_tracker = CostTracker(daily_budget=1.0, per_query_alert=0.005)
optimized_results = []

if optimized_agent is not None:
    optimized_app = optimized_agent["app"]
    print(f"\n{'=' * 70}")
    print(f"MEASURING: AFTER: Optimized (chunk=400, k=3)")
    print(f"{'=' * 70}")

    optimized_results = measure(optimized_app, optimized_tracker, "optimized")

else:
    print("Complete step 5 first.")

cached_tracker = CostTracker(daily_budget=1.0, per_query_alert=0.005)
cached_results = []

if optimized_results:
    print("\n\nDemonstrating semantic caching...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    cache = SemanticCache(embeddings, threshold=0.95)
    for query, result in optimized_results:
        cache.put(query, result)
    cached_results = measure(optimized_app, cached_tracker, "optimized+cache", cache=cache)
    print(f"\n  Cache hit rate: {cache.hit_rate:.0%} ({cache.hits} hits / {cache.misses} misses)")
else:
    print("Complete steps 5-6 first for caching demo.")

print(f"\n{'=' * 85}")
print("FULL COMPARISON: BASELINE -> OPTIMIZED -> OPTIMIZED + CACHE")
print(f"{'=' * 85}")

def safe_pct(before_val, after_val):
    return (before_val - after_val) / before_val * 100 if before_val else 0

if baseline_results and optimized_results and cached_results:
    b, o, c = baseline_tracker, optimized_tracker, cached_tracker
    rows = [
        ("Avg prompt tokens", b.avg_prompt, o.avg_prompt, c.avg_prompt, "{:.0f}"),
        ("Avg completion tokens", b.avg_completion, o.avg_completion, c.avg_completion, "{:.0f}"),
        ("Avg cost/query ($)", b.avg_cost, o.avg_cost, c.avg_cost, "{:.6f}"),
        ("Avg latency (ms)", b.avg_latency, o.avg_latency, c.avg_latency, "{:.0f}"),
        ("Total cost ($)", b.total_cost, o.total_cost, c.total_cost, "{:.6f}"),
    ]
    print(f"{'Metric':<24}{'BASELINE':>12}{'OPTIMIZED':>12}{'+ CACHE':>12}{'Savings %':>12}")
    for name, bv, ov, cv, fmt in rows:
        print(f"{name:<24}{fmt.format(bv):>12}{fmt.format(ov):>12}{fmt.format(cv):>12}"
              f"{safe_pct(bv, cv):>11.1f}%")
    print(f"\nCache hit rate: {cache.hit_rate:.0%}")
else:
    print("Complete steps above, then build comparison table here.")

print(f"\n{'=' * 85}")
print("PER-INTENT COST BREAKDOWN (Baseline)")
print(f"{'=' * 85}")

summary = baseline_tracker.intent_summary()
if summary:
    print(f"{'Intent':<16}{'Count':>6}{'Avg Prompt':>12}{'Avg Compl':>11}{'Avg Cost':>12}{'Avg ms':>9}")
    for intent, st in summary.items():
        print(f"{intent:<16}{st['count']:>6}{st['avg_prompt']:>12.0f}{st['avg_completion']:>11.0f}"
              f"${st['avg_cost']:>11.6f}{st['avg_latency_ms']:>9.0f}")
else:
    print("Complete steps above, then print intent breakdown here.")

print(f"\n{'=' * 85}")
print("QUALITY REGRESSION CHECK")
print(f"{'=' * 85}")

base_by_q = dict(baseline_results)
opt_by_q = dict(optimized_results)
passed = total = 0
for query, terms in QUALITY_CHECKS.items():
    if query not in opt_by_q:
        continue
    total += 1
    base_ok = any(t.lower() in base_by_q[query]["response"].lower() for t in terms)
    opt_ok = any(t.lower() in opt_by_q[query]["response"].lower() for t in terms)
    passed += opt_ok
    print(f"  {'PASS' if opt_ok else 'FAIL'} (baseline: {'PASS' if base_ok else 'FAIL'}) {query[:55]}")
if total:
    print(f"\n  Optimized quality: {passed}/{total} checks passed")

print(f"\n{'=' * 85}")
print("PROJECTED SAVINGS")
print(f"{'=' * 85}")

QPD = 1000
if baseline_results and optimized_results:
    base_daily = baseline_tracker.avg_cost * QPD
    print(f"  {'Config':<20}{'Daily':>10}{'Monthly':>11}{'Annual':>11}   Saved/yr")
    for name, trk in [("Baseline", baseline_tracker), ("Optimized", optimized_tracker),
                      ("Optimized + cache", cached_tracker)]:
        daily = trk.avg_cost * QPD
        saved = (base_daily - daily) * 365
        print(f"  {name:<20}${daily:>9.2f}${daily * 30:>10.2f}${daily * 365:>10.2f}   ${saved:,.2f}")
    print("  (cache hit rate in this demo is artificially high: it replays the same queries)")

print(f"\n{'=' * 85}")
print("AUDIT & COMPLIANCE")
print(f"{'=' * 85}")

print(f"  Audit log: {AUDIT_LOG_PATH}")
print("  JSONL, one record per query: trace_id, config, query, intent, tokens, cost, cache_hit, UTC timestamp.")
print("  Purpose: immutable-style record of every customer interaction for fintech compliance,")
print("  cost attribution per trace, incident investigation, and regulator audits.")
print("  Note: the log stores raw queries; apply PII redaction (Module C) before writing in production.")
