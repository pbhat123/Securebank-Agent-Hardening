"""
Agent Observability with LangSmith
-------------------------------------------------------
LangSmith tracing, run queries, and trace tree inspection.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"  # LANGCHAIN_API_KEY comes from .env

# Verify:
tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "false") == "true"
print(f"LangSmith tracing enabled: {tracing_enabled}")
if not tracing_enabled:
    print("WARNING: Set LANGCHAIN_TRACING_V2=true to enable tracing!")

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

agent = build_support_agent(collection_name="observability_exercise")
app = agent["app"]

print("\n" + "=" * 60)
print("SEGMENT 2: FIRST TRACES")
print("=" * 60)

queries = [
    "What is the overdraft fee?",
    "What is the balance on ACC-12345?",
    "I need to speak to a manager about fraud on my account!",
]

if app is not None:
    for query in queries:
        print(f"\nQuery: {query}")
        result = ask(app, query)
        print(f"  Intent:  {result['intent']}")
        print(f"  Answer:  {result['response'][:200]}")
        print(f"  Sources: {result['retrieved_sources']}")
else:
    print("Complete step 2 first.")

# Query 1 (Policy):
#   Agent selected: policy_agent
#   Total LLM calls: 2 (classify_intent + RAG generation)
#   Most expensive call tokens: ~1,000-1,500 prompt (RAG call w/ retrieved context)
#   Total latency: ~1.5-2.5s
#   Retrieved documents: account_fees.md (primary), possibly loan_policy.md

# Query 2 (Account):
#   Agent selected: account_agent
#   Total LLM calls: 2 (classify_intent + response generation)
#   Most expensive call tokens: ~200-400 prompt (small account record)
#   Total latency: ~0.8-1.2s

# Query 3 (Escalation):
#   Agent selected: escalation_agent
#   Total LLM calls: 2 (classify_intent + empathetic response)
#   Most expensive call tokens: ~100-200 prompt (no context)
#   Total latency: ~0.6-1.0s

print("\n" + "=" * 60)
print("SEGMENT 3: ERROR TRACING")
print("=" * 60)

if app is not None:
    error_query = "What is the balance on ACC-99999?"
    print(f"\nQuery: {error_query}")
    result = ask(app, error_query)
    print(f"  Intent: {result['intent']}")
    print(f"  Answer: {result['response'][:200]}")
else:
    print("Complete step 2 first.")

# Findings:
#   Supervisor routed to: account_agent (correct; ACC-XXXXX pattern detected)
#   Error origin in trace: account_agent node; MOCK_ACCOUNTS lookup returns None
#   Error flow: no exception. The node returns a graceful "account not found"
#               response, so the run tree shows all runs as successful and the
#               failure is visible only in the output (a silent failure).
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("BONUS: TAGGED RUNS")
print("=" * 60)

if app is not None:
    for tag, query in zip(["policy", "account", "escalation"], queries):
        result = app.invoke(
            {"query": query, "intent": "", "response": "", "context": "", "retrieved_sources": []},
            config={"tags": [f"agent-type:{tag}", "exercise"]},
        )
        print(f"[{tag}] intent={result['intent']} | {result['response'][:100]}")
    print(">>> Filter by tag 'exercise' in LangSmith.")
else:
    print("Complete step 2 first.")
