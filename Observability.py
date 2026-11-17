"""
Module A Exercise: Agent Observability with LangSmith
-------------------------------------------------------
Set up LangSmith tracing, run queries, and inspect the trace tree.

Segments covered:
  2. LangSmith setup & first traces
  3. Trace anatomy — debugging tool calls, latency, token counts
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# TODO 1: Enable LangSmith tracing
#
# Set these environment variables (or add them to your .env file):
#   LANGCHAIN_TRACING_V2 = "true"
#   LANGCHAIN_API_KEY    = "<your LangSmith API key>"
#
# Verify tracing is enabled by checking the env var.
# Free Developer plan: https://smith.langchain.com (1 seat, 5K traces/month)
# ---------------------------------------------------------------------------
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # LANGCHAIN_API_KEY comes from .env

# Verify:
tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "false") == "true"
print(f"LangSmith tracing enabled: {tracing_enabled}")
if not tracing_enabled:
    print("WARNING: Set LANGCHAIN_TRACING_V2=true to enable tracing!")


sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

# ---------------------------------------------------------------------------
# TODO 2: Build the multi-agent pipeline
#
# Use build_support_agent() with collection_name="observability_exercise"
# ---------------------------------------------------------------------------
agent = build_support_agent(collection_name="observability_exercise")
app = agent["app"]

# ---------------------------------------------------------------------------
# TODO 3: Run 3 different queries — one for each agent type
#
# Run these queries through the agent and print the results:
#   a) A policy question (e.g., "What is the overdraft fee?")
#   b) An account lookup (e.g., "What is the balance on ACC-12345?")
#   c) An escalation (e.g., "I need to speak to a manager about fraud!")
#
# For each, print: intent, response (first 200 chars), retrieved_sources
# ---------------------------------------------------------------------------
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
    print("Complete TODO 2 first.")


# ---------------------------------------------------------------------------
# TODO 4: Inspect traces in LangSmith
#
# Open https://smith.langchain.com and find your traces.
# For EACH of the 3 traces above, answer these questions:
#
# a) Which agent did the supervisor route to?
# b) How many total LLM calls were made? (supervisor + agent)
# c) How many prompt tokens were used in the most expensive call?
# d) What was the total latency of the trace?
# e) For the policy query: which documents were retrieved?
#
# Write your answers as comments below:
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TODO 5: Inject a deliberate failure and trace the error path
#
# Run a query with a non-existent account number (e.g., ACC-99999).
# Then find the trace in LangSmith and answer:
#   - Did the supervisor route correctly?
#   - Where in the trace tree does the "not found" response originate?
#   - How does error information flow through the run tree?
# ---------------------------------------------------------------------------
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
    print("Complete TODO 2 first.")

# Write your findings:
#   Supervisor routed to: account_agent (correct; ACC-XXXXX pattern detected)
#   Error origin in trace: account_agent node; MOCK_ACCOUNTS lookup returns None
#   Error flow: no exception. The node returns a graceful "account not found"
#               response, so the run tree shows all runs as successful and the
#               failure is visible only in the output (a silent failure).


# ---------------------------------------------------------------------------
# TODO 6 (BONUS): Tag your runs for monitoring
#
# Re-run the 3 queries from TODO 3, but this time add tags
# using the config parameter:
#   result = app.invoke(inputs, config={"tags": ["your-tag"]})
#
# Then filter by your tag in the LangSmith dashboard.
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
    print("Complete TODO 2 first.")
