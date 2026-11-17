# 🧠 SecureBank Agent Hardening

> Production-hardening a multi-agent FinTech support bot: tracing, evaluation, guardrails and PII redaction, and cost control.

---

## 1️⃣ Project Title & Value Proposition

**Agent hardening toolkit for AI engineers shipping a multi-agent banking support assistant, making it observable, measurable, safe, and cheap to run.**

A LangGraph support agent (supervisor + policy RAG + account lookup + escalation) is wrapped in four production layers, one script each:

| Script | Layer | Question it answers |
|---|---|---|
| `Observability.py` | 🔭 Tracing | What did the agent actually do? |
| `Evaluation.py` | 📏 Evals | Is it good, and did my change make it better? |
| `Gaurdrails.py` | 🛡️ Guardrails | Can it leak PII or be manipulated? |
| `Optimization.py` | 💸 Cost | What does it cost, and how do I cut it without losing quality? |

---

## 2️⃣ Background & Problem Context

- **Where it appears:** Bank and fintech customer support, where an LLM agent answers fee/policy questions, reads account data, and hands angry customers to humans.
- **Who feels the pain:** Support ops and the engineers on call. Every customer query hits the agent, so every failure mode repeats daily.
- **Why it's hard at scale:**
  - Failures are *silent*. A wrong routing or a "not found" reply still returns HTTP 200 and a green trace.
  - A single leaked SSN or competitor mention is a compliance incident.
  - Token cost grows linearly with traffic, and retrieval settings (chunk size, `top_k`) quietly dominate it.
- **What breaks with manual processes:** Eyeballing a few chats doesn't catch regressions, can't compare two retrieval configs, and leaves no audit trail.

---

## 3️⃣ Target User & Job To Be Done (JTBD)

- **Primary user:** AI / ML engineer responsible for a customer-facing LLM agent in a regulated domain.
- **Secondary users:** Engineering managers (cost and risk), compliance/audit reviewers (audit log), support ops (routing quality).
- **JTBD:** *Take a working prototype agent to a state where I can trace it, score it, block unsafe I/O, and prove what it costs, before it touches real customers.*

---

## 4️⃣ Why an Agentic Approach (Very Important)

The system under test is genuinely agentic, and that is what makes hardening necessary:

- **Why not a script or FAQ chatbot:** Queries mix three different jobs (policy lookup, account data, emotional escalation). A fixed workflow can't tell "What is the overdraft fee?" from "Someone drained my savings!" without understanding intent.
- **Where reasoning is required:**
  - The **supervisor** classifies intent per query and routes dynamically.
  - The **policy agent** retrieves and grounds answers in documents, and must say "I don't know" when context is missing.
  - The **account agent** extracts an account ID and decides between answer / ask-for-ID / not-found.
- **Dynamic decisions:** Which specialist runs, what to retrieve, whether to answer or escalate.

Each dynamic decision is a place the agent can be wrong without crashing, which is exactly what the four layers here detect and contain.

---

## 5️⃣ Agent Role, Scope & Autonomy Level

**Agent owns (end-to-end):** Classifying a customer query, retrieving or looking up the relevant data, and drafting the reply.

**Humans intervene:** Escalations. The escalation agent acknowledges and hands off to `support@securebank.com` / phone; it never resolves the issue itself.

**Restricted actions:**
- No policy claims outside retrieved documents.
- No SSN disclosure, no competitor comparisons, no financial or investment advice.
- Read-only: no money movement or account changes.

---

## 6️⃣ Agent Architecture & Components

```
Query → [Input guards] → Supervisor ─┬─ policy_agent     (RAG over policy docs)
                                     ├─ account_agent    (mock account DB)
                                     └─ escalation_agent (empathetic handoff)
                         → [Output guards: Guardrails AI + Presidio] → Response
```

**a) Planner / Decision layer:** The supervisor makes a single LLM call to pick one of `policy`, `account_status`, `escalation` (static, one-hop routing; falls back to `policy` on unexpected output).

**b) Executors / sub-agents:**
- Policy agent: Chroma retriever + `gpt-4o-mini`, grounded-only prompt.
- Account agent: regex ID extraction, then a lookup in a 3-account mock DB.
- Escalation agent: no tools, empathy-only prompt.

**c) Memory:** Stateless per query. Short-term state lives in the LangGraph `SupportState`; long-term "memory" is the in-memory vector store plus the JSONL audit log.

**d) Orchestration logic:** LangGraph `StateGraph` with conditional edges. There are no retries; the failure handling here is guards (fall back to a safe message) and graceful not-found replies.

---

## 7️⃣ End-to-End Agent Workflow

1. **Input ingestion:** Customer query arrives.
2. **Input guards:** Moderation API → regex filter → LLM injection classifier → Presidio redacts PII from the query.
3. **Context extraction / planning:** Supervisor classifies intent.
4. **Tool execution:** Policy retrieval (top-k chunks) *or* account lookup *or* escalation prompt.
5. **Validation / self-check:** Guardrails AI checks the output for SSN patterns, toxicity, and competitor names.
6. **Output generation:** Presidio redacts PII in the response.
7. **Fallback:** Any blocked step returns a fixed safe message pointing to human support.
8. **Observability:** Every step is traced in LangSmith; cost and audit records are written per query.

---

## 8️⃣ Tools, Models & Stack (With Rationale)

| Component | Role | Why |
|---|---|---|
| **LangGraph / LangChain** | Orchestration | Explicit graph makes routing inspectable and testable |
| **gpt-4o-mini** | Agent LLM + judge + injection classifier | Cheap and deterministic at `temperature=0`, good enough to expose pipeline issues |
| **Chroma + text-embedding-3-small** | Retrieval | In-process, zero infrastructure for a workshop-scale corpus |
| **LangSmith** | Tracing, datasets, experiments | Run-tree visibility and side-by-side experiment comparison |
| **DeepEval** | Faithfulness, relevancy, hallucination, G-Eval (empathy) | Pytest-style metrics for what simple evaluators can't score |
| **Guardrails AI** | Output validators (SSN regex, toxicity, competitors) | Declarative validators with fail-fast behaviour |
| **Presidio** | PII detection and redaction | Local ML/NER, no API key or data egress |
| **OpenAI Moderation API** | Input safety (violence, self-harm, harassment) | Free, intent-aware, ~100 ms |
| **tiktoken + `get_openai_callback`** | Token and cost accounting | Exact counts, per-query cost |

---

## 9️⃣ Evaluation Strategy & Metrics

`Evaluation.py` runs against a labeled LangSmith dataset (policy, account, escalation, out-of-scope cases, plus added edge cases):

- **Routing accuracy:** Supervisor intent vs labeled intent (binary).
- **Faithfulness:** LLM-as-judge, is the answer grounded in the retrieved context?
- **Correctness:** LLM-as-judge vs the expected answer (facts, not wording).
- **Retrieval quality:** MRR over 10 queries, including deliberately ambiguous ones.
- **DeepEval:** Faithfulness, answer relevancy, hallucination (threshold 0.7); G-Eval empathy on escalation responses.
- **Hill climbing:** `top_k=1` vs `top_k=5` with 200-char chunks, compared on correctness.
- **Cost / latency:** From `Optimization.py` (see below).

> 📌 *No benchmark numbers are committed to this repo yet.* Run the scripts to populate your own results; judge-based scores are approximate and vary by run.

---

## 🔟 Guardrails, Trust & Safety

Four strategies, each catching what the previous cannot:

1. **Regex** (<1 ms, free): SSN extraction attempts, investment advice, competitor names, harmful keywords.
2. **Moderation API:** Violence, self-harm, harassment by intent.
3. **ML/NER (Presidio):** Names, emails, phones, credit cards, SSNs, in both query and response.
4. **LLM-based:** Prompt-injection classifier (catches rephrasings regex misses) plus Guardrails AI output validators.

**Where the agent must stop:** Any blocked input or output returns a fixed fallback and routes the customer to human support.
**Override:** Customers are always pointed to a human channel; the guard thresholds are plain code and easy to tune.
**Logging / observability:** LangSmith traces, structured JSON logs with trace IDs, and a JSONL audit log (`audit_log.jsonl`) with query, intent, tokens, cost, and cache-hit per interaction.

---

## 1️⃣1️⃣ Failure Modes & Tradeoffs

- **Silent failures:** "Account not found" and wrong-route cases succeed technically; they're visible only in outputs, so evals are required.
- **SSN exposure by design:** The reference agent leaves the SSN last-4 in the account LLM context to demonstrate the leak; guardrails are the mitigation, not the fix. In production, strip PII *before* the LLM.
- **Presidio quirks:** Known dummy values (e.g. `123-45-6789`) are not detected; names can be over- or under-redacted.
- **Regex brittleness:** Keyword filters block innocent text (e.g. any mention of "invest") and miss rephrasings, hence the LLM layer.
- **LLM-judge variance:** Evaluator scores are non-deterministic and can disagree with human judgement.
- **Cache demo is optimistic:** It replays the queries it was seeded with, so the hit rate is inflated.
- **Tradeoffs accepted:** Every extra guard adds latency and ($0.001-level) cost; smaller chunks and lower `top_k` cut tokens but can drop key facts (checked via a quality-regression test).

---

## 1️⃣2️⃣ Results, Learnings & Insights

Expected, demonstrated behaviours (run it to see your own numbers):

- **What worked:** Tracing makes the 2-LLM-call-per-query structure obvious, and the policy path dominates tokens because it carries retrieved context.
- **What fails initially:** Ambiguous queries ("What interest rate will I get?") retrieve from the wrong document and score poorly on faithfulness and correctness.
- **Surprising:** With tiny chunks, `top_k` becomes the main correctness lever; with large chunks even `top_k=1` suffices.
- **Learning:** Defense in depth matters. Regex alone misses rephrased SSN requests; the LLM classifier catches them, and Presidio covers what both leave.

---

## 1️⃣3️⃣ Future Improvements & Iteration Plan

- Strip PII before the account LLM call instead of relying on output guards.
- Replace the in-memory cache with Redis plus a vector index; add TTL and per-intent cache policies.
- Add the agent module, policy documents and a `requirements.txt` so the repo runs standalone.
- Run evals in CI with regression thresholds; track results over time.
- Add human-in-the-loop approval for escalations and a real account backend with auth.
- Multi-turn memory and retry/fallback logic in the graph.

---

## 1️⃣4️⃣ Demo & Artifacts

```bash
# Run each script from the project root (needs OPENAI_API_KEY and LANGCHAIN_API_KEY in .env)
python Observability.py   # traces + tagged runs
python Evaluation.py      # evaluators, MRR, DeepEval, hill climbing
python Gaurdrails.py      # input/output guard pipeline
python Optimization.py    # baseline vs optimized vs cache, audit log
```

- **Prerequisites:** `langchain`, `langgraph`, `langsmith`, `chromadb`, `openai`, `tiktoken`, `deepeval`, `guardrails-ai`, `presidio-analyzer`, `presidio-anonymizer`, plus Guardrails Hub validators:
  `guardrails hub install hub://guardrails/regex_match hub://guardrails/toxic_language hub://guardrails/competitor_check`
- ⚠️ The scripts import `build_support_agent` / `ask` from a `project/fintech_support_agent.py` module that is **not included in this repo**; add it alongside the scripts (the scripts expect it at `../project`).
- **Artifacts:** LangSmith traces and experiments, and `audit_log.jsonl` (generated on run).

---

## 1️⃣5️⃣ Role-Based Signal (Mandatory)

- **For PMs:** Framing an agent launch around measurable quality, safety, and unit economics; explicit tradeoffs (accuracy vs cost vs latency) and an audit trail for trust.
- **For EMs:** System design across orchestration, evals, guardrails, and cost controls; a repeatable path from prototype to production readiness.
- **For SWEs:** Modular guard layers, deterministic evaluators plus LLM judges, structured logging with trace IDs, and robust fallbacks.

---

## 📄 License

MIT. See [LICENSE](LICENSE).
