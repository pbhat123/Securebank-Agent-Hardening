"""
Evaluation with LangSmith + DeepEval
---------------------------------------------------------
Implements evaluators (custom + LLM-as-judge), computes MRR (Mean Reciprocal Rank), uses DeepEval metrics (faithfulness, hallucination, answer relevancy), 
builds a G-Eval custom metric (empathy), enhances the dataset (adds edge-case examples), and does hill-climbing (top_k=1 vs top_k=5).
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langsmith.evaluation import evaluate
from langsmith import Client

from eval_dataset import EXERCISE_DATASET_NAME, EVAL_EXAMPLES, ensure_exercise_dataset
from eval_dataset import EXERCISE_HC_DATASET_NAME, ensure_exercise_hc_dataset

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

# --- Build pipeline ---
print("Building FinTech support agent...")
agent = build_support_agent(collection_name="eval_exercise")
app = agent["app"]
retriever = agent["retriever"]
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("Pipeline ready.\n")

ensure_exercise_dataset()

# ===================================================================
# LangSmith Evaluators
# ===================================================================

def _judge(prompt, key, **kw):
    """Call judge_llm with a prompt, parse {"score","reason"} JSON."""
    response = judge_llm.invoke(prompt.format_messages(**kw)).content.strip()
    try:
        parsed = json.loads(response[response.find("{"):response.rfind("}") + 1])
        return {"key": key, "score": float(parsed.get("score", 0.5)), "comment": parsed.get("reason", "")}
    except (json.JSONDecodeError, ValueError):
        return {"key": key, "score": 0.5}

def run_agent(inputs):
    question = inputs["question"]
    result = ask(app, question)
    return {
        "answer": result["response"],
        "intent": result["intent"],
        "context": result["context"],
        "retrieved_sources": result["retrieved_sources"],
    }

def routing_evaluator(run, example):
    score = 1.0 if run.outputs.get("intent", "") == example.outputs.get("intent", "") else 0.0
    return {"key": "routing_accuracy", "score": score}

def faithfulness_evaluator(run, example):
    answer = run.outputs.get("answer", "")
    context = run.outputs.get("context", "")
    question = example.inputs.get("question", "")

    if not answer:
        return {"key": "faithfulness", "score": 0.0}
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Assess whether the answer is faithful to the provided context.\n"
         "1.0 = every claim supported by context; 0.5 = some claims unsupported; "
         "0.0 = claims contradict or are absent from context.\n"
         "If the context is empty (escalation response), score 1.0 if the answer is a general "
         "empathetic handoff with no specific policy claims, else 0.5.\n"
         'Respond ONLY with JSON: {{"score": <float>, "reason": "<one sentence>"}}'),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:\n{answer}"),
    ])
    return _judge(prompt, "faithfulness",
                  context=context[:2000] if context else "(no context - escalation response)",
                  question=question, answer=answer)

def correctness_evaluator(run, example):
    actual = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    if not actual or not expected:
        return {"key": "correctness", "score": 0.0}
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Compare the AI's answer to the expected answer. Focus on factual accuracy, not wording.\n"
         "1.0 = all key facts correct; 0.5 = partially correct; 0.0 = key facts wrong or missing.\n"
         "For escalation responses, check empathy and contact info are present.\n"
         'Respond ONLY with JSON: {{"score": <float>, "reason": "<one sentence>"}}'),
        ("human", "Question: {question}\n\nExpected: {expected}\n\nActual: {actual}"),
    ])
    return _judge(prompt, "correctness", question=question, expected=expected, actual=actual)

results = evaluate(
    run_agent,
    data=EXERCISE_DATASET_NAME,
    evaluators=[routing_evaluator, faithfulness_evaluator, correctness_evaluator],
    experiment_prefix="exercise-eval-student",
    metadata={"model": "gpt-4o-mini"},
)
print("Evaluation complete. View results in LangSmith.\n")

# ===================================================================
# MRR (Mean Reciprocal Rank)
# ===================================================================

mrr_queries = [
    # Easy — clearly maps to one document
    {"query": "What credit score do I need for a personal loan?", "relevant_source": "loan_policy.md"},
    {"query": "How do I report identity theft?", "relevant_source": "fraud_policy.md"},
    # Ambiguous — wire transfer fees appear in BOTH account_fees.md and transfer_policy.md
    {"query": "How much does an international wire transfer cost?", "relevant_source": "transfer_policy.md"},
    {"query": "What are the wire transfer fees?", "relevant_source": "account_fees.md"},
    # Cross-domain — "interest rate" matches savings APY AND loan APR
    {"query": "What interest rate will I get?", "relevant_source": "account_fees.md"},
    {"query": "What is the APR on a used car loan?", "relevant_source": "loan_policy.md"},
    # Confusing — "late fee" could match overdraft fee OR loan late payment fee
    {"query": "What happens if I'm late on a payment?", "relevant_source": "loan_policy.md"},
    # Overlapping term — "card replacement" is in account_fees.md AND fraud_policy.md
    {"query": "How much does a replacement debit card cost?", "relevant_source": "account_fees.md"},
    # Vague — "daily limit" is only in transfer_policy.md but could match account_fees.md
    {"query": "What are the daily transaction limits?", "relevant_source": "transfer_policy.md"},
    # Specific but tricky — "two-factor authentication" in both fraud_policy.md and transfer_policy.md
    {"query": "When is two-factor authentication required?", "relevant_source": "fraud_policy.md"},
]

print("=" * 60)
print("SEGMENT 8: MRR COMPUTATION")
print("=" * 60)

reciprocal_ranks = []
for item in mrr_queries:
    docs = retriever.invoke(item["query"])
    rank = next((i for i, d in enumerate(docs, 1)
                 if d.metadata.get("source") == item["relevant_source"]), 0)
    rr = 1.0 / rank if rank else 0.0
    reciprocal_ranks.append(rr)
    print(f"  {item['query'][:50]:50s} | rank: {rank} | RR: {rr:.2f}")

mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
print(f"\n  MRR = {mrr:.4f}\n")

# ===================================================================
# DeepEval
# ===================================================================

print("=" * 60)
print("SEGMENT 9: DEEPEVAL METRICS")
print("=" * 60)

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric

eval_queries = [
    "What is the overdraft fee?",
    "What credit score do I need for a personal loan?",
    "How much does a domestic wire transfer cost?",
    "How long do I have to report unauthorized transactions?",
    "What interest rate will I get?",  # vague: likely to fail
]
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    HallucinationMetric(threshold=0.7),
]
for query in eval_queries:
    result = ask(app, query)
    ctx = [result["context"]] if result["context"] else ["No context retrieved."]
    tc = LLMTestCase(input=query, actual_output=result["response"],
                     retrieval_context=ctx, context=ctx)
    print(f"\n  Q: {query}\n  A: {result['response'][:150]}")
    for m in metrics:
        try:
            m.measure(tc)
            print(f"    {type(m).__name__}: {m.score:.2f} ({'PASS' if m.is_successful() else 'FAIL'})")
        except Exception as e:
            print(f"    {type(m).__name__}: SKIPPED ({type(e).__name__}: {str(e)[:100]})")
print()

# ===================================================================
# G-Eval
# ===================================================================

print("=" * 60)
print("SEGMENT 10: G-EVAL (EMPATHY)")
print("=" * 60)

escalation_queries = [
    "This is ridiculous! Someone withdrew $15,000 from my savings without my permission!",
    "I've been waiting 3 weeks for my fraud dispute to be resolved! This is unacceptable!",
    # Factual policy question — agent gives a dry/robotic answer with no empathy
    "What is the wire transfer fee?",
]

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

empathy_metric = GEval(
    name="Empathy",
    criteria=(
        "Evaluate whether the response shows genuine empathy for the customer's situation. "
        "It should: 1) acknowledge the frustration or distress, 2) validate feelings without "
        "being dismissive, 3) offer a clear next step (escalation, contact info), "
        "4) use warm, professional language. Score 0 if robotic or dismissive, 1 if genuinely empathetic."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)
for query in escalation_queries:
    result = ask(app, query)
    tc = LLMTestCase(input=query, actual_output=result["response"])
    empathy_metric.measure(tc)
    print(f"\n  Q: {query}\n  A: {result['response'][:150]}")
    print(f"  Empathy: {empathy_metric.score:.2f} ({'PASS' if empathy_metric.is_successful() else 'FAIL'})")
    print(f"  Reason: {(empathy_metric.reason or '')[:100]}")

# ===================================================================
# Dataset Enhancement
# ===================================================================

print("=" * 60)
print("SEGMENT 11: DATASET ENHANCEMENT")
print("=" * 60)

client = Client()

new_examples = [
    # a) multi-part question
    {"inputs": {"question": "What is the overdraft fee and is there a daily limit?"},
     "outputs": {"answer": "The overdraft fee is $35 per transaction, with a maximum of 3 overdraft fees per day ($105 maximum).",
                 "intent": "policy"}},
    # b) wrong account number
    {"inputs": {"question": "What is the balance on ACC-00000?"},
     "outputs": {"answer": "I couldn't find account ACC-00000 in our system.",
                 "intent": "account_status"}},
    # c) boundary case: policy says waived if balance is *above* $1,500
    {"inputs": {"question": "If I keep exactly $1,500 in my Premium Checking, do I still pay the monthly fee?"},
     "outputs": {"answer": "The $12.99 fee is waived only if the daily balance stays above $1,500, so exactly $1,500 does not qualify, unless you have a direct deposit of $500 or more per month.",
                 "intent": "policy"}},
]
existing = list(client.list_datasets(dataset_name=EXERCISE_DATASET_NAME))
if existing:
    client.create_examples(
        inputs=[e["inputs"] for e in new_examples],
        outputs=[e["outputs"] for e in new_examples],
        dataset_id=existing[0].id,
    )
    print(f"Added {len(new_examples)} examples to '{EXERCISE_DATASET_NAME}'.\n")

# ===================================================================
# Hill Climbing
# ===================================================================

print("=" * 60)
print("SEGMENT 12: HILL CLIMBING")
print("=" * 60)

# --- Hill climbing dataset ---
ensure_exercise_hc_dataset()

# --- Evaluators ---
def routing_evaluator_hc(run, example):
    predicted = run.outputs.get("intent", "")
    expected = example.outputs.get("intent", "")
    return {"key": "routing_accuracy", "score": 1.0 if predicted == expected else 0.0}

def keyword_correctness_hc(run, example):
    import re
    actual = run.outputs.get("answer", "").lower()
    expected = example.outputs.get("answer", "").lower()
    key_terms = re.findall(r"\$[\d,.]+|\d+(?:\.\d+)?%?|acc-\d+", expected)
    if not key_terms:
        return {"key": "keyword_correctness", "score": 0.5}
    matches = sum(1 for term in key_terms if term in actual)
    return {"key": "keyword_correctness", "score": round(matches / len(key_terms), 4)}

# --- Implement correctness_evaluator ---
def correctness_evaluator_hc(run, example):
    actual = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    if not actual or not expected:
        return {"key": "correctness", "score": 0.0}
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Compare the AI's answer to the expected answer. Focus on numbers, amounts, percentages, "
         "thresholds; wording does not matter.\n"
         "1.0 = all key facts correct; 0.5 = partial; 0.0 = wrong or missing.\n"
         'Respond ONLY with JSON: {{"score": <float>, "reason": "<one sentence>"}}'),
        ("human", "Question: {question}\n\nExpected: {expected}\n\nActual: {actual}"),
    ])
    return _judge(prompt, "correctness", question=question, expected=expected, actual=actual)

hc_evaluators = [routing_evaluator_hc, keyword_correctness_hc, correctness_evaluator_hc]

def make_runner(app_):
    def _run(inputs):
        r = ask(app_, inputs["question"])
        return {"answer": r["response"], "intent": r["intent"],
                "context": r["context"], "retrieved_sources": r["retrieved_sources"]}
    return _run

for top_k, name in [(1, "hill_climb_v1"), (5, "hill_climb_v2")]:
    print(f"\nBuilding agent (chunk_size=200, top_k={top_k})...")
    ag = build_support_agent(collection_name=name, chunk_size=200, chunk_overlap=20, top_k=top_k)
    evaluate(
        make_runner(ag["app"]),
        data=EXERCISE_HC_DATASET_NAME,
        evaluators=hc_evaluators,
        experiment_prefix=f"exercise-hc-topk{top_k}",
        metadata={"model": "gpt-4o-mini", "chunk_size": 200, "top_k": top_k},
    )
print(">>> Compare in LangSmith: Datasets -> fintech-exercise-hill-climb -> Compare")

print("Done.")
