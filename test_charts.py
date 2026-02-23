#!/usr/bin/env python3
"""Generate charts with synthetic data to verify visualization pipeline works."""

import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from physics_eval.questions import QUESTIONS
from physics_eval.visualizations import generate_all_charts

# Synthetic test data
models = ["claude-sonnet-4", "claude-opus-4", "gpt-4o", "gpt-4.1", "o3", "o4-mini",
          "gemini-2.5-pro", "gemini-2.5-flash", "deepseek-r1", "grok-3"]

random.seed(42)

grades = []
responses = []
for model in models:
    for q in QUESTIONS:
        correctness = random.choices([0, 1, 2, 3], weights=[0.15, 0.2, 0.3, 0.35])[0]
        rigor = random.choices([0, 1, 2], weights=[0.2, 0.4, 0.4])[0]
        score = correctness + rigor
        labels = {0: "Incorrect", 1: "Poor", 2: "Below Average", 3: "Adequate", 4: "Good", 5: "Excellent"}
        grades.append({
            "question_id": q["question_id"],
            "model_name": model,
            "score": score,
            "correctness_score": correctness,
            "rigor_score": rigor,
            "max_score": 5,
            "score_label": labels[score],
            "attempted": True,
            "hallucinated_formulas": random.random() < 0.1,
            "error_type": random.choice([None, "algebraic", "conceptual"]) if score < 4 else None,
            "justification": "Test grade",
            "model_solution": "Test solution",
            "ground_truth": q["solution_text"],
        })
        responses.append({
            "model_name": model,
            "question_id": q["question_id"],
            "full_response": "Test response",
            "extracted_answer": "Test answer",
            "latency_seconds": random.uniform(1, 30),
            "input_tokens": random.randint(500, 2000),
            "output_tokens": random.randint(200, 1500),
            "timestamp": "2025-01-01T00:00:00Z",
            "confidence": random.randint(30, 95),
        })

# Build summary
from physics_eval.models import estimate_cost
model_stats = {}
for model in models:
    mg = [g for g in grades if g["model_name"] == model]
    mr = [r for r in responses if r["model_name"] == model]
    total_score = sum(g["score"] for g in mg)
    max_score = len(mg) * 5
    accuracy = total_score / max_score if max_score else 0

    correct = sum(1 for g in mg if g["score"] >= 4)
    partial = sum(1 for g in mg if 1 <= g["score"] <= 3)
    wrong = sum(1 for g in mg if g["score"] == 0)

    model_stats[model] = {
        "total_score": total_score,
        "max_score": max_score,
        "accuracy": round(accuracy, 4),
        "accuracy_pct": round(accuracy * 100, 1),
        "correct": correct,
        "partial": partial,
        "incorrect": wrong,
        "avg_rigor": round(sum(g["rigor_score"] for g in mg) / len(mg), 2),
        "avg_correctness": round(sum(g["correctness_score"] for g in mg) / len(mg), 2),
        "avg_latency_seconds": round(sum(r["latency_seconds"] for r in mr) / len(mr), 2),
        "total_input_tokens": sum(r["input_tokens"] for r in mr),
        "total_output_tokens": sum(r["output_tokens"] for r in mr),
        "total_cost_usd": round(sum(estimate_cost(model, r["input_tokens"], r["output_tokens"]) for r in mr), 4),
        "avg_confidence": round(sum(r["confidence"] for r in mr) / len(mr), 1),
        "error_taxonomy": {},
    }

question_stats = {}
for q in QUESTIONS:
    qg = [g for g in grades if g["question_id"] == q["question_id"]]
    avg_score = sum(g["score"] for g in qg) / len(qg) if qg else 0
    question_stats[q["question_id"]] = {
        "avg_score": round(avg_score, 2),
        "num_correct": sum(1 for g in qg if g["score"] >= 4),
        "num_partial": sum(1 for g in qg if 1 <= g["score"] <= 3),
        "num_incorrect": sum(1 for g in qg if g["score"] == 0),
    }

summary = {
    "total_questions": len(QUESTIONS),
    "total_models": len(models),
    "total_api_calls": len(responses),
    "total_tokens_used": sum(r["input_tokens"] + r["output_tokens"] for r in responses),
    "model_stats": model_stats,
    "question_stats": question_stats,
}

paths = generate_all_charts(summary, grades, responses)
print(f"\nGenerated {len(paths)} charts:")
for p in paths:
    print(f"  ✓ {p}")
