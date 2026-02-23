"""
Automated grading using Claude Sonnet as a judge.
Scores model answers against ground truth solutions on a 0-5 scale.
Split into Correctness (0-3) and Mathematical Rigor (0-2).
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path


MAX_SCORE = 5


@dataclass
class Grade:
    question_id: str
    model_name: str
    score: int  # 0-5 total (correctness + rigor)
    correctness_score: int  # 0-3
    rigor_score: int  # 0-2
    max_score: int  # always 5
    score_label: str  # "Excellent", "Good", "Adequate", "Below Average", "Poor", "Incorrect"
    attempted: bool
    hallucinated_formulas: bool
    error_type: str | None
    justification: str
    model_solution: str
    ground_truth: str


PHYSICS_GRADING_PROMPT = """You are an expert physics grader for MIT's 8.033 (Special and General Relativity) course.
You must grade STRICTLY. Do not give benefit of the doubt.

You are given:
1. A QUESTION from a problem set or exam
2. A MODEL'S ANSWER (from an AI model being evaluated)
3. The GROUND TRUTH SOLUTION (from the official answer key)

Grade using TWO dimensions:

CORRECTNESS (0-3 points):
- 3: Final answer is exactly correct (matching ground truth up to equivalent reformulations). All intermediate steps are correct.
- 2: Final answer has a minor arithmetic or algebra error (e.g., dropped factor of 2, sign error) but the setup and method are fully correct. The error must be clearly identifiable and localized.
- 1: Correct general approach (right physics principles, right starting equations) but the final answer is wrong due to a significant error. OR the answer is partially correct (e.g., gets one of multiple required results right).
- 0: Wrong answer AND wrong approach. OR the model refused to answer. OR the model used fundamentally wrong physics.

MATHEMATICAL RIGOR (0-2 points):
- 2: All steps shown explicitly. No hand-waving. Equations properly derived, not just stated. Assumptions clearly stated. Tensor indices correct. Units/dimensions consistent throughout.
- 1: Most steps shown but some non-trivial steps skipped or hand-waved. Minor notational issues. Generally follows a logical path but with gaps.
- 0: Significant steps missing. Claims results without derivation. Sloppy notation that changes meaning. Circular reasoning. Just states the answer with no work.

TOTAL SCORE = CORRECTNESS + RIGOR (0-5)

STRICT RULES:
- "Close enough" is NOT acceptable. If ground truth says R = -12/l^2 and model says R = -12/l, that is WRONG (correctness 0-1).
- Correct answer with no work shown: correctness 3, rigor 0 = total 3.
- Correct approach with wrong answer: correctness 1-2 depending on severity, rigor scored independently.
- Phrases like "it can be shown that" or "after simplification" without showing the work: rigor cannot be 2.
- If the model gets the right answer but skips key derivation steps, rigor MUST be 0 or 1.

Respond with ONLY a JSON object (no markdown):
{{
    "correctness_score": <0-3>,
    "rigor_score": <0-2>,
    "score": <0-5, must equal correctness_score + rigor_score>,
    "score_label": "<Excellent|Good|Adequate|Below Average|Poor|Incorrect>",
    "attempted": <true/false>,
    "hallucinated_formulas": <true/false>,
    "error_type": <null | "algebraic" | "conceptual" | "hallucinated_formula" | "gave_up" | "misread" | "missing_steps" | "dimensional_error">,
    "justification": "<2-4 sentence explanation>"
}}

QUESTION:
{question}

MODEL'S ANSWER:
{model_answer}

GROUND TRUTH SOLUTION:
{ground_truth}

Remember: respond with ONLY the JSON object, no other text."""


ALGORITHMS_GRADING_PROMPT = """You are an expert algorithms grader for MIT's 6.1220 (Design and Analysis of Algorithms) course.
You must grade STRICTLY. Do not give benefit of the doubt.

You are given:
1. A QUESTION from a problem set or exam
2. A MODEL'S ANSWER (from an AI model being evaluated)
3. The GROUND TRUTH SOLUTION (from the official answer key)

Grade using TWO dimensions:

CORRECTNESS (0-3 points):
- 3: Algorithm/proof is correct. Complexity analysis matches ground truth. All edge cases handled.
- 2: Core algorithm/proof idea is correct but has a minor error (e.g., off-by-one, missing edge case, minor complexity analysis error).
- 1: Correct general approach but significant errors in the details. OR partially correct (gets runtime but not the algorithm, or vice versa).
- 0: Fundamentally wrong algorithm/proof, or no real attempt.

RIGOR (0-2 points):
- 2: Proof is formally structured (clear induction base/step, contradiction setup, etc.). Algorithm has clear pseudocode. Complexity analysis uses proper notation and is fully derived.
- 1: Proof/analysis has the right structure but skips some steps. Pseudocode is informal. Big-O analysis stated but not derived.
- 0: No formal structure. Claims without justification. Missing complexity analysis. Proof by handwaving.

TOTAL = CORRECTNESS + RIGOR (0-5)

STRICT RULES:
- If the algorithm has the wrong time complexity, correctness cannot be 3.
- If a proof is missing the base case or a key step, rigor cannot be 2.
- Stating "this runs in O(n log n)" without justification: rigor 0.

Respond with ONLY a JSON object (no markdown):
{{
    "correctness_score": <0-3>,
    "rigor_score": <0-2>,
    "score": <0-5, must equal correctness_score + rigor_score>,
    "score_label": "<Excellent|Good|Adequate|Below Average|Poor|Incorrect>",
    "attempted": <true/false>,
    "hallucinated_formulas": <true/false>,
    "error_type": <null | "algorithmic" | "complexity_analysis" | "proof_gap" | "gave_up" | "misread">,
    "justification": "<2-4 sentence explanation>"
}}

QUESTION:
{question}

MODEL'S ANSWER:
{model_answer}

GROUND TRUTH SOLUTION:
{ground_truth}

Remember: respond with ONLY the JSON object, no other text."""


SCORE_LABELS = {
    5: "Excellent",
    4: "Good",
    3: "Adequate",
    2: "Below Average",
    1: "Poor",
    0: "Incorrect",
}


async def grade_single(
    question_id: str,
    question_text: str,
    model_name: str,
    model_answer: str,
    ground_truth: str,
    grader_api_key: str,
    course: str = "8.033",
) -> Grade:
    """Grade a single model response using Claude Sonnet as judge."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=grader_api_key)

    if course == "6.1220":
        prompt_template = ALGORITHMS_GRADING_PROMPT
    else:
        prompt_template = PHYSICS_GRADING_PROMPT

    prompt = prompt_template.format(
        question=question_text,
        model_answer=model_answer,
        ground_truth=ground_truth,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Parse JSON — handle markdown code blocks if model wraps it
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)

            correctness = int(result["correctness_score"])
            rigor = int(result["rigor_score"])
            total = correctness + rigor

            return Grade(
                question_id=question_id,
                model_name=model_name,
                score=total,
                correctness_score=correctness,
                rigor_score=rigor,
                max_score=MAX_SCORE,
                score_label=result.get("score_label", SCORE_LABELS.get(total, "Unknown")),
                attempted=result["attempted"],
                hallucinated_formulas=result["hallucinated_formulas"],
                error_type=result.get("error_type"),
                justification=result["justification"],
                model_solution=model_answer,
                ground_truth=ground_truth,
            )

        except (json.JSONDecodeError, KeyError) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return Grade(
                    question_id=question_id,
                    model_name=model_name,
                    score=0,
                    correctness_score=0,
                    rigor_score=0,
                    max_score=MAX_SCORE,
                    score_label="Grading Error",
                    attempted=True,
                    hallucinated_formulas=False,
                    error_type=None,
                    justification=f"Grading failed after {max_retries} attempts: {e}. Raw: {text[:200]}",
                    model_solution=model_answer,
                    ground_truth=ground_truth,
                )
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** (attempt + 1))
            else:
                return Grade(
                    question_id=question_id,
                    model_name=model_name,
                    score=0,
                    correctness_score=0,
                    rigor_score=0,
                    max_score=MAX_SCORE,
                    score_label="Grading Error",
                    attempted=True,
                    hallucinated_formulas=False,
                    error_type=None,
                    justification=f"API error: {e}",
                    model_solution=model_answer,
                    ground_truth=ground_truth,
                )


async def grade_all(
    responses: list,  # list of ModelResponse
    questions: list[dict],  # question dicts with solution_text
    grader_api_key: str,
    max_concurrent: int = 5,
) -> list[Grade]:
    """Grade all model responses. Rate-limited to max_concurrent at a time."""
    question_map = {q["question_id"]: q for q in questions}
    semaphore = asyncio.Semaphore(max_concurrent)

    async def grade_with_limit(resp):
        async with semaphore:
            q = question_map.get(resp.question_id)
            if not q:
                return None
            if resp.error:
                return Grade(
                    question_id=resp.question_id,
                    model_name=resp.model_name,
                    score=0,
                    correctness_score=0,
                    rigor_score=0,
                    max_score=MAX_SCORE,
                    score_label="Error",
                    attempted=False,
                    hallucinated_formulas=False,
                    error_type="gave_up",
                    justification=f"Model returned error: {resp.error}",
                    model_solution="",
                    ground_truth=q["solution_text"],
                )
            course = q.get("course", "8.033")
            return await grade_single(
                question_id=resp.question_id,
                question_text=q["question_text"],
                model_name=resp.model_name,
                model_answer=resp.full_response,
                ground_truth=q["solution_text"],
                grader_api_key=grader_api_key,
                course=course,
            )

    tasks = [grade_with_limit(r) for r in responses]
    grades = await asyncio.gather(*tasks)
    return [g for g in grades if g is not None]


def save_grades(grades: list[Grade], output_path: str = "results/grades.json"):
    """Save all grades to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(g) for g in grades], f, indent=2)
    return output_path


def compute_summary(grades: list[Grade], responses: list) -> dict:
    """Compute aggregate statistics."""
    from physics_eval.models import estimate_cost, MODEL_REGISTRY

    models = sorted(set(g.model_name for g in grades))
    questions = sorted(set(g.question_id for g in grades))

    # Per-model stats
    model_stats = {}
    for model in models:
        mg = [g for g in grades if g.model_name == model]
        mr = [r for r in responses if r.model_name == model]

        total_score = sum(g.score for g in mg)
        max_score = len(mg) * MAX_SCORE
        accuracy = total_score / max_score if max_score > 0 else 0

        # Score distribution (0-5 scale)
        excellent = sum(1 for g in mg if g.score == 5)
        good = sum(1 for g in mg if g.score == 4)
        adequate = sum(1 for g in mg if g.score == 3)
        below_avg = sum(1 for g in mg if g.score == 2)
        poor = sum(1 for g in mg if g.score == 1)
        wrong = sum(1 for g in mg if g.score == 0)

        # Backward-compatible buckets
        correct = sum(1 for g in mg if g.score >= 4)  # 4-5 = "correct"
        partial = sum(1 for g in mg if 1 <= g.score <= 3)  # 1-3 = "partial"

        # Rigor stats
        avg_rigor = sum(g.rigor_score for g in mg) / len(mg) if mg else 0
        avg_correctness = sum(g.correctness_score for g in mg) / len(mg) if mg else 0

        avg_latency = sum(r.latency_seconds for r in mr) / len(mr) if mr else 0
        total_input_tokens = sum(r.input_tokens for r in mr)
        total_output_tokens = sum(r.output_tokens for r in mr)
        total_cost = sum(estimate_cost(model, r.input_tokens, r.output_tokens) for r in mr)

        # Confidence calibration
        confidences = [r.confidence for r in mr if r.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        # Error taxonomy
        error_counts = {}
        for g in mg:
            if g.error_type:
                error_counts[g.error_type] = error_counts.get(g.error_type, 0) + 1

        model_stats[model] = {
            "total_score": total_score,
            "max_score": max_score,
            "accuracy": round(accuracy, 4),
            "accuracy_pct": round(accuracy * 100, 1),
            "correct": correct,
            "partial": partial,
            "incorrect": wrong,
            "score_distribution": {
                "excellent": excellent,
                "good": good,
                "adequate": adequate,
                "below_average": below_avg,
                "poor": poor,
                "incorrect": wrong,
            },
            "avg_rigor": round(avg_rigor, 2),
            "avg_correctness": round(avg_correctness, 2),
            "avg_latency_seconds": round(avg_latency, 2),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost_usd": round(total_cost, 4),
            "avg_confidence": round(avg_confidence, 1) if avg_confidence else None,
            "error_taxonomy": error_counts,
        }

    # Per-question difficulty
    question_stats = {}
    for qid in questions:
        qg = [g for g in grades if g.question_id == qid]
        avg_score = sum(g.score for g in qg) / len(qg) if qg else 0
        question_stats[qid] = {
            "avg_score": round(avg_score, 2),
            "num_correct": sum(1 for g in qg if g.score >= 4),
            "num_partial": sum(1 for g in qg if 1 <= g.score <= 3),
            "num_incorrect": sum(1 for g in qg if g.score == 0),
        }

    # Overall
    total_api_calls = len(responses)
    total_tokens = sum(r.input_tokens + r.output_tokens for r in responses)

    return {
        "total_questions": len(questions),
        "total_models": len(models),
        "total_api_calls": total_api_calls,
        "total_tokens_used": total_tokens,
        "model_stats": model_stats,
        "question_stats": question_stats,
    }


def save_summary(summary: dict, output_path: str = "results/summary.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    return output_path
