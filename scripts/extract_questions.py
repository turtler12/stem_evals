#!/usr/bin/env python3
"""
Extract questions and solutions from all course PDFs.
Uses pdfplumber for text extraction + Claude Sonnet for structured parsing.

Usage:
    python scripts/extract_questions.py                  # Extract all
    python scripts/extract_questions.py --course 8.033   # Extract one course
    python scripts/extract_questions.py --dry-run        # Show what would be extracted
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("pdfplumber not installed. Run: pip install pdfplumber")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("anthropic not installed. Run: pip install anthropic")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "extracted_questions.json"

# PDF pairs: each entry maps questions PDF to solutions PDF
PDF_PAIRS = {
    "8.033": {
        "psets": [
            {
                "questions": f"8.033 psets fall 2025/Problem Set {n}.pdf",
                "solutions": f"8.033 pset solutions fall 2025/Problem Set {n} Solutions.pdf",
                "id_prefix": f"8.033-ps{n}",
                "source_type": "pset",
            }
            for n in range(1, 12)
        ],
        "exams": [
            {
                "questions": "8.033 exams/Final Practice Problems.pdf",
                "solutions": "8.033 exams/8_033_final_practice_solutions.pdf",
                "id_prefix": "8.033-final-practice",
                "source_type": "exam",
            },
            {
                "questions": "8.033 exams/Practice Problems.pdf",
                "solutions": "8.033 exams/Solutions to Practice Problems.pdf",
                "id_prefix": "8.033-midterm1-practice",
                "source_type": "exam",
            },
            {
                "questions": "8.033 exams/Practice Problems for midterm 2.pdf",
                "solutions": "8.033 exams/Midterm 2 Practice Problems Solutions.pdf",
                "id_prefix": "8.033-midterm2-practice",
                "source_type": "exam",
            },
        ],
    },
    "6.1220": {
        "psets": [
            {
                "questions": f"6.1220 psets fall 2023/pset{n}.pdf",
                "solutions": f"6.1220 pset solutions fall 2023/pset{n}_solutions.pdf",
                "id_prefix": f"6.1220-ps{n}",
                "source_type": "pset",
            }
            for n in range(1, 11)
        ],
        "exams": [
            {
                "questions": "6.1220 exams/2024_Spring_Quiz_1.pdf",
                "solutions": "6.1220 exams/2024_Spring_Quiz_1_Solutions (1).pdf",
                "id_prefix": "6.1220-quiz1",
                "source_type": "exam",
            },
            {
                "questions": "6.1220 exams/2024_Spring_Quiz_2 (1).pdf",
                "solutions": "6.1220 exams/2024_Spring_Quiz_2_Solutions.pdf",
                "id_prefix": "6.1220-quiz2",
                "source_type": "exam",
            },
        ],
    },
}

EXTRACTION_PROMPT = """You are extracting individual questions from an MIT course document.

Course: {course}
Document type: {source_type}
Source file: {filename}

Here is the raw text extracted from the QUESTIONS PDF:
---
{questions_text}
---

Here is the raw text from the SOLUTIONS PDF:
---
{solutions_text}
---

Parse this into individual questions. For each question:
1. Identify the question number and any subparts (a), (b), (c), etc.
2. Extract the FULL question text including all setup, context, and constraints. Each sub-part must be self-contained (include the problem setup/context in each sub-part).
3. Match it with the corresponding solution from the solutions document.
4. Assign topic tags appropriate for the course.

Return a JSON array where each element has:
{{
  "question_number": "1a",
  "question_text": "...",
  "solution_text": "...",
  "topic_tags": ["tag1", "tag2"],
  "difficulty_estimate": "easy|medium|hard",
  "requires_figure": false
}}

IMPORTANT RULES:
- Preserve all mathematical notation exactly as it appears (LaTeX, Unicode, etc.)
- If a question REQUIRES a figure/diagram that cannot be described in text, set requires_figure to true
- If a figure is referenced but the question can be understood without it, set requires_figure to false and describe the figure setup in the question text
- Split multi-part questions into separate entries (1a, 1b, 1c, etc.)
- Each sub-part must include enough context to be answered independently
- Do NOT skip any questions
- For {course_type} topic tags, use these categories:
{tag_categories}

Return ONLY the JSON array, no other text."""

PHYSICS_TAGS = """  - special_relativity, general_relativity, lorentz_transformations, time_dilation, length_contraction
  - velocity_addition, doppler_effect, 4-vectors, 4-momentum, 4-velocity
  - tensor_computation, curvature, geodesics, schwarzschild, kruskal
  - particle_physics, electromagnetic, acceleration, gravitational_time_dilation
  - conceptual, computation, proof, short_answer"""

ALGORITHMS_TAGS = """  - sorting, searching, graph_algorithms, dynamic_programming, greedy
  - divide_and_conquer, amortized_analysis, randomized_algorithms, hashing
  - bfs, dfs, shortest_path, minimum_spanning_tree, network_flow
  - recurrence_relations, asymptotic_analysis, lower_bounds
  - data_structures, binary_search_tree, heap, hash_table
  - np_completeness, reduction, approximation_algorithms
  - proof, computation, algorithm_design, analysis"""


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


async def extract_questions_from_pair(
    client: anthropic.AsyncAnthropic,
    course: str,
    pair: dict,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Extract questions from a single PDF pair using Claude."""
    async with semaphore:
        q_path = DATA_DIR / pair["questions"]
        s_path = DATA_DIR / pair["solutions"]

        if not q_path.exists():
            print(f"  [SKIP] Questions PDF not found: {q_path}")
            return []
        if not s_path.exists():
            print(f"  [SKIP] Solutions PDF not found: {s_path}")
            return []

        print(f"  Extracting: {pair['questions']}")

        q_text = extract_pdf_text(q_path)
        s_text = extract_pdf_text(s_path)

        if not q_text.strip():
            print(f"  [SKIP] No text extracted from: {q_path}")
            return []

        course_type = "physics" if course == "8.033" else "algorithms"
        tag_cats = PHYSICS_TAGS if course == "8.033" else ALGORITHMS_TAGS

        prompt = EXTRACTION_PROMPT.format(
            course=course,
            source_type=pair["source_type"],
            filename=pair["questions"],
            questions_text=q_text[:15000],  # Truncate very long PDFs
            solutions_text=s_text[:15000],
            course_type=course_type,
            tag_categories=tag_cats,
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=8192,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()

                # Handle markdown code blocks
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()

                raw_questions = json.loads(text)

                # Post-process: add standardized fields
                processed = []
                for q in raw_questions:
                    if q.get("requires_figure", False):
                        continue  # Skip figure-dependent questions

                    qnum = str(q["question_number"]).lower().replace(" ", "")
                    question_id = f"{pair['id_prefix']}-{qnum}"

                    processed.append({
                        "course": course,
                        "source_file": pair["questions"],
                        "question_id": question_id,
                        "question_text": q["question_text"],
                        "solution_text": q["solution_text"],
                        "topic_tags": q.get("topic_tags", []),
                        "source_type": pair["source_type"],
                        "difficulty_estimate": q.get("difficulty_estimate", "medium"),
                    })

                print(f"    -> {len(processed)} questions extracted (skipped {len(raw_questions) - len(processed)} figure-dependent)")
                return processed

            except (json.JSONDecodeError, KeyError) as e:
                if attempt < max_retries - 1:
                    print(f"    [RETRY] Parse error: {e}")
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"    [ERROR] Failed after {max_retries} attempts: {e}")
                    return []
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    [RETRY] API error: {e}")
                    await asyncio.sleep(2 ** (attempt + 1))
                else:
                    print(f"    [ERROR] API failed: {e}")
                    return []


async def run_extraction(courses: list[str], max_concurrent: int = 3):
    """Extract questions from all specified courses."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    all_questions = []

    for course in courses:
        if course not in PDF_PAIRS:
            print(f"Unknown course: {course}")
            continue

        print(f"\n{'='*60}")
        print(f"Course: {course}")
        print(f"{'='*60}")

        pairs = PDF_PAIRS[course]["psets"] + PDF_PAIRS[course]["exams"]

        tasks = [
            extract_questions_from_pair(client, course, pair, semaphore)
            for pair in pairs
        ]
        results = await asyncio.gather(*tasks)

        for questions in results:
            all_questions.extend(questions)

    # Deduplicate by question_id
    seen = set()
    deduped = []
    for q in all_questions:
        if q["question_id"] not in seen:
            seen.add(q["question_id"])
            deduped.append(q)

    return deduped


def main():
    parser = argparse.ArgumentParser(description="Extract questions from course PDFs")
    parser.add_argument("--course", type=str, default="all",
                        help="Course to extract (8.033, 6.1220, or all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be extracted without calling API")
    parser.add_argument("--max-concurrent", type=int, default=3,
                        help="Max concurrent API calls")
    args = parser.parse_args()

    courses = list(PDF_PAIRS.keys()) if args.course == "all" else [args.course]

    if args.dry_run:
        for course in courses:
            print(f"\nCourse: {course}")
            pairs = PDF_PAIRS[course]["psets"] + PDF_PAIRS[course]["exams"]
            for pair in pairs:
                q_path = DATA_DIR / pair["questions"]
                s_path = DATA_DIR / pair["solutions"]
                q_exists = q_path.exists()
                s_exists = s_path.exists()
                status = "OK" if (q_exists and s_exists) else "MISSING"
                print(f"  [{status}] {pair['id_prefix']}: {pair['questions']}")
                if not q_exists:
                    print(f"           Missing: {q_path}")
                if not s_exists:
                    print(f"           Missing: {s_path}")
        return

    new_questions = asyncio.run(run_extraction(courses, args.max_concurrent))

    # Merge with existing extracted questions (preserve other courses)
    existing = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)
        # Remove questions from courses we just re-extracted
        existing = [q for q in existing if q["course"] not in courses]

    all_questions = existing + new_questions

    # Deduplicate by question_id
    seen = set()
    deduped = []
    for q in all_questions:
        if q["question_id"] not in seen:
            seen.add(q["question_id"])
            deduped.append(q)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"  New questions extracted: {len(new_questions)}")
    print(f"  Total questions (merged): {len(deduped)}")
    for c in sorted(set(q["course"] for q in deduped)):
        course_qs = [q for q in deduped if q["course"] == c]
        print(f"  {c}: {len(course_qs)} questions")
    print(f"  Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
