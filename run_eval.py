#!/usr/bin/env python3
"""
STEMEval Benchmark v2.0
CLI runner for physics/algorithms model evaluation pipeline.

Usage:
    python run_eval.py --models all --questions all
    python run_eval.py --models claude-sonnet-4,gpt-4o --course 8.033
    python run_eval.py --models all --course 6.1220
    python run_eval.py --list-models
    python run_eval.py --check-keys
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from physics_eval.questions import get_questions, get_formula_sheet, save_questions_json
from physics_eval.models import (
    MODEL_REGISTRY, call_model, get_available_models, get_missing_keys,
    estimate_cost, ModelResponse, load_cached, CACHE_DIR
)
from physics_eval.grading import grade_all, save_grades, compute_summary, save_summary, Grade
from physics_eval.visualizations import generate_all_charts
from physics_eval.twitter import generate_twitter_thread


def check_keys():
    """Check which API keys are available."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="API Key Status")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="magenta")
    table.add_column("Env Var", style="yellow")
    table.add_column("Status", style="bold")

    for name, (provider, model_id) in MODEL_REGISTRY.items():
        from physics_eval.models import API_KEY_ENV
        env = API_KEY_ENV[provider]
        has_key = bool(os.environ.get(env))
        status = "[green]✓ Set[/green]" if has_key else "[red]✗ Missing[/red]"
        table.add_row(name, provider, env, status)

    console.print(table)

    missing = get_missing_keys()
    if missing:
        console.print(f"\n[yellow]To set missing keys, run:[/yellow]")
        seen = set()
        for model, env in missing.items():
            if env not in seen:
                console.print(f"  export {env}='your-key-here'")
                seen.add(env)


async def run_evaluation(model_names: list[str], questions: list[dict], max_concurrent: int = 3):
    """Run evaluation with live progress dashboard."""
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    total_tasks = len(model_names) * len(questions)
    completed = 0
    results: list[ModelResponse] = []
    running_scores: dict[str, list[int]] = {m: [] for m in model_names}

    # Track per-model status
    model_status = {m: {"done": 0, "total": len(questions), "errors": 0, "cached": 0, "last_error": ""} for m in model_names}

    def make_table():
        table = Table(title="STEMEval Benchmark v2.0 — Live Progress", border_style="bright_blue")
        table.add_column("Model", style="cyan", width=20)
        table.add_column("Progress", style="white", width=15)
        table.add_column("Success", style="green", width=12)
        table.add_column("Errors", style="red", width=8)
        table.add_column("Cached", style="yellow", width=8)
        table.add_column("Status", style="bold", width=12)
        table.add_column("Last Error", style="red", width=40, no_wrap=True)

        for m in model_names:
            s = model_status[m]
            pct = f"{s['done']}/{s['total']}"
            scores = running_scores[m]
            if scores:
                success = sum(1 for x in scores if x is not None)
                acc = f"{success}/{len(scores)}"
            else:
                acc = "—"
            status = "[green]✓ Done[/green]" if s["done"] == s["total"] else "[yellow]Running...[/yellow]"
            last_err = s["last_error"][:40] if s["last_error"] else ""
            table.add_row(m, pct, acc, str(s["errors"]), str(s["cached"]), status, last_err)

        return table

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(model_name: str, question: dict, live: Live):
        nonlocal completed
        async with semaphore:
            # Check cache
            cached = load_cached(model_name, question["question_id"])
            if cached:
                model_status[model_name]["cached"] += 1
                model_status[model_name]["done"] += 1
                running_scores[model_name].append(1)  # Placeholder
                results.append(cached)
                completed += 1
                live.update(make_table())
                return

            course = question.get("course", "8.033")
            formula_sheet = get_formula_sheet(course)
            resp = await call_model(model_name, question["question_id"],
                                    question["question_text"], formula_sheet, course)
            results.append(resp)
            model_status[model_name]["done"] += 1
            if resp.error:
                model_status[model_name]["errors"] += 1
                model_status[model_name]["last_error"] = resp.error
                running_scores[model_name].append(None)
            else:
                running_scores[model_name].append(1)
            completed += 1
            live.update(make_table())

    console.print(Panel(
        f"[bold cyan]STEMEval Benchmark v2.0[/bold cyan]\n"
        f"Models: {len(model_names)} | Questions: {len(questions)} | Total tasks: {total_tasks}",
        border_style="bright_blue"
    ))

    with Live(make_table(), console=console, refresh_per_second=4) as live:
        tasks = []
        for model in model_names:
            for q in questions:
                tasks.append(run_one(model, q, live))
        await asyncio.gather(*tasks)

    console.print(f"\n[green]✓ Evaluation complete![/green] {len(results)} responses collected.\n")
    return results


async def run_grading(responses: list[ModelResponse], questions: list[dict]):
    """Run grading with progress display."""
    from rich.console import Console
    console = Console()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]Error: ANTHROPIC_API_KEY required for grading (uses Claude Sonnet as judge)[/red]")
        sys.exit(1)

    # Filter out responses with errors
    valid_responses = [r for r in responses if not r.error]

    # Check how many are already cached
    from physics_eval.grading import load_cached_grade
    cached_count = sum(1 for r in valid_responses if load_cached_grade(r.model_name, r.question_id))
    new_count = len(valid_responses) - cached_count
    console.print(f"[cyan]Grading {len(valid_responses)} responses with Claude Sonnet (0-5 strict rubric)...[/cyan]")
    if cached_count > 0:
        console.print(f"[yellow]  {cached_count} already graded (cached), {new_count} new to grade[/yellow]")

    grades = await grade_all(valid_responses, questions, api_key, max_concurrent=5)

    console.print(f"[green]✓ Grading complete![/green] {len(grades)} grades assigned.\n")
    return grades


def generate_outputs(summary: dict, grades: list, responses: list):
    """Generate all output files."""
    from rich.console import Console
    console = Console()

    # Save grades
    grades_dicts = [asdict(g) if hasattr(g, '__dataclass_fields__') else g for g in grades]
    save_grades(grades if all(hasattr(g, '__dataclass_fields__') for g in grades) else [],
                "results/grades.json")
    # Also save as dicts for visualization
    with open("results/grades.json", "w") as f:
        json.dump(grades_dicts, f, indent=2)

    # Save summary
    save_summary(summary)

    # Save responses
    responses_dicts = [asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in responses]
    with open("results/all_responses.json", "w") as f:
        json.dump(responses_dicts, f, indent=2)

    # Generate charts
    console.print("[cyan]Generating charts...[/cyan]")
    chart_paths = generate_all_charts(summary, grades_dicts, responses_dicts)
    for p in chart_paths:
        console.print(f"  [green]✓[/green] {p}")

    # Generate Twitter thread
    twitter_path = generate_twitter_thread(summary)
    console.print(f"  [green]✓[/green] {twitter_path}")

    # Save questions
    save_questions_json()

    console.print(f"\n[bold green]All outputs saved to results/[/bold green]")


async def main():
    parser = argparse.ArgumentParser(
        description="STEMEval Benchmark v2.0 — Evaluate AI models on MIT physics & algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated model names or 'all' (default: all available)")
    parser.add_argument("--questions", type=str, default="all",
                        help="Comma-separated question IDs or 'all'")
    parser.add_argument("--course", type=str, default=None,
                        help="Filter by course: 8.033, 6.1220, or omit for all")
    parser.add_argument("--list-models", action="store_true", help="List all models")
    parser.add_argument("--check-keys", action="store_true", help="Check API key status")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, just regenerate charts from cached results")
    parser.add_argument("--skip-grading", action="store_true",
                        help="Skip grading, use cached grades")
    parser.add_argument("--clear-grades", action="store_true",
                        help="Delete cached grades (use when grading rubric changes)")
    parser.add_argument("--max-concurrent", type=int, default=3,
                        help="Max concurrent API calls (default: 3)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results)")

    args = parser.parse_args()

    if args.list_models:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="Available Models")
        table.add_column("Name", style="cyan")
        table.add_column("Provider", style="magenta")
        table.add_column("Model ID", style="yellow")
        for name, (provider, mid) in MODEL_REGISTRY.items():
            table.add_row(name, provider, mid)
        console.print(table)
        return

    if args.check_keys:
        check_keys()
        return

    if args.clear_grades:
        grades_path = Path("results/grades.json")
        cache_dir = Path("results/grades_cache")
        cleared = False
        if grades_path.exists():
            backup = Path("results/grades_v1_backup.json")
            grades_path.rename(backup)
            print(f"Backed up old grades to {backup}.")
            cleared = True
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print("Cleared grades cache directory.")
            cleared = True
        if not cleared:
            print("No cached grades to clear.")
        return

    # Resolve models
    if args.models == "all":
        model_names = get_available_models()
        if not model_names:
            from rich.console import Console
            Console().print("[red]No API keys set! Run --check-keys to see what's needed.[/red]")
            return
    else:
        model_names = [m.strip() for m in args.models.split(",")]
        for m in model_names:
            if m not in MODEL_REGISTRY:
                print(f"Error: Unknown model '{m}'. Use --list-models to see options.")
                return

    # Resolve questions (with optional course filter)
    if args.questions == "all":
        questions = get_questions(course=args.course)
    else:
        qids = [q.strip() for q in args.questions.split(",")]
        questions = get_questions(qids, course=args.course)
        if not questions:
            print(f"Error: No matching questions found for IDs: {qids}")
            return

    if not questions:
        print("No questions found. Run 'python scripts/extract_questions.py' to extract questions from PDFs.")
        return

    from rich.console import Console
    console = Console()
    courses_in_qs = sorted(set(q.get("course", "8.033") for q in questions))
    console.print(f"\n[bold]STEMEval Benchmark v2.0[/bold]")
    console.print(f"Models: {', '.join(model_names)}")
    console.print(f"Questions: {len(questions)} ({', '.join(courses_in_qs)})")
    console.print(f"Grading: 0-5 scale (Correctness 0-3 + Rigor 0-2)")
    console.print()

    # Step 1: Run evaluation
    if args.skip_eval:
        console.print("[yellow]Skipping evaluation, loading cached responses...[/yellow]")
        responses = []
        for model in model_names:
            for q in questions:
                cached = load_cached(model, q["question_id"])
                if cached:
                    responses.append(cached)
        console.print(f"Loaded {len(responses)} cached responses.\n")
    else:
        responses = await run_evaluation(model_names, questions, args.max_concurrent)

    if not responses:
        console.print("[red]No responses collected. Nothing to grade.[/red]")
        return

    # Step 1b: Also load cached responses from ALL models (not just the ones we ran)
    # so the leaderboard includes previous runs
    all_model_names = list(MODEL_REGISTRY.keys())
    for model in all_model_names:
        if model in model_names:
            continue  # already included
        for q in questions:
            cached = load_cached(model, q["question_id"])
            if cached:
                responses.append(cached)
    # Deduplicate by (model_name, question_id)
    seen = set()
    deduped = []
    for r in responses:
        key = (r.model_name, r.question_id)
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    responses = deduped
    # Report which models are in the combined results
    models_in_results = sorted(set(r.model_name for r in responses if not r.error))
    console.print(f"[cyan]Combined results include {len(responses)} responses from: {', '.join(models_in_results)}[/cyan]\n")

    # Step 2: Grade
    if args.skip_grading:
        console.print("[yellow]Skipping grading, loading cached grades...[/yellow]")
        grades_path = Path("results/grades.json")
        if grades_path.exists():
            with open(grades_path) as f:
                grades_dicts = json.load(f)
            grades = [Grade(**g) for g in grades_dicts]
        else:
            console.print("[red]No cached grades found. Run without --skip-grading first.[/red]")
            return
    else:
        grades = await run_grading(responses, questions)

    # Step 3: Compute summary
    summary = compute_summary(grades, responses)

    # Step 4: Generate outputs
    generate_outputs(summary, grades, responses)

    # Print summary table
    from rich.table import Table
    table = Table(title="Final Results (0-5 Scale)")
    table.add_column("Model", style="cyan")
    table.add_column("Score %", style="green")
    table.add_column("Avg Score", style="green")
    table.add_column("Avg Correct.", style="blue")
    table.add_column("Avg Rigor", style="magenta")
    table.add_column("Correct", style="green")
    table.add_column("Partial", style="yellow")
    table.add_column("Wrong", style="red")
    table.add_column("Cost", style="white")

    for model in sorted(summary["model_stats"].keys(),
                        key=lambda m: summary["model_stats"][m]["accuracy_pct"], reverse=True):
        s = summary["model_stats"][model]
        n_questions = s["correct"] + s["partial"] + s["incorrect"]
        avg_score = s["total_score"] / n_questions if n_questions else 0
        table.add_row(
            model,
            f"{s['accuracy_pct']:.1f}%",
            f"{avg_score:.1f}/5",
            f"{s['avg_correctness']:.1f}/3",
            f"{s['avg_rigor']:.1f}/2",
            str(s["correct"]),
            str(s["partial"]),
            str(s["incorrect"]),
            f"${s['total_cost_usd']:.4f}",
        )

    console.print(table)
    console.print("\n[bold green]Done! Check results/ for all outputs.[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
