"""
Auto-generate a Twitter thread draft from eval results.
"""

import json
import os


def generate_twitter_thread(summary: dict, output_path: str = "results/twitter_thread.md") -> str:
    """Generate a Twitter thread markdown file."""
    stats = summary["model_stats"]
    ranked = sorted(stats.items(), key=lambda x: x[1]["accuracy_pct"], reverse=True)

    total_qs = summary["total_questions"]
    total_models = summary["total_models"]
    total_tokens = summary["total_tokens_used"]

    lines = []
    lines.append("# STEMEval Benchmark v2.0 — Twitter Thread Draft\n")

    # Tweet 1: Hero
    lines.append("## Tweet 1 (attach: chart8_summary.png)\n")
    lines.append(f"We tested {total_models} frontier AI models on MIT's 8.033 (Relativity) and 6.1220 (Algorithms).\n")
    lines.append(f"{total_qs} questions. Real problem sets and exams. Strict 0-5 grading.\n")
    if ranked:
        lines.append(f"The best physicist? {ranked[0][0]} at {ranked[0][1]['accuracy_pct']:.1f}% accuracy.\n")
    lines.append("Thread below with full results.\n")

    # Tweet 2: Leaderboard
    lines.append("\n## Tweet 2 (attach: chart1_leaderboard.png)\n")
    lines.append("Full leaderboard:\n")
    for i, (model, s) in enumerate(ranked):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
        lines.append(f"{medal} {model}: {s['accuracy_pct']:.1f}%")
    lines.append("")

    # Tweet 3: Heatmap
    lines.append("\n## Tweet 3 (attach: chart3_heatmap.png)\n")
    lines.append("This heatmap shows every model × every question.\n")
    lines.append("Green = correct, yellow = partial, red = wrong.\n")
    lines.append("You can see each model's 'fingerprint' — where it's strong and where it fails.\n")

    # Tweet 4: Hardest questions
    lines.append("\n## Tweet 4 (attach: chart6_hardest.png)\n")
    q_stats = summary["question_stats"]
    hardest = sorted(q_stats.items(), key=lambda x: x[1]["avg_score"])[:3]
    lines.append("The questions that stumped everyone:\n")
    for qid, qs in hardest:
        short = qid.split("-")[-1]
        lines.append(f"- Q{short}: avg score {qs['avg_score']:.2f}/2.0")
    lines.append("\nThese are the questions where even the best models struggle.\n")

    # Tweet 5: Cost vs accuracy
    lines.append("\n## Tweet 5 (attach: chart7_cost_accuracy.png)\n")
    if ranked:
        cheapest = min(stats.items(), key=lambda x: x[1]["total_cost_usd"] if x[1]["total_cost_usd"] > 0 else float('inf'))
        lines.append(f"Cost matters. {cheapest[0]} costs ${cheapest[1]['total_cost_usd']:.3f} for the full eval.\n")
        lines.append("Is the most expensive model worth it? Check the scatter plot.\n")

    # Tweet 6: Head to head
    lines.append("\n## Tweet 6 (attach: chart5_head_to_head.png)\n")
    lines.append("Pairwise win rates: for each pair, what % of questions did Model A beat Model B?\n")
    lines.append("Some surprising matchups in here.\n")

    # Tweet 7: Radar
    lines.append("\n## Tweet 7 (attach: chart4_radar.png)\n")
    lines.append("Multi-dimensional comparison of the top 5:\n")
    lines.append("Accuracy, consistency, speed, cost efficiency, and confidence calibration.\n")
    lines.append("No single model dominates every dimension.\n")

    # Tweet 8: Wrap up
    lines.append("\n## Tweet 8\n")
    lines.append(f"Full stats: {total_qs} questions, {summary['total_api_calls']} API calls, {total_tokens:,} tokens.\n")
    lines.append("All questions from MIT's actual course materials.\n")
    lines.append("Full dataset and methodology: [link]\n")
    lines.append("\nSTEMEval Benchmark v2.0\n")

    # Performance badges
    if ranked:
        lines.append("\n## Performance Badges\n")
        lines.append(f"🏆 Best Overall: {ranked[0][0]}")
        fastest = min(stats.items(), key=lambda x: x[1]["avg_latency_seconds"])
        lines.append(f"⚡ Fastest: {fastest[0]} ({fastest[1]['avg_latency_seconds']:.1f}s avg)")
        best_value = max(stats.items(),
                         key=lambda x: x[1]["accuracy_pct"] / max(x[1]["total_cost_usd"], 0.001))
        lines.append(f"💰 Best Value: {best_value[0]}")

        # Best at hard questions
        q_stats_sorted = sorted(q_stats.items(), key=lambda x: x[1]["avg_score"])
        hard_qs = [q[0] for q in q_stats_sorted[:5]]
        # Would need grades to compute this properly
        lines.append(f"🧠 Best at Hard Questions: (see heatmap)")

        # Most calibrated
        calibrated = [(m, s) for m, s in stats.items() if s["avg_confidence"] is not None]
        if calibrated:
            best_cal = min(calibrated, key=lambda x: abs((x[1]["avg_confidence"] or 50)/100 - x[1]["accuracy"]))
            lines.append(f"🎯 Most Calibrated: {best_cal[0]}")

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)
    return output_path
