"""
Twitter-ready visualization suite for STEMEval Benchmark.
Light-themed, 16:9, 2x resolution charts. Clean and modern.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# ── Light theme ──
BG_COLOR = "#ffffff"
CARD_BG = "#f6f8fa"
TEXT_COLOR = "#1f2328"
TEXT_SECONDARY = "#656d76"
GRID_COLOR = "#d1d9e0"
BORDER_COLOR = "#d1d9e0"

ACCENT_COLORS = {
    "anthropic": "#7c3aed",   # purple
    "openai": "#059669",      # green
    "google": "#2563eb",      # blue
    "deepseek": "#ea580c",    # orange
    "xai": "#dc2626",         # red
    "qwen": "#0891b2",        # cyan
}

MODEL_PROVIDERS = {
    "claude-sonnet-4": "anthropic",
    "claude-opus-4": "anthropic",
    "gpt-4o": "openai",
    "gpt-4.1": "openai",
    "o3": "openai",
    "o4-mini": "openai",
    "gemini-2.5-pro": "google",
    "gemini-2.5-flash": "google",
    "deepseek-r1": "deepseek",
    "grok-3": "xai",
    "qwen": "qwen",
}

WATERMARK = "STEMEval Benchmark v2.0"


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": BORDER_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 12,
    })


def _add_watermark(fig):
    fig.text(0.98, 0.02, WATERMARK, fontsize=9, color=TEXT_SECONDARY,
             ha="right", va="bottom", alpha=0.5, style="italic")


def _model_color(model_name: str) -> str:
    provider = MODEL_PROVIDERS.get(model_name, "openai")
    return ACCENT_COLORS.get(provider, "#6b7280")


def _save(fig, name: str, output_dir: str = "results/charts"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _clean_axes(ax):
    """Remove top/right spines, lighten remaining ones."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER_COLOR)
    ax.spines["bottom"].set_color(BORDER_COLOR)


def chart1_leaderboard(summary: dict, output_dir: str = "results/charts") -> str:
    """Chart 1 — Overall Leaderboard (Hero Chart)"""
    _setup_style()
    stats = summary["model_stats"]
    models = sorted(stats.keys(), key=lambda m: stats[m]["accuracy_pct"])
    accuracies = [stats[m]["accuracy_pct"] for m in models]
    colors = [_model_color(m) for m in models]

    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.barh(range(len(models)), accuracies, color=colors, height=0.55,
                   edgecolor="white", linewidth=0.5)

    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                f"{acc:.1f}%", va="center", fontsize=14, fontweight="bold", color=TEXT_COLOR)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=14, fontweight="medium")
    ax.set_xlabel("Accuracy (%)", fontsize=14, color=TEXT_SECONDARY)
    ax.set_title("Which AI Model is the Best Physicist?",
                 fontsize=22, fontweight="bold", pad=20, color=TEXT_COLOR)
    ax.set_xlim(0, max(accuracies) * 1.15 if accuracies else 100)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    _clean_axes(ax)

    # Legend for providers
    handles = [mpatches.Patch(color=c, label=p.title()) for p, c in ACCENT_COLORS.items()
               if any(MODEL_PROVIDERS.get(m) == p for m in models)]
    ax.legend(handles=handles, loc="lower right", fontsize=10, framealpha=0.9,
              edgecolor=BORDER_COLOR, facecolor=BG_COLOR)

    _add_watermark(fig)
    fig.tight_layout()
    return _save(fig, "chart1_leaderboard", output_dir)


def chart2_course_breakdown(summary: dict, grades: list[dict], output_dir: str = "results/charts") -> str:
    """Chart 2 — Course Breakdown (grouped bar chart by course)."""
    _setup_style()
    stats = summary["model_stats"]
    models = sorted(stats.keys(), key=lambda m: stats[m]["accuracy_pct"], reverse=True)

    # Group questions by course
    course_scores = {}
    for g in grades:
        qid = g["question_id"]
        course = qid.split("-")[0]
        model = g["model_name"]
        course_scores.setdefault(course, {}).setdefault(model, []).append(g["score"])

    courses = sorted(course_scores.keys())
    course_colors = ["#2563eb", "#059669", "#ea580c", "#7c3aed"]

    x = np.arange(len(models))
    width = 0.8 / max(len(courses), 1)

    fig, ax = plt.subplots(figsize=(16, 9))
    for i, course in enumerate(courses):
        accs = []
        for m in models:
            scores = course_scores.get(course, {}).get(m, [])
            acc = (sum(scores) / (len(scores) * 5) * 100) if scores else 0
            accs.append(acc)
        offset = (i - len(courses)/2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=course,
               color=course_colors[i % len(course_colors)], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Model", fontsize=14, color=TEXT_SECONDARY)
    ax.set_ylabel("Accuracy (%)", fontsize=14, color=TEXT_SECONDARY)
    ax.set_title("Performance by Course", fontsize=22, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=12)
    ax.legend(fontsize=12, framealpha=0.9, edgecolor=BORDER_COLOR, facecolor=BG_COLOR)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    _clean_axes(ax)

    _add_watermark(fig)
    fig.tight_layout()
    return _save(fig, "chart2_course_breakdown", output_dir)


def chart3_difficulty_heatmap(summary: dict, grades: list[dict], output_dir: str = "results/charts") -> str:
    """Chart 3 — Difficulty Heatmap (models x questions)."""
    _setup_style()

    models = sorted(set(g["model_name"] for g in grades),
                    key=lambda m: summary["model_stats"].get(m, {}).get("accuracy_pct", 0), reverse=True)
    questions = sorted(set(g["question_id"] for g in grades))

    # Sort questions by difficulty (avg score across models)
    q_avg = {}
    for qid in questions:
        scores = [g["score"] for g in grades if g["question_id"] == qid]
        q_avg[qid] = sum(scores) / len(scores) if scores else 0
    questions = sorted(questions, key=lambda q: q_avg[q], reverse=True)

    # Build matrix
    matrix = np.full((len(models), len(questions)), np.nan)
    for g in grades:
        if g["model_name"] in models and g["question_id"] in questions:
            mi = models.index(g["model_name"])
            qi = questions.index(g["question_id"])
            matrix[mi, qi] = g["score"]

    fig, ax = plt.subplots(figsize=(max(16, len(questions) * 0.55), max(9, len(models) * 0.9)))

    # Red-Yellow-Green colormap (6 levels for 0-5)
    cmap = sns.color_palette(["#dc2626", "#ea580c", "#f59e0b", "#84cc16", "#16a34a", "#059669"], as_cmap=True)
    sns.heatmap(matrix, ax=ax, cmap=cmap, vmin=0, vmax=5,
                xticklabels=[q.split("-")[-1] for q in questions],
                yticklabels=models,
                cbar_kws={"label": "Score (0-5: Correctness + Rigor)", "shrink": 0.6},
                linewidths=1.5, linecolor="white", annot=True, fmt=".0f",
                annot_kws={"fontsize": 9, "fontweight": "bold", "color": "white"})

    ax.set_title("Model Performance Fingerprint", fontsize=22, fontweight="bold", pad=20)
    ax.set_xlabel("Question", fontsize=14, color=TEXT_SECONDARY)
    ax.set_ylabel("Model", fontsize=14, color=TEXT_SECONDARY)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', labelsize=12)

    _add_watermark(fig)
    fig.tight_layout()
    return _save(fig, "chart3_heatmap", output_dir)


def chart4_radar(summary: dict, responses: list[dict], output_dir: str = "results/charts") -> str:
    """Chart 4 — Radar/Spider Chart for top 5 models."""
    _setup_style()
    stats = summary["model_stats"]
    top_models = sorted(stats.keys(), key=lambda m: stats[m]["accuracy_pct"], reverse=True)[:5]

    categories = ["Overall\nAccuracy", "Consistency", "Speed", "Cost\nEfficiency", "Confidence\nCalibration"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    for model in top_models:
        s = stats[model]
        accuracy = s["accuracy_pct"] / 100
        consistency = max(0, 1 - (s["incorrect"] / max(s["correct"] + s["partial"] + s["incorrect"], 1)))
        all_latencies = [stats[m]["avg_latency_seconds"] for m in stats]
        max_lat = max(all_latencies) if all_latencies else 1
        speed = 1 - (s["avg_latency_seconds"] / max_lat) if max_lat > 0 else 0.5
        all_costs = [stats[m]["total_cost_usd"] for m in stats if stats[m]["total_cost_usd"] > 0]
        max_cost = max(all_costs) if all_costs else 1
        cost_eff = 1 - (s["total_cost_usd"] / max_cost) if max_cost > 0 else 0.5
        if s["avg_confidence"] is not None:
            cal = 1 - abs(s["avg_confidence"]/100 - accuracy)
        else:
            cal = 0.5

        values = [accuracy, consistency, speed, cost_eff, cal]
        values += values[:1]

        color = _model_color(model)
        ax.plot(angles, values, "o-", linewidth=2.5, label=model, color=color, markersize=6)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, color=TEXT_COLOR, fontweight="medium")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=9, color=TEXT_SECONDARY)
    ax.spines["polar"].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, alpha=0.4)
    ax.set_title("Top 5 Models — Multi-Dimensional Comparison",
                 fontsize=18, fontweight="bold", pad=30, color=TEXT_COLOR)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11,
              framealpha=0.9, edgecolor=BORDER_COLOR, facecolor=BG_COLOR)

    _add_watermark(fig)
    return _save(fig, "chart4_radar", output_dir)


def chart5_head_to_head(grades: list[dict], output_dir: str = "results/charts") -> str:
    """Chart 5 — Head-to-Head Pairwise Win-Rate Matrix."""
    _setup_style()
    models = sorted(set(g["model_name"] for g in grades))
    n = len(models)

    win_matrix = np.zeros((n, n))
    questions = sorted(set(g["question_id"] for g in grades))

    for qid in questions:
        q_grades = {g["model_name"]: g["score"] for g in grades if g["question_id"] == qid}
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i != j and m1 in q_grades and m2 in q_grades:
                    if q_grades[m1] > q_grades[m2]:
                        win_matrix[i, j] += 1

    total_qs = len(questions)
    win_pct = win_matrix / total_qs * 100 if total_qs > 0 else win_matrix

    fig, ax = plt.subplots(figsize=(max(12, n * 1.4), max(10, n * 1.2)))
    mask = np.eye(n, dtype=bool)
    sns.heatmap(win_pct, ax=ax, annot=True, fmt=".0f", mask=mask,
                xticklabels=models, yticklabels=models,
                cmap="RdYlGn", center=50, vmin=0, vmax=100,
                linewidths=2, linecolor="white",
                cbar_kws={"label": "Win Rate %", "shrink": 0.7},
                annot_kws={"fontsize": 12, "fontweight": "bold"})

    ax.set_title("Head-to-Head Win Rate (%)\n(row beats column)",
                 fontsize=20, fontweight="bold", pad=20)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', rotation=0, labelsize=12)

    _add_watermark(fig)
    fig.tight_layout()
    return _save(fig, "chart5_head_to_head", output_dir)


def chart6_hardest_questions(summary: dict, grades: list[dict], output_dir: str = "results/charts") -> str:
    """Chart 6 — The 10 Hardest Questions."""
    _setup_style()
    q_stats = summary["question_stats"]
    hardest = sorted(q_stats.items(), key=lambda x: x[1]["avg_score"])[:10]

    questions = [q[0].split("-")[-1] for q in hardest]
    full_ids = [q[0] for q in hardest]
    avg_scores = [q[1]["avg_score"] for q in hardest]

    fig, ax = plt.subplots(figsize=(16, 9))
    colors = ["#dc2626" if s < 1.5 else "#f59e0b" if s < 3.0 else "#16a34a" for s in avg_scores]
    bars = ax.barh(range(len(questions)), avg_scores, color=colors, height=0.55,
                   edgecolor="white", linewidth=0.5)

    for i, (bar, score, fid) in enumerate(zip(bars, avg_scores, full_ids)):
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
                f"{score:.2f} / 5.0", va="center", fontsize=12, fontweight="bold", color=TEXT_COLOR)

    ax.set_yticks(range(len(questions)))
    ax.set_yticklabels([f"Q{q}" for q in questions], fontsize=13, fontweight="medium")
    ax.set_xlabel("Average Score (out of 5)", fontsize=14, color=TEXT_SECONDARY)
    ax.set_title("The Hardest Questions (Lowest Average Score)",
                 fontsize=22, fontweight="bold", pad=20)
    ax.set_xlim(0, 5.5)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    _clean_axes(ax)

    _add_watermark(fig)
    fig.tight_layout()
    return _save(fig, "chart6_hardest", output_dir)


def chart7_cost_vs_accuracy(summary: dict, output_dir: str = "results/charts") -> str:
    """Chart 7 — Cost vs. Accuracy Scatter."""
    _setup_style()
    stats = summary["model_stats"]

    fig, ax = plt.subplots(figsize=(16, 9))
    for model, s in stats.items():
        cost = s["total_cost_usd"]
        acc = s["accuracy_pct"]
        lat = s["avg_latency_seconds"]
        color = _model_color(model)
        size = max(80, 400 / max(lat, 0.1))  # Bigger = faster
        ax.scatter(cost, acc, s=size, color=color, edgecolors="white",
                   linewidth=2, zorder=5, alpha=0.9)
        ax.annotate(model, (cost, acc), textcoords="offset points", xytext=(10, 8),
                    fontsize=11, fontweight="medium", color=TEXT_COLOR)

    ax.set_xlabel("Estimated Cost ($)", fontsize=14, color=TEXT_SECONDARY)
    ax.set_ylabel("Accuracy (%)", fontsize=14, color=TEXT_SECONDARY)
    ax.set_title("Cost vs. Accuracy (bubble size = speed)",
                 fontsize=22, fontweight="bold", pad=20)
    ax.grid(alpha=0.3, linestyle="--")
    _clean_axes(ax)

    _add_watermark(fig)
    fig.tight_layout()
    return _save(fig, "chart7_cost_accuracy", output_dir)


def chart8_summary_card(summary: dict, output_dir: str = "results/charts") -> str:
    """Chart 8 — Twitter Thread Summary Card."""
    _setup_style()
    stats = summary["model_stats"]
    ranked = sorted(stats.items(), key=lambda x: x[1]["accuracy_pct"], reverse=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background card with subtle border
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, facecolor=CARD_BG,
                          edgecolor=BORDER_COLOR, linewidth=2,
                          transform=ax.transAxes, zorder=0, clip_on=False)
    ax.add_patch(rect)

    # Title
    ax.text(0.5, 0.92, "STEMEval Benchmark v2.0", fontsize=30, fontweight="bold",
            ha="center", va="top", color=TEXT_COLOR, transform=ax.transAxes)
    ax.text(0.5, 0.85, "MIT 8.033 (Relativity) + 6.1220 (Algorithms)", fontsize=16,
            ha="center", va="top", color=TEXT_SECONDARY, transform=ax.transAxes)

    # Top 3 models
    medals = ["1st", "2nd", "3rd"]
    medal_colors = ["#ca8a04", "#71717a", "#b45309"]
    for i, (model, s) in enumerate(ranked[:3]):
        y = 0.72 - i * 0.14
        ax.text(0.15, y, f"{medals[i]}", fontsize=24, fontweight="bold",
                ha="center", va="center", color=medal_colors[i], transform=ax.transAxes)
        ax.text(0.25, y, model, fontsize=21, fontweight="bold",
                ha="left", va="center", color=_model_color(model), transform=ax.transAxes)
        ax.text(0.85, y, f"{s['accuracy_pct']:.1f}%", fontsize=24, fontweight="bold",
                ha="right", va="center", color=TEXT_COLOR, transform=ax.transAxes)

    # Divider
    ax.plot([0.1, 0.9], [0.32, 0.32], color=BORDER_COLOR, linewidth=1,
            transform=ax.transAxes, clip_on=False)

    # Verdict
    if len(ranked) >= 2:
        verdict = f"{ranked[0][0]} leads with {ranked[0][1]['accuracy_pct']:.1f}% accuracy"
    else:
        verdict = "Results pending..."
    ax.text(0.5, 0.26, verdict, fontsize=15, ha="center", va="center",
            color=TEXT_SECONDARY, style="italic", transform=ax.transAxes)

    # Key stats
    total_qs = summary["total_questions"]
    total_calls = summary["total_api_calls"]
    total_tokens = summary["total_tokens_used"]
    stats_text = f"{total_qs} questions  |  {total_calls} API calls  |  {total_tokens:,} tokens"
    ax.text(0.5, 0.18, stats_text, fontsize=13, ha="center", va="center",
            color=TEXT_SECONDARY, transform=ax.transAxes)

    # Badges
    if ranked:
        badges = []
        best = ranked[0][0]
        badges.append(f"Best Overall: {best}")
        fastest = min(stats.items(), key=lambda x: x[1]["avg_latency_seconds"])
        badges.append(f"Fastest: {fastest[0]}")
        best_value = max(stats.items(),
                         key=lambda x: x[1]["accuracy_pct"] / max(x[1]["total_cost_usd"], 0.001))
        badges.append(f"Best Value: {best_value[0]}")

        badge_text = "  |  ".join(badges)
        ax.text(0.5, 0.08, badge_text, fontsize=11, ha="center", va="center",
                color="#2563eb", fontweight="medium", transform=ax.transAxes)

    _add_watermark(fig)
    return _save(fig, "chart8_summary", output_dir)


def generate_all_charts(summary: dict, grades: list[dict], responses: list[dict],
                        output_dir: str = "results/charts") -> list[str]:
    """Generate all 8 charts. Returns list of file paths."""
    paths = []
    paths.append(chart1_leaderboard(summary, output_dir))
    paths.append(chart2_course_breakdown(summary, grades, output_dir))
    paths.append(chart3_difficulty_heatmap(summary, grades, output_dir))
    paths.append(chart4_radar(summary, responses, output_dir))
    paths.append(chart5_head_to_head(grades, output_dir))
    paths.append(chart6_hardest_questions(summary, grades, output_dir))
    paths.append(chart7_cost_vs_accuracy(summary, output_dir))
    paths.append(chart8_summary_card(summary, output_dir))
    return paths
