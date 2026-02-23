# STEMEval Benchmark v2.0 — Twitter Thread Draft

## Tweet 1 (attach: chart8_summary.png)

We tested 8 frontier AI models on MIT's 8.033 (Relativity) and 6.1220 (Algorithms).

1 questions. Real problem sets and exams. Strict 0-5 grading.

The best physicist? claude-opus-4 at 0.0% accuracy.

Thread below with full results.


## Tweet 2 (attach: chart1_leaderboard.png)

Full leaderboard:

🥇 claude-opus-4: 0.0%
🥈 claude-sonnet-4: 0.0%
🥉 gemini-2.5-flash: 0.0%
4. gemini-2.5-pro: 0.0%
5. gpt-4.1: 0.0%
6. gpt-4o: 0.0%
7. o3: 0.0%
8. o4-mini: 0.0%


## Tweet 3 (attach: chart3_heatmap.png)

This heatmap shows every model × every question.

Green = correct, yellow = partial, red = wrong.

You can see each model's 'fingerprint' — where it's strong and where it fails.


## Tweet 4 (attach: chart6_hardest.png)

The questions that stumped everyone:

- Q1a: avg score 0.00/2.0

These are the questions where even the best models struggle.


## Tweet 5 (attach: chart7_cost_accuracy.png)

Cost matters. o4-mini costs $0.218 for the full eval.

Is the most expensive model worth it? Check the scatter plot.


## Tweet 6 (attach: chart5_head_to_head.png)

Pairwise win rates: for each pair, what % of questions did Model A beat Model B?

Some surprising matchups in here.


## Tweet 7 (attach: chart4_radar.png)

Multi-dimensional comparison of the top 5:

Accuracy, consistency, speed, cost efficiency, and confidence calibration.

No single model dominates every dimension.


## Tweet 8

Full stats: 1 questions, 2461 API calls, 6,232,864 tokens.

All questions from MIT's actual course materials.

Full dataset and methodology: [link]


STEMEval Benchmark v2.0


## Performance Badges

🏆 Best Overall: claude-opus-4
⚡ Fastest: gpt-4o (9.0s avg)
💰 Best Value: claude-opus-4
🧠 Best at Hard Questions: (see heatmap)
🎯 Most Calibrated: o3