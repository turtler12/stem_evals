# STEMEval Benchmark

Evaluate AI models on MIT course problems — 8.033 (Special & General Relativity) and 6.1220 (Design and Analysis of Algorithms).

270 questions extracted from problem sets and exams, graded on a 0-5 scale by Claude Sonnet as judge.

## Models

| Model | Provider |
|-------|----------|
| claude-sonnet-4 | Anthropic |
| claude-opus-4 | Anthropic |
| gpt-4o | OpenAI |
| gpt-4.1 | OpenAI |
| o3 | OpenAI |
| o4-mini | OpenAI |
| gemini-2.5-pro | Google |
| gemini-2.5-flash | Google |
| deepseek-r1 | DeepSeek |
| grok-3 | xAI |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set API keys:

```bash
export ANTHROPIC_API_KEY='...'
export OPENAI_API_KEY='...'
export GOOGLE_API_KEY='...'
export DEEPSEEK_API_KEY='...'
export XAI_API_KEY='...'
```

Check which keys are configured:

```bash
python run_eval.py --check-keys
```

## Usage

Run all models:

```bash
python run_eval.py --models all
```

Run specific models:

```bash
python run_eval.py --models claude-sonnet-4,o3
```

Run a single model without grading (collect responses only):

```bash
python run_eval.py --models deepseek-r1 --skip-grading
```

Grade all cached responses:

```bash
python run_eval.py --skip-eval
```

Filter by course:

```bash
python run_eval.py --models all --course 8.033
python run_eval.py --models all --course 6.1220
```

## Grading

Each response is graded by Claude Sonnet on two dimensions:

- **Correctness** (0-3): Is the final answer right?
- **Rigor** (0-2): Are all steps shown with proper derivations?

Total score = Correctness + Rigor (0-5).

## Project Structure

```
run_eval.py                  # Main CLI entry point
physics_eval/
  models.py                  # Multi-provider API calls, caching, retry logic
  questions.py               # Question loading and formula sheets
  grading.py                 # Claude-as-judge grading pipeline
  visualizations.py          # Chart generation
  twitter.py                 # Twitter thread generator
scripts/
  extract_questions.py       # Extract questions from PDF problem sets
data/
  extracted_questions.json   # 270 extracted questions
  8.033 psets fall 2025/     # Physics problem sets
  6.1220 psets fall 2023/    # Algorithms problem sets
results/
  raw_responses/             # Cached model responses (one JSON per model+question)
  grades.json                # Grading results
  summary.json               # Aggregate statistics
```

## Outputs

After a full run, `results/` contains:

- `summary.json` — per-model accuracy, cost, latency, error taxonomy
- `grades.json` — individual grades with justifications
- `all_responses.json` — all model responses
- `twitter_thread.md` — generated summary thread
