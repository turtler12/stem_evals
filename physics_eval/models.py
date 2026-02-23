"""
Multi-model evaluation runner.
Supports Anthropic, OpenAI, Google, DeepSeek, xAI, and Qwen models.
Async with retry logic, caching, and parallel execution.
"""

import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

import httpx

# Model registry: name -> (provider, api_model_id)
MODEL_REGISTRY = {
    "claude-sonnet-4": ("anthropic", "claude-sonnet-4-20250514"),
    "claude-opus-4": ("anthropic", "claude-opus-4-20250514"),
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4.1": ("openai", "gpt-4.1"),
    "o3": ("openai", "o3"),
    "o4-mini": ("openai", "o4-mini"),
    "gemini-2.5-pro": ("google", "gemini-2.5-pro"),
    "gemini-2.5-flash": ("google", "gemini-2.5-flash"),
    "deepseek-r1": ("deepseek", "deepseek-reasoner"),
    "grok-3": ("xai", "grok-3"),
    "qwen": ("qwen", "qwen-max"),
}

# Env var names for API keys
API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "qwen": "QWEN_API_KEY",
}

# Pricing per 1M tokens (input, output) in USD — approximate
MODEL_PRICING = {
    "claude-sonnet-4": (3.0, 15.0),
    "claude-opus-4": (15.0, 75.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4.1": (2.0, 8.0),
    "o3": (2.0, 8.0),
    "o4-mini": (1.1, 4.4),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.15, 0.6),
    "deepseek-r1": (0.55, 2.19),
    "grok-3": (3.0, 15.0),
    "qwen": (2.0, 8.0),
}

CACHE_DIR = Path("results/raw_responses")

# Per-provider semaphores to avoid rate limiting
# Reasoning models (o3, o4-mini) share tight OpenAI rate limits
_provider_semaphores: dict[str, asyncio.Semaphore] = {}

def _get_provider_semaphore(provider: str) -> asyncio.Semaphore:
    """Get or create a per-provider concurrency semaphore."""
    if provider not in _provider_semaphores:
        limits = {
            "openai": 2,      # OpenAI reasoning models have tight rate limits
            "anthropic": 5,
            "google": 3,
            "deepseek": 2,
            "xai": 2,
            "qwen": 2,
        }
        _provider_semaphores[provider] = asyncio.Semaphore(limits.get(provider, 2))
    return _provider_semaphores[provider]


@dataclass
class ModelResponse:
    model_name: str
    question_id: str
    full_response: str
    extracted_answer: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    timestamp: str
    confidence: int | None = None  # 0-100 self-reported confidence
    error: str | None = None


def _cache_key(model_name: str, question_id: str) -> str:
    return f"{model_name}__{question_id}"


def _cache_path(model_name: str, question_id: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{_cache_key(model_name, question_id)}.json"


def load_cached(model_name: str, question_id: str) -> ModelResponse | None:
    path = _cache_path(model_name, question_id)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return ModelResponse(**data)
    return None


def save_cache(response: ModelResponse):
    path = _cache_path(response.model_name, response.question_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(response), f, indent=2)


def build_physics_prompt(question_text: str, formula_sheet: str) -> str:
    return f"""You are solving a physics problem from MIT's 8.033 (Relativity) course.

{formula_sheet}

PROBLEM:
{question_text}

INSTRUCTIONS:
- Show all your work step by step.
- Clearly state any assumptions you make.
- Use proper mathematical notation.
- At the end, put your final answer inside a box like this: \\boxed{{your answer}}
- If the answer is conceptual/qualitative, still put a concise final answer in \\boxed{{}}.
- After your boxed answer, on a new line, state your confidence in your answer as a percentage: "Confidence: X%" where X is 0-100.
"""


def build_algorithms_prompt(question_text: str) -> str:
    return f"""You are solving an algorithms problem from MIT's 6.1220 (Design and Analysis of Algorithms) course.

PROBLEM:
{question_text}

INSTRUCTIONS:
- Show all your work step by step.
- For proofs, state your proof technique (induction, contradiction, etc.) and be rigorous.
- For algorithm design, provide pseudocode and analyze time/space complexity.
- For recurrences, show your work using Master theorem or substitution method.
- Clearly state any assumptions you make.
- At the end, put your final answer inside a box like this: \\boxed{{your answer}}
- If the answer is a proof, put a concise summary of the key insight in \\boxed{{}}.
- After your boxed answer, on a new line, state your confidence: "Confidence: X%" where X is 0-100.
"""


def build_physics_prompt_messages(question_text: str, formula_sheet: str) -> list[dict]:
    """Return messages list for chat-style APIs (physics)."""
    return [
        {"role": "system", "content": "You are an expert physicist solving MIT-level special and general relativity problems. Show detailed work and provide a final boxed answer."},
        {"role": "user", "content": build_physics_prompt(question_text, formula_sheet)},
    ]


def build_algorithms_prompt_messages(question_text: str) -> list[dict]:
    """Return messages list for chat-style APIs (algorithms)."""
    return [
        {"role": "system", "content": "You are an expert in algorithms and data structures solving MIT-level algorithm design and analysis problems. Show detailed work, provide rigorous proofs where needed, and give a final boxed answer."},
        {"role": "user", "content": build_algorithms_prompt(question_text)},
    ]


def extract_boxed_answer(text: str) -> str:
    """Extract the content inside \\boxed{...} from model response."""
    import re
    # Try to find \boxed{...} with brace matching
    patterns = [
        r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        r'\\boxed\{(.+?)\}',
        r'\$\\boxed\{(.+?)\}\$',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()  # Take last boxed answer
    return "[No boxed answer found]"


def extract_confidence(text: str) -> int | None:
    """Extract confidence percentage from model response."""
    import re
    match = re.search(r'[Cc]onfidence:\s*(\d+)%', text)
    if match:
        return int(match.group(1))
    return None


# ============ Provider-specific API calls ============

async def call_anthropic(model_id: str, messages: list[dict], api_key: str) -> tuple[str, int, int]:
    """Call Anthropic API. Returns (response_text, input_tokens, output_tokens)."""
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=api_key)
    system_msg = messages[0]["content"] if messages[0]["role"] == "system" else ""
    user_msgs = [m for m in messages if m["role"] != "system"]

    response = await client.messages.create(
        model=model_id,
        max_tokens=4096,
        system=system_msg,
        messages=user_msgs,
    )
    text = response.content[0].text
    return text, response.usage.input_tokens, response.usage.output_tokens


async def call_openai(model_id: str, messages: list[dict], api_key: str) -> tuple[str, int, int]:
    """Call OpenAI API."""
    import openai
    client = openai.AsyncOpenAI(api_key=api_key)

    kwargs = dict(
        model=model_id,
        messages=messages,
    )
    # o3 and o4-mini are reasoning models — don't pass max_tokens, use max_completion_tokens
    if model_id in ("o3", "o4-mini"):
        kwargs["max_completion_tokens"] = 16384
        # Reasoning models don't support system messages in the same way
        # Convert system message to user context
        if messages[0]["role"] == "system":
            combined = messages[0]["content"] + "\n\n" + messages[1]["content"]
            kwargs["messages"] = [{"role": "user", "content": combined}]
    else:
        kwargs["max_tokens"] = 4096

    response = await client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content
    usage = response.usage
    return text, usage.prompt_tokens, usage.completion_tokens


async def call_google(model_id: str, messages: list[dict], api_key: str) -> tuple[str, int, int]:
    """Call Google Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_id)

    # Combine system + user messages
    prompt = "\n\n".join(m["content"] for m in messages)

    response = await asyncio.to_thread(
        model.generate_content, prompt
    )
    text = response.text
    # Approximate token counts
    input_tokens = int(len(prompt.split()) * 1.3)
    output_tokens = int(len(text.split()) * 1.3)
    if hasattr(response, 'usage_metadata'):
        um = response.usage_metadata
        input_tokens = getattr(um, 'prompt_token_count', input_tokens)
        output_tokens = getattr(um, 'candidates_token_count', output_tokens)
    return text, input_tokens, output_tokens


async def call_openai_compatible(model_id: str, messages: list[dict], api_key: str, base_url: str) -> tuple[str, int, int]:
    """Call OpenAI-compatible API (DeepSeek, xAI, Qwen)."""
    import openai
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    response = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=4096,
    )
    text = response.choices[0].message.content
    usage = response.usage
    return text, usage.prompt_tokens, usage.completion_tokens


PROVIDER_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "xai": "https://api.x.ai/v1",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
}


async def call_model(model_name: str, question_id: str, question_text: str,
                     formula_sheet: str | None = None, course: str = "8.033") -> ModelResponse:
    """Call a specific model with retry logic."""
    provider, model_id = MODEL_REGISTRY[model_name]
    api_key_env = API_KEY_ENV[provider]
    api_key = os.environ.get(api_key_env)

    if not api_key:
        return ModelResponse(
            model_name=model_name,
            question_id=question_id,
            full_response="",
            extracted_answer="",
            latency_seconds=0,
            input_tokens=0,
            output_tokens=0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            error=f"Missing API key: {api_key_env}",
        )

    # Check cache first
    cached = load_cached(model_name, question_id)
    if cached:
        return cached

    if course == "6.1220":
        messages = build_algorithms_prompt_messages(question_text)
    else:
        messages = build_physics_prompt_messages(question_text, formula_sheet or "")

    provider_sem = _get_provider_semaphore(provider)
    max_retries = 6
    for attempt in range(max_retries):
        try:
            async with provider_sem:
                start = time.monotonic()

                if provider == "anthropic":
                    text, in_tok, out_tok = await call_anthropic(model_id, messages, api_key)
                elif provider == "openai":
                    text, in_tok, out_tok = await call_openai(model_id, messages, api_key)
                elif provider == "google":
                    text, in_tok, out_tok = await call_google(model_id, messages, api_key)
                else:
                    base_url = PROVIDER_BASE_URLS[provider]
                    text, in_tok, out_tok = await call_openai_compatible(model_id, messages, api_key, base_url)

                elapsed = time.monotonic() - start

            response = ModelResponse(
                model_name=model_name,
                question_id=question_id,
                full_response=text,
                extracted_answer=extract_boxed_answer(text),
                latency_seconds=round(elapsed, 2),
                input_tokens=in_tok,
                output_tokens=out_tok,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                confidence=extract_confidence(text),
            )
            save_cache(response)
            return response

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower()
            if attempt < max_retries - 1:
                if is_rate_limit:
                    # Backoff for rate limits: 10s, 20s, 40s, 60s, 90s + jitter
                    wait = min(90, 10 * (2 ** attempt)) + random.uniform(0, 5)
                else:
                    wait = 2 ** (attempt + 1)
                await asyncio.sleep(wait)
            else:
                return ModelResponse(
                    model_name=model_name,
                    question_id=question_id,
                    full_response="",
                    extracted_answer="",
                    latency_seconds=0,
                    input_tokens=0,
                    output_tokens=0,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    error=err_str,
                )


def get_available_models() -> list[str]:
    """Return models that have API keys set."""
    available = []
    for name, (provider, _) in MODEL_REGISTRY.items():
        env = API_KEY_ENV[provider]
        if os.environ.get(env):
            available.append(name)
    return available


def get_missing_keys() -> dict[str, str]:
    """Return dict of model_name -> env_var for missing keys."""
    missing = {}
    for name, (provider, _) in MODEL_REGISTRY.items():
        env = API_KEY_ENV[provider]
        if not os.environ.get(env):
            missing[name] = env
    return missing


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a single query."""
    if model_name not in MODEL_PRICING:
        return 0.0
    in_price, out_price = MODEL_PRICING[model_name]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
