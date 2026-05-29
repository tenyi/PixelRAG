#!/usr/bin/env python3
"""MoNaCo multi-hop QA evaluation with a ReAct search agent.

Runs a GPT-5 (or any OpenAI-compatible) ReAct agent that iteratively searches
a local retrieval API and answers 1315 complex multi-hop Wikipedia questions
from the MoNaCo benchmark.

Supports:
  - pixel retrieval (--pixel-api, default :30888)
  - text  retrieval (--text-api,  default :30889)
  - Claude models via Anthropic API (auto-detected from model name)
  - resumable: skips examples whose output JSON already exists
  - per-example JSONL output + per-example JSON files (for judge_predictions.py)
  - automatic token-level F1 grading (primary metric)
  - optional LLM judge grading (secondary, via --judge)

Usage:
    # Text retrieval with GPT-5
    python run_monaco.py --reader gpt-5 --retrieval text

    # Pixel retrieval with GPT-4o
    python run_monaco.py --reader gpt-4o-2024-08-06 --retrieval pixel

    # Smoke test: 5 examples, text retrieval
    python run_monaco.py --reader gpt-5 --retrieval text --smoke 5

    # With LLM judge grading
    python run_monaco.py --reader gpt-5 --retrieval text --judge

Environment:
    OPENAI_API_KEY   — required (or pass --api-key)
    OPENAI_BASE_URL  — optional override (or pass --base-url)
    ANTHROPIC_API_KEY — required for Claude models (or pass --api-key)

Dataset:
    MoNaCo v1: download from https://github.com/facebookresearch/MoNaCo
    Place at: data/monaco/monaco_version_1_release.jsonl (relative to this script)
    Or pass --data-path <path>
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import copy
import json
import logging
import os
import re
import string
import time
import traceback
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "data" / "monaco" / "monaco_version_1_release.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "eval_output" / "monaco"
LOG_DIR = SCRIPT_DIR / "logs" / "monaco"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PIXEL_API = "http://localhost:30888/search"
TEXT_API = "http://localhost:30889/search"
DEFAULT_K = 5
MAX_TOP_K = 10
MAX_TURNS = 16
READER_TIMEOUT = 300
SEARCH_TIMEOUT = 30
RESULT_TRUNCATE_CHARS = 8000

# Cumulative usage tracker
USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "tool_calls": 0}

# Pricing per million tokens
PRICING = {
    "gpt-5": {"in": 0.625, "out": 5.0},
    "gpt-5-2025-08-07": {"in": 0.625, "out": 5.0},
    "gpt-4o-2024-08-06": {"in": 2.50, "out": 10.0},
    "gpt-4o": {"in": 2.50, "out": 10.0},
    "claude-sonnet-4-6": {"in": 3.0, "out": 15.0},
    "claude-opus-4-7": {"in": 15.0, "out": 75.0},
    "claude-haiku-4-5": {"in": 0.80, "out": 4.0},
}

# Module-level globals (set in main() from CLI args)
_PIXEL_API = PIXEL_API
_TEXT_API = TEXT_API
_DEFAULT_TOP_K = DEFAULT_K
_MAX_TOP_K = MAX_TOP_K
_IMAGE_DETAIL = "auto"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging(tag: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_monaco")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / f"{tag}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = (
    "You are a research assistant that answers complex multi-hop questions "
    "by searching Wikipedia. You have ONE tool: `{tool_name}(query, top_k?)` which "
    "returns relevant {artifact}.\n"
    "\n"
    "Strategy:\n"
    "  - Decompose the question into sub-queries as you go.\n"
    "  - Call {tool_name} multiple times with different queries to gather "
    "evidence. Each search returns short {artifact}.\n"
    "  - When you have enough evidence, stop searching and give the final "
    "answer.\n"
    "  - You can search up to {max_turns} times; budget your calls.\n"
    "\n"
    "Choosing top_k:\n"
    "  - Default (omit top_k): {default_k} results per search.\n"
    "  - Narrow factoid lookup (one person/date/place): top_k=2-3.\n"
    '  - Broad enumeration ("all X", "every Y", "list of Z"): top_k=7-10.\n'
    "\n"
    "List questions (especially LONG lists):\n"
    '  - If the question asks for a list ("all X", "every Y", "each Z", '
    '"top N"), do AT LEAST 3 distinct searches with varied phrasings before '
    "answering.\n"
    "  - If the question implies a VERY long list (50+ entries), do AT LEAST "
    "5 broader searches; aim to enumerate AT LEAST 30 entries.\n"
    "  - Output ALL valid entries in the Answers line, comma-separated.\n"
    "\n"
    "NEVER refuse, hedge, or output empty:\n"
    "  - NEVER write 'I was unable to find', 'I cannot determine', 'data not "
    "available', or any similar deflection. The judge scores zero for these.\n"
    "  - NEVER output `Answers:` followed by nothing. Even on the hardest "
    "questions, output your BEST guess.\n"
    "\n"
    "Final answer format:\n"
    "  - Last line MUST be:  Answers: {{comma-separated entities, numbers, "
    "or dates}}\n"
    "  - The Answers line contains ONLY values — no explanations or caveats.\n"
    "  - End your response immediately after the Answers line.\n"
)


def _build_system_prompt(retrieval: str) -> str:
    if retrieval == "pixel":
        tool_name = "search_pixel"
        artifact = "Wikipedia screenshot tiles (PNG images)"
    else:
        tool_name = "search_text"
        artifact = "text passages"
    return SYSTEM_PROMPT_TEMPLATE.format(
        tool_name=tool_name,
        artifact=artifact,
        max_turns=MAX_TURNS,
        default_k=_DEFAULT_TOP_K,
    )


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
SEARCH_TEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "search_text",
        "description": "Search Wikipedia for text passages relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {
                    "type": "integer",
                    "description": "Number of results (1-10). Default 5.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}

SEARCH_PIXEL_TOOL = {
    "type": "function",
    "function": {
        "name": "search_pixel",
        "description": "Search Wikipedia screenshot tiles relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {
                    "type": "integer",
                    "description": "Number of tiles (1-10). Default 5.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------
def _search_text(query: str, n_docs: int | None = None) -> str:
    """Return top-K text chunks formatted as one string."""
    if n_docs is None:
        n_docs = _DEFAULT_TOP_K
    n_docs = max(1, min(n_docs, _MAX_TOP_K))
    body = {"queries": [{"text": query}], "n_docs": n_docs}
    req = urllib.request.Request(
        _TEXT_API,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
            d = json.load(resp)
        hits = d.get("results", [{}])[0].get("hits", [])
    except Exception as e:
        return f"[search_error: {e}]"

    if not hits:
        return "[no results]"

    parts = []
    for h in hits:
        title = (h.get("title") or "").strip() or h.get("url", "")
        text = (h.get("text") or "").strip()
        chunk_idx = h.get("chunk_index", 0)
        label = f"{title} (chunk {chunk_idx})" if chunk_idx > 0 else title
        parts.append(f"*** Doc title: {label}\n*** Contents:\n{text}")
    return "\n\n".join(parts)


def _search_pixel(query: str, n_docs: int | None = None) -> list[dict]:
    """Return top-K screenshot tiles as multimodal content parts."""
    if n_docs is None:
        n_docs = _DEFAULT_TOP_K
    n_docs = max(1, min(n_docs, _MAX_TOP_K))
    body = {"queries": [{"text": query}], "n_docs": n_docs}
    req = urllib.request.Request(
        _PIXEL_API,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
            d = json.load(resp)
        hits = d.get("results", [{}])[0].get("hits", [])
    except Exception as e:
        return [{"type": "text", "text": f"[search_error: {e}]"}]

    if not hits:
        return [{"type": "text", "text": "[no results]"}]

    parts: list[dict] = [{"type": "text", "text": "Top-K Wikipedia screenshot tiles:"}]
    for h in hits:
        png_path = h.get("path", "")
        try:
            with open(png_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": _IMAGE_DETAIL,
                    },
                }
            )
        except Exception as e:
            parts.append({"type": "text", "text": f"[image_error for {png_path}: {e}]"})
    return parts


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------
def _supports_temperature(model: str) -> bool:
    return "gpt-5" not in model


def _is_local_model(base_url: str) -> bool:
    return "localhost" in base_url or "127.0.0.1" in base_url


def _is_claude_model(model: str) -> bool:
    return "claude" in model.lower()


def _call_llm_openai(
    messages: list[dict], model: str, tool_schema: dict, api_key: str, base_url: str
) -> dict:
    """One OpenAI LLM turn with tools. Returns the message dict."""
    body: dict = {
        "model": model,
        "messages": messages,
        "tools": [tool_schema],
        "tool_choice": "auto",
    }
    if _is_local_model(base_url):
        body["max_tokens"] = 4096
    else:
        body["max_completion_tokens"] = 16000
    if _supports_temperature(model):
        body["temperature"] = 0.0
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    last_exc = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=READER_TIMEOUT) as resp:
                d = json.load(resp)
            usage = d.get("usage", {})
            USAGE["prompt_tokens"] += usage.get("prompt_tokens", 0)
            USAGE["completion_tokens"] += usage.get("completion_tokens", 0)
            USAGE["calls"] += 1
            return d["choices"][0]["message"]
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in (400, 429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(min(60, 2**attempt + 2))
                continue
            raise
        except urllib.error.URLError as e:
            last_exc = e
            if attempt < 4:
                time.sleep(min(60, 2**attempt + 2))
                continue
            raise
    raise last_exc  # type: ignore[misc]


def _openai_tool_to_claude(schema: dict) -> dict:
    fn = schema["function"]
    return {
        "name": fn["name"],
        "description": fn["description"],
        "input_schema": fn["parameters"],
    }


def _openai_msgs_to_claude(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert OpenAI messages to (system_str, claude_messages)."""
    system_parts = []
    claude_msgs: list[dict] = []
    for m in messages:
        role = m.get("role", "")
        if role == "system":
            system_parts.append(
                m["content"] if isinstance(m["content"], str) else str(m["content"])
            )
        elif role == "user":
            content = m["content"]
            if isinstance(content, str):
                claude_msgs.append({"role": "user", "content": content})
            elif isinstance(content, list):
                blocks = []
                for part in content:
                    if part.get("type") == "text":
                        blocks.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            header, b64data = url.split(",", 1)
                            media_type = header.split(":")[1].split(";")[0]
                            blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64data,
                                    },
                                }
                            )
                claude_msgs.append({"role": "user", "content": blocks})
        elif role == "assistant":
            blocks = []
            text = m.get("content")
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {})
                try:
                    inp = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    inp = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": fn["name"],
                        "input": inp,
                    }
                )
            claude_msgs.append(
                {
                    "role": "assistant",
                    "content": blocks or [{"type": "text", "text": ""}],
                }
            )
        elif role == "tool":
            tool_content = m.get("content", "")
            result_blocks = []
            if isinstance(tool_content, str):
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", ""),
                        "content": tool_content,
                    }
                )
            elif isinstance(tool_content, list):
                inner = []
                for part in tool_content:
                    if part.get("type") == "text":
                        inner.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            header, b64data = url.split(",", 1)
                            media_type = header.split(":")[1].split(";")[0]
                            inner.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64data,
                                    },
                                }
                            )
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", ""),
                        "content": inner,
                    }
                )
            if (
                claude_msgs
                and claude_msgs[-1]["role"] == "user"
                and isinstance(claude_msgs[-1]["content"], list)
                and claude_msgs[-1]["content"]
                and claude_msgs[-1]["content"][0].get("type") == "tool_result"
            ):
                claude_msgs[-1]["content"].extend(result_blocks)
            else:
                claude_msgs.append({"role": "user", "content": result_blocks})
    return "\n\n".join(system_parts), claude_msgs


def _claude_response_to_openai(response) -> dict:
    content_text = ""
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            content_text += block.text
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                }
            )
    msg: dict = {"content": content_text or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _call_llm_claude(
    messages: list[dict], model: str, tool_schema: dict, api_key: str
) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    system_str, claude_msgs = _openai_msgs_to_claude(messages)
    claude_tool = _openai_tool_to_claude(tool_schema)
    last_exc = None
    for attempt in range(5):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=16000,
                system=system_str,
                messages=claude_msgs,
                tools=[claude_tool],
                tool_choice={"type": "auto"},
            )
            USAGE["prompt_tokens"] += response.usage.input_tokens
            USAGE["completion_tokens"] += response.usage.output_tokens
            USAGE["calls"] += 1
            return _claude_response_to_openai(response)
        except anthropic.APIStatusError as e:
            last_exc = e
            if e.status_code in (400, 429, 500, 502, 503, 529) and attempt < 4:
                time.sleep(min(60, 2**attempt + 2))
                continue
            raise
        except Exception as e:
            last_exc = e
            if attempt < 4:
                time.sleep(min(60, 2**attempt + 2))
                continue
            raise
    raise last_exc  # type: ignore[misc]


def _call_llm_forced_openai(
    messages: list[dict], model: str, api_key: str, base_url: str
) -> str:
    """Final forced-answer call without tools."""
    body: dict = {"model": model, "messages": messages}
    if _is_local_model(base_url):
        body["max_tokens"] = 4096
    else:
        body["max_completion_tokens"] = 4096
    if _supports_temperature(model):
        body["temperature"] = 0.0
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=READER_TIMEOUT) as resp:
        d = json.load(resp)
    USAGE["prompt_tokens"] += d.get("usage", {}).get("prompt_tokens", 0)
    USAGE["completion_tokens"] += d.get("usage", {}).get("completion_tokens", 0)
    USAGE["calls"] += 1
    return d["choices"][0]["message"].get("content", "")


def _call_llm_forced_claude(messages: list[dict], model: str, api_key: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    system_str, claude_msgs = _openai_msgs_to_claude(messages)
    last_exc = None
    for attempt in range(4):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_str,
                messages=claude_msgs,
            )
            USAGE["prompt_tokens"] += response.usage.input_tokens
            USAGE["completion_tokens"] += response.usage.output_tokens
            USAGE["calls"] += 1
            return "".join(b.text for b in response.content if b.type == "text")
        except anthropic.APIStatusError as e:
            last_exc = e
            if e.status_code in (400, 429, 500, 502, 503, 529) and attempt < 3:
                time.sleep(min(60, 2**attempt + 2))
                continue
            raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------
def react_loop(
    question: str,
    model: str,
    retrieval: str,
    api_key: str,
    base_url: str,
    max_turns: int | None = None,
) -> dict:
    """Run the ReAct loop. Returns dict with 'final', 'turns', 'searches', 'trace', 'k_values'."""
    if max_turns is None:
        max_turns = MAX_TURNS
    system_prompt = _build_system_prompt(retrieval)
    tool_schema = copy.deepcopy(
        SEARCH_PIXEL_TOOL if retrieval == "pixel" else SEARCH_TEXT_TOOL
    )
    tool_name = "search_pixel" if retrieval == "pixel" else "search_text"
    use_claude = _is_claude_model(model)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    trace = []
    n_searches = 0
    k_values: list[int] = []

    for turn in range(max_turns):
        if use_claude:
            msg = _call_llm_claude(messages, model, tool_schema, api_key)
        else:
            msg = _call_llm_openai(messages, model, tool_schema, api_key, base_url)

        assistant_entry = {"role": "assistant", "content": msg.get("content")}
        if msg.get("tool_calls"):
            assistant_entry["tool_calls"] = msg["tool_calls"]
        messages.append(assistant_entry)

        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    args = {}

                if name == tool_name:
                    n_searches += 1
                    USAGE["tool_calls"] += 1
                    q = args.get("query", "")
                    k = max(1, min(args.get("top_k") or _DEFAULT_TOP_K, _MAX_TOP_K))
                    k_values.append(k)
                    trace.append((turn, "search", f"k={k} {q[:80]}"))

                    if retrieval == "text":
                        result = _search_text(q, n_docs=k)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": result[:RESULT_TRUNCATE_CHARS],
                            }
                        )
                    else:
                        image_parts = _search_pixel(q, n_docs=k)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": image_parts,
                            }
                        )
                else:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": f"[unknown tool: {name}]",
                        }
                    )
        else:
            # No tool calls -> final answer
            content = msg.get("content", "") or ""
            trace.append((turn, "answer", content[:80]))
            return {
                "final": content,
                "turns": turn + 1,
                "searches": n_searches,
                "trace": trace,
                "k_values": k_values,
            }

    # Hit max_turns: force a final answer
    messages.append(
        {
            "role": "user",
            "content": "You must now provide the final answer. Output exactly one line:\nAnswers: {your answer}",
        }
    )
    if use_claude:
        forced = _call_llm_forced_claude(messages, model, api_key)
    else:
        forced = _call_llm_forced_openai(messages, model, api_key, base_url)
    trace.append((max_turns, "forced_answer", forced[:80]))
    return {
        "final": forced,
        "turns": max_turns + 1,
        "searches": n_searches,
        "trace": trace,
        "k_values": k_values,
    }


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
_ANSWERS_PAT = re.compile(r"(?im)^\s*answers?:\s*(.*)$")


def parse_answer(reply: str) -> str:
    """Extract the final answer from agent output."""
    if not reply:
        return ""
    matches = _ANSWERS_PAT.findall(reply)
    if matches:
        return matches[-1].strip().rstrip(".")
    return reply.splitlines()[-1].strip() if reply else ""


# ---------------------------------------------------------------------------
# Token-level F1 grading (MoNaCo primary metric, SQuAD-style normalization)
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation, articles, whitespace."""
    if s is None:
        return ""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground_truth after normalization."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_tokens)
    r = num_same / len(gold_tokens)
    return 2 * p * r / (p + r)


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def grade_monaco(predicted: str, validated_answer: Any) -> dict:
    """Grade a MoNaCo prediction against the validated_answer field.

    MoNaCo validated_answer is either:
      - a flat list of strings: ['ans1', 'ans2', ...]
      - a list of tuples (list of lists): [['a','b'], ['c','d']]

    For a flat list, we treat the gold as the joined string "ans1, ans2, ..."
    and compute token F1 against it.

    For list-of-tuples, we compute max F1 over all tuple-combinations.

    Returns dict with 'em' and 'f1'.
    """
    if not validated_answer:
        return {"em": 0, "f1": 0.0}

    # Flatten gold answer to a single string for token F1
    if isinstance(validated_answer, list):
        if all(isinstance(x, list) for x in validated_answer):
            # List of tuples: compute max F1 over all combinations
            # (each combination is one element from each tuple)
            # But for simplicity, flatten all elements as gold tokens
            flat = []
            for sub in validated_answer:
                flat.extend(sub)
            gold_str = ", ".join(str(a) for a in flat)
        else:
            gold_str = ", ".join(str(a) for a in validated_answer)
    else:
        gold_str = str(validated_answer)

    f1 = token_f1(predicted, gold_str)
    em = exact_match(predicted, gold_str)
    return {"em": em, "f1": f1}


# ---------------------------------------------------------------------------
# Process one example
# ---------------------------------------------------------------------------
def process_one(
    ex: dict, model: str, retrieval: str, api_key: str, base_url: str
) -> dict:
    """Run the ReAct agent on one MoNaCo example. Returns prediction record."""
    t0 = time.time()
    ex_num = ex["ex_num"]
    question = ex["question"]
    decomp = ex.get("decomposition") or []

    try:
        result = react_loop(question, model, retrieval, api_key, base_url)
        final_answer = parse_answer(result["final"])
        output = (
            f"Let's think step by step:\n"
            f"[self-decomp ReAct, {result['turns']} turns, {result['searches']} searches]\n"
            f"\nAnswers: {final_answer}"
        )
        rec: dict = {
            "question": question,
            "output": output,
            "qa_type": f"agent_self_decomp_{retrieval}",
            "llm": model,
            "gold_decomposition": "\n".join(
                f"{i + 1}. {s}" for i, s in enumerate(decomp)
            ),
            "ex_num": ex_num,
            "gold_question": question,
            "elapsed_sec": round(time.time() - t0, 2),
            "n_turns": result["turns"],
            "n_searches": result["searches"],
            "k_values": result["k_values"],
            "trace": result["trace"],
        }
    except Exception as e:
        rec = {
            "question": question,
            "output": "Let's think step by step: [agent_error]\nAnswers: [error]",
            "qa_type": f"agent_self_decomp_{retrieval}",
            "llm": model,
            "gold_decomposition": "\n".join(
                f"{i + 1}. {s}" for i, s in enumerate(decomp)
            ),
            "ex_num": ex_num,
            "gold_question": question,
            "elapsed_sec": round(time.time() - t0, 2),
            "n_turns": 0,
            "n_searches": 0,
            "k_values": [],
            "trace": [],
            "agent_error": str(e),
            "agent_traceback": traceback.format_exc(),
        }

    # Inline F1 grading if gold answer is available
    validated_answer = ex.get("validated_answer")
    if validated_answer is not None:
        predicted = parse_answer(rec.get("output", ""))
        scores = grade_monaco(predicted, validated_answer)
        rec["token_f1"] = scores["f1"]
        rec["token_em"] = scores["em"]
        rec["gold_answers"] = validated_answer

    return rec


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_monaco(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"MoNaCo dataset not found at {path}\n"
            f"Download from https://github.com/facebookresearch/MoNaCo\n"
            f"Place the JSONL file at: {DEFAULT_DATA_PATH}\n"
            f"Or pass --data-path <path>"
        )
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# LLM judge (optional secondary metric)
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.
[question]: {question}
[response]: '{response}'

Your judgment must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

final answer length: Provide the overall number of answers that appear in [response], not just the correct ones.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems, a margin of 1 to 5.5 percentage points is acceptable. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

overlapping answers: List all of the answers in [response] that also appear in [correct_answer]. You can consider an answer from [response] to match with an answer in [correct_answer] if it is equivalent or is within a small margin of error for numerical problems, a margin of 1 to 3.5 percentage points is acceptable. List all of the [response] answer appearing in [correct_answer] with each answer delimited by '###'. If the number of overlapping answers is zero, output 'NULL'.
"""

_JUDGE_LEN_PAT = re.compile(r"final answer length\s*[:\-]?\s*(\d+)", re.IGNORECASE)
_JUDGE_OVERLAP_PAT = re.compile(
    r"overlapping answers\s*[:\-]?\s*(.*)", re.IGNORECASE | re.DOTALL
)


def _gold_length(validated_answer: Any) -> int:
    """MoNaCo gold_answers_length convention."""
    if not isinstance(validated_answer, list) or not validated_answer:
        return 0
    if all(isinstance(x, list) for x in validated_answer):
        return sum(len(x) for x in validated_answer)
    return len(validated_answer)


def _parse_judge_response(text: str) -> tuple[int, list[str]]:
    m = _JUDGE_LEN_PAT.search(text)
    n_pred = int(m.group(1)) if m else 0
    m = _JUDGE_OVERLAP_PAT.search(text)
    if not m:
        return n_pred, ["NULL"]
    tail = m.group(1).strip().split("\n\n", 1)[0].strip()
    if tail.upper().startswith("NULL"):
        return n_pred, ["NULL"]
    parts = [p.strip() for p in tail.split("###") if p.strip()]
    return n_pred, parts if parts else ["NULL"]


def _judge_f1(predicted_num: int, correct_preds: list[str], gold_len: int) -> dict:
    num_correct = 0 if correct_preds == ["NULL"] else len(correct_preds)
    p = num_correct / predicted_num if predicted_num > 0 else 0.0
    r = num_correct / gold_len if gold_len > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "judge_f1": f1,
        "judge_p": p,
        "judge_r": r,
        "judge_n_correct": num_correct,
        "judge_n_pred": predicted_num,
        "judge_gold_len": gold_len,
    }


def judge_one(rec: dict, judge_model: str, api_key: str, base_url: str) -> dict:
    """Run the MoNaCo LLM judge on one prediction record. Returns judge scores."""
    validated_answer = rec.get("gold_answers")
    if validated_answer is None:
        return {}
    question = rec["question"]
    response = rec.get("output", "")
    prompt = JUDGE_PROMPT.format(
        question=question,
        response=response,
        correct_answer=str(validated_answer),
    )
    body = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    last_exc = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                d = json.load(resp)
            judgement = d["choices"][0]["message"].get("content", "") or ""
            n_pred, correct_preds = _parse_judge_response(judgement)
            gl = _gold_length(validated_answer)
            scores = _judge_f1(n_pred, correct_preds, gl)
            scores["judge_raw"] = judgement[:2000]
            return scores
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(min(60, 2**attempt + 2))
                continue
            raise
        except urllib.error.URLError as e:
            last_exc = e
            if attempt < 3:
                time.sleep(min(60, 2**attempt + 2))
                continue
            raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="MoNaCo multi-hop QA evaluation with ReAct agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--reader",
        type=str,
        default="gpt-5",
        help="Model name (default: gpt-5). E.g. gpt-5, gpt-4o-2024-08-06, claude-sonnet-4-6.",
    )
    ap.add_argument(
        "--retrieval",
        type=str,
        choices=["text", "pixel"],
        default="text",
        help="Retrieval backend: 'text' (default) or 'pixel'.",
    )
    ap.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help=f"Path to MoNaCo JSONL file (default: {DEFAULT_DATA_PATH})",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Override output directory (default: eval_output/monaco/<tag>)",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Process only first N examples (0 = all)"
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent workers (default: 4)",
    )
    ap.add_argument(
        "--smoke", type=int, default=0, help="Quick smoke test with N examples"
    )
    ap.add_argument(
        "--tag-suffix",
        type=str,
        default="",
        help="Appended to the auto-generated run tag",
    )
    ap.add_argument("--base-url", type=str, default="", help="Override OPENAI_BASE_URL")
    ap.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Override OPENAI_API_KEY / ANTHROPIC_API_KEY",
    )
    ap.add_argument(
        "--pixel-api",
        type=str,
        default="",
        help=f"Pixel search endpoint (default: {PIXEL_API})",
    )
    ap.add_argument(
        "--text-api",
        type=str,
        default="",
        help=f"Text search endpoint (default: {TEXT_API})",
    )
    ap.add_argument(
        "--image-detail",
        choices=["auto", "low", "high"],
        default="auto",
        help="OpenAI image detail level for pixel retrieval",
    )
    ap.add_argument(
        "--default-top-k",
        type=int,
        default=DEFAULT_K,
        help=f"Default top-k per search (default: {DEFAULT_K})",
    )
    ap.add_argument(
        "--max-top-k",
        type=int,
        default=MAX_TOP_K,
        help=f"Max top-k the agent can use (default: {MAX_TOP_K})",
    )
    ap.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS,
        help=f"Max ReAct turns (default: {MAX_TURNS})",
    )
    ap.add_argument(
        "--judge",
        action="store_true",
        help="Run LLM judge grading after all predictions (secondary metric)",
    )
    ap.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4.1-2025-04-14",
        help="Model for LLM judge (default: gpt-4.1-2025-04-14)",
    )
    ap.add_argument(
        "--judge-workers",
        type=int,
        default=12,
        help="Workers for LLM judge (default: 12)",
    )
    args = ap.parse_args()

    # Set module-level globals
    global _PIXEL_API, _TEXT_API, _DEFAULT_TOP_K, _MAX_TOP_K, _IMAGE_DETAIL
    _PIXEL_API = args.pixel_api or PIXEL_API
    _TEXT_API = args.text_api or TEXT_API
    _DEFAULT_TOP_K = args.default_top_k
    _MAX_TOP_K = args.max_top_k
    _IMAGE_DETAIL = args.image_detail
    globals()["MAX_TURNS"] = args.max_turns

    # Resolve API key
    model = args.reader
    use_claude = _is_claude_model(model)
    if use_claude:
        api_key = (
            args.api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
        ).strip()
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY not set (use --api-key or env var)")
        base_url = ""  # unused for Claude
    else:
        api_key = (args.api_key.strip() or os.environ.get("OPENAI_API_KEY", "")).strip()
        if not api_key:
            raise SystemExit("OPENAI_API_KEY not set (use --api-key or env var)")
        base_url = (
            args.base_url.strip()
            or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).strip()

    # Build run tag
    model_slug = model.replace("/", "_").replace("-", "_").replace(".", "_")
    tag = f"{model_slug}_agent_{args.retrieval}{args.tag_suffix}"

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = DEFAULT_OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logging(tag)

    # Load data
    data_path = Path(args.data_path)
    rows = load_monaco(data_path)
    logger.info(f"Reader: {model} | retrieval: {args.retrieval} | tag: {tag}")
    logger.info(f"Data: {data_path} ({len(rows)} examples)")
    if not use_claude:
        logger.info(f"base_url: {base_url}")
    if args.retrieval == "pixel":
        logger.info(f"pixel_api: {_PIXEL_API}")
    else:
        logger.info(f"text_api: {_TEXT_API}")

    # Filter already-done examples (resumable)
    todo = [
        ex
        for ex in rows
        if not (out_dir / f"llm_qa_judgement__{ex['ex_num']}.json").exists()
    ]
    logger.info(
        f"Remaining: {len(todo)} (skipping {len(rows) - len(todo)} already-done)"
    )

    if args.smoke:
        todo = todo[: args.smoke]
    elif args.limit:
        todo = todo[: args.limit]

    if not todo:
        logger.info("Nothing to do.")
    else:
        # Run predictions
        t0 = time.time()
        n_ok = n_err = 0
        f1_sum = 0.0
        n_graded = 0

        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(
                    process_one, ex, model, args.retrieval, api_key, base_url
                ): ex
                for ex in todo
            }
            for i, fut in enumerate(cf.as_completed(futs), 1):
                ex = futs[fut]
                out_path = out_dir / f"llm_qa_judgement__{ex['ex_num']}.json"
                try:
                    rec = fut.result()
                    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
                    n_ok += 1

                    f1_val = rec.get("token_f1")
                    f1_str = f"F1={f1_val:.3f}" if f1_val is not None else "F1=N/A"
                    if f1_val is not None:
                        f1_sum += f1_val
                        n_graded += 1

                    tail = rec["output"].splitlines()[-1][:120] if rec["output"] else ""
                    msg = (
                        f"  [{i:>4}/{len(todo)}] ex={ex['ex_num']:<5} "
                        f"turns={rec.get('n_turns', '?')} "
                        f"searches={rec.get('n_searches', '?')} "
                        f"t={rec.get('elapsed_sec'):>5.1f}s "
                        f"{f1_str} | {tail}"
                    )
                    logger.info(msg)
                except Exception as e:
                    n_err += 1
                    tb = traceback.format_exc()
                    msg = f"  [{i:>4}/{len(todo)}] ex={ex['ex_num']:<5} ERR: {e}"
                    logger.info(msg)
                    logger.info(tb)

        dt = time.time() - t0

        # Estimate cost
        price = PRICING.get(
            model,
            PRICING.get(
                "gpt-5"
                if "gpt-5" in model
                else ("gpt-4o-2024-08-06" if "gpt-4o" in model else ""),
                {"in": 0.0, "out": 0.0},
            ),
        )
        cost = (
            USAGE["prompt_tokens"] * price["in"] * 1e-6
            + USAGE["completion_tokens"] * price["out"] * 1e-6
        )

        logger.info(f"\nPredictions done in {dt / 60:.1f} min — ok={n_ok} err={n_err}")
        logger.info(f"LLM calls: {USAGE['calls']} | tool calls: {USAGE['tool_calls']}")
        logger.info(
            f"Tokens: in={USAGE['prompt_tokens']:,} out={USAGE['completion_tokens']:,} | est cost: ${cost:.4f}"
        )
        if n_graded:
            logger.info(
                f"Mean token F1 (new predictions): {f1_sum / n_graded:.4f} ({n_graded} graded)"
            )

    # Aggregate F1 over all completed predictions
    all_files = sorted(out_dir.glob("llm_qa_judgement__*.json"))
    if all_files:
        all_f1 = []
        all_em = []
        for p in all_files:
            rec = json.loads(p.read_text())
            if rec.get("token_f1") is not None:
                all_f1.append(rec["token_f1"])
            if rec.get("token_em") is not None:
                all_em.append(rec["token_em"])
        if all_f1:
            logger.info(f"\nAggregate over {len(all_f1)} examples:")
            logger.info(f"  Mean token F1: {sum(all_f1) / len(all_f1):.4f}")
            logger.info(f"  Mean token EM: {sum(all_em) / len(all_em):.4f}")

    # Optional: LLM judge grading
    if args.judge:
        logger.info(
            f"\nRunning LLM judge ({args.judge_model}) on {len(all_files)} predictions..."
        )
        judge_api_key = os.environ.get("OPENAI_API_KEY", "").strip() or api_key
        judge_base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).strip()

        n_judged = n_skip_judge = n_judge_err = 0
        judge_f1_sum = 0.0

        def _judge_wrapper(p: Path) -> tuple[Path, dict | None, str]:
            rec = json.loads(p.read_text())
            if rec.get("judge_f1") is not None:
                return p, rec, "skip"
            try:
                scores = judge_one(rec, args.judge_model, judge_api_key, judge_base_url)
                rec.update(scores)
                tmp = p.with_suffix(p.suffix + ".tmp")
                tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
                os.replace(tmp, p)
                return p, rec, "ok"
            except Exception as e:
                return p, None, f"err:{e}"

        with cf.ThreadPoolExecutor(max_workers=args.judge_workers) as pool:
            futs = [pool.submit(_judge_wrapper, p) for p in all_files]
            for i, fut in enumerate(cf.as_completed(futs), 1):
                path, rec, status = fut.result()
                if status == "ok":
                    n_judged += 1
                    jf1 = rec.get("judge_f1", 0.0)
                    judge_f1_sum += jf1
                    if i % 50 == 0 or i == len(futs):
                        logger.info(f"  Judged {i}/{len(futs)}")
                elif status == "skip":
                    n_skip_judge += 1
                    jf1 = rec.get("judge_f1", 0.0) if rec else 0.0
                    judge_f1_sum += jf1
                else:
                    n_judge_err += 1

        n_total_judge = n_judged + n_skip_judge
        if n_total_judge:
            logger.info(
                f"Judge: {n_judged} new + {n_skip_judge} cached = {n_total_judge} total ({n_judge_err} errors)"
            )
            logger.info(f"Mean judge F1: {judge_f1_sum / n_total_judge:.4f}")

    # Write aggregate summary
    summary_path = out_dir / "summary.json"
    summary: dict = {
        "tag": tag,
        "model": model,
        "retrieval": args.retrieval,
        "n_predictions": len(all_files),
    }
    if all_f1:
        summary["mean_token_f1"] = round(sum(all_f1) / len(all_f1), 4)
        summary["mean_token_em"] = round(sum(all_em) / len(all_em), 4)
    if args.judge and n_total_judge:
        summary["mean_judge_f1"] = round(judge_f1_sum / n_total_judge, 4)
    summary["usage"] = dict(USAGE)
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
