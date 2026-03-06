"""
assistant.py — AIVoiceAssistant with LangGraph HITL + Multi-turn Param Collection
==================================================================================
KEY CHANGES vs previous version:
  • AssistantState gains `pending_tool_intent` — stores the tool name and
    already-collected args between turns when a clarification question is asked.
  • agent_node checks pending_tool_intent FIRST on every turn.
    If it's set, the user's current message is interpreted as an answer to the
    pending clarification question, NOT as a brand-new query.
  • _extract_param_value() does robust extraction from noisy STT text:
      "a 7 glass"   → class "7"
      "class seven" → class "7"
      "roll 12"     → roll_number "12"
      "just 5"      → roll_number "5"

Flow with intent tracking:
  Turn 1  User: "Check my result"
          → LLM proposes get_exam_result({})
          → PARAM GUARD: class_name missing
          → state.pending_tool_intent = {tool:"get_exam_result", args:{}}
          → Returns: "Which class are you in?"

  Turn 2  User: "a 7 glass"   ← STT misheard "Class 7"
          → pending_tool_intent IS SET
          → bypass LLM entirely
          → _extract_param_value("class_name", "a 7 glass") → "7"
          → args = {class_name:"7"} | roll_number still missing
          → Returns: "What's your roll number?"

  Turn 3  User: "12"
          → pending_tool_intent IS SET
          → _extract_param_value("roll_number", "12") → "12"
          → args = {class_name:"7", roll_number:"12"} — all present ✅
          → Clears intent → queues get_exam_result(class_name="7", roll_number="12")
"""

import base64
import json
import os
import re
import tempfile
import time
import uuid
from typing import Annotated, Dict, List, Literal, Optional, TypedDict
import operator
import asyncio
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from reasoning_engine.llm.chat_llm import llm_loader
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import yaml


# ══════════════════════════════════════════════════════════════
# ENV CONFIG
# ══════════════════════════════════════════════════════════════


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))




with open("configs/model_config.yaml",    "r", encoding="utf-8") as f:
    model_config    = yaml.safe_load(f)

with open("configs/tools_config.yaml",    "r", encoding="utf-8") as f:
    tools_config    = yaml.safe_load(f)

with open("configs/pipeline_config.yaml", "r", encoding="utf-8") as f:
    pipeline_config = yaml.safe_load(f)




# ── Quick references ──────────────────────────────────────────────────────

# model_config.yaml
GROQ_LLM        = model_config["GROQ_LLM"]
GROQ_STT        = model_config["GROQ_STT"]
GROQ_API_KEY    = model_config["GROQ_API_KEY"]
GROQ_LLM_MODEL  = model_config["GROQ_LLM_MODEL"]
OLLAMA_BASE_URL = model_config["OLLAMA_BASE_URL"]
OLLAMA_MODEL    = model_config["OLLAMA_MODEL"]
LLM_MAX_TOKENS  = model_config["LLM_MAX_TOKENS"]
LLM_RETRIES     = model_config["LLM_RETRIES"]
_STT_CFG        = model_config["stt"]
_LLM_CFG        = model_config          # whole file is flat

LOCAL_WHISPER_MODEL = model_config['LOCAL_WHISPER_MODEL']

# Resolve to absolute path — eliminates ambiguity completely
WHISPER_PATH =     os.path.join(_PROJECT_ROOT, LOCAL_WHISPER_MODEL["path"])                    # project root


print(f"[DEBUG] WHISPER_PATH = {WHISPER_PATH}")

print(f"[DEBUG] _PROJECT_ROOT = {_PROJECT_ROOT}")
print(f"[DEBUG] __file__ = {__file__}")

# tools_config.yaml
AUTO_APPROVE_TOOLS  = set(tools_config["auto_approve_tools"])
_WRONG_FOR_RESULT   = set(tools_config["wrong_for_result"])
_WRONG_FOR_SEARCH   = set(tools_config["wrong_for_search"])
_AUDIT_CFG          = tools_config["audit"]

# pipeline_config.yaml
_WORD_TO_NUM    = pipeline_config["spoken_numbers"]
_PARAM_CFG      = pipeline_config["param_schema"]
_PAT            = pipeline_config           # patterns are top-level keys




# ═════════════════════════════════════════════════════════════
# STT CLIENT SETUP
# ══════════════════════════════════════════════════════════════

if GROQ_STT:
    from groq import Groq as GroqSTT
    stt_client = GroqSTT(api_key=GROQ_API_KEY)
    _whisper   = None
    print("[STT] Using Groq Whisper")
else:
    from faster_whisper import WhisperModel
    print(f"[STT] Loading faster-whisper from: ")
    _load_start = time.perf_counter()
    _whisper = WhisperModel(
        model_size_or_path=str(WHISPER_PATH),
        device=LOCAL_WHISPER_MODEL["device"],
        compute_type=LOCAL_WHISPER_MODEL["compute_type"],
        cpu_threads=LOCAL_WHISPER_MODEL["cpu_threads"],
        num_workers=LOCAL_WHISPER_MODEL["num_workers"],
    )
    print(f"[STT] faster-whisper ready in {time.perf_counter() - _load_start:.2f}s")
    stt_client = None


def _build_keyword_regex(words: list) -> re.Pattern:
    escaped = [re.escape(w) for w in words]
    return re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.I)


def _build_result_regex(keywords: list, exclude: list) -> re.Pattern:
    kw  = '|'.join(re.escape(w) for w in keywords)
    exc = '|'.join(re.escape(w) for w in exclude)
    return re.compile(rf'\b({kw})\b(?!.*\b({exc})\b)', re.I)


_RESULT_KEYWORDS = _build_result_regex(
    pipeline_config["result_keywords"],
    pipeline_config["result_exclude_keywords"]
)


_SEARCH_KEYWORDS = _build_keyword_regex(pipeline_config["search_keywords"])
_REFUSAL_PHRASES = _build_keyword_regex(pipeline_config["refusal_phrases"])
_QUESTION_START  = _build_keyword_regex(pipeline_config["question_starters"])
_CASUAL_PHRASES  = _build_keyword_regex(pipeline_config["casual_phrases"])
# ══════════════════════════════════════════════════════════════
# PATTERNS & HELPERS
# ══════════════════════════════════════════════════════════════


_ROLL_PATTERN = re.compile(r'\b(\d{3,6})\b')

_CLASS_PATTERN   = re.compile(
    r'\b(?:class|std|standard|grade|klass)?\s*([1-9]|1[0-2])\b', re.I
)
_SECTION_PATTERN = re.compile(r'\b(?:section|sec)?\s*([A-Ea-e])\b')




# ══════════════════════════════════════════════════════════════
# PARAM SCHEMA — single source of truth
# ══════════════════════════════════════════════════════════════

# AFTER — built from pipeline_config
def _build_param_schema(cfg: dict) -> dict:
    schema = {}
    for param, spec in cfg.items():
        entry = {
            "description": spec["description"],
            "examples":    spec.get("examples", []),
            "invalid":     spec.get("invalid", []),
            "regex":       re.compile(spec["regex_pattern"], re.I),
        }
        if param == "class_name":
            mn, mx = spec["min_value"], spec["max_value"]
            entry["validate"] = lambda v, mn=mn, mx=mx: (
                bool(re.match(r'^\d{1,2}$', str(v).strip())) and
                mn <= int(str(v).strip()) <= mx
            )
            entry["clean"] = lambda v: str(v).strip().rstrip(".").strip()

        elif param == "roll_number":
            ml = spec["max_length"]
            entry["validate"] = lambda v, ml=ml: bool(
                re.match(rf'^[A-Za-z0-9]{{1,{ml}}}$', str(v).strip())
            )
            entry["clean"] = lambda v: re.sub(r'[\s\-,]+', '', str(v)).strip()

        elif param == "section":
            entry["validate"] = lambda v: bool(re.match(r'^[A-Ea-e]$', str(v).strip()))
            entry["clean"]    = lambda v: str(v).strip().upper()

        else:
            entry["validate"] = lambda v: bool(str(v).strip())
            entry["clean"]    = lambda v: str(v).strip()

        schema[param] = entry
    return schema

PARAM_SCHEMA = _build_param_schema(_PARAM_CFG)




def _get_tools():
    from reasoning_engine.tools.registry import ALL_TOOLS, FUNCTION_MAP   # ← re-import every call
    return ALL_TOOLS, FUNCTION_MAP


def _normalize_spoken(text: str) -> str:
    """Replace spoken/ordinal words with digits."""
    for word, digit in _WORD_TO_NUM.items():
        text = re.sub(rf'\b{word}\b', digit, text, flags=re.I)
    return text


# ══════════════════════════════════════════════════════════════
# LAYER 1 — Fast regex extraction
# ══════════════════════════════════════════════════════════════

def _extract_by_regex(param: str, text: str) -> Optional[str]:
    schema = PARAM_SCHEMA.get(param)
    if not schema:
        return None

    text = _normalize_spoken(text)
    match = schema["regex"].search(text)
    if not match:
        return None

    # Take first non-None group
    raw = next((g for g in match.groups() if g), None)
    if not raw:
        return None

    cleaned = schema["clean"](raw)
    if schema["validate"](cleaned):
        return cleaned

    return None


# ══════════════════════════════════════════════════════════════
# LAYER 2 — LLM extraction with strict schema-aware prompt
# ══════════════════════════════════════════════════════════════

async def _extract_by_llm(tool_name: str, missing_params: list, user_text: str) -> dict:
    if not missing_params:
        return {}

    param_lines = []
    for p in missing_params:
        schema = PARAM_SCHEMA.get(p, {})
        desc    = schema.get("description", p)
        examples = ", ".join(f'"{e}"' for e in schema.get("examples", []))
        invalid  = ", ".join(f'"{e}"' for e in schema.get("invalid",  []))
        line = f'- "{p}": {desc}. Valid examples: {examples}.'
        if invalid:
            line += f' NEVER use: {invalid}.'
        param_lines.append(line)

    prompt = f"""Extract parameter values from the user's spoken message.
Return ONLY a valid JSON object. No explanation, no markdown.

Parameters needed:
{chr(10).join(param_lines)}

User said: "{_normalize_spoken(user_text)}"

Rules:
- Use null if a value is not clearly present
- Never guess or invent values
- class_name must be a digit 1-12 only — never a subject name or word

Return: {{{", ".join(f'"{p}": ...' for p in missing_params)}}}"""

    from reasoning_engine.llm.chat_llm import llm_loader
    from langchain_core.messages import HumanMessage

    _llm = await llm_loader.load()
    response = _llm.invoke([HumanMessage(content=prompt)])

    try:
        text = response.content.strip()
        text = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.MULTILINE).strip()
        raw_extracted = json.loads(text)

        result = {}
        for p, v in raw_extracted.items():
            if v is None or str(v).lower() in {"null", "none", "unknown", "n/a"}:
                continue
            schema = PARAM_SCHEMA.get(p)
            if not schema:
                result[p] = str(v)
                continue
            cleaned = schema["clean"](str(v))
            if schema["validate"](cleaned):
                result[p] = cleaned
            else:
                print(f"[LLM EXTRACT] Rejected '{p}'='{v}' — failed validation")

        return result

    except Exception as e:
        print(f"[LLM EXTRACT] Parse failed: {e}")
        return {}


# ══════════════════════════════════════════════════════════════
# LAYER 3 — Combined: regex first, LLM for remainder
# ══════════════════════════════════════════════════════════════

async def extract_params(tool_name: str, missing_params: list, user_text: str) -> dict:
    """
    Multi-layer param extraction:
      1. Regex  — fast, always runs first
      2. LLM    — only for params regex couldn't find
      3. Validate + clean — applied to ALL extracted values

    Returns dict of successfully extracted + validated params only.
    """
    result = {}
    still_missing = []

    # Layer 1 — regex
    for param in missing_params:
        val = _extract_by_regex(param, user_text)
        if val:
            print(f"[REGEX] '{param}' → '{val}'")
            result[param] = val
        else:
            still_missing.append(param)

    # Layer 2 — LLM for anything regex missed
    if still_missing:
        llm_result = await _extract_by_llm(tool_name, still_missing, user_text)
        for p, v in llm_result.items():
            print(f"[LLM]   '{p}' → '{v}'")
            result[p] = v

    print(f"[EXTRACT] final: {result}")
    return result


async def _extract_params_with_llm(tool_name: str, missing_params: list, user_text: str) -> dict:
    """Use LLM to extract parameter values from noisy STT text."""

    param_descriptions = {
        "class_name": "class or grade number (e.g. 6, 7, 8, 9, 10)",
        "roll_number": "student roll number (numeric)",
    }

    params_desc = "\n".join(
        f"- {p}: {param_descriptions.get(p, p)}"
        for p in missing_params
    )

    prompt = f"""Extract these values from the user's message. Return ONLY valid JSON, nothing else.

    Parameters to extract:
    {params_desc}

    User said: "{user_text}"

    Rules:
    - If a value is not mentioned, use null
    - class_name MUST be a number between 1-12 ONLY. Never use subject names, words, or anything else.
      Valid: "6", "7", "10"  Invalid: "English", "Math", "unknown", "second"
    - If class_name cannot be identified as a number, use null
    - roll_number: could be alphanumeric, merge if split ("234 330" → "234330")

    Return JSON like: {{"class_name": "6", "roll_number": null}}"""

    _langchain_llm = await llm_loader.load()
    from langchain_core.messages import HumanMessage
    response = _langchain_llm.invoke([HumanMessage(content=prompt)])

    try:
        text = response.content.strip()
        # Strip markdown fences if present
        text = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.MULTILINE).strip()
        extracted = json.loads(text)
        # Filter out nulls
        return {k: str(v) for k, v in extracted.items() if v is not None}
    except Exception as e:
        print(f"[LLM EXTRACT] Failed to parse: {e}")
        return {}




def _normalize_spoken_numbers(text: str) -> str:
    """Replace spoken number words with digits so regex patterns match."""
    for word, digit in _WORD_TO_NUM.items():
        text = re.sub(rf'\b{word}\b', digit, text, flags=re.I)
    return text


def _extract_param_value(param_name: str, user_text: str) -> Optional[str]:
    """
    Robustly extract a specific param value from noisy STT text.

    class_name  : "a 7 glass"   → "7"
                  "class seven" → "7"
                  "I am in 10"  → "10"
    roll_number : "roll 12"     → "12"
                  "234 330"     → "234330"
                  "just 5"      → "5"
    section     : "section B"   → "B"
    """
    text = _normalize_spoken_numbers(user_text)

    if param_name == "class_name":
        match = _CLASS_PATTERN.search(text)
        if match:
            return match.group(1)
        # Fallback: any 1-2 digit number in 1-12 range
        fallback = re.search(r'\b(1[0-2]|[1-9])\b', text)
        return fallback.group(1) if fallback else None

    elif param_name == "roll_number":
        # Merge digit groups split by spaces/dashes: "234 330" → "234330"
        cleaned = re.sub(r'(\d)[,\s\-]+(\d)', r'\1\2', text)
        match   = _ROLL_PATTERN.search(cleaned)
        if match:
            return match.group(1)

            # fallback: any alphanumeric string of length 1+
        fallback = re.search(r'\b([A-Za-z0-9]{1,10})\b', cleaned)
        return fallback.group(1) if fallback else None

    elif param_name == "section":
        match = _SECTION_PATTERN.search(text)
        return match.group(1).upper() if match else None

    return None


def _extract_roll_number(text: str) -> Optional[str]:
    """Legacy helper used by fallback heuristics."""
    return _extract_param_value("roll_number", text)


def _make_forced_tool_call(tool_name: str, args: dict) -> dict:
    fake_id   = f"fallback_{uuid.uuid4().hex[:8]}"
    args_json = json.dumps(args)
    return {
        "pending_tool_calls": [
            {"id": fake_id, "name": tool_name, "arguments": args_json}
        ],
        "raw_assistant_message": {
            "role": "assistant",
            "tool_calls": [{"id": fake_id, "type": "function",
                            "function": {"name": tool_name, "arguments": args_json}}],
        },
        "final_response":      "",
        "function_called":     None,
        "function_results":    None,
        "conversation_update": None,
        "pending_tool_intent": None,  # clear intent once tool is queued
    }


def _make_clarification_response(question: str, tool_name: str, collected_args: dict) -> dict:
    """
    Ask the user for a missing param and store intent so the next turn
    knows which tool we are collecting for and what's already been gathered.
    """
    assistant_dict = {"role": "assistant", "content": question}
    return {
        "pending_tool_calls":    None,
        "raw_assistant_message": assistant_dict,
        "final_response":        question,
        "function_called":       None,
        "function_results":      None,
        "conversation_update":   [assistant_dict],
        "pending_tool_intent": {
            "tool_name":      tool_name,
            "collected_args": collected_args,
        },
    }


def _should_search(user_text: str, model_reply: str = "") -> bool:
    if _CASUAL_PHRASES.match(user_text.strip()):
        return False
    if _SEARCH_KEYWORDS.search(user_text):
        return True
    if model_reply and _REFUSAL_PHRASES.search(model_reply):
        if _QUESTION_START.search(user_text.strip()) and len(user_text.split()) >= 3:
            return True
    return False


def _strip_markdown(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\n{2,}', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.M)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.M)
    return text.strip()

# ══════════════════════════════════════════════════════════════
# UNIFIED LLM CALL
# ══════════════════════════════════════════════════════════════

async def llm_chat(messages: list, tools=None, max_tokens=4000,retries=2):
    _SKIP_CONTENT = {"unknown", "n/a", "null", "undefined", "not provided"}

    max_tokens = max_tokens or model_config["LLM_MAX_TOKENS"]
    retries = retries or model_config["LLM_RETRIES"]

    def _is_safe_message(m: dict) -> bool:
        """Filter out messages that would cause Groq 400 errors."""
        role = m.get("role", "")
        content = m.get("content") or ""

        # Assistant messages with tool_calls but no content are fine
        # Assistant messages with empty content AND no tool_calls → skip
        if role == "assistant" and not content and not m.get("tool_calls"):
            return False

        # Tool result messages — must be valid JSON
        if role == "tool":
            try:
                parsed = json.loads(content)
                # If the tool was called with placeholder args, the result
                # may reference "unknown" — still pass it, result is fine
                return True
            except (json.JSONDecodeError, TypeError):
                return False

        return True

    lc_messages = []
    for m in filter(_is_safe_message, messages):
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif role == "tool":
            lc_messages.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", "")))
        else:
            lc_messages.append(HumanMessage(content=content))

    _langchain_llm = await llm_loader.load()
    llm = _langchain_llm.bind_tools(tools) if tools else _langchain_llm

    for attempt in range(retries + 1):
        try:
            response = llm.invoke(lc_messages)
            return response  # ← returns AIMessage directly
        except Exception as e:
            if "tool_use_failed" in str(e) and attempt < retries:
                print(f"[LLM] Tool use failed, retry {attempt + 1}/{retries}...")
                await asyncio.sleep(model_config["LLM_RETRY_DELAY_SECONDS"])
                continue
            raise

    class _FakeChoice:
        def __init__(self, msg): self.message = msg

    class _FakeMessage:
        def __init__(self, r):
            self.content    = r.content
            self.role       = "assistant"
            self.tool_calls = [
                type("TC", (), {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": type("F", (), {
                        "name":      tc["name"],
                        "arguments": json.dumps(tc.get("args", {}))
                    })()
                })()
                for tc in (r.tool_calls or [])
            ]

    class _FakeResponse:
        def __init__(self, r):
            self.choices = [_FakeChoice(_FakeMessage(r))]

    return _FakeResponse(response)

# ══════════════════════════════════════════════════════════════
# STATE DEFINITION
# ══════════════════════════════════════════════════════════════

class AssistantState(TypedDict):
    messages:              Annotated[list, operator.add]
    user_message:          str
    pending_tool_calls:    Optional[list]
    raw_assistant_message: Optional[dict]
    human_decision:        Optional[str]
    edited_arguments:      Optional[str]
    final_response:        str
    function_called:       Optional[str]
    function_results:      Optional[list]
    conversation_update:   Optional[list]
    pending_tool_intent:   Optional[dict]   # NEW: {"tool_name": str, "collected_args": dict}
    language:              str              # ✅ add this


# ══════════════════════════════════════════════════════════════
# GRAPH NODES
# ══════════════════════════════════════════════════════════════

async def agent_node(state: AssistantState) -> dict:
    """
    First LLM pass — with intent-aware param collection.

    Priority:
      1. pending_tool_intent set → extract answer from user text (no LLM call)
      2. LLM proposes tool call → validate params → ask or proceed
      3. LLM gives plain reply   → return as-is
    """
    try:

        from app.tools.validator import (
            validate_tool_params,
            REQUIRED_TOOL_PARAMS,
            _is_blank )

    except ImportError:
        validate_tool_params = lambda tool_name, args: None  # noqa
        REQUIRED_TOOL_PARAMS = {}
        _is_blank = lambda v: not v  # noqa

    tools, _  = _get_tools()
    user_text = state["user_message"]

    # ══════════════════════════════════════════════════════════════════════════
    # BRANCH A — We are mid-collection: interpret user reply as param answer
    # ══════════════════════════════════════════════════════════════════════════
    intent = state.get("pending_tool_intent")
    # BRANCH A — mid-collection

    if intent:
        tool_name = intent["tool_name"]
        collected = dict(intent.get("collected_args", {}))
        spec = REQUIRED_TOOL_PARAMS.get(tool_name, {})
        required = spec.get("required", [])
        missing = [p for p in required if _is_blank(collected.get(p))]

        # ── LLM extracts all missing params at once ──
        extracted = await extract_params(tool_name, missing, user_text)
        print(f"[LLM EXTRACT] tool={tool_name} | extracted={extracted}")

        collected.update(extracted)

        still_missing = [p for p in required if _is_blank(collected.get(p))]

        if still_missing:
            ask_map = spec.get("ask_if_missing", {})
            question = ask_map.get(still_missing[0],
                                   f"What is the {still_missing[0].replace('_', ' ')}?")
            return _make_clarification_response(question, tool_name, collected)

        print(f"[LLM EXTRACT] all params collected: {collected}")
        return _make_forced_tool_call(tool_name, collected)

    # ══════════════════════════════════════════════════════════════════════════
    # BRANCH B — Normal LLM turn
    # ══════════════════════════════════════════════════════════════════════════
    messages   = state["messages"] + [{"role": "user", "content": user_text}]
    response   = await llm_chat(messages, tools=tools, max_tokens=4000
                                )
    # Guard: block tool calls with missing/placeholder values

    tool_calls = response.tool_calls or []   # LangChain AIMessage has .tool_calls list
    model_text = response.content or ""

    # No tool call proposed
    if not tool_calls:
        roll = _extract_param_value("roll_number", user_text)

        # AFTER
        if _RESULT_KEYWORDS.search(user_text):
            class_val = _extract_param_value("class_name", user_text)
            roll = _extract_param_value("roll_number", user_text)
            if class_val and roll:
                return _make_forced_tool_call("get_exam_result", {"class_name": class_val, "roll_number": roll})
            else:
                collected = {}
                if class_val: collected["class_name"] = class_val
                if roll:      collected["roll_number"] = roll
                return _make_clarification_response(
                    "Please tell me the class and roll number.",
                    "get_exam_result",
                    collected,)

        if _should_search(user_text, model_text):
            return _make_forced_tool_call("searxng_search", {"query": user_text})
        content        = model_text or "Sorry, can you say that again?"
        assistant_dict = {"role": "assistant", "content": content}
        return {
            "pending_tool_calls":    None,
            "raw_assistant_message": assistant_dict,
            "final_response":        content,
            "function_called":       None,
            "function_results":      None,
            "conversation_update":   [assistant_dict],
            "pending_tool_intent":   None,
        }

    # Tool calls proposed — validate and HEAL before executing
    serialised_calls = [
        {
            "id": tc["id"],
            "name": tc["name"],
            "arguments": json.dumps(tc.get("args", {}))
        }
        for tc in tool_calls
    ]

    for tc_dict in serialised_calls:
        tool_name = tc_dict["name"]
        try:
            args = json.loads(tc_dict["arguments"])
            if not isinstance(args, dict):
                args = {}
        except json.JSONDecodeError:
            args = {}

        # ── HEAL: try to extract any placeholder/blank param from user text ──
        spec = REQUIRED_TOOL_PARAMS.get(tool_name, {})
        required = spec.get("required", [])
        healed = False

        for param in required:
            if _is_blank(args.get(param)):
                extracted = _extract_param_value(param, user_text)
                if extracted:
                    args[param] = extracted
                    healed = True
                    print(f"[HEAL] '{tool_name}'.{param} healed from user text → '{extracted}'")

        if healed:
            tc_dict["arguments"] = json.dumps(args)

        # ── GUARD: after healing, check if still missing ──────────────────────
        missing_question = validate_tool_params(tool_name, args)
        if missing_question:
            collected = {p: args[p] for p in required if not _is_blank(args.get(p))}
            # Find first still-missing param and use its ask_if_missing question
            ask_map = spec.get("ask_if_missing", {})
            first_missing = next((p for p in required if _is_blank(args.get(p))), None)
            question = ask_map.get(first_missing, missing_question)
            print(f"[PARAM GUARD] '{tool_name}' still missing '{first_missing}' after heal")
            return _make_clarification_response(question, tool_name, collected)



    # All params present — check for wrong tool selection
    called_names = {tc["name"] for tc in tool_calls}

    if called_names & _WRONG_FOR_RESULT:
        roll = _extract_param_value("roll_number", user_text)
        if _RESULT_KEYWORDS.search(user_text) and roll:
            return _make_forced_tool_call("check_result", {"roll_number": roll})
    if called_names & _WRONG_FOR_SEARCH:
        if _should_search(user_text) and not _RESULT_KEYWORDS.search(user_text):
            return _make_forced_tool_call("searxng_search", {"query": user_text})

    raw_assistant_dict = {
        "role": "assistant",
        "tool_calls": [
            {
                "id":   tc["id"],
                "type": "function",
                "function": {
                    "name":      tc["name"],
                    "arguments": json.dumps(tc.get("args", {}))
                }
            }
            for tc in tool_calls
        ],
    }



    return {
        "pending_tool_calls":    serialised_calls,
        "raw_assistant_message": raw_assistant_dict,
        "final_response":        "",
        "function_called":       None,
        "function_results":      None,
        "conversation_update":   None,
        "pending_tool_intent":   None,
    }


def human_review_node(state: AssistantState) -> dict:
    """⛔ INTERRUPT POINT."""
    return {}


async def execute_node(state: AssistantState) -> dict:
    tools, function_map = _get_tools()
    decision = state.get("human_decision") or "execute"
    pending  = state.get("pending_tool_calls") or []

    if decision == "discard":
        msg = "Action discarded. How else can I help you?"
        return {
            "final_response":      msg,
            "function_called":     None,
            "function_results":    None,
            "conversation_update": [{"role": "assistant", "content": msg}],
            "pending_tool_calls":  None,
            "pending_tool_intent": None,
        }

    if decision == "edit" and state.get("edited_arguments"):
        try:
            new_args = json.loads(state["edited_arguments"])
            for tc in pending:
                tc["arguments"] = json.dumps(new_args)
        except json.JSONDecodeError:
            print("[WARN] edited_arguments is not valid JSON — ignoring")

    function_results = []
    tool_messages    = []

    for tc in pending:
        fname = tc["name"]
        try:
            fargs = json.loads(tc["arguments"]) if tc["arguments"].strip() else {}
            if not isinstance(fargs, dict):
                fargs = {}
        except json.JSONDecodeError:
            fargs = {}

        if fname == "searxng_search" and not fargs.get("query"):
            fargs["query"] = state["user_message"]

        for key, val in list(fargs.items()):
            if isinstance(val, str):
                if val.lower() in ("true", "false"):
                    fargs[key] = val.lower() == "true"
                elif val in ("1", "0"):
                    fargs[key] = val == "1"

        result = (
            function_map[fname](**fargs)
            if fname in function_map
            else {"success": False, "message": f"Unknown function: {fname}"}
        )
        function_results.append({"tool_call_id": tc["id"], "function_name": fname, "result": result})
        tool_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})


    search_called = any(r["function_name"] == "searxng_search" for r in function_results)
    if search_called:
        for r in function_results:
            if r["function_name"] == "searxng_search":
                res = r.get("result", {})
                final_text = res["message"] if res.get("success") and res.get(
                    "message") else "I couldn't find anything on that topic."
                break
        return {
            "final_response": final_text,
            "function_called": "searxng_search",
            "function_results": function_results,
            "conversation_update": [state["raw_assistant_message"]] + tool_messages + [
                {"role": "assistant", "content": final_text}],
            "pending_tool_calls": None,
            "pending_tool_intent": None,
        }


# ── All other tools — second LLM call to summarize ────────────────────────

    tool_result_lines = []
    for r in function_results:
        res   = r.get("result", {})
        fname = r["function_name"]
        tool_result_lines.append(f"Tool '{fname}' returned: {json.dumps(res)}")

    summarizer_prompt = (
        "You are a voice assistant. Summarize the tool result below in "
        "1-2 natural spoken sentences. No markdown, no bullet points. "
        "Do NOT ask for more information — the tool already ran successfully.\n\n"
        + "\n".join(tool_result_lines)
    )

    messages = [
        {"role": "system", "content": summarizer_prompt},
        {"role": "user",   "content": state["user_message"]},
    ]

    final_resp = await llm_chat(messages, tools=None, max_tokens=256)
    final_text = _strip_markdown(final_resp.content or "")

    if not final_text:
        for r in function_results:
            res = r.get("result", {})
            final_text = res.get("message", "Action completed.")
            break

    return {
        "final_response":      final_text,
        "function_called":     function_results[0]["function_name"] if function_results else None,
        "function_results":    function_results,
        "conversation_update": [state["raw_assistant_message"]] + tool_messages + [{"role": "assistant", "content": final_text}],
        "pending_tool_calls":  None,
        "pending_tool_intent": None,
    }
# ══════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════

def route_after_agent(state: AssistantState) -> Literal["human_review", "execute", "end"]:
    pending = state.get("pending_tool_calls")
    if not pending:
        return "end"
    if all(tc["name"] in AUTO_APPROVE_TOOLS for tc in pending):
        return "execute"
    return "human_review"


def route_after_review(state: AssistantState) -> Literal["execute"]:
    return "execute"

# ══════════════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════════════

def build_graph() -> tuple:
    from langgraph.checkpoint.memory import MemorySaver
    memory  = MemorySaver()
    builder = StateGraph(AssistantState)
    builder.add_node("agent",        agent_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("execute",      execute_node)
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent", route_after_agent,
        {"human_review": "human_review", "execute": "execute", "end": END},
    )
    builder.add_conditional_edges(
        "human_review", route_after_review,
        {"execute": "execute"},
    )
    builder.add_edge("execute", END)
    graph = builder.compile(checkpointer=memory, interrupt_before=["human_review"])
    return graph, memory


_graph, _memory = build_graph()

# ══════════════════════════════════════════════════════════════
# AIVoiceAssistant
# ══════════════════════════════════════════════════════════════

class AIVoiceAssistant:

    def __init__(self):
        import speech_recognition as sr
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold         = _STT_CFG["energy_threshold"]
        self.recognizer.dynamic_energy_threshold =  _STT_CFG["dynamic_energy_threshold"]

    def process_audio(self, base64_audio: str, language: str = "en") -> str:
        tmp_path = None
        try:
            audio_bytes = base64.b64decode(base64_audio)
            tmp_path    = tempfile.mktemp(suffix=".webm")
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)
            if GROQ_STT:
                with open(tmp_path, "rb") as audio_file:
                    transcription = stt_client.audio.transcriptions.create(
                        model="whisper-large-v3", file=audio_file,
                        response_format="text", language=language,
                    )
                return transcription
            else:
                t0 = time.perf_counter()

                segments_gen, info = _whisper.transcribe(
                    tmp_path,
                    language=language,
                    beam_size=_STT_CFG["beam_size"],
                    best_of=_STT_CFG["best_of"],
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": _STT_CFG["min_silence_duration_ms"],
                        "speech_pad_ms": _STT_CFG["speech_pad_ms"],
                    },
                    condition_on_previous_text=_STT_CFG["condition_on_previous_text"],
                    word_timestamps=_STT_CFG["word_timestamps"],
                )
                segments = list(segments_gen)
                elapsed  = time.perf_counter() - t0
                if not segments:
                    return "[WARN] Empty transcription"
                text = " ".join(s.text.strip() for s in segments).strip()
                print(f"[STT] {elapsed:.2f}s | lang={info.language} ({info.language_probability:.0%}) | text='{text[:60]}'")
                return text if text else "[WARN] Empty transcription"
        except Exception as e:
            return f"[ERROR] {type(e).__name__}: {e}"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    async def process_command(
        self,
        user_message: str,
        conversation_history: List[Dict],
        thread_id: str = "default",
        hitl_mode: str = "auto",
        forced_intent: dict = None,
    ) -> Dict:

        config = {"configurable": {"thread_id": thread_id}}

        # ── Carry forward pending_tool_intent from previous turn ────────────
        prev_intent = None
        try:
            current_state = _graph.get_state(config)
            if current_state and current_state.values:
                prev_intent = current_state.values.get("pending_tool_intent")
        except Exception:
            pass

        initial_state: AssistantState = {
            "messages":              conversation_history,
            "user_message":          user_message,
            "pending_tool_calls":    None,
            "raw_assistant_message": None,
            "human_decision":        None,
            "edited_arguments":      None,
            "final_response":        "",
            "function_called":       None,
            "function_results":      None,
            "conversation_update":   None,
            "pending_tool_intent":   prev_intent,   # ← injected here
            "language": "English",
        }

        try:
            final_state = None
            async for event in _graph.astream(initial_state, config, stream_mode="values"):
                final_state = event

            if final_state is None:
                return {"status": "error", "message": "Graph produced no output"}

            graph_state = _graph.get_state(config)
            if graph_state.next and "human_review" in graph_state.next:
                return {
                    "status":              "pending",
                    "message":             "⏸️ Action requires your approval before executing.",
                    "pending_tool_calls":  final_state.get("pending_tool_calls", []),
                    "thread_id":           thread_id,
                    "function_called":     None,
                    "conversation_update": None,
                }

            return self._make_success_response(final_state)

        except Exception as e:
            import traceback
            traceback.print_exc()
            err_str = str(e)
            if "tool_use_failed" in err_str or "Failed to call a function" in err_str:
                clean_msg = "I had trouble processing that request. Please try again."
            elif "rate_limit" in err_str or "413" in err_str:
                clean_msg = "I'm a bit busy right now. Please try again in a moment."
            elif "400" in err_str or "401" in err_str:
                clean_msg = "Something went wrong. Please try again."
            else:
                clean_msg = "Sorry, I couldn't complete that. Please try again."
            return {"status": "error", "message": clean_msg, "function_called": None}

    async def resume_command(
        self,
        thread_id: str,
        decision: str,
        edited_arguments: str = None,
    ) -> Dict:
        config = {"configurable": {"thread_id": thread_id}}
        _graph.update_state(
            config,
            {"human_decision": decision, "edited_arguments": edited_arguments},
            as_node="human_review",
        )
        try:
            final_state = None
            async for event in _graph.astream(None, config, stream_mode="values"):
                final_state = event
            if final_state is None:
                return {"status": "error", "message": "Resume produced no output"}
            return self._make_success_response(final_state)
        except Exception as e:
            import traceback
            traceback.print_exc()
            err_str = str(e)
            if "tool_use_failed" in err_str or "Failed to call a function" in err_str:
                clean_msg = "I had trouble processing that request. Please try again."
            elif "rate_limit" in err_str or "413" in err_str:
                clean_msg = "I'm a bit busy right now. Please try again in a moment."
            else:
                clean_msg = "Sorry, I couldn't complete that. Please try again."
            return {"status": "error", "message": clean_msg, "function_called": None}

    def get_thread_history(self, thread_id: str) -> List[Dict]:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = _graph.get_state(config)
            return state.values.get("messages", []) if state else []
        except Exception:
            return []

    def clear_thread(self, thread_id: str) -> bool:
        try:
            config = {"configurable": {"thread_id": thread_id}}
            _graph.update_state(config, {
                "messages": [], "user_message": "",
                "pending_tool_calls": None, "raw_assistant_message": None,
                "human_decision": None, "edited_arguments": None,
                "final_response": "", "function_called": None,
                "function_results": None, "conversation_update": None,
                "pending_tool_intent": None,
            })
            return True
        except Exception as e:
            print(f"[WARN] clear_thread failed: {e}")
            return False

    @staticmethod
    def _make_success_response(state: dict) -> dict:
        return {
            "status":              "success",
            "message":             state.get("final_response", ""),
            "function_called":     state.get("function_called"),
            "function_results":    state.get("function_results"),
            "conversation_update": state.get("conversation_update"),
        }