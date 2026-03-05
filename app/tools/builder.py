# app/tools/builder.py

import logging
from app.tools.validator import REQUIRED_TOOL_PARAMS

logger = logging.getLogger(__name__)

_rag_manager = None





def get_manager():
    if _rag_manager is None:
        raise RuntimeError("RAG tools not initialized. Call init_rag_tools() first.")
    return _rag_manager


def build_tool_description(tool_name: str, base_description: str) -> str:
    spec         = REQUIRED_TOOL_PARAMS.get(tool_name, {})
    instructions = spec.get("param_instructions", {})
    required     = spec.get("required", [])

    if not instructions:
        return base_description

    lines = [base_description, "PARAMETER RULES:"]
    for param, rule in instructions.items():
        tag = "[REQUIRED]" if param in required else "[optional]"
        lines.append(f"- {param} {tag}: {rule}")

    return " ".join(lines)