# app/tools/validator.py

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PLACEHOLDER_VALUES = {
    "unknown", "n/a", "na", "none", "null", "undefined",
    "not provided", "not specified", "not given", "missing",
    "?", "...", "placeholder", "your_roll_number", "your_class",
}



# ── Param specs ───────────────────────────────────────────────────────────────
REQUIRED_TOOL_PARAMS = {

    "get_class_timetable": {
        "required": ["class_name"],
        "ask_if_missing": {
            "class_name": "Which class do you want the timetable for?",
        },
        "param_instructions": {
            "class_name": "Digit only: '7', '8', '10'.",
            "subject":    "Full subject name. Omit if not mentioned.",
            "day":        "Full day name. Omit if not mentioned.",
        },
    },

    "get_exam_timetable": {
        "required": ["class_name"],
        "ask_if_missing": {
            "class_name": "Which class do you want the exam timetable for?",
        },
        "param_instructions": {
            "class_name": "Digit only: '7', '8', '10'.",
            "subject":    "Subject name if mentioned. Omit if unknown.",
        },
    },

    "get_exam_result": {
        "required": ["class_name", "roll_number"],
        "ask_if_missing": {
            "class_name":  "Which class is the student in?",
            "roll_number": "What is your roll number? e.g. STU004",
        },
        "param_instructions": {
            "class_name":  "Digit only: '7', '10'.",
            "roll_number": "Strip dashes/spaces: 'STU-004' → 'STU004'.",
        },
    },

    "get_notice": {
        "required": [],
        "ask_if_missing": {},
        "param_instructions": {
            "category":    "One of: 'exam','event','holiday','fee','general','emergency'.",
            "target_id":   "Specific ID if mentioned. Omit otherwise.",
            "priority":    "One of: 'high','medium','low'.",
            "status":      "Defaults to 'active'.",
        },
    },
}





# ── Helpers ───────────────────────────────────────────────────────────────────
def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in _PLACEHOLDER_VALUES
    return False




def validate_tool_params(tool_name: str, arguments: dict) -> Optional[str]:
    spec = REQUIRED_TOOL_PARAMS.get(tool_name)
    if not spec:
        return None
    for param in spec["required"]:
        if _is_blank(arguments.get(param)):
            question = spec["ask_if_missing"].get(
                param, f"What is the {param.replace('_', ' ')}?"
            )
            logger.info(f"[PARAM VALIDATION] '{tool_name}' missing '{param}' → {question}")
            return question
    return None


