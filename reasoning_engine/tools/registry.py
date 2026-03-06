# app/tools/registry.py

from reasoning_engine.tools.builder import build_tool_description
from reasoning_engine.tools.functions import (
    get_class_timetable,
    get_exam_timetable,
    get_exam_result,
    get_notice,
    send_message,
    get_messages,
    get_current_time,
    get_current_date,
    searxng_search)



RAG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_class_timetable",
            "description": build_tool_description(
                "get_class_timetable",
                "Get weekly class timetable/periods for a class. "
                "Use for: 'Timetable for Class 6?', 'When is Maths on Monday?'. "
                "OMIT optional parameters if unknown — never pass null or placeholders."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {"type": "string", "description": "Class number e.g. '6', '7', '10'"},
                    "subject":    {"type": "string", "description": "Subject e.g. 'English' — omit if unknown"},
                    "day":        {"type": "string", "description": "Day e.g. 'Monday' — omit if unknown"},
                },
                "required": ["class_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exam_result",
            "description": build_tool_description(
                "get_exam_result",
                "Get a student's exam result. "
                "BOTH class_name AND roll_number are MANDATORY. "
                "DO NOT call this function if either is missing — ask the user first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name":  {"type": "string", "description": "Class number — REQUIRED"},
                    "roll_number": {"type": "string", "description": "Student roll number — REQUIRED"},
                },
                "required": ["class_name", "roll_number"],
            },
        },
    },

    {
        "name": "get_exam_timetable",
        "description": build_tool_description(
            "get_exam_timetable",
            "Get exam schedule/dates for a class. "
            "Use for: 'When is Maths exam?', 'Exam schedule for Class 10?'. "
            "IMPORTANT: class_name is REQUIRED. "
            "If the user does NOT mention a class number, do NOT guess or assume. "
            "Instead ask: 'Which class are you in?' before calling this tool. "
            "OMIT subject if not mentioned — never pass null."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "class_name": {
                    "type": "string",
                    "description": "Class number e.g. '6', '8', '10'. ONLY set if user explicitly mentions it. Never guess."
                },
                "subject": {
                    "type": "string",
                    "description": "Subject name — omit if not mentioned"
                },
            },
            "required": ["class_name"],
        },
    },

{
    "type": "function",
    "function": {
        "name": "get_notice",
        "description": build_tool_description(
            "get_notice",
            "ALWAYS use this tool when the user asks about ANYTHING related to school "
            "information, announcements, schedules, or updates — including sports, events, "
            "exams, holidays, fees, emergencies, or any school activity. "
            "Do NOT answer from memory. Call this tool first, even if unsure. "
            "Examples: 'Any sports events?', 'Is there a holiday?', 'Any exam notices?', "
            "'What events are happening this month?', 'Any announcements?'. "
            "All parameters are optional — omit what is not clearly mentioned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Only set if user clearly mentions a category. "
                        "Map user words to: "
                        "'event' for sports/games/competitions/activities/tournaments, "
                        "'exam' for tests/exams/assessments, "
                        "'holiday' for holidays/breaks/vacations, "
                        "'fee' for fees/payments/dues, "
                        "'emergency' for urgent/emergency/alert, "
                        "'general' for everything else. "
                        "If unsure, omit this field entirely."
                    ),
                    "enum": ["exam", "event", "holiday", "fee", "general", "emergency"]
                },
                "priority": {
                    "type": "string",
                    "description": "Only set if user mentions urgency or priority level.",
                    "enum": ["high", "medium", "low"]
                },
                "status": {
                    "type": "string",
                    "description": "Notice status — always default to 'active' unless user asks for old/archived notices.",
                    "enum": ["active", "inactive", "archived"],
                    "default": "active"
                },
            },
            "required": [],
        },
    },
}

]




ALL_TOOLS = [

    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "ONLY use when the user explicitly asks what time it is RIGHT NOW. "
                "Example triggers: 'what time is it', 'current time', 'tell me the time'. "
                "DO NOT call this for dates, results, reminders, or general questions."
            ),
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": (
                "ONLY use when the user explicitly asks what today's date is. "
                "Example triggers: 'what is today's date', 'what day is it today'. "
                "DO NOT call this for time, results, or anything else."
            ),
            "parameters": {"type": "object", "properties": {}}
        }
    },

    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": (
                "Use ONLY when the user explicitly says 'send a message', 'text someone', or 'message [name]'. "
                "DO NOT call this when the user just mentions a name or introduces themselves."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Who to send the message to"},
                    "content": {"type": "string", "description": "The message content"}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_messages",
            "description": "Use when the user asks to see or read their messages.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "searxng_search",
            "description": (
                "Use when the user asks about current news, recent events, live information, "
                "or anything that requires up-to-date data from the internet. "
                "Example triggers: 'latest news', 'what happened today', 'search for...', "
                "'who won the match', 'current weather'. "
                "DO NOT use for general knowledge you already know."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "num_results": {"type": "integer", "default": 2}
                },
                "required": ["query"]
            }
        }
    },
] + RAG_TOOLS






FUNCTION_MAP = {
    "send_message":    send_message,
    "get_messages":    get_messages,
    "get_current_time": get_current_time,
    "get_current_date": get_current_date,
    "searxng_search":  searxng_search,
    "get_class_timetable":get_class_timetable,
    "get_exam_timetable":get_exam_timetable,
    "get_exam_result" :get_exam_result,
    "get_notice" :get_notice
}



