"""
school_rag_tools.py
===================
Three RAG-backed tool functions that plug directly into the existing
voice assistant's tools list and function_map.

CHANGES:
  • get_exam_result now requires class_name AND roll_number (both mandatory)
  • Tool description updated to enforce param collection before calling
  • Added REQUIRED_TOOL_PARAMS registry used by assistant.py for validation
"""

from collections import defaultdict
import asyncio
import logging
from typing import Optional
import sqlite3
from app.config.config import DB_PATH
logger = logging.getLogger(__name__)
from app.utils.tools_input.preprocess import normalize_class_name




# ── Module-level singleton — injected at startup ──────────────────────────────
_rag_manager = None


def init_rag_tools(llm, embeddings, base_vector_db_path: str, yaml_path: str):
    global _rag_manager
    from app.rag_pipeline.rag.faiss_loader import SchoolFaissLoader
    from app.rag_pipeline.rag.rag_manager  import SchoolRAGManager

    loader       = SchoolFaissLoader(
        base_path  = base_vector_db_path,
        yaml_path  = yaml_path,
        embeddings = embeddings,
    )
    _rag_manager = SchoolRAGManager(llm=llm, faiss_loader=loader)
    logger.info("[RAG Tools] Initialized successfully.")


def _get_manager():
    if _rag_manager is None:
        raise RuntimeError(
            "RAG tools not initialized. Call init_rag_tools() at startup first."
        )
    return _rag_manager


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)




# ══════════════════════════════════════════════════════════════════════════════
#  REQUIRED PARAMS REGISTRY
#  Used by assistant.py agent_node to validate BEFORE any tool is executed.
#
#  Format per tool:
#    "required"     : params that MUST be present (non-empty)
#    "ask_if_missing": human-readable question to ask when each param is missing
# ══════════════════════════════════════════════════════════════════════════════



REQUIRED_TOOL_PARAMS = {

    "get_class_timetable": {
        "required": ["class_name"],
        "ask_if_missing": {
            "class_name": "Which class do you want the timetable for?",
        },
        "param_instructions": {
            "class_name": "Digit only: '7', '8', '10'. NEVER 'your class', 'my class', or subject names.",
            "subject":    "Full subject name as spoken: 'English', 'Mathematics', 'Science'. Omit if not mentioned.",
            "day":        "Full day name: 'Monday', 'Tuesday' etc. Omit if not mentioned.",
        },
    },

    "get_exam_timetable": {
        "required": ["class_name"],
        "ask_if_missing": {
            "class_name": "Which class do you want the exam timetable for? e.g. 3",
        },
        "param_instructions": {
            "class_name": "Digit only: '7', '8', '10'. Never a subject or placeholder.",
            "subject":    "Subject name if mentioned: 'Mathematics', 'Science'. Omit if unknown.",
        },
    },

    "get_exam_result": {
        "required": ["class_name", "roll_number"],
        "ask_if_missing": {
            "class_name":  "Which class is the student in?",
            "roll_number": "What is your roll number? e.g. STU004",
        },
        "param_instructions": {
            "class_name":  "Digit only: '7', '10'. Never a name or placeholder.",
            "roll_number": "Alphanumeric ID as spoken. Strip dashes and spaces: 'STU-004' → 'STU004', '234 330' → '234330'.",
        },
    },

"get_notice": {
    "required": [],  # No mandatory params — notices can be browsed freely
    "ask_if_missing": {},
    "param_instructions": {
        "category":    "One of: 'exam', 'event', 'holiday', 'fee', 'general', 'emergency'. Omit if not mentioned.",
        "target_type": "One of: 'all', 'grade', 'class', 'student', 'teacher'. Omit if not mentioned.",
        "target_id":   "Specific class/student/teacher ID if mentioned. Omit if not mentioned.",
        "priority":    "One of: 'high', 'medium', 'low'. Omit if not mentioned.",
        "status":      "One of: 'active', 'inactive', 'archived'. Defaults to 'active' if not mentioned.",
    },
},

}






# LLM placeholder values that must be treated as "not provided"
_PLACEHOLDER_VALUES = {
    "unknown", "n/a", "na", "none", "null", "undefined",
    "not provided", "not specified", "not given", "missing",
    "?", "...", "placeholder", "your_roll_number", "your_class",
}


def _is_blank(value) -> bool:
    """True if value is missing, empty, or a known LLM placeholder."""
    if value is None:
        return True
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return not cleaned or cleaned in _PLACEHOLDER_VALUES
    return False


def validate_tool_params(tool_name: str, arguments: dict) -> Optional[str]:
    """
    Check if all required params for tool_name are present in arguments.
    Also rejects LLM placeholder values like "unknown", "n/a", etc.

    Returns:
        None              → all required params present, safe to execute
        str (question)    → a missing/placeholder param found; ask the user this
    """
    spec = REQUIRED_TOOL_PARAMS.get(tool_name)
    if not spec:
        return None  # no validation rule → allow

    for param in spec["required"]:
        value = arguments.get(param)
        if _is_blank(value):
            question = spec["ask_if_missing"].get(
                param, f"What is the {param.replace('_', ' ')}?"
            )
            logger.info(f"[PARAM VALIDATION] '{tool_name}' missing '{param}' → asking: {question}")
            return question  # return the FIRST missing param question

    return None  # all present



def build_tool_description(tool_name: str, base_description: str) -> str:
    """
    Appends param-level instructions from REQUIRED_TOOL_PARAMS
    directly into the tool description the LLM reads.

    Call this when building RAG_TOOLS instead of writing instructions manually.
    """
    spec = REQUIRED_TOOL_PARAMS.get(tool_name, {})
    instructions = spec.get("param_instructions", {})
    required     = spec.get("required", [])

    if not instructions:
        return base_description

    lines = [base_description, "PARAMETER RULES:"]

    for param, rule in instructions.items():
        tag = "[REQUIRED]" if param in required else "[optional]"
        lines.append(f"- {param} {tag}: {rule}")

    return " ".join(lines)







def get_exam_timetable(
    class_name: str,
    section: Optional[str] = None,
    subject: Optional[str] = None,
    chat_history: list = None,
) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        # 🔍 DEBUG 1 — what's actually in the DB
        raw = conn.execute("SELECT id, exam_id, class_id, subject_id, exam_date FROM exam_timetable LIMIT 5").fetchall()
        print(f"[DEBUG] exam_timetable rows: {[dict(r) for r in raw]}")

        classes = conn.execute("SELECT id, name, grade FROM classes LIMIT 10").fetchall()
        print(f"[DEBUG] classes: {[dict(r) for r in classes]}")

        # 🔍 DEBUG 2 — what params we're querying with
        print(f"[DEBUG] get_exam_timetable called with: class_name={repr(class_name)}, subject={repr(subject)}")

        conditions = []
        params = []

        if class_name:
            conditions.append("""(
                c.name    LIKE ? OR 
                c.grade   LIKE ? OR
                et.class_id LIKE ?
            )""")
            params.extend([f"%{class_name}%", f"%{class_name}%", f"%{class_name}%"])

        if subject:
            conditions.append("sub.name LIKE ?")
            params.append(f"%{subject}%")

        where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"""
        SELECT
            c.name        AS class_name,
            e.id          AS exam_id,
            e.name        AS exam_name,
            et.exam_date,
            sub.name      AS subject_name,
            et.start_time,
            et.end_time,
            et.duration_min,
            et.room,
            t.name        AS invigilator,
            et.status
        FROM exam_timetable et
        LEFT JOIN exams e      ON et.exam_id    = e.id
        LEFT JOIN subjects sub ON et.subject_id = sub.id
        LEFT JOIN classes c    ON et.class_id   = c.id
        LEFT JOIN teachers t   ON et.invigilator = t.id
        {where_clause}
        ORDER BY et.exam_date ASC, et.start_time ASC
        """

        # 🔍 DEBUG 3 — log the exact SQL and params
        print(f"[DEBUG] SQL: {sql}")
        print(f"[DEBUG] params: {params}")

        rows = conn.execute(sql, params).fetchall()

        # 🔍 DEBUG 4 — what came back
        print(f"[DEBUG] rows returned: {len(rows)}")
        if rows:
            print(f"[DEBUG] first row: {dict(rows[0])}")

        if not rows:
            return {
                "success": False,
                "message": "No timetable found.",
                "data": {}
            }

        # 🔥 GROUP BY EXAM
        grouped = defaultdict(lambda: {
            "exam_name": "",
            "schedule": []
        })

        for row in rows:
            exam_id = row["exam_id"]

            grouped[exam_id]["exam_name"] = row["exam_name"]

            grouped[exam_id]["schedule"].append({
                "subject": row["subject_name"],
                "date": row["exam_date"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "duration_min": row["duration_min"],
                "room": row["room"],
                "invigilator": row["invigilator"],
                "status": row["status"]
            })

        conn.close()

        return {
            "success": True,
            "message": "Exam timetable retrieved successfully.",
            "data": {
                "class": class_name,
                "exams": list(grouped.values())
            }
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "data": {}
        }





def get_exam_result(
    class_name:   str,
    roll_number:  str,
    student_name: Optional[str] = None,
    chat_history: list          = None,
) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        conditions = []
        params     = []

        if roll_number:
            conditions.append("UPPER(TRIM(s.id)) = UPPER(TRIM(?))")
            params.append(str(roll_number).strip())

        if class_name and not roll_number:
            # Only filter by class when roll_number is NOT given
            normalized = normalize_class_name(class_name)
            conditions.append("(s.class_id LIKE ? OR c.name LIKE ? OR c.grade LIKE ?)")
            params.extend([f"%{normalized}%", f"%{normalized}%", f"%{normalized}%"])

        if student_name:
            conditions.append("s.name LIKE ?")
            params.append(f"%{student_name}%")

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else "WHERE 1=1"

        sql = f"""
        SELECT
            s.id           AS roll_number,
            s.name         AS student_name,
            c.name         AS class_name,
            s.class_id     AS class_id,
            e.name         AS exam_name,
            e.exam_date,
            sub.name       AS subject_name,
            er.score,
            er.max_score,
            er.percentage,
            er.grade_letter,
            er.pass_fail
        FROM students s
        LEFT JOIN classes c   ON s.class_id = c.id
        INNER JOIN exam_results er ON s.id = er.student_id
        INNER JOIN exams e         ON er.exam_id = e.id
        INNER JOIN subjects sub    ON er.subject_id = sub.id
        {where_clause}
        ORDER BY e.exam_date DESC, sub.name
        """

        cursor = conn.execute(sql, params)
        rows   = [dict(row) for row in cursor.fetchall()]
        conn.close()

        logger.info(f"[get_exam_result] Found {len(rows)} rows for roll={roll_number}, class={class_name}")

        return {
            "success": True,
            "message": rows,
            "data": {
                "class":       class_name,
                "roll_number": roll_number,
                "answer":      rows,
            },
        }

    except Exception as e:
        logger.error(f"[get_exam_result] Error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Could not retrieve exam result for Class {class_name}, Roll {roll_number}.",
            "data":    {},
        }


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL FUNCTION 3: get_class_timetable
# ══════════════════════════════════════════════════════════════════════════════

def get_class_timetable(
        class_name: str,
        subject: Optional[str] = None,
        day: Optional[str] = None,
        chat_history: list = None
) -> dict:

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # ── Base SQL query with LEFT JOINs ─────────────
        sql = """
            SELECT tt.id, c.name AS class_name, s.name AS subject_name, 
                   t.name AS teacher_name, tt.day, tt.start_time, tt.end_time, tt.room
            FROM timetable tt
            LEFT JOIN classes c ON tt.class_id = c.id
            LEFT JOIN subjects s ON tt.subject_id = s.id
            LEFT JOIN teachers t ON tt.teacher_id = t.id
            WHERE UPPER(c.name) LIKE ?
        """

        params = [normalize_class_name(class_name)]


        if subject:
            # Use LIKE for partial matches
            sql += " AND UPPER(s.name) LIKE ?"
            params.append(f"%{subject.upper()}%")
        if day:
            sql += " AND UPPER(tt.day) LIKE ?"
            params.append(f"%{day.upper()}%")

        sql += " ORDER BY tt.day, tt.start_time"

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        # ── Transform rows to dict ───────────────────
        timetable_data = [
            {
                "timetable_id": row[0],
                "class": row[1],
                "subject": row[2],
                "teacher": row[3],
                "day": row[4],
                "start_time": row[5],
                "end_time": row[6],
                "room": row[7]
            }
            for row in rows
        ]

        cursor.close()
        conn.close()


        logger.info(f"[timetable is ---->] : {timetable_data}")

        if not timetable_data:
            message = "No timetable found for the given filters."
        else:
            message = "Class timetable retrieved successfully."

        return {
            "success": True,
            "message": message,
            "data": {
                "class": class_name,
                "subject": subject,
                "day": day,
                "timetable": timetable_data
            }
        }

    except Exception as e:
        logger.error(f"[get_class_timetable] Error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Could not retrieve timetable for Class {class_name}.",
            "data": {}
        }




def get_notice(
    category:    Optional[str] = None,
    target_type: Optional[str] = None,
    target_id:   Optional[str] = None,
    priority:    Optional[str] = None,
    status:      Optional[str] = "active",
    chat_history: list         = None,
) -> dict:
    """
    Retrieve notices from the DB with optional filters.
    Mirrors the pattern of get_class_timetable / get_exam_timetable.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        sql = """
            SELECT
                n.id,
                n.title,
                n.content,
                n.category,
                n.target_type,
                n.target_id,
                n.created_by,
                t.name       AS created_by_name,
                n.created_at,
                n.expires_at,
                n.status,
                n.priority
            FROM notices n
            LEFT JOIN teachers t ON n.created_by = t.id
            WHERE 1=1
        """
        params = []

        # ── status filter (default = active) ──────────────────────────────
        if status:
            sql += " AND LOWER(n.status) = LOWER(?)"
            params.append(status.strip())

        # ── category filter ───────────────────────────────────────────────
        if category:
            sql += " AND LOWER(n.category) = LOWER(?)"
            params.append(category.strip())

        # ── target_type filter ────────────────────────────────────────────
        if target_type:
            sql += " AND LOWER(n.target_type) = LOWER(?)"
            params.append(target_type.strip())

        # ── target_id filter ──────────────────────────────────────────────
        if target_id:
            sql += " AND (n.target_id = ? OR n.target_type = 'all')"
            params.append(str(target_id).strip())

        # ── priority filter ───────────────────────────────────────────────
        if priority:
            sql += " AND LOWER(n.priority) = LOWER(?)"
            params.append(priority.strip())

        # ── exclude expired notices ───────────────────────────────────────
        sql += """
            AND (
                n.expires_at IS NULL
                OR n.expires_at = ''
                OR DATE(n.expires_at) >= DATE('now')
            )
            """

        sql += " ORDER BY n.priority DESC, n.created_at DESC"

        rows = conn.execute(sql, params).fetchall()
        conn.close()

        logger.info(f"[get_notice] Found {len(rows)} notices | "
                    f"category={category}, target_type={target_type}, "
                    f"target_id={target_id}, priority={priority}, status={status}")

        if not rows:
            return {
                "success": False,
                "message": "No notices found for the given filters.",
                "data": {}
            }

        notices = [
            {
                "id":              row["id"],
                "title":           row["title"],
                "content":         row["content"],
                "category":        row["category"],
                "target_type":     row["target_type"],
                "target_id":       row["target_id"],
                "created_by":      row["created_by_name"] or row["created_by"],
                "created_at":      row["created_at"],
                "expires_at":      row["expires_at"],
                "status":          row["status"],
                "priority":        row["priority"],
            }
            for row in rows
        ]

        return {
            "success": True,
            "message": f"{len(notices)} notice(s) retrieved successfully.",
            "data": {
                "filters": {
                    "category":    category,
                    "target_type": target_type,
                    "target_id":   target_id,
                    "priority":    priority,
                    "status":      status,
                },
                "count":   len(notices),
                "notices": notices,
            }
        }

    except Exception as e:
        logger.error(f"[get_notice] Error: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"Could not retrieve notices: {str(e)}",
            "data": {}
        }





# ══════════════════════════════════════════════════════════════════════════════
#  TOOL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Function map ──────────────────────────────────────────────────────────────
RAG_FUNCTION_MAP = {
    "get_exam_timetable":  get_exam_timetable,
    "get_exam_result":     get_exam_result,
    "get_class_timetable": get_class_timetable,
}