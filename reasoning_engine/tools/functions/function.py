from collections import defaultdict
import asyncio
import logging
from typing import Optional
import sqlite3
logger = logging.getLogger(__name__)
from reasoning_engine.tools_input.preprocess import normalize_class_name,_clean_content
import yaml



with open("configs/database_config.yaml",    "r", encoding="utf-8") as f:
    database_config    = yaml.safe_load(f)

DB_PATH = database_config['SQL_DB_PATH']




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
    category:     Optional[str] = None,
    priority:     Optional[str] = None,
    status:       Optional[str] = "active",
    chat_history: list          = None,
) -> dict:
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
                n.created_at,
                n.expires_at,
                n.status,
                n.priority
            FROM notices n
            WHERE 1=1
        """
        params = []

        # ✅ status filter
        if status:
            sql += " AND LOWER(n.status) = LOWER(?)"
            params.append(status.strip())

        # ✅ category filter (what the notice is about)
        if category:
            sql += " AND LOWER(n.category) = LOWER(?)"
            params.append(category.strip())

        # ✅ priority filter
        if priority:
            sql += " AND LOWER(n.priority) = LOWER(?)"
            params.append(priority.strip())

        # ✅ target_type & target_id removed as search filters
        # they are visibility/permission metadata, not search criteria

        # ✅ auto-expiry check
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
                    f"category={category}, priority={priority}, status={status}")

        if not rows:
            return {
                "success": False,
                "message": "No notices found for the given filters.",
                "data": {}
            }

        notices = [
            {
                "id":          row["id"],
                "title":       row["title"],
                "content":     row["content"],
                "category":    row["category"],
                "target_type": row["target_type"],   # still returned for context
                "target_id":   row["target_id"],     # still returned for context
                "created_at":  row["created_at"],
                "expires_at":  row["expires_at"],
                "status":      row["status"],
                "priority":    row["priority"],
            }
            for row in rows
        ]

        return {
            "success": True,
            "message": f"{len(notices)} notice(s) retrieved successfully.",
            "data": {
                "filters": {
                    "category": category,
                    "priority": priority,
                    "status":   status,
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




from datetime import datetime, timedelta
import re
import requests

reminders = []
messages  = []

# ==============================================================
# REMINDER FUNCTIONS
# ==============================================================

def set_reminder(reminder_text: str, duration_minutes: int = 5) -> dict:
    reminder_time = datetime.now() + timedelta(minutes=duration_minutes)
    reminder = {
        "id":      len(reminders) + 1,
        "text":    reminder_text,
        "time":    reminder_time.strftime("%Y-%m-%d %H:%M"),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "active":  True,
    }
    reminders.append(reminder)
    return {
        "success": True,
        "message": f"Reminder set for {reminder_time.strftime('%I:%M %p')}",
        "data":    reminder,
    }

def get_reminders() -> dict:
    active = [r for r in reminders if r.get("active", True)]
    return {"success": True, "message": f"You have {len(active)} active reminder(s)", "data": active}

def delete_reminder(reminder_id: int) -> dict:
    for reminder in reminders:
        if reminder["id"] == reminder_id:
            reminder["active"] = False
            return {"success": True, "message": f"Reminder {reminder_id} deleted", "data": reminder}
    return {"success": False, "message": f"Reminder {reminder_id} not found"}

# ==============================================================
# MESSAGE FUNCTIONS
# ==============================================================

def send_message(content: str, recipient: str = "default") -> dict:
    message = {
        "id":        len(messages) + 1,
        "recipient": recipient,
        "content":   content,
        "time":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    messages.append(message)
    return {"success": True, "message": f"Message sent to {recipient}", "data": message}

def get_messages() -> dict:
    return {"success": True, "message": f"You have {len(messages)} message(s)", "data": messages}

# ==============================================================
# DATE / TIME
# ==============================================================

def get_current_time() -> dict:
    t = datetime.now().strftime("%I:%M %p")
    return {"success": True, "message": f"The current time is {t}", "data": {"time": t}}

def get_current_date() -> dict:
    d = datetime.now().strftime("%A, %B %d, %Y")
    return {"success": True, "message": f"Today is {d}", "data": {"date": d}}

# # ==============================================================
# # EXAM RESULT
# # ==============================================================
#
# def check_result(roll_number: str) -> dict:
#     if not roll_number or str(roll_number).strip().lower() in (
#         "unknown", "none", "n/a", "null", "not provided", ""
#     ):
#         return {"success": False, "message": "Roll number is required.", "data": {}}
#
#     results_db = {
#         "2026": {"name": "Ali Khan",              "marks": 450, "grade": "A"},
#         "2025": {"name": "Sara Ahmed",            "marks": 380, "grade": "B"},
#         "2024": {"name": "Vivaan Singh",          "marks": 410, "grade": "A"},
#         "2023": {"name": "Sarah Maria Fernandes", "marks": 370, "grade": "B"},
#     }
#
#     rn = str(roll_number).strip()
#     if rn in results_db:
#         r = results_db[rn]
#         return {
#             "success": True,
#             "message": f"{r['name']}, roll {rn}: {r['marks']} marks, Grade {r['grade']}.",
#             "data":    {"roll_number": rn, **r},
#         }
#
#
#     return {
#         "success": False,
#         "message": f"No result found for roll number {rn}.",
#         "data":    {},
#     }

# ==============================================================
# SEARXNG SEARCH
# ==============================================================

def searxng_search(query: str, num_results: int = 3) -> dict:
    """
    Search using local SearXNG.
    Returns a single clean plain-text message ready for TTS.
    No markdown, no URLs, no Unicode symbols.
    """
    try:
        response = requests.get(
            "http://localhost:8080/search",
            params={"q": query, "format": "json", "language": "en"},
            timeout=10,
        )
        raw_results = response.json().get("results", [])

        if not raw_results:
            return {
                "success": False,
                "message": f"I couldn't find anything about {query}.",
                "data":    [],
            }

        # Pick the best result: prefer ones with content over title-only
        best_content = ""
        for r in raw_results[:num_results]:
            content = _clean_content(r.get("content", ""))
            if len(content) > 60:           # has a real summary
                best_content = content
                break

        # Fallback: use title of first result
        if not best_content:
            best_content = _clean_content(raw_results[0].get("title", ""))

        if not best_content:
            return {
                "success": False,
                "message": f"I couldn't find clear information about {query}.",
                "data":    [],
            }

        # Store clean structured data too (for debugging)
        clean_data = []
        for r in raw_results[:num_results]:
            clean_data.append({
                "title":   _clean_content(r.get("title", "")),
                "summary": _clean_content(r.get("content", "")),
            })

        return {
            "success": True,
            "message": best_content,   # ← plain text, TTS-ready
            "data":    clean_data,
        }

    except Exception as e:
        return {"success": False, "message": f"Search failed: {str(e)}", "data": []}


