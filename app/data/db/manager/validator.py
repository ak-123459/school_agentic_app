"""
STEP 4 — VALIDATOR (Smart Upsert)
==================================
NEVER rejects due to missing records.
DB empty (first upload) OR DB full → always works.
"""

import uuid, re
from app.data.db.mysql_db.connection import get_connection
from datetime import datetime


class Validator:

    def validate(self, extracted: dict, doc_type: str) -> dict:
        data = extracted.get("data", {})
        fn   = {
            "exam_result":    self._exam_result,
            "exam_timetable": self._exam_timetable,
            "notice":         self._notice,
            "attendance":     self._attendance,
            "fee":            self._fee,
            "event":          self._event,
            "timetable":      self._timetable,
            "library":        self._library,
            "student":        self._student,
            "teacher":        self._teacher,
        }.get(doc_type)

        if not fn:
            return {"valid": False, "errors": [f"Unknown doc_type: {doc_type}"]}
        return fn(data)

    # ── 1. Exam Result → exam_results + report_cards ───
    def _exam_result(self, data):
        warnings, errors = [], []
        if not data.get("students"):
            return {"valid": False, "errors": ["No students in document"]}

        conn = get_connection()
        enriched = []

        for s in data["students"]:
            name = (s.get("name") or "").strip()
            if not name: continue

            self._upsert_class(conn, data.get("class_name"), data.get("grade"))

            db_student, s_status = self._upsert_student(
                conn,
                s,
                grade=data.get("grade"),
                class_name=data.get("class_name")
            )

            if s_status == "created":
                warnings.append(f"New student created: '{name}'")

            enriched_subjects = []
            for subj in s.get("subjects", []):
                sname = (subj.get("name") or "").strip()
                if not sname: continue

                db_subj, sub_status = self._upsert_subject(
                    conn, sname, data.get("grade"))
                if sub_status == "created":
                    warnings.append(f"New subject created: '{sname}'")

                score, max_score = subj.get("score"), subj.get("max_score")
                if score is None or max_score is None:
                    warnings.append(f"Missing score for {name}/{sname}"); continue
                if score > max_score:
                    errors.append(f"Score {score} > max {max_score}"); continue

                enriched_subjects.append({**subj, "subject_id": db_subj["id"]})

            enriched.append({
                **s,
                "student_id": db_student["id"],
                "subjects":   enriched_subjects
            })

        exam_info = self._upsert_exam(conn, data)
        conn.commit(); conn.close()

        return {
            "valid":    len(errors) == 0 and len(enriched) > 0,
            "data":     {**data, "students": enriched, "exam_info": exam_info},
            "warnings": warnings, "errors": errors
        }

    # ── 2. Exam Timetable → exam_timetable ─────────────
    def _exam_timetable(self, data):
        warnings, errors = [], []
        if not data.get("slots"):
            return {"valid": False, "errors": ["No slots found"]}

        conn = get_connection()

        # ✅ Only upsert class if class_name is actually present
        if data.get("class_name"):
            self._upsert_class(conn, data.get("class_name"), data.get("grade"))

        enriched = []

        for slot in data["slots"]:
            if not slot.get("exam_date"):
                warnings.append(f"No date for {slot.get('subject')} — skip"); continue
            if not slot.get("start_time") or not slot.get("end_time"):
                warnings.append(f"No time for {slot.get('subject')} — skip"); continue
            if not self._vtime(slot["start_time"]) or not self._vtime(slot["end_time"]):
                errors.append(f"Bad time format: {slot.get('subject')}"); continue

            # ── Normalize before comparing (fixes "01:00" vs "11:30") ──
            start_norm = self._normalize_time(slot["start_time"])
            end_norm   = self._normalize_time(slot["end_time"])

            if end_norm <= start_norm:
                errors.append(f"end_time <= start_time: {slot.get('subject')}"); continue

            db_subj, status = self._upsert_subject(conn, slot["subject"], data.get("grade"))
            if status == "created":
                warnings.append(f"New subject created: '{slot['subject']}'")

            enriched.append({
                **slot,
                "start_time": start_norm,   # ← store normalized 24h time
                "end_time":   end_norm,      # ← store normalized 24h time
                "subject_id": db_subj["id"]
            })

        exam_info = self._upsert_exam(conn, data)
        conn.commit(); conn.close()

        return {
            "valid":    len(errors) == 0 and len(enriched) > 0,
            "data":     {**data, "slots": enriched, "exam_info": exam_info},
            "warnings": warnings, "errors": errors
        }

    # ── 3. Notice → notices ─────────────────────────────
    def _notice(self, data):
        warnings, errors = [], []
        if not data.get("title"):
            errors.append("Notice title is required")

        valid_cats = ["exam","event","holiday","fee","general",
                      "emergency","transport","library"]
        if data.get("category") not in valid_cats:
            data["category"] = "general"
            warnings.append("Unknown category → 'general'")

        if data.get("target_type") not in ["all","grade","class","student","teacher"]:
            data["target_type"] = "all"
            warnings.append("Unknown target_type → 'all'")

        if data.get("expires_at") and not self._vdate(data["expires_at"]):
            data["expires_at"] = None
            warnings.append("Invalid expires_at → null")

        return {"valid": len(errors)==0, "data": data,
                "warnings": warnings, "errors": errors}

    # ── 4. Attendance → attendance ──────────────────────
    def _attendance(self, data):
        warnings, errors = [], []
        if not data.get("date"):
            return {"valid": False, "errors": ["Date is required"]}
        if not self._vdate(data["date"]):
            return {"valid": False, "errors": [f"Bad date: {data['date']}"]}

        conn = get_connection()
        # With:
        if data.get("class_name"):
            self._upsert_class(conn, data.get("class_name"), data.get("grade"))

        enriched = []

        for r in data.get("records", []):
            name = (r.get("student_name") or "").strip()
            if not name: continue

            db_student, status = self._upsert_student(
                conn, name, data.get("grade"), data.get("class_name"))
            if status == "created":
                warnings.append(f"New student created: '{name}'")

            s = (r.get("status") or "absent").lower()
            if s not in ["present","absent","late","excused"]:
                s = "absent"; warnings.append(f"Bad status for {name} → 'absent'")

            enriched.append({
                **r,
                "student_id": db_student["id"],
                "class_id":   db_student.get("class_id") or data.get("class_name"),
                "status":     s
            })

        conn.commit(); conn.close()
        return {
            "valid":    len(errors)==0 and len(enriched)>0,
            "data":     {**data, "records": enriched},
            "warnings": warnings, "errors": errors
        }

    # ── 5. Fee → fees ───────────────────────────────────
    def _fee(self, data):
        warnings, errors = [], []
        name = (data.get("student_name") or "").strip()
        if not name:
            return {"valid": False, "errors": ["Student name required"]}

        conn = get_connection()
        db_student, status = self._upsert_student(
            conn, name, class_name=data.get("class_name"))
        if status == "created":
            warnings.append(f"New student created: '{name}'")

        for fee in data.get("fees", []):
            if not fee.get("amount") or fee["amount"] <= 0:
                errors.append(f"Invalid amount: {fee.get('amount')}")

        conn.commit(); conn.close()
        return {
            "valid":    len(errors)==0,
            "data":     {**data, "student_id": db_student["id"]},
            "warnings": warnings, "errors": errors
        }

    # ── 6. Event → events ───────────────────────────────
    def _event(self, data):
        warnings, errors = [], []
        if not data.get("title"):
            errors.append("Event title is required")
        if not data.get("event_date"):
            errors.append("Event date is required")
        elif not self._vdate(data["event_date"]):
            errors.append(f"Bad event_date: {data['event_date']}")

        valid_types = ["holiday","sports","exam","trip","ceremony"]
        if data.get("type") not in valid_types:
            data["type"] = "general"; warnings.append("Unknown type → 'general'")

        return {"valid": len(errors)==0, "data": data,
                "warnings": warnings, "errors": errors}

    # ── 7. Timetable → timetable (class schedule) ───────
    def _timetable(self, data):
        warnings, errors = [], []
        if not data.get("slots"):
            return {"valid": False, "errors": ["No timetable slots found"]}

        conn = get_connection()

        # With:
        if data.get("class_name"):
            self._upsert_class(conn, data.get("class_name"), data.get("grade"))

        enriched = []

        valid_days = ["monday","tuesday","wednesday","thursday",
                      "friday","saturday","sunday"]

        for slot in data["slots"]:
            day = (slot.get("day") or "").lower()
            if day not in valid_days:
                warnings.append(f"Invalid day '{slot.get('day')}' — skip"); continue
            if not slot.get("start_time") or not slot.get("end_time"):
                warnings.append(f"No time for {slot.get('subject')} — skip"); continue

            db_subj, s_status = self._upsert_subject(conn, slot["subject"], data.get("grade"))
            if s_status == "created":
                warnings.append(f"New subject created: '{slot['subject']}'")

            teacher_id = None
            if slot.get("teacher_name"):
                db_teacher, t_status = self._upsert_teacher(conn, slot["teacher_name"])
                if t_status == "created":
                    warnings.append(f"New teacher created: '{slot['teacher_name']}'")
                teacher_id = db_teacher["id"]

            enriched.append({
                **slot,
                "subject_id":  db_subj["id"],
                "teacher_id":  teacher_id,
                "class_id":    data.get("class_name")
            })

        conn.commit(); conn.close()
        return {
            "valid":    len(errors)==0 and len(enriched)>0,
            "data":     {**data, "slots": enriched},
            "warnings": warnings, "errors": errors
        }

    # ── 8. Library → library_books + library_loans ──────
    def _library(self, data):
        warnings, errors = [], []

        conn = get_connection()
        enriched_loans = []

        for loan in data.get("loans", []):
            db_book, b_status = self._upsert_book(conn, loan["book_title"])
            if b_status == "created":
                warnings.append(f"New book created: '{loan['book_title']}'")

            db_student, s_status = self._upsert_student(conn, loan["student_name"])
            if s_status == "created":
                warnings.append(f"New student created: '{loan['student_name']}'")

            if not loan.get("due_date"):
                warnings.append(f"No due_date for {loan['student_name']} — skip"); continue

            enriched_loans.append({
                **loan,
                "book_id":    db_book["id"],
                "student_id": db_student["id"]
            })

        conn.commit(); conn.close()
        return {
            "valid":    True,
            "data":     {**data, "loans": enriched_loans},
            "warnings": warnings, "errors": errors
        }

    # ── 9. Student → students ───────────────────────────
    def _student(self, data):
        warnings, errors = [], []
        if not data.get("students"):
            return {"valid": False, "errors": ["No students found"]}

        conn = get_connection()
        enriched = []

        for s in data["students"]:
            name = (s.get("name") or "").strip()
            if not name: continue

            if s.get("class_name"):
                self._upsert_class(conn, s["class_name"], s.get("grade"))

            db_student, status = self._upsert_student(conn, name, s.get("grade"), s.get("class_name"))

            if status == "found":
                warnings.append(f"Student '{name}' already exists — will update")

            enriched.append({**s, "id": db_student["id"]})

        conn.commit(); conn.close()
        return {
            "valid":    len(enriched)>0,
            "data":     {**data, "students": enriched},
            "warnings": warnings, "errors": errors
        }

    # ── 10. Teacher → teachers ──────────────────────────
    def _teacher(self, data):
        warnings, errors = [], []
        if not data.get("teachers"):
            return {"valid": False, "errors": ["No teachers found"]}

        conn = get_connection()
        enriched = []

        for t in data["teachers"]:
            name = (t.get("name") or "").strip()
            if not name: continue

            db_teacher, status = self._upsert_teacher(conn, name)
            if status == "found":
                warnings.append(f"Teacher '{name}' already exists — will update")

            enriched.append({**t, "id": db_teacher["id"]})

        conn.commit(); conn.close()
        return {
            "valid":    len(enriched)>0,
            "data":     {**data, "teachers": enriched},
            "warnings": warnings, "errors": errors
        }

    # ══════════════════════════════════════════════════
    #  UPSERT HELPERS — Find or Create
    # ══════════════════════════════════════════════════

    def _upsert_student(self, conn, student_data, grade=None, class_name=None):
        student_id = student_data.get("student_id")
        name = (student_data.get("name") or "").strip()

        if not student_id:
            return None, "missing student_id"

        row = conn.execute(
            "SELECT * FROM students WHERE id=?", (student_id,)
        ).fetchone()

        if row:
            return dict(row), "found"

        if not name:
            return None, "missing_name"

        conn.execute("""
            INSERT INTO students (id, name, grade, class_id, status, enrolled_date)
            VALUES (?, ?, ?, ?, 'active', DATE('now'))
        """, (student_id, name, grade, class_name))

        new_row = conn.execute(
            "SELECT * FROM students WHERE id=?", (student_id,)
        ).fetchone()

        return dict(new_row), "created"

    def _upsert_subject(self, conn, name, grade=None):
        nl = name.strip().lower()
        q  = "SELECT * FROM subjects WHERE LOWER(name)=?"
        p  = [nl]
        if grade: q += " AND grade=?"; p.append(str(grade))
        row = conn.execute(q, p).fetchone()
        if row: return dict(row), "found"

        sid = f"SUB_{uuid.uuid4().hex[:8].upper()}"
        conn.execute("INSERT INTO subjects (id,name,grade) VALUES (?,?,?)",
                     (sid, name.strip(), grade))
        return {"id": sid, "name": name.strip()}, "created"

    def _upsert_class(self, conn, class_name, grade):
        if not class_name: return None, "skipped"
        row = conn.execute(
            "SELECT * FROM classes WHERE name=?", (class_name,)).fetchone()
        if row: return dict(row), "found"

        yr = f"{datetime.now().year}-{datetime.now().year+1}"
        conn.execute("""
            INSERT INTO classes (id,name,grade,academic_year)
            VALUES (?,?,?,?)
        """, (class_name, class_name, grade, yr))
        return {"id": class_name, "name": class_name}, "created"

    def _upsert_exam(self, conn, data):
        exam_name = data.get("exam_name") or \
                    f"{(data.get('exam_type') or 'Exam').title()} {datetime.now().year}"
        row = conn.execute(
            "SELECT id FROM exams WHERE name=?", (exam_name,)).fetchone()
        if row: return {"exists": True, "exam_id": row["id"]}

        eid = f"EXAM_{uuid.uuid4().hex[:8].upper()}"
        conn.execute("""
            INSERT INTO exams (id,name,type,grade,exam_date,academic_year)
            VALUES (?,?,?,?,?,?)
        """, (eid, exam_name, data.get("exam_type","general"),
              data.get("grade"), data.get("exam_date"),
              data.get("academic_year", f"{datetime.now().year}-{datetime.now().year+1}")))
        return {"exists": False, "exam_id": eid, "name": exam_name}

    def _upsert_teacher(self, conn, name):
        nl  = name.strip().lower()
        row = conn.execute(
            "SELECT * FROM teachers WHERE LOWER(name)=?", (nl,)).fetchone()
        if row: return dict(row), "found"

        tid = f"TCH_{uuid.uuid4().hex[:8].upper()}"
        conn.execute("""
            INSERT INTO teachers (id,name,status)
            VALUES (?,?, 'active')
        """, (tid, name.strip()))
        return {"id": tid, "name": name.strip()}, "created"

    def _upsert_book(self, conn, title):
        tl  = title.strip().lower()
        row = conn.execute(
            "SELECT * FROM library_books WHERE LOWER(title)=?", (tl,)).fetchone()
        if row: return dict(row), "found"

        bid = f"BK_{uuid.uuid4().hex[:8].upper()}"
        conn.execute("""
            INSERT INTO library_books (id,title,quantity,available)
            VALUES (?,?,1,1)
        """, (bid, title.strip()))
        return {"id": bid, "title": title.strip()}, "created"

    # ══════════════════════════════════════════════════
    #  TIME HELPERS
    # ══════════════════════════════════════════════════

    def _normalize_time(self, t: str) -> str:
        """
        Converts ambiguous short times to 24-hour HH:MM.

        School day rule: hours 01–08 are treated as PM (13:00–20:00).
        Hours 09–23 are kept as-is (already unambiguous in school context).

        Why: LLMs extract "11:30 - 01:00 PM" from docx as "01:00",
             dropping the PM. This restores the correct 24h value.

        Examples:
          "01:00"  →  "13:00"   ✅ 1 PM end of Class 5 exam
          "09:00"  →  "09:00"   ✅ 9 AM start unchanged
          "11:30"  →  "11:30"   ✅ unchanged
          "10:15"  →  "10:15"   ✅ unchanged
        """
        h, m = map(int, t.strip().split(":"))
        if 1 <= h <= 8:
            h += 12          # 01:00 → 13:00, 08:30 → 20:30
        return f"{h:02d}:{m:02d}"

    def _vdate(self, d):
        try: datetime.strptime(d, "%Y-%m-%d"); return True
        except: return False

    def _vtime(self, t):
        """Validates HH:MM format (accepts both 12h and 24h values)."""
        try: datetime.strptime(t.strip(), "%H:%M"); return True
        except: return False