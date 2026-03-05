"""
STEP 5 — INSERTER
Insert validated data into the correct table(s).
Zero LLM calls. Rollback on any error.
"""
import json
from app.data.db.mysql_db.connection import get_connection
from datetime import datetime


class Inserter:

    def insert(self, validated: dict, doc_type: str, created_by: str = None) -> dict:

        if not validated.get("valid"):
            return {"success": False, "reason": "Validation failed"}

        data = validated["data"]
        fn = {
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
            return {"success": False, "reason": f"No inserter for {doc_type}"}

        return fn(data, created_by)

    # ── 1. exam_results + report_cards ─────────────────
    def _exam_result(self, data, created_by):
        conn = get_connection()
        inserted = 0
        try:
            exam_id = data["exam_info"]["exam_id"]

            for student in data["students"]:
                sid = student["student_id"]

                for subj in student["subjects"]:
                    score     = subj["score"]
                    max_score = subj["max_score"]
                    pct       = round((score / max_score) * 100, 2) if max_score else 0
                    pf        = "pass" if pct >= 50 else "fail"

                    conn.execute("""
                        INSERT OR REPLACE INTO exam_results
                        (student_id,exam_id,subject_id,score,max_score,
                         percentage,grade_letter,pass_fail,remarks)
                        VALUES (?,?,?,?,?,?,?,?,?)
                    """, (sid, exam_id, subj["subject_id"],
                          score, max_score, pct,
                          self._grade(pct), pf, subj.get("remarks")))
                    inserted += 1

                # Insert report card (aggregated)
                if student.get("total_score") is not None:
                    pct = student.get("percentage") or \
                          round((student["total_score"] / student["total_max"]) * 100, 2) \
                          if student.get("total_max") else 0

                    conn.execute("""
                        INSERT OR REPLACE INTO report_cards
                        (student_id,exam_id,total_score,total_max,percentage,
                         rank,grade_letter,pass_fail,teacher_remarks,generated_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                    """, (sid, exam_id,
                          student["total_score"], student.get("total_max"),
                          pct, student.get("rank"),
                          self._grade(pct),
                          student.get("pass_fail","pass" if pct>=50 else "fail"),
                          student.get("remarks"),
                          datetime.now().isoformat()))

            conn.commit()
            return {"success": True, "inserted": inserted,
                    "table": "exam_results + report_cards"}
        except Exception as e:
            conn.rollback()
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    # ── 2. exam_timetable ───────────────────────────────
    def _exam_timetable(self, data, created_by):
        conn = get_connection()
        inserted = 0
        try:
            exam_id  = data["exam_info"]["exam_id"]
            class_id = data.get("class_name")
            grade    = data.get("grade")

            for slot in data["slots"]:
                conn.execute("""
                    INSERT INTO exam_timetable
                    (exam_id,subject_id,class_id,grade,
                     exam_date,start_time,end_time,duration_min,
                     room,invigilator,status,notes)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (exam_id, slot["subject_id"], class_id, grade,
                      slot["exam_date"], slot["start_time"], slot["end_time"],
                      slot.get("duration_min"), slot.get("room"),
                      slot.get("invigilator"), "scheduled", slot.get("notes")))
                inserted += 1

            conn.commit()
            return {"success": True, "inserted": inserted, "table": "exam_timetable"}
        except Exception as e:
            conn.rollback()
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    # ── 3. notices ──────────────────────────────────────
    def _notice(self, data, created_by):
        conn = get_connection()
        try:
            conn.execute("""
                INSERT INTO notices
                (title,content,category,target_type,target_id,
                 created_at,expires_at,status,priority)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (data["title"], data.get("content"),
                  data.get("category","general"),
                  data.get("target_type","all"),
                  data.get("target_id"),
                  datetime.now().isoformat(),
                  data.get("expires_at"), "active",
                  data.get("priority","medium")))
            conn.commit()
            return {"success": True, "inserted": 1, "table": "notices"}
        except Exception as e:
            conn.rollback()
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    # ── 4. attendance ───────────────────────────────────
    def _attendance(self, data, created_by):
        conn = get_connection()
        inserted = 0
        try:
            for r in data["records"]:
                conn.execute("""
                    INSERT INTO attendance
                    (student_id,class_id,date,status,remarks)
                    VALUES (?,?,?,?,?)
                """, (r["student_id"], r["class_id"],
                      data["date"], r["status"], r.get("remarks")))
                inserted += 1
            conn.commit()
            return {"success": True, "inserted": inserted, "table": "attendance"}
        except Exception as e:
            conn.rollback()
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    # ── 5. fees ─────────────────────────────────────────
    def _fee(self, data, created_by):
        conn = get_connection()
        inserted = 0
        try:
            for fee in data["fees"]:
                conn.execute("""
                    INSERT INTO fees
                    (student_id,amount,fee_type,due_date,paid_date,status)
                    VALUES (?,?,?,?,?,?)
                """, (data["student_id"], fee["amount"],
                      fee.get("fee_type","tuition"),
                      fee.get("due_date"), fee.get("paid_date"),
                      fee.get("status","unpaid")))
                inserted += 1
            conn.commit()
            return {"success": True, "inserted": inserted, "table": "fees"}
        except Exception as e:
            conn.rollback()
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    # ── 6. events ───────────────────────────────────────
    def _event(self, data, created_by):
        conn = get_connection()
        try:
            conn.execute("""
                INSERT INTO events
                (title,description,event_date,type)
                VALUES (?,?,?,?)
            """, (data["title"], data.get("description"),
                  data["event_date"], data.get("type","general")))
            conn.commit()
            return {"success": True, "inserted": 1, "table": "events"}
        except Exception as e:
            conn.rollback()
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    # ── 7. timetable (class schedule) ───────────────────
    def _timetable(self, data, created_by):
        conn = get_connection()
        inserted = 0
        try:
            for slot in data["slots"]:
                conn.execute("""
                    INSERT INTO timetable
                    (class_id,subject_id,teacher_id,day,start_time,end_time,room)
                    VALUES (?,?,?,?,?,?,?)
                """, (slot["class_id"], slot["subject_id"],
                      slot.get("teacher_id"),
                      slot["day"].capitalize(),
                      slot["start_time"], slot["end_time"],
                      slot.get("room")))
                inserted += 1
            conn.commit()
            return {"success": True, "inserted": inserted, "table": "timetable"}
        except Exception as e:
            conn.rollback()
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    # ── 8. library_books + library_loans ────────────────
    def _library(self, data, created_by):
        conn = get_connection()
        inserted = 0
        try:
            # Insert standalone books if any
            for book in data.get("books", []):
                conn.execute("""
                    INSERT OR IGNORE INTO library_books
                    (id,title,author,subject,grade,quantity,available)
                    VALUES (?,?,?,?,?,?,?)
                """, (f"BK_{book['title'][:8].upper().replace(' ','_')}",
                      book["title"], book.get("author"),
                      book.get("subject"), book.get("grade"),
                      book.get("quantity",1), book.get("quantity",1)))
                inserted += 1

            # Insert loans
            for loan in data.get("loans", []):
                conn.execute("""
                    INSERT INTO library_loans
                    (book_id,student_id,issued_date,due_date,return_date,status)
                    VALUES (?,?,?,?,?,?)
                """, (loan["book_id"], loan["student_id"],
                      loan.get("issued_date", datetime.now().strftime("%Y-%m-%d")),
                      loan["due_date"],
                      loan.get("return_date"),
                      loan.get("status","issued")))
                inserted += 1

            conn.commit()
            return {"success": True, "inserted": inserted,
                    "table": "library_books + library_loans"}

        except Exception as e:

            conn.rollback()
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    # ── 9. students ─────────────────────────────────────
    def _student(self, data, created_by):
        conn = get_connection()
        inserted = 0
        try:
            for s in data["students"]:
                conn.execute("""
                    INSERT OR REPLACE INTO students
                    (id,name,dob,gender,grade,class_id,
                     parent_name,parent_phone,parent_email,
                     address,enrolled_date,status)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (s["id"], s["name"],
                      s.get("dob"), s.get("gender"),
                      s.get("grade"), s.get("class_name"),
                      s.get("parent_name"), s.get("parent_phone"),
                      s.get("parent_email"), s.get("address"),
                      s.get("enrolled_date", datetime.now().strftime("%Y-%m-%d")),
                      "active"))
                inserted += 1
            conn.commit()
            return {"success": True, "inserted": inserted, "table": "students"}
        except Exception as e:
            conn.rollback()
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    # ── 10. teachers ────────────────────────────────────
    def _teacher(self, data, created_by):
        conn = get_connection()
        inserted = 0
        try:
            for t in data["teachers"]:
                conn.execute("""
                    INSERT OR REPLACE INTO teachers
                    (id,name,email,phone,
                     subjects,class_id,joined_date,status)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (t["id"], t["name"],
                      t.get("email"), t.get("phone"),
                      json.dumps(t.get("subjects", [])),
                      t.get("class_name"),
                      t.get("joined_date"),
                      t.get("status","active")))
                inserted += 1
            conn.commit()
            return {"success": True, "inserted": inserted, "table": "teachers"}
        except Exception as e:
            conn.rollback()
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    def _grade(self, pct):
        if pct >= 90: return "A+"
        if pct >= 80: return "A"
        if pct >= 70: return "B"
        if pct >= 60: return "C"
        if pct >= 50: return "D"
        return "F"