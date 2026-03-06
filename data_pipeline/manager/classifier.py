"""
STEP 2 — CLASSIFIER (Robust Two-Stage)
=======================================
Stage 1 → Structural fingerprint (0ms, no LLM, catches ~80% of docs)
Stage 2 → Small LLM fallback ONLY for genuinely ambiguous docs

Your tables mapped:
  exam_result    → exam_results + report_cards
  exam_timetable → exam_timetable
  notice         → notices
  attendance     → attendance
  fee            → fees
  event          → events
  timetable      → timetable  (regular class schedule)
  library        → library_books / library_loans
  student        → students   (registration doc)
  teacher        → teachers   (registration doc)

Why two-stage?
  - Rule/keyword scoring breaks on edge cases (timetable vs exam_timetable).
  - Document SHAPE is unique — a weekly grid with days+periods
    cannot be an exam timetable regardless of keywords.
  - LLM only fires for the ~20% that are genuinely ambiguous.
"""

import re
import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  KNOWN CATEGORIES
# ─────────────────────────────────────────────────────────────
CATEGORIES = [

    "exam_result",
    "exam_timetable",
    "timetable",
    "attendance",
    "fee",
    "notice",
    "event",
    "library",
    "student",
    "teacher",

]


# load_prompt.py
def load_classifier_system_prompt(file_path="prompts/classifier_system_prompt"):
    with open(file_path, "r", encoding="utf-8") as f:
        classifier_prompt = f.read()
    return classifier_prompt

#  load classifier_prompt
classifier_prompt = load_classifier_system_prompt()


# ─────────────────────────────────────────────────────────────
#  CLASSIFIER
# ─────────────────────────────────────────────────────────────
class Classifier:

    async def classify(self, text: str, llm, use_llm_fallback: bool = True) -> dict:
        """
        Main entry point.
        Returns: {doc_type, confidence, method, ...}
        """
        # ── Stage 1: structural fingerprint (instant) ─────────
        result = self._structural_classify(text)
        if result:
            logger.info(f"   ✅ {result['doc_type']} "
                        f"(confidence={result['confidence']:.2f}, method=structural)")
            return result

        # ── Stage 2: small LLM for ambiguous docs ─────────────
        if use_llm_fallback:
            result = await self._llm_classify(text, llm)
            logger.info(f"   ✅ {result['doc_type']} "
                        f"(confidence={result.get('confidence', 0):.2f}, method=llm)")
            return result

        return {"doc_type": "unknown", "confidence": 0.0, "method": "failed"}

    # ══════════════════════════════════════════════════════════
    #  STAGE 1 — STRUCTURAL FINGERPRINT
    # ══════════════════════════════════════════════════════════

    def _structural_classify(self, text: str) -> dict | None:
        """
        Analyzes document SHAPE — not keywords.
        Returns a classification dict or None (→ go to LLM).
        """
        fp = self._fingerprint(text)
        decision = self._decision_tree(fp)

        if decision:
            return {
                "doc_type":    decision["doc_type"],
                "confidence":  decision["confidence"],
                "method":      "structural",
                "fingerprint": fp,
            }
        return None

    def _fingerprint(self, text: str) -> dict:
        """
        Extract objective signals from the document.
        Counts matter — single hits are noise, multiple hits are signal.
        """
        t = text.lower()

        # ── Day-of-week signals ───────────────────────────────
        day_matches = re.findall(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", t)
        unique_days = len(set(day_matches))      # how many distinct days appear
        total_days  = len(day_matches)           # total occurrences

        # ── Period signals ────────────────────────────────────
        period_matches = re.findall(r"\bperiod\s*\d+\b", t)
        period_count   = len(period_matches)

        # ── Calendar date signals ─────────────────────────────
        # Strict: YYYY-MM-DD or DD/MM/YYYY — not just any number
        calendar_dates = re.findall(
            r"\b\d{4}-\d{2}-\d{2}\b"
            r"|\b\d{1,2}/\d{1,2}/\d{4}\b"
            r"|\b\d{1,2}-\d{1,2}-\d{4}\b", text)
        has_calendar_date = len(calendar_dates) >= 1

        # ── Score / marks signals ─────────────────────────────
        score_matches = re.findall(r"\b\d{1,3}\s*/\s*\d{2,3}\b", text)  # 85/100
        score_count   = len(score_matches)

        # ── Grade / result signals ────────────────────────────
        has_grade_col = bool(re.search(
            r"\b(pass|fail|grade|percentage|obtained marks|total marks)\b", t))

        # ── Money / fee signals ───────────────────────────────
        money_matches = re.findall(
            r"(rs\.?|pkr|aed|usd|sar|\$|£)\s*[\d,]+"
            r"|\b\d+\.\d{2}\b", t)
        has_money = len(money_matches) >= 1

        # ── Exam admin signals ────────────────────────────────
        has_exam_admin = bool(re.search(
            r"\b(invigilator|hall ticket|exam hall|date sheet"
            r"|examination schedule|admit card)\b", t))

        # ── Attendance signals ────────────────────────────────
        attendance_hits = len(re.findall(
            r"\b(present|absent|p/a|attendance percentage|days present|days absent)\b", t
        ))

        # ── Notice / announcement signals ─────────────────────
        notice_hits = len(re.findall(
            r"\b(notice|circular|announcement|hereby|informed|"
            r"attention|kindly|parents are requested)\b", t))

        # ── Library signals ───────────────────────────────────
        library_hits = len(re.findall(
            r"\b(isbn|book title|author name|library card|"
            r"borrower|accession no|overdue fine|"
            r"books? issued|books? returned|return slip)\b", t
        ))

        # ── Student registration signals ──────────────────────
        student_reg_hits = len(re.findall(
            r"\b(admission form|enrollment|date of birth|guardian|"
            r"parent name|admission no|registration form)\b", t))

        # ── Teacher registration signals ──────────────────────
        teacher_reg_hits = len(re.findall(
            r"\b(joining date|designation|qualification|"
            r"staff profile|employee id)\b", t))

        return {
            "unique_days":        unique_days,
            "total_days":         total_days,
            "period_count":       period_count,
            "has_calendar_date":  has_calendar_date,
            "calendar_date_count":len(calendar_dates),
            "score_count":        score_count,
            "has_grade_col":      has_grade_col,
            "has_money":          has_money,
            "has_exam_admin":     has_exam_admin,
            "attendance_hits":    attendance_hits,
            "notice_hits":        notice_hits,
            "library_hits":       library_hits,
            "student_reg_hits":   student_reg_hits,
            "teacher_reg_hits":   teacher_reg_hits,
        }

    def _decision_tree(self, fp: dict) -> dict | None:
        """
        Deterministic decision tree on fingerprint signals.
        Most specific / unambiguous checks first.
        Returns None if ambiguous → LLM takes over.
        """

        # ── 1. CLASS TIMETABLE ────────────────────────────────
        # Weekly grid: 3+ distinct days + 3+ period numbers, NO calendar dates
        # This is the most common false-positive: catches your CSV/docx problem
        if (fp["unique_days"] >= 3
                and fp["period_count"] >= 3
                and not fp["has_calendar_date"]):
            return {"doc_type": "timetable", "confidence": 0.99}

        # Days grid without explicit "period" labels (e.g. just time slots per day)
        if (fp["unique_days"] >= 4
                and not fp["has_calendar_date"]
                and not fp["has_exam_admin"]
                and fp["score_count"] == 0):
            return {"doc_type": "timetable", "confidence": 0.95}

        # ── 2. EXAM RESULT ────────────────────────────────────
        # Scores (85/100) + grade/pass column → unambiguous
        if fp["score_count"] >= 2 and fp["has_grade_col"]:
            return {"doc_type": "exam_result", "confidence": 0.97}

        # Many scores even without explicit grade column
        if fp["score_count"] >= 4:
            return {"doc_type": "exam_result", "confidence": 0.93}

        # ── 3. EXAM TIMETABLE ─────────────────────────────────
        # Calendar dates + exam admin signals → unambiguous
        if fp["has_calendar_date"] and fp["has_exam_admin"]:
            return {"doc_type": "exam_timetable", "confidence": 0.97}

        # Calendar dates + no scores + not a weekly grid
        if (fp["has_calendar_date"]
                and fp["score_count"] == 0
                and fp["unique_days"] < 3):
            return {"doc_type": "exam_timetable", "confidence": 0.90}

        # Move ABOVE attendance check
        if fp["has_money"] and fp["notice_hits"] >= 1:
            return {"doc_type": "fee", "confidence": 0.96}  # fee notice specifically

        if fp["has_money"]:
            return {"doc_type": "fee", "confidence": 0.93}

        if fp["has_money"] and fp["attendance_hits"] == 0:
            return {"doc_type": "fee", "confidence": 0.93}

        # ── 5. ATTENDANCE ─────────────────────────────────────
        if fp["attendance_hits"] >= 3 and fp["score_count"] == 0:
            return {"doc_type": "attendance", "confidence": 0.92}

        # ── 6. LIBRARY ────────────────────────────────────────
        if fp["library_hits"] >= 3:
            return {"doc_type": "library", "confidence": 0.93}

        # ── 7. STUDENT REGISTRATION ───────────────────────────
        if fp["student_reg_hits"] >= 2:
            return {"doc_type": "student", "confidence": 0.92}

        # ── 8. TEACHER REGISTRATION ───────────────────────────
        if fp["teacher_reg_hits"] >= 2:
            return {"doc_type": "teacher", "confidence": 0.92}

        # ── 9. NOTICE ─────────────────────────────────────────
        if fp["notice_hits"] >= 2:
            return {"doc_type": "notice", "confidence": 0.88}

        # ── Ambiguous → LLM ───────────────────────────────────
        return None

    # ══════════════════════════════════════════════════════════
    #  STAGE 2 — SMALL LLM FALLBACK
    # ══════════════════════════════════════════════════════════

    async def _llm_classify(self, text: str, llm) -> dict:
        """
        Only fires for genuinely ambiguous documents (~20%).
        Sends first 800 chars only — enough to classify, cheaper, faster.
        Hard rules in system prompt prevent timetable/exam confusion.
        """
        snippet = text[:800].strip()

        prompt = f"""Classify this school document.

Document:
\"\"\"
{snippet}
\"\"\"

Reply ONLY with JSON (no markdown):
{{"doc_type": "<category>", "confidence": <0.0-1.0>, "reason": "<one line>"}}"""

        try:
            response = await llm.ainvoke([
                SystemMessage(content=classifier_prompt),
                HumanMessage(content=prompt),
            ])

            content = response.content.strip()
            content = re.sub(r"```json|```", "", content).strip()
            result  = json.loads(content)

            doc_type = result.get("doc_type", "unknown")
            # Guard: reject hallucinated categories
            if doc_type not in CATEGORIES:
                logger.warning(f"LLM returned unknown category '{doc_type}' → unknown")
                doc_type = "unknown"

            return {
                "doc_type":   doc_type,
                "confidence": float(result.get("confidence", 0.5)),
                "method":     "llm",
                "reason":     result.get("reason", ""),
            }

        except json.JSONDecodeError as e:
            logger.error(f"LLM classify JSON parse error: {e}")
            return {"doc_type": "unknown", "confidence": 0.0,
                    "method": "llm_failed", "error": str(e)}
        except Exception as e:
            logger.error(f"LLM classify error: {e}")
            return {"doc_type": "unknown", "confidence": 0.0,
                    "method": "llm_failed", "error": str(e)}