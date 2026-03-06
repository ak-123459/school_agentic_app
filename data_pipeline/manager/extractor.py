"""
STEP 3 — EXTRACTOR
ONE LLM call per document.
Strict JSON schema per doc type → maps directly to your DB columns.
Now uses LLMLoader → switch provider via LLM_PROVIDER env var:
    groq | nvidia | ollama | anthropic
"""
import json
import re
import asyncio
from langchain_core.messages import HumanMessage


SCHEMAS = {

    # → exam_results + report_cards tables
    "exam_result": {
        "prompt": """Extract exam results. Return ONLY this JSON:
{
  "exam_name":     "string or null",
  "exam_type":     "annual|midterm|unit_test|quiz or null",
  "exam_date":     "YYYY-MM-DD or null",
  "academic_year": "string or null",
  "class_name":    "string or null",
  "grade":         "string or null",
  "students": [
    {
      "name":        "string",
      "student_id":  "string or null",
      "subjects": [
        {"name": "string", "score": number, "max_score": number}
      ],
      "pass_fail":    "pass|fail or null"
    }
  ]
}"""
    },

    # → exam_timetable table
    "exam_timetable": {
        "prompt": """Extract exam timetable. Return ONLY this JSON:
{
  "exam_name":     "string or null",
  "exam_type":     "annual|midterm|unit_test|quiz or null",
  "academic_year": "string or null",
  "class_name":    "string or null",
  "slots": [
    {
      "subject":      "string",
      "exam_date":    "YYYY-MM-DD",
      "start_time":   "HH:MM",
      "end_time":     "HH:MM",
      "duration_min": number or null,
      "room":         "string or null",
      "invigilator":  "string or null",
      "notes":        "string or null"
    }
  ]
}"""
    },

    # → notices table
    "notice": {
        "prompt": """Extract notice/announcement. Return ONLY this JSON:
{
  "title":       "string",
  "content":     "string",
  "category":    "exam|event|holiday|fee|general|emergency|transport|library",
  "target_type": "all|grade|class|student|teacher",
  "target_id":   "grade number or class name or null",
  "priority":    "high|medium|low",
  "expires_at":  "YYYY-MM-DD or null",
  "school_name": "string or null"
}"""
    },

    # → attendance table
    "attendance": {
        "prompt": """Extract attendance records. Return ONLY this JSON:
{
  "date":       "YYYY-MM-DD",
  "class_name": "string or null",
  "grade":      "string or null",
  "records": [
    {
      "student_name": "string",
      "status":       "present|absent|late|excused",
      "remarks":      "string or null"
    }
  ]
}"""
    },

    # → fees table
    "fee": {
        "prompt": """Extract fee information. Return ONLY this JSON:
{
  "student_name": "string or null",
  "class_name":   "string or null",
  "fees": [
    {
      "fee_type": "tuition|transport|library|activity",
      "amount":   number,
      "due_date": "YYYY-MM-DD or null",
      "paid_date":"YYYY-MM-DD or null",
      "status":   "paid|unpaid|partial"
    }
  ]
}"""
    },

    # → events table
    "event": {
        "prompt": """Extract event information. Return ONLY this JSON:
{
  "title":       "string",
  "description": "string or null",
  "event_date":  "YYYY-MM-DD",
  "type":        "holiday|sports|exam|trip|ceremony",
  "school_name": "string or null"
}"""
    },

    # → timetable table (regular class schedule)
    "timetable": {
        "prompt": """Extract class timetable. Return ONLY this JSON:
{
  "class_name": "string or null",
  "grade":      "string or null",
  "slots": [
    {
      "subject":     "string",
      "teacher_name":"string or null",
      "day":         "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday",
      "start_time":  "HH:MM",
      "end_time":    "HH:MM",
      "room":        "string or null"
    }
  ]
}"""
    },

    # → library_books + library_loans tables
    "library": {
        "prompt": """Extract library information. Return ONLY this JSON:
{
  "type": "book_list|loan_record",
  "books": [
    {
      "title":     "string",
      "author":    "string or null",
      "subject":   "string or null",
      "grade":     "string or null",
      "quantity":  number or null
    }
  ],
  "loans": [
    {
      "book_title":    "string",
      "student_name":  "string",
      "issued_date":   "YYYY-MM-DD or null",
      "due_date":      "YYYY-MM-DD or null",
      "return_date":   "YYYY-MM-DD or null",
      "status":        "issued|returned|overdue"
    }
  ]
}"""
    },

    # → students table
    "student": {
        "prompt": """Extract student registration data. Return ONLY this JSON:
{
  "students": [
    {
      "name":         "string",
      "dob":          "YYYY-MM-DD or null",
      "gender":       "M|F or null",
      "grade":        "string or null",
      "class_name":   "string or null",
      "parent_name":  "string or null",
      "parent_phone": "string or null",
      "parent_email": "string or null",
      "address":      "string or null",
      "enrolled_date":"YYYY-MM-DD or null"
    }
  ]
}"""
    },

    # → teachers table
    "teacher": {
        "prompt": """Extract teacher/staff registration data. Return ONLY this JSON:
{
  "teachers": [
    {
      "name":        "string",
      "email":       "string or null",
      "phone":       "string or null",
      "subjects":    ["list of subject names"],
      "class_name":  "string or null",
      "joined_date": "YYYY-MM-DD or null",
      "status":      "active|inactive or null"
    }
  ]
}"""
    }
}


class Extractor:

    CHUNK_SIZE     = 8000   # chars per chunk for first attempt
    FALLBACK_CHUNK = 4000   # smaller chunks if first attempt fails

    async def extract(self, text: str, doc_type: str, llm) -> dict:
        if doc_type not in SCHEMAS:
            return {"success": False, "error": f"No schema for: {doc_type}"}

        # ── ATTEMPT 1: full doc (up to CHUNK_SIZE) ─────────────────────────
        result = await self._extract_chunk(text[:self.CHUNK_SIZE], doc_type, llm)

        if result["success"]:
            return result

        print(f"[EXTRACTOR] Full-doc extraction failed: {result.get('error')} — falling back to chunking")

        # ── ATTEMPT 2: split into smaller chunks and merge ──────────────────
        chunks      = self._split_chunks(text, self.FALLBACK_CHUNK)
        all_results = []

        for i, chunk in enumerate(chunks):
            print(f"[EXTRACTOR] Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            chunk_result = await self._extract_chunk(chunk, doc_type, llm)
            if chunk_result["success"]:
                all_results.append(chunk_result["data"])
            else:
                print(f"[EXTRACTOR] Chunk {i+1} failed: {chunk_result.get('error')}")

        if not all_results:
            return {"success": False, "error": "All chunks failed to extract"}

        merged = self._merge_results(all_results, doc_type)
        return {"success": True, "data": merged, "doc_type": doc_type}


    async def _extract_chunk(self, snippet: str, doc_type: str, llm) -> dict:
        """Single LLM call for one chunk."""
        schema = SCHEMAS[doc_type]["prompt"]

        prompt = f"""{schema}

Rules:
- Return ONLY valid JSON — no explanation, no markdown backticks
- Use null for missing values, never guess or invent data
- Dates: YYYY-MM-DD format only
- Times: HH:MM 24h format only
- Numbers must be numeric type, not strings
- If a field is not found in this section, use null

Document section:
{snippet}"""

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content  = response.content.strip()
            content  = re.sub(r"```json|```", "", content).strip()
            parsed   = json.loads(content)
            return {"success": True, "data": parsed, "doc_type": doc_type}

        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON parse error: {e} | Raw: {content[:200]}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


    def _split_chunks(self, text: str, chunk_size: int) -> list[str]:
        """
        Split text into chunks at natural boundaries (newlines preferred).
        Adds overlap so records split across chunk boundaries are not lost.
        """
        OVERLAP = 200   # chars of overlap between chunks
        chunks  = []
        start   = 0

        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                # Try to cut at last newline before end
                newline_pos = text.rfind("\n", start, end)
                if newline_pos > start:
                    end = newline_pos

            chunks.append(text[start:end])
            start = end - OVERLAP   # overlap to catch split records

        return chunks


    def _merge_results(self, results: list[dict], doc_type: str) -> dict:
        """
        Merge multiple chunk results into a single coherent dict.
        List fields (students, slots, records, etc.) are concatenated.
        Scalar fields take the first non-null value found.
        Duplicates are removed by converting to JSON strings.
        """
        if not results:
            return {}

        merged = {}

        for result in results:
            for key, val in result.items():

                if isinstance(val, list):
                    # Concatenate lists
                    merged.setdefault(key, [])
                    merged[key].extend(val)

                elif val is not None:
                    # Scalar: take first non-null value
                    if key not in merged or merged[key] is None:
                        merged[key] = val

        # ── Deduplicate list items by JSON fingerprint ──────────────────────
        for key, val in merged.items():
            if isinstance(val, list):
                seen    = set()
                deduped = []
                for item in val:
                    fingerprint = json.dumps(item, sort_keys=True)
                    if fingerprint not in seen:
                        seen.add(fingerprint)
                        deduped.append(item)
                merged[key] = deduped

        return merged