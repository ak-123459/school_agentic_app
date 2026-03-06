"""
PIPELINE ORCHESTRATOR
=====================
Call: pipeline.run(file_path)
Works whether DB is empty (day 1) or full.
LLM loaded ONCE here → passed to steps that need it.
"""
import os, json, asyncio
from datetime import datetime
from data_pipeline.manager.preprocessor import Preprocessor  # ← separate file
from data_pipeline.manager.classifier   import Classifier
from data_pipeline.manager.extractor    import Extractor
from data_pipeline.manager.validator    import Validator
from data_pipeline.manager.inserter     import Inserter
from reasoning_engine.llm.chat_llm      import llm_loader
import logging


log = logging.getLogger("aptal_edu")  # ← same logger as knowledge_ingest_api.py


class DocumentPipeline:

    def __init__(self):
        self.pre = Preprocessor()
        self.cls = Classifier()
        self.ext = Extractor()
        self.val = Validator()
        self.ins = Inserter()

    # ── called by FastAPI (async) ──────────────────────────
    async def run_async(self, file_path: str,
                        created_by: str = None,
                        force_doc_type: str = None) -> dict:


        log.info(f"🚀 Pipeline started → {os.path.basename(file_path)}")

        pipeline_log = {  # ← renamed from log → pipeline_log
            "file": os.path.basename(file_path),
            "started_at": datetime.now().isoformat(),
            "steps": {}
        }

        # ── Load LLM ONCE here, pass to any step that needs it ──
        llm = await llm_loader.load()

        # STEP 1 — Extract text (no LLM needed)
        log.info("1️⃣  Extracting text...")
        pre = self.pre.extract(file_path)
        pipeline_log["steps"]["preprocess"] = {
            "type":  pre["file_type"],
            "chars": len(pre["text"]),
            "error": pre.get("error")
        }
        if pre.get("error"):
            return self._fail(pipeline_log, f"Preprocessing: {pre['error']}")
        log.info(f"   ✅ {len(pre['text'])} chars extracted")

        # STEP 2 — Classify (no LLM — rules based)
        log.info("2️⃣  Classifying...")
        if force_doc_type:
            clf = {"doc_type": force_doc_type,
                   "confidence": "forced", "method": "manual"}
        else:
            clf = await self.cls.classify(pre["text"],llm=llm)

        pipeline_log["steps"]["classify"] = clf
        doc_type = clf["doc_type"]

        if doc_type == "unknown":
            return self._fail(pipeline_log, "Could not classify document")

        log.info(f"   ✅ {doc_type.upper()} "
              f"(confidence={clf['confidence']}, method={clf['method']})")

        # STEP 3 — Extract structured data (LLM passed in ✅)
        log.info("3️⃣  Extracting structured data (1 LLM call)...")
        ext = await self.ext.extract(pre["text"], doc_type, llm)
        pipeline_log["steps"]["extract"] = {"success": ext.get("success"),
                                   "error":   ext.get("error")}
        if not ext.get("success"):
            return self._fail(pipeline_log, f"Extraction: {ext.get('error')}")
        log.info("   ✅ Structured data extracted")

        log.info(f"Extracted data is :-> {ext}")

        # STEP 4 — Validate (no LLM needed)
        log.info("4️⃣  Validating & resolving DB references...")
        val = self.val.validate(ext, doc_type)
        pipeline_log["steps"]["validate"] = {
            "valid":    val.get("valid"),
            "warnings": val.get("warnings", []),
            "errors":   val.get("errors", [])
        }
        for w in val.get("warnings", []):
            log.info(f"   ⚠️  {w}")
        if not val.get("valid"):
            return self._fail(pipeline_log, f"Validation: {'; '.join(val.get('errors', []))}")
        log.info("   ✅ Validation passed")

        # STEP 5 — Insert (no LLM needed)
        log.info("5️⃣  Inserting into database...")
        ins = self.ins.insert(val, doc_type, created_by)
        pipeline_log["steps"]["insert"] = ins
        if not ins.get("success"):
            return self._fail(pipeline_log, f"Insert: {ins.get('error')}")

        log.info(f"   ✅ {ins['inserted']} records → {ins['table']}")

        llm_calls = 1 if clf["method"] == "rules" else 2
        pipeline_log.update({
            "success":     True,
            "doc_type":    doc_type,
            "inserted":    ins["inserted"],
            "table":       ins["table"],
            "llm_calls":   llm_calls,
            "warnings":    val.get("warnings", []),
            "finished_at": datetime.now().isoformat()
        })
        log.info(f"\n✅ Done! LLM calls used: {llm_calls}")
        return pipeline_log

    # ── called by scripts / tests (sync) ──────────────────
    def run(self, file_path: str,
            created_by: str = None,
            force_doc_type: str = None) -> dict:
        return asyncio.run(self.run_async(
            file_path, created_by, force_doc_type
        ))

    def _fail(self, pipeline_log: dict, reason):
        log.error(f"💥 Pipeline failed: {reason}")  # module-level logger
        pipeline_log.update({"success": False, "error": reason})
        return pipeline_log

# ── Quick test ─────────────────────────────────────────
if __name__ == "__main__":
    sample = "sample_notice.txt"
    with open(sample, "w") as f:
        f.write("""
NOTICE — Al Noor Academy
Grade 9 Midterm Examinations will commence from March 10, 2026.
Students must carry their admit cards.
Principal, Al Noor Academy
        """)

    pipeline = DocumentPipeline()
    result   = pipeline.run(
        file_path  = sample,
        created_by = "TCH_001"
    )
    log.info("\n📋 Result:")
    print(json.dumps(result, indent=2))