"""
EDUMIND AI — FastAPI Document Upload Server
"""
import os
import json
import asyncio
import tempfile
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # ← load .env before anything else

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fast_api.log", mode="a", encoding="utf-8"),
    ]
)
log = logging.getLogger("aptal_edu")

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# ── Import pipeline ───────────────────────────────────────────────────────────
try:
    from app.data.db.manager.pipeline import DocumentPipeline
    PIPELINE_AVAILABLE = True
    log.info("✅ Pipeline imported successfully")
except Exception as e:
    PIPELINE_AVAILABLE = False
    log.error(f"❌ Pipeline import failed: {e}")
    log.error(traceback.format_exc())

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EduMind AI — Document Pipeline API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global exception logger ───────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    log.info(f"→ {request.method} {request.url}")
    try:
        response = await call_next(request)
        log.info(f"← {response.status_code} {request.url}")
        return response
    except Exception as e:
        log.error(f"💥 Unhandled error on {request.url}: {e}")
        log.error(traceback.format_exc())
        raise

# ── Pipeline singleton ────────────────────────────────────────────────────────
try:
    pipeline = DocumentPipeline() if PIPELINE_AVAILABLE else None
    if pipeline:
        log.info("✅ Pipeline instance created")
except Exception as e:
    pipeline = None
    log.error(f"❌ Pipeline instantiation failed: {e}")
    log.error(traceback.format_exc())

# ── Helpers ───────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".jpg", ".jpeg", ".png"}

# ▼▼▼ ADD 1: size constant — change this one number to adjust the limit ▼▼▼
MAX_FILE_SIZE = 50 * 1024          # 50 KB  (50 × 1024 bytes)
# ▲▲▲ ─────────────────────────────────────────────────────────────────── ▲▲▲


def _validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    return ext


# ▼▼▼ ADD 2: size check inside _save_upload — runs for BOTH /upload routes ▼▼▼
async def _save_upload(file: UploadFile) -> str:
    ext     = _validate_extension(file.filename)
    content = await file.read()                    # read once, reuse below

    # ── Size guard ────────────────────────────────
    if len(content) > MAX_FILE_SIZE:
        size_kb  = len(content) / 1024
        limit_kb = MAX_FILE_SIZE / 1024
        log.warning(f"🚫 Rejected '{file.filename}': {size_kb:.1f} KB > {limit_kb:.0f} KB limit")
        raise HTTPException(
            status_code=413,           # 413 = Payload Too Large (correct HTTP code)
            detail={
                "error":    "FILE_TOO_LARGE",
                "message":  f"File '{file.filename}' is {size_kb:.1f} KB. Maximum allowed size is {limit_kb:.0f} KB.",
                "size_kb":  round(size_kb, 1),
                "limit_kb": limit_kb,
            }
        )
    # ─────────────────────────────────────────────

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(content)
    tmp.close()
    log.info(f"📁 Saved upload: {file.filename} ({len(content)/1024:.1f} KB) → {tmp.name}")
    return tmp.name
# ▲▲▲ ─────────────────────────────────────────────────────────────────── ▲▲▲


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":             "ok",
        "pipeline_available": PIPELINE_AVAILABLE,
        "max_upload_kb":      MAX_FILE_SIZE // 1024,   # ← visible in health check
        "timestamp":          datetime.now().isoformat(),
    }

# ── 1. Standard upload ────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    school_id: str = Form("SCH_001"),
    created_by: Optional[str] = Form(None),
    force_doc_type: Optional[str] = Form(None),
):
    tmp_path = await _save_upload(file)   # ← size check happens here, raises 413 if too big
    log.info(f"📤 /upload → {file.filename} | school={school_id}")

    try:
        if not PIPELINE_AVAILABLE:
            await asyncio.sleep(1)
            return JSONResponse({
                "success": True, "file": file.filename,
                "school_id": school_id, "doc_type": "exam_result",
                "inserted": 5, "table": "exam_results",
                "llm_calls": 1, "warnings": [], "demo_mode": True,
            })

        result = await pipeline.run_async(
            file_path=tmp_path,
            school_id=school_id,
            created_by=created_by,
            force_doc_type=force_doc_type,
        )
        result["original_filename"] = file.filename

        if result.get("success"):
            log.info(f"✅ {file.filename} → {result.get('doc_type')} | {result.get('inserted')} rows → {result.get('table')}")
        else:
            log.error(f"❌ {file.filename} failed: {result.get('error')}")

        return JSONResponse(result)

    except HTTPException:
        raise   # re-raise 413 / 415 as-is — don't swallow them

    except Exception as e:
        log.error(f"💥 /upload exception for {file.filename}: {e}")
        log.error(traceback.format_exc())
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── 2. Streaming upload (SSE) ─────────────────────────────────────────────────
@app.post("/upload/stream")
async def upload_document_stream(
    file: UploadFile = File(...),
    school_id: str = Form("SCH_001"),
    created_by: Optional[str] = Form(None),
    force_doc_type: Optional[str] = Form(None),
):
    # ▼▼▼ ADD 3: for streaming, catch 413 early and return SSE error event ▼▼▼
    try:
        tmp_path = await _save_upload(file)   # raises 413 if too big
    except HTTPException as e:
        if e.status_code == 413:
            # Return a proper SSE error so the frontend stream handler gets it
            async def size_error():
                err = json.dumps({
                    "step":    "done",
                    "success": False,
                    "error":   e.detail["error"],
                    "message": f"❌ {e.detail['message']}",
                    "progress": 100,
                })
                yield f"data: {err}\n\n"
            return StreamingResponse(
                size_error(),
                media_type="text/event-stream",
                status_code=413,
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        raise   # re-raise other HTTP errors (415 etc.)
    # ▲▲▲ ─────────────────────────────────────────────────────────────────── ▲▲▲

    filename = file.filename
    log.info(f"📤 /upload/stream → {filename} | school={school_id}")

    async def event_stream():
        steps = [
            ("preprocess", f"📄 Extracting text from {filename}...",         20),
            ("classify",   "🧠 Classifying document type...",                 40),
            ("extract",    "⚡ Extracting structured data (1 LLM call)...",   60),
            ("validate",   "🔒 Validating & resolving DB references...",       80),
            ("insert",     "💾 Inserting records into database...",            95),
        ]

        try:
            if not PIPELINE_AVAILABLE:
                for step_id, msg, prog in steps:
                    payload = json.dumps({"step": step_id, "message": msg, "progress": prog})
                    yield f"data: {payload}\n\n"
                    await asyncio.sleep(0.9)
                done = json.dumps({
                    "step": "done", "success": True, "file": filename,
                    "doc_type": "exam_result", "inserted": 5,
                    "table": "exam_results", "llm_calls": 1, "warnings": [],
                    "message": "✅ All records inserted successfully!",
                    "demo_mode": True,
                })
                yield f"data: {done}\n\n"
                return

            for step_id, msg, prog in steps:
                log.info(f"  ⚙️  [{filename}] {msg}")
                payload = json.dumps({"step": step_id, "message": msg, "progress": prog})
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.3)

            result = await pipeline.run_async(
                file_path=tmp_path,
                created_by=created_by,
                force_doc_type=force_doc_type,
            )

            result["original_filename"] = filename
            result["step"]     = "done"
            result["progress"] = 100

            if result.get("success"):
                msg = f"✅ {result.get('inserted', 0)} records inserted into <strong>{result.get('table','DB')}</strong>"
                log.info(f"✅ [{filename}] {result.get('doc_type')} | {result.get('inserted')} rows → {result.get('table')}")
            else:
                msg = f"❌ {result.get('error', 'Unknown error')}"
                log.error(f"❌ [{filename}] pipeline failed: {result.get('error')}")

            result["message"] = msg
            yield f"data: {json.dumps(result)}\n\n"

        except Exception as exc:
            log.error(f"💥 /upload/stream exception for {filename}: {exc}")
            log.error(traceback.format_exc())
            err = json.dumps({
                "step": "done", "success": False,
                "error": str(exc),
                "message": f"❌ Server error: {exc}",
                "progress": 100,
            })
            yield f"data: {err}\n\n"

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                log.info(f"🗑️  Cleaned up temp file for {filename}")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fast_api:app", host="0.0.0.0", port=8001, reload=True)