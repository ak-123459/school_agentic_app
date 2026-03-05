"""
api_main.py
===========
FastAPI server — correct startup sequence:

  1. Load LLM
  2. Load Embedder
  3. CREATE all per-key FAISS indexes  (skips already-existing ones)
  4. Init RAG tools  (SchoolFaissLoader + SchoolRAGManager)
  5. Load default index for ChatManager

Each tool call then loads its OWN specific index on demand via
SchoolFaissLoader.get_retriever(key_name).
"""

from fastapi import FastAPI, HTTPException
from app.assistant import ChatManager
from app.models.schemas import ChatInput, ChatOutput
from dotenv import load_dotenv
import os
import logging
import traceback
import time
from fastapi.middleware.cors import CORSMiddleware
import yaml

from app.settings import MODEL_CONFIG_PATH, EMBD_MODEL_DIR, DOCS_PATH, VECTOR_STORE_PATH
from app.src.llm.chat_llm import llm_loader
from app.src.embedder.embedder_factory import EMBFactory
from app.src.vector_database.vector_db_factory import VECTORDBFactory
from app.rag_pipeline.tools.school_rag_tools import init_rag_tools

SCHOOL_YAML_PATH = os.path.join(os.path.dirname(__file__), "app/config/school_vector_indexes.yaml")

print("=== Starting the app ===")
load_dotenv()
logger = logging.getLogger("uvicorn")

llm        = None
vector_db  = None
manager    = None

GROQ_API_KEY = os.getenv('GROQ_API_KEY')


def load_config(config_path=MODEL_CONFIG_PATH):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])



@app.on_event("startup")
async def on_startup():
    global llm, vector_db, manager

    try:
        model_config  = load_config()
        chat_llm_args = model_config['chat_llm_args']
        chat_llm_args['api_key'] = GROQ_API_KEY
        db_args       = model_config['db_args']
        db_args['vector_store_path'] = VECTOR_STORE_PATH
        db_args['docs_path']         = DOCS_PATH
        embedder_args = model_config['embedder_args']
        embedder_args['model_path']  = EMBD_MODEL_DIR

        # ── STEP 1: Load LLM ──────────────────────────────────────────────
        logger.info("⏳ [STARTUP] Step 1/4 — Loading LLM...")
        t0  = time.perf_counter()
        llm = await llm_loader.load()
        logger.info(f"✅ [STARTUP] LLM ready  ({(time.perf_counter()-t0)*1000:.0f} ms)")

        # ── STEP 2: Load Embedder ─────────────────────────────────────────
        logger.info("⏳ [STARTUP] Step 2/4 — Loading embedding model...")
        t0            = time.perf_counter()
        embedder_pipe = EMBFactory.create_embedder_model_pipeline(
            embedder_args['type'], **embedder_args
        )
        embedding_model = await embedder_pipe.load_model()
        db_args['embedding_model'] = embedding_model
        logger.info(f"✅ [STARTUP] Embedder ready  ({(time.perf_counter()-t0)*1000:.0f} ms)")

        # ── STEP 3: Create all per-key FAISS indexes ──────────────────────
        #
        #  This is the KEY step. For each entry in school_vector_indexes.yaml
        #  it reads docs/<key_name>/ and writes vector_db/<key_name>/index.faiss
        #  Already-existing indexes are skipped (pass overwrite=True to rebuild).
        #
        logger.info("⏳ [STARTUP] Step 3/4 — Creating per-key FAISS indexes...")
        logger.info(f"   docs root  : {DOCS_PATH}")
        logger.info(f"   index root : {VECTOR_STORE_PATH}")
        t0             = time.perf_counter()
        vector_db_pipe = VECTORDBFactory.create_vector_db_pipeline(**db_args)
        result         = vector_db_pipe.create_all_key_indexes(overwrite=False)
        logger.info(
            f"✅ [STARTUP] Indexes ready — "
            f"created: {result['created']}, skipped: {result['skipped']}, failed: {result['failed']}"
            f"  ({(time.perf_counter()-t0)*1000:.0f} ms)"
        )

        if result['failed'] > 0:
            logger.warning(
                f"⚠️  [STARTUP] {result['failed']} key(s) failed — "
                f"check that docs/<key_name>/ subfolders exist with content."
            )

        # ── STEP 4: Init RAG tools ────────────────────────────────────────
        #
        #  SchoolFaissLoader is created here.
        #  It does NOT load any indexes yet — each index is loaded lazily
        #  on the FIRST tool call that needs it, then cached in memory.
        #
        logger.info("⏳ [STARTUP] Step 4/4 — Initialising RAG tools...")
        t0 = time.perf_counter()
        init_rag_tools(
            llm                 = llm,
            embeddings          = embedding_model,
            base_vector_db_path = str(VECTOR_STORE_PATH),
            yaml_path           = SCHOOL_YAML_PATH,
        )
        logger.info(f"✅ [STARTUP] RAG tools ready  ({(time.perf_counter()-t0)*1000:.0f} ms)")

        # ── Load default index for ChatManager ────────────────────────────
        logger.info("⏳ [STARTUP] Loading default index for ChatManager...")
        t0        = time.perf_counter()
        vector_db = await vector_db_pipe.load_faiss_db("get_exam_timetable")
        manager   = ChatManager(llm, vector_db.as_retriever())
        _         = vector_db.as_retriever().invoke("startup test")
        logger.info(f"✅ [STARTUP] ChatManager ready  ({(time.perf_counter()-t0)*1000:.0f} ms)")

        logger.info("🚀 [STARTUP] All components initialized. Server is ready.")

    except Exception:
        logger.error("❌ [STARTUP] Failed:\n%s", traceback.format_exc())
        raise RuntimeError("Startup initialization failed.")


@app.get("/")
def health_check():
    return {'health': 'ok'}


@app.post("/chat", response_model=ChatOutput)
async def get_response(request: ChatInput):
    global manager
    try:
        logger.info(f"📥 Received query: {request.query}")
        t0       = time.perf_counter()
        response = await manager.run(request.query, request.last_3_turn)
        logger.info(f"⚡ Latency: {(time.perf_counter()-t0)*1000:.0f} ms")
        logger.info(f"📤 Response: {response}")
        return {'response': response}
    except Exception:
        logger.error("❌ Error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")