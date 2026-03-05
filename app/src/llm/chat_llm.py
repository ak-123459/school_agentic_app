# from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import logging
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)

# ── ENV CONFIG ──────────────────────────────────────────────
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "groq").lower()   # groq | nvidia | ollama
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_LLM_MODEL  = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL    = os.getenv("NVIDIA_MODEL", "dracarys-llama-3.1-70b-instruct")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "5000"))


# ── FACTORY ─────────────────────────────────────────────────
class LLMLoader:
    """
    Single instance LLM loader.
    Switch provider via LLM_PROVIDER env var: groq | nvidia | ollama
    """

    _instance = None  # singleton cache


    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._llm = None
        return cls._instance


    async def load(self):
        """Returns cached LLM if already loaded, else builds it."""
        if self._llm is not None:
            return self._llm

        if LLM_PROVIDER == "groq":
            self._llm = ChatGroq(
                model=GROQ_LLM_MODEL,
                api_key=GROQ_API_KEY,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            logging.info(f"[LLM] Groq -> {GROQ_LLM_MODEL}")

        elif LLM_PROVIDER == "nvidia":
            self._llm = ChatNVIDIA(
                model=NVIDIA_MODEL,
                api_key=NVIDIA_API_KEY,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            logging.info(f"[LLM] NVIDIA -> {NVIDIA_MODEL}")

        elif LLM_PROVIDER == "ollama":
            self._llm = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=LLM_TEMPERATURE,
            )
            logging.info(f"[LLM] Ollama -> {OLLAMA_MODEL}")





        else:
            raise ValueError(f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. Use groq | nvidia | ollama")

        return self._llm






# ── GLOBAL SINGLETON ─────────────────────────────────────────
llm_loader =  LLMLoader()