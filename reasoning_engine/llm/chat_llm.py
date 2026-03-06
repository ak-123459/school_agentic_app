# from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import logging
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import yaml


logging.basicConfig(level=logging.INFO)


# model_config.yaml
with open("configs/model_config.yaml",    "r", encoding="utf-8") as f:
    model_config    = yaml.safe_load(f)




GROQ_LLM        = model_config["GROQ_LLM"]
GROQ_STT        = model_config["GROQ_STT"]
GROQ_API_KEY    = model_config["GROQ_API_KEY"]
GROQ_LLM_MODEL  = model_config["GROQ_LLM_MODEL"]
OLLAMA_BASE_URL = model_config["OLLAMA_BASE_URL"]
OLLAMA_MODEL    = model_config["OLLAMA_MODEL"]
LLM_MAX_TOKENS  = model_config["LLM_MAX_TOKENS"]
LLM_RETRIES     = model_config["LLM_RETRIES"]
LLM_PROVIDER = model_config['LLM_PROVIDER']
LLM_TEMPERATURE = model_config['LLM_TEMPERATURE']
NVIDIA_API_KEY = model_config['NVIDIA_API_KEY']
NVIDIA_MODEL = model_config['NVIDIA_MODEL']


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