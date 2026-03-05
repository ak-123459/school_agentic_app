"""
rag_manager.py
==============
SchoolRAGManager — one RAG pipeline per school function key.

Each call:
  1. Loads the FAISS index for the requested function key  (cached)
  2. Builds a history-aware retriever chain
  3. Builds a QA chain with a key-specific prompt
  4. Invokes the full RAG chain and returns a clean plain-text answer

This is intentionally separate from ChatManager (Open-RAGA) so both
can coexist and be called from the same voice assistant.

Usage (from a tool function):
    answer = await rag_manager.query(
        function_key = "get_exam_timetable",
        query        = "When is the Maths exam for Class 8?",
        chat_history = [],
        filters      = {"class": "8", "section": "A"},
    )
"""

import logging
from typing import Any, Dict, List

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

from .faiss_loader import SchoolFaissLoader
from .prompt_templates import (
    build_contextualize_prompt,
    build_answer_prompt,
    EXAM_TIMETABLE_PROMPT,
    EXAM_RESULT_PROMPT,
    CLASS_TIMETABLE_PROMPT,
)

logger = logging.getLogger(__name__)




# ── Per-key custom prompts ─────────────────────────────────────────────────────
KEY_PROMPT_MAP = {
    "get_exam_timetable":  EXAM_TIMETABLE_PROMPT,
    "get_exam_result":     EXAM_RESULT_PROMPT,
    "get_class_timetable": CLASS_TIMETABLE_PROMPT,
    "get_grades":          EXAM_RESULT_PROMPT,
}







class SchoolRAGManager:
    """
    Scalable RAG manager for school voice assistant.

    One instance shared across all tool calls.
    Internally caches one RAG chain per function key.
    """

    def __init__(self, llm, faiss_loader: SchoolFaissLoader):
        self.llm           = llm
        self.faiss_loader  = faiss_loader
        self._chain_cache: Dict[str, Any] = {}   # key_name → rag_chain

    # ── Chain builder ─────────────────────────────────────────────────────────

    def _build_chain(self, key_name: str, search_kwargs: dict = None):
        """Build and cache a full RAG chain for the given function key."""

        if key_name in self._chain_cache:
            return self._chain_cache[key_name]

        logger.info(f"[RAG] Building chain for key: {key_name}")

        # 1. Retriever
        retriever = self.faiss_loader.get_retriever(
            key_name,
            search_kwargs=search_kwargs or {"k": 2}
        )

        # 2. History-aware retriever
        contextualize_prompt = build_contextualize_prompt()
        history_retriever    = create_history_aware_retriever(
            self.llm, retriever, contextualize_prompt
        )

        # 3. Answer chain — use key-specific prompt if available
        custom_prompt = KEY_PROMPT_MAP.get(key_name)
        answer_prompt = build_answer_prompt(custom_prompt)
        qa_chain      = create_stuff_documents_chain(self.llm, answer_prompt)

        # 4. Full RAG chain
        rag_chain = create_retrieval_chain(history_retriever, qa_chain)

        self._chain_cache[key_name] = rag_chain
        logger.info(f"[RAG] Chain ready for key: {key_name}")
        return rag_chain

    # ── Chat history formatter ────────────────────────────────────────────────

    @staticmethod
    def _format_history(chat_history: List[Dict]) -> list:
        """Convert list of dicts → LangChain HumanMessage / AIMessage."""
        if not chat_history:
            return []
        formatted = []
        for msg in chat_history:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(HumanMessage(content=content))
            elif role == "assistant":
                formatted.append(AIMessage(content=content))

        # Ensure strict alternation (merge consecutive same-role messages)
        fixed = []
        for msg in formatted:
            if fixed and type(msg) == type(fixed[-1]):
                fixed[-1].content += " " + msg.content
            else:
                fixed.append(msg)
        return fixed

    # ── Filter query builder ──────────────────────────────────────────────────

    @staticmethod
    def _inject_filters(query: str, filters: Dict) -> str:
        """
        Append filter context to the query so the retriever/LLM focuses on
        the right class, section, or subject.

        filters keys: class, section, subject, roll_number, student_name
        """
        if not filters:
            return query

        parts = []
        if filters.get("class"):
            parts.append(f"Class {filters['class']}")
        if filters.get("section"):
            parts.append(f"Section {filters['section']}")
        if filters.get("subject"):
            parts.append(f"Subject: {filters['subject']}")
        if filters.get("roll_number"):
            parts.append(f"Roll Number: {filters['roll_number']}")
        if filters.get("student_name"):
            parts.append(f"Student: {filters['student_name']}")

        if parts:
            query = f"{query} [{', '.join(parts)}]"
        return query

    # ── Main query method ─────────────────────────────────────────────────────

    async def query(
        self,
        function_key: str,
        query:        str,
        chat_history: List[Dict] = None,
        filters:      Dict       = None,
        search_k:     int        = 4,
    ) -> str:
        """
        Run a RAG query for the given function key.

        Args:
            function_key : e.g. "get_exam_timetable" or int 7
            query        : user's question
            chat_history : list of {role, content} dicts
            filters      : {"class": "8", "section": "A", "subject": "Maths"}
            search_k     : number of docs to retrieve

        Returns:
            Plain-text answer string (TTS-ready)
        """
        try:
            # Resolve int → string key
            if isinstance(function_key, int):
                function_key = self.faiss_loader._key_map.get(function_key, str(function_key))

            # Inject filters into query for better retrieval
            enriched_query = self._inject_filters(query, filters or {})
            logger.info(f"[RAG] key={function_key} | query={enriched_query[:80]}")

            # Build or fetch cached chain
            chain = self._build_chain(function_key, search_kwargs={"k": search_k})

            # Format history
            history = self._format_history(chat_history or [])

            # Invoke chain
            result = chain.invoke({
                "input":        enriched_query,
                "chat_history": history,
            })

            answer = result.get("answer", "").strip()

            if not answer:
                return "I don't have that information right now."

            logger.info(f"[RAG] Answer: {answer[:100]}")
            return answer

        except FileNotFoundError as e:
            logger.error(f"[RAG] Index missing: {e}")
            return "That information is not available in the database yet."

        except Exception as e:
            logger.error(f"[RAG] Error in query: {e}", exc_info=True)
            return "Sorry, I couldn't retrieve that information right now."

    def clear_chain_cache(self, key: str = None):
        """Clear cached chains (useful after index update)."""
        if key:
            self._chain_cache.pop(key, None)
        else:
            self._chain_cache.clear()
        logger.info(f"[RAG] Chain cache cleared: {'all' if not key else key}")