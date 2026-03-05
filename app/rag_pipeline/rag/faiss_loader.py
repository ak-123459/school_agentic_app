"""
faiss_loader.py
===============
Loads FAISS indexes per school_function_key using school_vector_indexes.yaml.

Each function key maps to its own sub-folder:
    <BASE_VECTOR_DB_PATH>/<key_name_without_prefix>/
        index.faiss
        index.pkl

Usage:
    loader = SchoolFaissLoader(
        base_path   = "/root/Open-RAGA/app/data/vector_db",
        yaml_path   = "school_vector_indexes.yaml",
        embeddings  = your_embedding_model,
    )
    retriever = loader.get_retriever("get_exam_timetable")
    retriever = loader.get_retriever(7)          # by int index too
"""

import os
import yaml
import logging
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


class SchoolFaissLoader:
    """
    Manages per-function FAISS index loading with caching.
    Reads key → folder mapping from school_vector_indexes.yaml.
    """

    def __init__(self, base_path: str, yaml_path: str, embeddings):
        self.base_path  = base_path
        self.embeddings = embeddings
        self._cache: dict = {}                          # key_name → FAISS instance
        self._key_map   = self._load_yaml(yaml_path)   # int → key_name
        self._name_map  = {v: k for k, v in self._key_map.items()}  # key_name → int

    # ── YAML loading ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_yaml(yaml_path: str) -> dict:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        raw = config.get("school_function_keys", {})
        return {int(k): v for k, v in raw.items()}

    # ── Key resolution ────────────────────────────────────────────────────────

    def _resolve_key_name(self, key) -> str:
        """Accept int index or string key name, always return key_name string."""
        if isinstance(key, int):
            if key not in self._key_map:
                raise KeyError(f"No function key found for index {key}")
            return self._key_map[key]
        return key  # already a string

    @staticmethod
    def _folder_name(key_name: str) -> str:

        return key_name

    def _index_path(self, key_name: str) -> str:
        return os.path.join(self.base_path, self._folder_name(key_name))

    def _index_exists(self, key_name: str) -> bool:
        path = self._index_path(key_name)
        return os.path.exists(os.path.join(path, "index.faiss"))

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, key, force_reload: bool = False) -> FAISS:
        """
        Load (and cache) the FAISS index for a function key.

        Args:
            key          : int index or str key name
            force_reload : bypass cache and reload from disk

        Returns:
            FAISS vectorstore instance
        """
        key_name = self._resolve_key_name(key)

        if not force_reload and key_name in self._cache:
            logger.debug(f"[FAISS] Cache hit: {key_name}")
            return self._cache[key_name]

        index_path = self._index_path(key_name)
        if not self._index_exists(key_name):
            raise FileNotFoundError(
                f"FAISS index not found for key '{key_name}' at: {index_path}\n"
                f"Run create_dirs.py and ingest documents first."
            )

        logger.info(f"[FAISS] Loading index: {key_name} from {index_path}")
        db = FAISS.load_local(
            folder_path=index_path,
            index_name="index",
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self._cache[key_name] = db
        return db

    def get_retriever(self, key, search_kwargs: dict = None):
        """
        Return a LangChain retriever for a function key.

        Args:
            key           : int index or str key name
            search_kwargs : e.g. {"k": 4, "filter": {...}}
        """
        db = self.load(key)
        kwargs = search_kwargs or {"k": 4}
        return db.as_retriever(search_kwargs=kwargs)

    def list_available(self) -> list:
        """Return list of dicts showing which indexes exist on disk."""
        result = []
        for idx, key_name in self._key_map.items():
            exists = self._index_exists(key_name)
            result.append({
                "index":   idx,
                "key":     key_name,
                "folder":  self._folder_name(key_name),
                "exists":  exists,
                "cached":  key_name in self._cache,
            })
        return result

    def clear_cache(self, key=None):
        """Clear one or all cached indexes."""
        if key:
            self._cache.pop(self._resolve_key_name(key), None)
        else:
            self._cache.clear()
        logger.info(f"[FAISS] Cache cleared: {'all' if not key else key}")