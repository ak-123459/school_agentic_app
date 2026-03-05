import os
import logging
from typing import List

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from .embedding_interface import EMBInterface

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Available ONNX model filenames (in priority order within the model directory)
ONNX_MODEL_FILES = [
    "model_O4.onnx"
]


class ONNXEmbeddings(Embeddings):
    """
    LangChain-compatible wrapper around a SentenceTransformer
    loaded with an ONNX backend.
    """

    def __init__(self, model: SentenceTransformer):
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()


class Huggingface_embedders(EMBInterface):

    def __init__(self, **kwargs):
        self.model_name   = kwargs.get("model_name")
        self.model_path   = str(kwargs.get("model_path"))
        self.model_kwargs = kwargs.get("model_kwargs", {})
        self.encode_kwargs = kwargs.get("encode_kwargs", {})

        # ONNX settings
        # Pass  use_onnx=True  and (optionally)  onnx_file="model_O3.onnx"
        self.use_onnx  = kwargs.get("use_onnx", False)
        self.onnx_file = kwargs.get("onnx_file", None)   # None → auto-detect

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_onnx_file(self) -> str:
        """
        Return the full path to the ONNX file to use.

        Priority:
          1. Caller explicitly set  onnx_file  → use that.
          2. Auto-detect: walk ONNX_MODEL_FILES in order and return the
             first one that exists inside self.model_path.
          3. Raise FileNotFoundError if none found.
        """
        if self.onnx_file:
            explicit = os.path.join(self.model_path, self.onnx_file)
            if not os.path.exists(explicit):
                raise FileNotFoundError(
                    f"Specified ONNX file not found: {explicit}"
                )
            return explicit

        for candidate in ONNX_MODEL_FILES:
            full = os.path.join(self.model_path, candidate)
            if os.path.exists(full):
                logging.info(f"Auto-detected ONNX file: {full}")
                return full

        raise FileNotFoundError(
            f"No ONNX model file found in '{self.model_path}'. "
            f"Checked: {ONNX_MODEL_FILES}"
        )

    def _load_onnx_model(self) -> ONNXEmbeddings:
        """Load SentenceTransformer with ONNX backend and wrap it."""
        onnx_path = self._resolve_onnx_file()
        logging.info(f"Loading ONNX embedding model from: {onnx_path}")

        # SentenceTransformer supports backend="onnx" and model_kwargs for the
        # onnx session.  We point model_name to the directory so it picks up
        # the tokeniser/config alongside the .onnx file.
        model = SentenceTransformer(
            str(self.model_path),
            backend="onnx",
            model_kwargs={"file_name": str(os.path.basename(onnx_path))},
        )

        return ONNXEmbeddings(model)

    def _load_standard_model(self) -> HuggingFaceEmbeddings:
        """Load a normal HuggingFaceEmbeddings model."""
        logging.info("Loading standard HuggingFace embedding model...")
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs,
        )

    def _download_model(self) -> None:
        """Download model from HuggingFace Hub and save locally."""
        os.makedirs(self.model_path, exist_ok=True)
        logging.info(f"Downloading embedding model '{self.model_name}' ...")
        model = SentenceTransformer(self.model_name)
        model.save(str(self.model_path))
        logging.info(f"Model saved at: {self.model_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def load_model(self) -> HuggingFaceEmbeddings | ONNXEmbeddings:
        """
        Load the embedding model.

        Flow
        ────
        1. If the model directory does NOT exist → download from HuggingFace.
        2. If use_onnx=True  → load with ONNX backend (ONNXEmbeddings wrapper).
        3. Otherwise         → load with HuggingFaceEmbeddings (PyTorch).
        """

        model_exists = os.path.exists(self.model_path)

        if not model_exists:
            self._download_model()

        if self.use_onnx:
            return self._load_onnx_model()

        return self._load_standard_model()