from .embedders_llm import Huggingface_embedders


class EMBFactory:

    @staticmethod
    def create_embedder_model_pipeline(emb_type: str, **kwargs):
        """
        Factory to create an embedding pipeline.

        Parameters
        ----------
        emb_type : str
            "huggingface"  – standard PyTorch embeddings
            "onnx"         – same class but with use_onnx=True forced on

        kwargs (common)
        ---------------
        model_name    : str   – HuggingFace model id, e.g. "BAAI/bge-large-en-v1.5"
        model_path    : str   – local directory to save / load the model
        model_kwargs  : dict  – extra kwargs forwarded to the model loader
        encode_kwargs : dict  – extra kwargs for encoding (e.g. normalize_embeddings)

        kwargs (ONNX-specific)
        ----------------------
        onnx_file : str  (optional)
            Explicit ONNX filename inside model_path, e.g. "model_O3.onnx".
            When omitted the first available file from the known list is used.

        Examples
        --------
        # Standard PyTorch
        embedder = EMBFactory.create_embedder_model_pipeline(
            "huggingface",
            model_name="BAAI/bge-large-en-v1.5",
            model_path="./models/bge",
        )

        # ONNX – auto-detect model file
        embedder = EMBFactory.create_embedder_model_pipeline(
            "onnx",
            model_name="BAAI/bge-large-en-v1.5",
            model_path="./models/bge",
        )

        # ONNX – pick a specific optimised file
        embedder = EMBFactory.create_embedder_model_pipeline(
            "onnx",
            model_name="BAAI/bge-large-en-v1.5",
            model_path="./models/bge",
            onnx_file="model_qint8_avx512.onnx",
        )

        model = await embedder.load_model()
        """

        if emb_type == "huggingface":
            return Huggingface_embedders(**kwargs)

        if emb_type == "onnx":
            # Force ONNX mode regardless of what the caller put in kwargs
            kwargs["use_onnx"] = True
            return Huggingface_embedders(**kwargs)

        raise ValueError(
            f"Unknown embedding type '{emb_type}'. "
            "Supported types: 'huggingface', 'onnx'."
        )