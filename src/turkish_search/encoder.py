"""All inference loads local files only. Network access is explicit in download()."""
import hashlib
import os
from pathlib import Path
import numpy as np

MODEL_ID = "intfloat/multilingual-e5-small"


def bundle_fingerprint(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Local model missing: {path}. Run download-model while online first.")
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts and p.name != "download-record.json" and not any(part.startswith(".") for part in p.relative_to(path).parts))
    if not files:
        raise ValueError("Model directory is empty")
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode())
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def load_tokenizer(path):
    offline()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError("Install semantic dependencies: pip install -e '.[semantic]'") from exc
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


class Encoder:
    def __init__(self, path, backend="torch", onnx_file=None):
        offline()
        self.fingerprint = bundle_fingerprint(path)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ValueError("Install semantic dependencies: pip install -e '.[semantic]'") from exc
        import torch
        threads = int(os.environ.get("TR_SEARCH_THREADS", "2"))
        if threads < 1:
            raise ValueError("TR_SEARCH_THREADS must be positive")
        torch.set_num_threads(threads)
        kwargs = {"file_name": onnx_file} if onnx_file else {}
        self.model = SentenceTransformer(str(path), local_files_only=True, device="cpu",
                                         backend=backend, model_kwargs=kwargs)
        self.backend, self.onnx_file = backend, onnx_file

    def encode(self, texts, role, batch_size=32):
        if role not in {"query", "passage"}:
            raise ValueError("role must be query or passage")
        inputs = [f"{role}: {text}" for text in texts]
        for text in inputs:
            if len(self.model.tokenizer.encode(text, add_special_tokens=True)) > self.model.max_seq_length:
                raise ValueError("Input exceeds model token limit; split passage or shorten question")
        return np.asarray(self.model.encode(inputs, batch_size=batch_size,
                          normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def download(path, revision=None):
    # This is the only deliberately network-enabled model operation.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from sentence_transformers import SentenceTransformer
    path = Path(path)
    if path.exists():
        raise ValueError("Model destination already exists; choose a new directory")
    model = SentenceTransformer(MODEL_ID, revision=revision, device="cpu")
    model.save(str(path))
    return {"model": MODEL_ID, "revision_requested": revision,
            "resolved_revision": getattr(model[0].auto_model.config, "_commit_hash", None),
            "bundle_fingerprint": bundle_fingerprint(path)}


def quantize(source, destination, config="arm64"):
    import shutil
    import tempfile
    offline()
    if Path(destination).exists():
        raise ValueError("Quantized destination already exists; choose a new directory")
    from sentence_transformers import SentenceTransformer, export_dynamic_quantized_onnx_model
    # Isolate exporter side effects: the original indexed bundle must not change.
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        "*.safetensors", "pytorch_model*.bin", "onnx", ".*"))
    try:
        with tempfile.TemporaryDirectory(prefix="tr-search-export-") as scratch:
            work = Path(scratch) / "model"
            shutil.copytree(source, work, ignore=shutil.ignore_patterns("onnx", ".*"))
            model = SentenceTransformer(str(work), local_files_only=True, backend="onnx", device="cpu")
            export_dynamic_quantized_onnx_model(model, quantization_config=config,
                                              model_name_or_path=str(destination))
    except Exception:
        shutil.rmtree(destination)
        raise
    return {"bundle_fingerprint": bundle_fingerprint(destination), "config": config,
            "onnx_files": [p.relative_to(destination).as_posix() for p in Path(destination).rglob("*.onnx")]}
