"""CPU inference for the same encoder and reranker used by the local app."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from langchain_core.embeddings import Embeddings
from tokenizers import Tokenizer

from backend.retrieval import RERANKER, RERANKER_REVISION


class OnnxEncoder:
    def __init__(self, model: Path, tokenizer: Path, max_length: int, threads: int):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=max_length)
        self.inputs = {entry.name for entry in self.session.get_inputs()}

    def encode(self, texts):
        rows = self.tokenizer.encode_batch(texts)
        arrays = {
            "input_ids": np.asarray([e.ids for e in rows], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask for e in rows], dtype=np.int64),
            "token_type_ids": np.asarray([e.type_ids for e in rows], dtype=np.int64),
        }
        result = self.session.run(None, {key: arrays[key] for key in self.inputs})[0]
        return result, arrays["attention_mask"]


class MiniLM(Embeddings):
    def __init__(self, directory: Path, threads: int = 2):
        self.encoder = OnnxEncoder(directory / "minilm.onnx", directory / "tokenizer.json", 256, threads)

    def embed_documents(self, texts):
        vectors = []
        for offset in range(0, len(texts), 32):
            batch = [text.replace("\n", " ") for text in texts[offset:offset + 32]]
            states, mask = self.encoder.encode(batch)
            weights = mask[:, :, None]
            pooled = (states * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1)
            vectors.extend((pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)).tolist())
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class OnnxReranker:
    def __init__(self, threads: int):
        from huggingface_hub import hf_hub_download

        def file(name):
            return Path(hf_hub_download(RERANKER, name, revision=RERANKER_REVISION, token=False))

        self.encoder = OnnxEncoder(file("onnx/model.onnx"), file("tokenizer.json"), 384, threads)

    def predict(self, pairs, *, batch_size=8, show_progress_bar=False):
        return [
            float(score)
            for offset in range(0, len(pairs), batch_size)
            for score in self.encoder.encode(list(pairs[offset:offset + batch_size]))[0].reshape(-1)
        ]
