"""
embedders.py — Embedding wrappers for 3 model groups
Group A: Instruction-tuned  (e.g. e5-instruct, Qwen3, NV-Embed)
Group B: Sentence Transformer (e.g. BGE-M3, SBERT, GTE)
Group C: Base Contextual BERT (e.g. WangchanBERTa, XLM-R, mBERT)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import DEVICE, DTYPE, BATCH_SIZE, DEFAULT_TEMPLATES, free_memory

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════

@dataclass
class EmbedConfig:
    """ตั้งค่าการ encode สำหรับแต่ละ model group"""
    group:        str
    templates:    list[str] = field(default_factory=list)
    instruction:  str  = ""
    query_prefix: str  = ""
    normalize:    bool = True
    batch_size:   int  = field(default_factory=lambda: BATCH_SIZE)
    # float16 บน GPU: ลด VRAM ครึ่งหนึ่ง แต่ยังได้ cosine แม่นยำ
    use_fp16:     bool = field(default_factory=lambda: DEVICE == "cuda")


# ══════════════════════════════════════════════════════════════
# Base
# ══════════════════════════════════════════════════════════════

class EmbedderBase(ABC):
    """
    Abstract base — ทุก subclass ต้อง implement _prepare() เท่านั้น
    encode(), _aggregate(), free() ใช้ร่วมกัน
    """

    def __init__(self, model_name: str, config: EmbedConfig) -> None:
        self.model_name = model_name
        self.config     = config
        log.info("Loading %s on %s", model_name, DEVICE)

        # เพิ่มใน __init__ ของ EmbedderBase
        if "gte-multilingual" in model_name:
            # force CPU เพื่อหลีกเลี่ยง CUDA assert
            device_override = "cpu"
        else:
            device_override = DEVICE

        self.model = SentenceTransformer(
            model_name,
            device            = device_override,
            trust_remote_code = True,
        )
        
        # cast weights เป็น fp16 บน GPU เพื่อลด VRAM
        if config.use_fp16 and DEVICE == "cuda":
            self.model = self.model.half()

    # ── public ────────────────────────────────────────────────

    def encode(self, words: list[str]) -> np.ndarray:
        """
        Encode word list → L2-normalized float32 ndarray (n, dim)
        float32 เสมอ เพราะ metric คำนวณบน CPU ด้วย numpy
        """
        sentences = self._prepare(words)
        with torch.inference_mode():
            vecs = self.model.encode(
                sentences,
                normalize_embeddings = self.config.normalize,
                batch_size           = self.config.batch_size,
                show_progress_bar    = False,
                convert_to_numpy     = True,
            ).astype(np.float32)   # ← always float32 สำหรับ metric
        return self._aggregate(vecs, n_words=len(words))

    def free(self) -> None:
        """ลบ model weights ออกจาก RAM/VRAM ทันที"""
        try:
            del self.model
        except AttributeError:
            pass
        free_memory()
        log.info("Freed %s", self.model_name)

    # ── abstract ──────────────────────────────────────────────

    @abstractmethod
    def _prepare(self, words: list[str]) -> list[str]:
        """แปลง word list → sentence list ตาม format ของแต่ละ group"""
        ...

    # ── private ───────────────────────────────────────────────

    def _aggregate(self, vecs: np.ndarray, n_words: int) -> np.ndarray:
        """
        Mean-pool vectors ข้าม templates แล้ว re-normalize
        ถ้ามีแค่ 1 sentence ต่อคำ → return ตรงๆ
        """
        n_templates = len(vecs) // n_words
        if n_templates == 1:
            return vecs

        # (n_words, n_templates, dim) → mean → (n_words, dim)
        pooled = vecs.reshape(n_words, n_templates, -1).mean(axis=1)

        if self.config.normalize:
            norms  = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.clip(norms, 1e-10, None)

        return pooled

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.model_name}', group={self.config.group!r})"


# ══════════════════════════════════════════════════════════════
# Concrete subclasses
# ══════════════════════════════════════════════════════════════

class InstructionEmbedder(EmbedderBase):
    """
    Group A — ใส่ instruction prefix ก่อนทุกคำ
    ไม่ใส่ → vector generic → bias score ต่ำผิดปกติ
    """
    def _prepare(self, words: list[str]) -> list[str]:
        instr = self.config.instruction
        return [f"Instruct: {instr}\nQuery: {w}" for w in words]


class SentenceEmbedder(EmbedderBase):
    """
    Group B — query prefix หรือ bare word
    E5 non-instruct ต้องการ 'query: ' prefix
    BGE/SBERT/GTE ใช้ bare ได้
    """
    def _prepare(self, words: list[str]) -> list[str]:
        prefix = self.config.query_prefix
        return [f"{prefix}{w}" for w in words]


class BERTEmbedder(EmbedderBase):
    """
    Group C — หลาย template แล้ว mean-pool
    วาง templates ก่อน words เพื่อให้ _aggregate reshape ถูก
    order: word1_t1, word1_t2, ..., word2_t1, word2_t2, ...
    """
    def _prepare(self, words: list[str]) -> list[str]:
        templates = self.config.templates or DEFAULT_TEMPLATES
        return [tmpl.format(w) for w in words for tmpl in templates]


# ══════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════

_I = InstructionEmbedder
_S = SentenceEmbedder
_B = BERTEmbedder

MODEL_REGISTRY: dict[str, dict] = {
    # ── Group A — Instruction-tuned ──────────────────────────
    "intfloat/multilingual-e5-large-instruct": {
        "cls": _I, "short": "e5-large-instruct",
        "cfg": EmbedConfig(group="A", instruction="จงหาความสัมพันธ์ทางความหมายของคำ"),
    },
    "Qwen/Qwen3-Embedding-8B": {
        "cls": _I, "short": "Qwen3-8B",
        "cfg": EmbedConfig(group="A", instruction="Retrieve semantically similar text"),
    },
    "Qwen/Qwen3-Embedding-4B": {
        "cls": _I, "short": "Qwen3-4B",
        "cfg": EmbedConfig(group="A", instruction="Retrieve semantically similar text"),
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "cls": _I, "short": "Qwen3-0.6B",
        "cfg": EmbedConfig(group="A", instruction="Retrieve semantically similar text"),
    },
    "jinaai/jina-embeddings-v3": {
        "cls": _I, "short": "jina-v3",
        "cfg": EmbedConfig(group="A", instruction="Retrieve semantically similar words"),
    },
    "nvidia/NV-Embed-v2": {
        "cls": _I, "short": "NV-Embed-v2",
        "cfg": EmbedConfig(group="A", instruction="Given a word, retrieve its semantic meaning"),
    },
    "nomic-ai/nomic-embed-text-v2-moe": {
        "cls": _I, "short": "nomic-v2-moe",
        "cfg": EmbedConfig(group="A", instruction="search_query"),
    },
    # ── Group B — Sentence Transformer ───────────────────────
    "intfloat/multilingual-e5-large": {
        "cls": _S, "short": "e5-large",
        "cfg": EmbedConfig(group="B", query_prefix="query: "),
    },
    "intfloat/multilingual-e5-base": {
        "cls": _S, "short": "e5-base",
        "cfg": EmbedConfig(group="B", query_prefix="query: "),
    },
    "intfloat/multilingual-e5-small": {
        "cls": _S, "short": "e5-small",
        "cfg": EmbedConfig(group="B", query_prefix="query: "),
    },
    "BAAI/bge-m3": {
        "cls": _S, "short": "BGE-M3",
        "cfg": EmbedConfig(group="B"),
    },
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "cls": _S, "short": "mpnet-base",
        "cfg": EmbedConfig(group="B"),
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "cls": _S, "short": "MiniLM-L12",
        "cfg": EmbedConfig(group="B"),
    },
    "Alibaba-NLP/gte-multilingual-base": {
        "cls": _S, "short": "GTE-base",
        "cfg": EmbedConfig(group="B"),
    },
    # ── Group C — Base Contextual BERT ───────────────────────
    "sentence-transformers/LaBSE": {
        "cls": _B, "short": "LaBSE",
        "cfg": EmbedConfig(group="C", templates=DEFAULT_TEMPLATES),
    },
    "FacebookAI/xlm-roberta-large": {
        "cls": _B, "short": "XLM-R-large",
        "cfg": EmbedConfig(group="C", templates=DEFAULT_TEMPLATES),
    },
    "FacebookAI/xlm-roberta-base": {
        "cls": _B, "short": "XLM-R-base",
        "cfg": EmbedConfig(group="C", templates=DEFAULT_TEMPLATES),
    },
    "google-bert/bert-base-multilingual-cased": {
        "cls": _B, "short": "mBERT",
        "cfg": EmbedConfig(group="C", templates=DEFAULT_TEMPLATES),
    },
    "ibm-granite/granite-embedding-97m-multilingual-r2": {
        "cls": _B, "short": "granite-97m",
        "cfg": EmbedConfig(group="C", templates=DEFAULT_TEMPLATES),
    },
    "clicknext/phayathaibert": {
        "cls": _B, "short": "PhayaThaiBERT",
        "cfg": EmbedConfig(group="C", templates=DEFAULT_TEMPLATES),
    },
    "airesearch/wangchanberta-base-att-spm-uncased": {
        "cls": _B, "short": "WangchanBERTa",
        "cfg": EmbedConfig(group="C", templates=DEFAULT_TEMPLATES),
    },
    "mrpeerat/simcse-xlm-roberta-base": {
        "cls": _B, "short": "SimCSE-Thai",
        "cfg": EmbedConfig(group="C", templates=DEFAULT_TEMPLATES),
    },
}


# ══════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════

class EmbedderFactory:
    """
    สร้าง embedder จาก model_name
    ถ้า model ไม่อยู่ใน registry → raise ValueError พร้อม list ที่มี
    """

    @staticmethod
    def create(
        model_name: str,
        config_override: Optional[EmbedConfig] = None,
    ) -> EmbedderBase:
        if model_name not in MODEL_REGISTRY:
            available = "\n  ".join(MODEL_REGISTRY)
            raise ValueError(
                f"Unknown model: '{model_name}'\n"
                f"Available:\n  {available}"
            )
        entry  = MODEL_REGISTRY[model_name]
        config = config_override or entry["cfg"]
        return entry["cls"](model_name, config)

    @staticmethod
    def list_models(group: Optional[str] = None) -> list[str]:
        if group is None:
            return list(MODEL_REGISTRY)
        return [n for n, e in MODEL_REGISTRY.items() if e["cfg"].group == group]

    @staticmethod
    def get_short(model_name: str) -> str:
        return MODEL_REGISTRY[model_name]["short"]

    @staticmethod
    def get_group(model_name: str) -> str:
        return MODEL_REGISTRY[model_name]["cfg"].group
