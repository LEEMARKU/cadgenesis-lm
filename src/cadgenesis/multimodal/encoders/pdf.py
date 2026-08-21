"""cadgenesis.multimodal.encoders.pdf
====================================
PDF encoder (technical specifications, datasheets, build manuals, drawings).

A PDF is normalised into a :class:`PDFDocument` — per-page text, per-page
geometry statistics (line / rect / curve counts) and embedded-image counts.
The text pages are summarised into a hash-based bag-of-ngrams descriptor,
the page geometry into a per-page histogram, and both are concatenated and
mapped through an MLP into the shared raw feature space.

Text extraction tries ``pypdf``/``PyPDF2`` first and falls back to a naive
page-pattern reader; geometry statistics are always computed from the raw
page objects.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.base import MultimodalEncoder

_TEXT_VOCAB = 2048
_PAGE_TEXT_DIM = 256
_PAGE_GEOM_KINDS: tuple[str, ...] = ("line", "rect", "curve", "text", "image")
_MAX_PAGES = 64
_GEOM_BINS = 8
_DESCRIPTOR_SIZE = _PAGE_TEXT_DIM + _GEOM_BINS * len(_PAGE_GEOM_KINDS) + 8


def _token_hash(token: str) -> int:
    return int.from_bytes(hashlib.sha1(token.encode("utf-8", "ignore")).digest()[:8], "big")


@dataclass
class PDFPage:
    """One page of a PDF."""

    number: int
    text: str = ""
    geometry: dict[str, int] = field(default_factory=lambda: {kind: 0 for kind in _PAGE_GEOM_KINDS})
    image_count: int = 0

    def add_geometry(self, kind: str, count: int = 1) -> None:
        if kind in self.geometry:
            self.geometry[kind] += count
        else:
            self.geometry[kind] = count


@dataclass
class PDFDocument:
    """Normalised representation of a PDF file."""

    name: str = ""
    pages: list[PDFPage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)


def extract_text_naive(raw: bytes) -> list[str]:
    """Best-effort text extraction without third-party libraries.

    Uses the ``BT``/``ET`` text-object markers and extracts any readable
    ASCII runs.  This is intentionally crude; prefer installing ``pypdf``.
    """
    try:
        text = raw.decode("utf-8", errors="ignore")
    except (UnicodeDecodeError, ValueError):
        text = ""
    pages: list[str] = []
    current: list[str] = []
    in_text = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("BT"):
            in_text = True
            continue
        if stripped.startswith("ET"):
            in_text = False
            continue
        if not in_text:
            continue
        current.extend(
            match.strip()
            for match in re.findall(r"\(([^()\\]*(?:\\.[^()\\]*)*)\)", stripped)
            if match.strip()
        )
        if ("END_PAGE" in stripped or "PAGEBREAK" in stripped) and current:
            pages.append(" ".join(current))
            current = []
    if current:
        pages.append(" ".join(current))
    return pages


def parse_pdf_file(path: str | Path) -> PDFDocument:
    """Parse a PDF on disk into a :class:`PDFDocument`.

    Uses ``pypdf``/``PyPDF2`` when available; otherwise falls back to
    :func:`extract_text_naive`.  Page geometry is always collected by
    inspecting the raw page streams.
    """
    path = Path(path)
    raw = path.read_bytes()
    document = PDFDocument(name=path.name)
    texts: list[str] = []

    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]

        reader = PdfReader(str(path))
        texts.extend(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found]

            reader = PdfReader(str(path))
            texts = [page.extract_text() or "" for page in reader.pages]
        except ImportError:
            texts = extract_text_naive(raw)

    # Page-geometry histogram from raw page streams.
    page_streams = re.split(rb"/Type\s*/Page", raw)
    for i, page_text in enumerate(texts):
        page = PDFPage(number=i, text=page_text)
        if i < len(page_streams) - 1:
            stream = page_streams[i + 1]
            page.geometry["line"] = len(re.findall(rb"/L\b|/l\b", stream))
            page.geometry["rect"] = len(re.findall(rb"/Re\b|/rect\b", stream, re.I))
            page.geometry["curve"] = len(re.findall(rb"/C\b|/curve\b", stream, re.I))
            page.geometry["text"] = len(re.findall(rb"/Tj|/TJ", stream))
            page.geometry["image"] = len(re.findall(rb"/XObject\b", stream))
            page.image_count = page.geometry["image"]
        document.pages.append(page)

    if not document.pages:
        document.pages = [PDFPage(number=0, text=" ".join(texts))]
    return document


def _page_text_descriptor(page: PDFPage) -> torch.Tensor:
    """Hash-based bag-of-ngrams descriptor of a page's text."""
    vec = torch.zeros(_PAGE_TEXT_DIM, dtype=torch.float32)
    tokens = re.findall(r"[a-zA-Z0-9]+", page.text.lower())
    for token in tokens:
        vec[_token_hash(token) % _PAGE_TEXT_DIM] += 1.0
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]}_{tokens[i + 1]}"
        vec[_token_hash(bigram) % _PAGE_TEXT_DIM] += 1.0
    norm = vec.sum()
    if norm > 0:
        vec /= norm
    return vec


def pdf_document_descriptor(document: PDFDocument) -> torch.Tensor:
    """Deterministic descriptor of a whole PDF document.

    Layout: mean page-text bag-of-ngrams (256) + geometry histogram
    (line/rect/curve/text/image x 8 percentile bins = 40) + 8 stats.
    """
    vec = torch.zeros(_DESCRIPTOR_SIZE, dtype=torch.float32)
    offset = 0

    pages = document.pages[:_MAX_PAGES]
    if pages:
        text_mean = sum(_page_text_descriptor(p) for p in pages) / len(pages)
        vec[offset : offset + _PAGE_TEXT_DIM] = text_mean
    offset += _PAGE_TEXT_DIM

    geom_values: dict[str, list[int]] = {k: [] for k in _PAGE_GEOM_KINDS}
    for page in pages:
        for kind in _PAGE_GEOM_KINDS:
            geom_values[kind].append(page.geometry.get(kind, 0))
    for i, kind in enumerate(_PAGE_GEOM_KINDS):
        values = sorted(geom_values[kind])
        if not values:
            continue
        for bin_index in range(_GEOM_BINS):
            idx = min(bin_index * len(values) // _GEOM_BINS, len(values) - 1)
            vec[offset + i * _GEOM_BINS + bin_index] = math.log1p(values[idx])

    stats_offset = offset + _GEOM_BINS * len(_PAGE_GEOM_KINDS)
    vec[stats_offset + 0] = math.log1p(len(document.pages))
    vec[stats_offset + 1] = sum(math.log1p(sum(page.geometry.values())) for page in pages)
    vec[stats_offset + 2] = sum(p.image_count for p in pages)
    vec[stats_offset + 3] = sum(len(p.text.split()) for p in pages)
    vec[stats_offset + 4] = 1.0 if document.metadata.get("title") else 0.0
    vec[stats_offset + 5] = 1.0 if document.metadata.get("author") else 0.0
    vec[stats_offset + 6] = len(pages)
    vec[stats_offset + 7] = 1.0 if document.name else 0.0
    return vec


class PDFEncoder(MultimodalEncoder):
    """Encoder for the ``pdf`` modality."""

    modality: ClassVar[Modality] = Modality.PDF

    def __init__(
        self,
        feature_dim: int = 384,
        hidden_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(feature_dim=feature_dim)
        self.net = nn.Sequential(
            nn.Linear(_DESCRIPTOR_SIZE, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[-1] != _DESCRIPTOR_SIZE:
            raise ValueError(
                f"pdf encoder expects (B, {_DESCRIPTOR_SIZE}) descriptors; got {tuple(x.shape)}"
            )
        return self.net(x)

    def encode(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return self.forward(inputs)
        if isinstance(inputs, (str, Path)):
            inputs = [parse_pdf_file(inputs)]
        elif isinstance(inputs, PDFDocument):
            inputs = [inputs]
        items = list(inputs)
        if not items:
            raise ValueError("cannot encode an empty PDF batch")
        descriptors = torch.stack([pdf_document_descriptor(d) for d in items])
        return self.forward(descriptors)


__all__ = [
    "PDFDocument",
    "PDFEncoder",
    "PDFPage",
    "extract_text_naive",
    "parse_pdf_file",
    "pdf_document_descriptor",
]
