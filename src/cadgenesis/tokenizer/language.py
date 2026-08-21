"""
cadgenesis.tokenizer.language
==============================
Language (text) tokenizer for the Autonomous CAD Tokenizer.

Purpose
-------
Handles the text / natural-language side of the hybrid tokenizer.  Provides:

1. **LegacyWordTokenizer** — backward-compatible drop-in replacement for
   data.py's ``LangTokenizer``.  Word-level, minimal, builds vocab from corpus.
   Used for unit tests and the existing Colab notebook.

2. **BPETokenizer** — production-grade subword tokenizer wrapper.  Designed
   to wrap any HuggingFace ``tokenizers`` library tokenizer (e.g. trained
   BPE, WordPiece, or SentencePiece) and integrate its vocabulary into the
   ``CADVocabulary`` LANGUAGE family slots.

The public API is intentional so both tokenizers are interchangeable via
duck typing — both implement ``encode(text) → List[int]`` and
``decode(ids) → str``.

Architecture
------------
::

    LanguageTokenizerBase  (ABC)
    ├── LegacyWordTokenizer   ← existing data.py LangTokenizer, upgraded
    └── BPETokenizer          ← HuggingFace tokenizers wrapper

Interfaces
----------
    tokenizer.encode(text: str) → List[int]
    tokenizer.decode(ids: List[int]) → str
    tokenizer.vocab_size → int
    tokenizer.save(path)
    tokenizer.load(path)

Algorithms
----------
    LegacyWordTokenizer: regex word splitting, O(n) per text
    BPETokenizer:        HuggingFace byte-pair encoding, O(n) amortized

Complexity
----------
    Both: O(n) where n = number of characters in the input text.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LanguageTokenizerBase(ABC):
    """Duck-type interface for all language tokenizer implementations."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode a text string into a list of integer token ids."""
        ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode a list of integer token ids back to a string."""
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Size of the language vocabulary."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the tokenizer to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> LanguageTokenizerBase:
        """Load a tokenizer from disk."""
        ...


# ---------------------------------------------------------------------------
# LegacyWordTokenizer — backward compatible with data.py's LangTokenizer
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z]+|\d+\.\d+|\d+|[.,]")


class LegacyWordTokenizer(LanguageTokenizerBase):
    """
    Word-level tokenizer.  Backward-compatible with the original
    ``LangTokenizer`` in ``data.py``.

    Improvements over the original:
      - Inherits from ``LanguageTokenizerBase`` (duck-typed interface)
      - ``decode()`` method (not in original)
      - ``save()`` / ``load()`` methods
      - ``vocab_size`` property
      - Thread-safe (vocab building is idempotent after initial build)
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self) -> None:
        self.tok2id: dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }
        self.id2tok: dict[int, str] = {v: k for k, v in self.tok2id.items()}

    def build_vocab(self, texts: list[str]) -> None:
        """Build vocabulary from a list of text strings (idempotent)."""
        word_set: set = set()
        for text in texts:
            word_set.update(_WORD_RE.findall(text.lower()))
        for word in sorted(word_set):
            if word not in self.tok2id:
                idx = len(self.tok2id)
                self.tok2id[word] = idx
                self.id2tok[idx] = word

    def encode(self, text: str) -> list[int]:
        return [
            self.tok2id.get(w, self.tok2id[self.UNK_TOKEN]) for w in _WORD_RE.findall(text.lower())
        ]

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id2tok.get(i, self.UNK_TOKEN) for i in ids]
        return " ".join(t for t in tokens if t not in (self.PAD_TOKEN,))

    @property
    def vocab_size(self) -> int:
        return len(self.tok2id)

    def __len__(self) -> int:  # legacy compat: len(lang_tok)
        return self.vocab_size

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                {"type": "LegacyWordTokenizer", "tok2id": self.tok2id},
                fh,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> LegacyWordTokenizer:
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        tok = cls()
        tok.tok2id = data["tok2id"]
        tok.id2tok = {int(v): k for k, v in tok.tok2id.items()}
        return tok


# ---------------------------------------------------------------------------
# BPETokenizer — HuggingFace tokenizers wrapper
# ---------------------------------------------------------------------------


class BPETokenizer(LanguageTokenizerBase):
    """
    Production BPE tokenizer wrapping the HuggingFace ``tokenizers`` library.

    Supports:
      - Loading any pretrained HF tokenizer (BERT, GPT-2, LLaMA, etc.)
      - Training a fresh BPE tokenizer from a corpus of CAD design texts
      - Integrating the BPE vocabulary into CADVocabulary's LANGUAGE slots

    Usage::

        # Option A: Load a pretrained HF tokenizer
        tok = BPETokenizer.from_pretrained("gpt2")

        # Option B: Train from scratch on a CAD corpus
        tok = BPETokenizer.train_from_corpus(
            texts=cad_texts,
            vocab_size=32_000,
            save_path="outputs/cad_bpe"
        )

        ids = tok.encode("Create a box 50mm wide")
        text = tok.decode(ids)

    Note
    ----
    The ``tokenizers`` and ``transformers`` packages are optional at import
    time.  An ``ImportError`` with installation instructions is raised only
    when a ``BPETokenizer`` is actually instantiated, not at module load.
    This keeps the rest of the tokenizer package usable without GPU
    dependencies in lightweight environments.
    """

    def __init__(self, hf_tokenizer) -> None:
        """
        Parameters
        ----------
        hf_tokenizer : tokenizers.Tokenizer or transformers.PreTrainedTokenizerFast
            An already-initialized HuggingFace tokenizer object.
        """
        self._tok = hf_tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> BPETokenizer:
        """
        Load a pretrained HuggingFace tokenizer.

        Parameters
        ----------
        model_name_or_path : str
            HF Hub model name (e.g. ``"gpt2"``) or local path to a saved
            tokenizer directory.
        """
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "BPETokenizer requires the 'transformers' package. "
                "Install with: pip install transformers"
            ) from exc
        hf_tok = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(hf_tok)

    @classmethod
    def train_from_corpus(
        cls,
        texts: list[str],
        vocab_size: int = 32_000,
        save_path: str | None = None,
        special_tokens: list[str] | None = None,
    ) -> BPETokenizer:
        """
        Train a fresh BPE tokenizer from a corpus of texts.

        Parameters
        ----------
        texts : List[str]
            Training corpus (one document per string).
        vocab_size : int
            Target vocabulary size.
        save_path : str, optional
            If given, save the trained tokenizer here.
        special_tokens : List[str], optional
            Extra special tokens to add (beyond the defaults).
        """
        try:
            from tokenizers import (  # type: ignore
                Tokenizer,
                models,
                pre_tokenizers,
                trainers,
            )
        except ImportError as exc:
            raise ImportError(
                "BPETokenizer.train_from_corpus requires the 'tokenizers' package. "
                "Install with: pip install tokenizers"
            ) from exc

        _defaults = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<mask>"]
        _specials = _defaults + (special_tokens or [])

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=_specials,
            min_frequency=2,
            show_progress=False,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)

        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(Path(save_path) / "tokenizer.json"))

        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        out = self._tok.encode(text)
        # HF PreTrainedTokenizer returns a different object than Tokenizer
        if hasattr(out, "ids"):
            return out.ids
        if hasattr(out, "input_ids"):
            return out.input_ids
        return list(out)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

    @property
    def vocab_size(self) -> int:
        if hasattr(self._tok, "vocab_size"):
            return self._tok.vocab_size
        if hasattr(self._tok, "get_vocab_size"):
            return self._tok.get_vocab_size()
        return len(self._tok.get_vocab())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self._tok, "save_pretrained"):
            self._tok.save_pretrained(str(path))
        elif hasattr(self._tok, "save"):
            self._tok.save(str(path / "tokenizer.json"))

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        return cls.from_pretrained(str(path))
