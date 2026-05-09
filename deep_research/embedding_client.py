#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BASE_URL = "https://aimpapi.midea.com/t-aigc/aimp-text-embedding/v1"
_DEFAULT_MODEL = "Qwen3-Embedding-4B"
_DEFAULT_TIMEOUT = 120.0


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"), override=False)
    except Exception:
        pass


def _parse_optional_int(value: str) -> Optional[int]:
    text = (value or "").strip()
    if not text:
        return None
    return int(text)


def get_embedding_settings() -> Dict[str, object]:
    _load_env()
    return {
        "base_url": (os.getenv("EMBED_BASE_URL", "").strip() or _DEFAULT_BASE_URL).rstrip("/"),
        "api_key": os.getenv("EMBED_API_KEY", "").strip() or os.getenv("QWEN_API_KEY", "").strip(),
        "model": os.getenv("EMBED_MODEL", "").strip() or _DEFAULT_MODEL,
        "user": os.getenv("MIDEA_AIGC_USER", "").strip(),
        "dimensions": _parse_optional_int(os.getenv("EMBED_DIMENSIONS", "")),
        "timeout": float(os.getenv("EMBED_TIMEOUT", str(_DEFAULT_TIMEOUT)).strip() or _DEFAULT_TIMEOUT),
    }


class OnlineEmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        user: str,
        dimensions: Optional[int] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not api_key:
            raise ValueError("EMBED_API_KEY 未设置，无法调用在线 embedding 服务。")
        if not user:
            raise ValueError("MIDEA_AIGC_USER 未设置，无法调用在线 embedding 服务。")
        if not base_url:
            raise ValueError("EMBED_BASE_URL 未设置，无法调用在线 embedding 服务。")
        if not model:
            raise ValueError("EMBED_MODEL 未设置，无法调用在线 embedding 服务。")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            default_headers={"AIGC-USER": user},
            timeout=timeout,
            max_retries=2,
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        clean_texts = [text if isinstance(text, str) else str(text) for text in texts]
        request_args = {"model": self.model, "input": clean_texts}
        if self.dimensions is not None:
            request_args["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**request_args)
        data = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        vectors = self._embed([text])
        return vectors[0] if vectors else []


def build_embedding_client(
    *,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> OnlineEmbeddingClient:
    settings = get_embedding_settings()
    return OnlineEmbeddingClient(
        base_url=str(settings["base_url"]),
        api_key=str(settings["api_key"]),
        model=model or str(settings["model"]),
        user=str(settings["user"]),
        dimensions=dimensions if dimensions is not None else settings["dimensions"],
        timeout=float(settings["timeout"]),
    )


def embed_documents(texts: List[str], *, model: Optional[str] = None, dimensions: Optional[int] = None) -> List[List[float]]:
    client = build_embedding_client(model=model, dimensions=dimensions)
    return client.embed_documents(texts)


def embed_query(text: str, *, model: Optional[str] = None, dimensions: Optional[int] = None) -> List[float]:
    client = build_embedding_client(model=model, dimensions=dimensions)
    return client.embed_query(text)