"""HTTP client for the shared inference pool (thin miners)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import httpx

from alpharidge_ai import config


class InferencePoolClient:
    def __init__(self, base_url: str = None, timeout: float = None):
        self.base_url = (base_url or config.INFERENCE_POOL_URL or "").rstrip("/")
        self.timeout = float(
            timeout
            if timeout is not None
            else getattr(config, "INFERENCE_POOL_TIMEOUT", 600.0)
        )
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self):
        self._client.close()

    def health(self) -> dict:
        r = self._client.get("/health")
        r.raise_for_status()
        return r.json()

    def analyze_tweet(self, text: str) -> Optional[dict]:
        r = self._client.post("/v1/tweet", json={"text": text})
        r.raise_for_status()
        data = r.json()
        return data.get("analysis")

    def analyze_tweets_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        r = self._client.post("/v1/tweets/batch", json={"items": items})
        r.raise_for_status()
        return r.json().get("results") or []

    def analyze_telegram(
        self, messages: List[dict], asset_id: Optional[int] = None
    ) -> Optional[dict]:
        r = self._client.post(
            "/v1/telegram", json={"messages": messages, "asset_id": asset_id}
        )
        r.raise_for_status()
        return r.json().get("analysis")

    def analyze_telegram_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        r = self._client.post("/v1/telegram/batch", json={"items": items})
        r.raise_for_status()
        return r.json().get("results") or []

    def analyze_articles_batch(
        self,
        articles: List[Dict[str, Any]],
        *,
        miner_hotkey: Optional[str] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        r = self._client.post(
            "/v1/articles/batch",
            json={
                "articles": articles,
                "miner_hotkey": miner_hotkey,
            },
        )
        r.raise_for_status()
        data = r.json() or {}
        if progress is not None:
            total = len(articles)
            done = sum(
                1 for item in (data.get("results") or []) if item.get("analysis") is not None
            )
            progress(done, total)
        return data

