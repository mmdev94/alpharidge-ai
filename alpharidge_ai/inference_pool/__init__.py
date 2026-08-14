"""Shared inference pool: one process loads heavy analyzers; thin miners HTTP in."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import urlparse

# Determinism / meta-init before model loads
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from alpharidge_ai.analyzer.aspect_sentiment import install_meta_init_guard

install_meta_init_guard()

from alpharidge_ai.analyzer import (
    setup_analyzer,
    setup_article_intelligence_analyzer,
    setup_news_analyzer,
    setup_telegram_analyzer,
)
from alpharidge_ai.inference_pool.mapping import intel_to_analysis_dict, triage_only_analysis_dict

log = logging.getLogger("inference_pool")


class InferenceEngine:
    """Owns analyzers once; serves concurrent jobs via a thread pool."""

    def __init__(self, workers: int = 4):
        self.workers = max(1, int(workers))
        self._executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="pool")
        log.info("loading analyzers workers=%s", self.workers)
        t0 = time.time()
        self.tweet_analyzer = setup_analyzer()
        self.telegram_analyzer = setup_telegram_analyzer()
        self.news_analyzer = setup_news_analyzer()

        # V2 + triage are mandatory (subnet 3.5.0); fail startup rather than
        # silently falling back.
        try:
            self.article_intel = setup_article_intelligence_analyzer()
            log.info("article intel ready")
        except Exception as e:
            raise RuntimeError(
                "ArticleIntelligence (V2) analyzer failed to initialize; "
                f"triage requires V2. Fix analyzer setup and restart: {e}"
            ) from e

        try:
            from alpharidge_ai.analyzer.asset_extractor import AssetExtractor
            from alpharidge_ai.analyzer.triage_stage import TriageStage

            self.triage_stage = TriageStage(AssetExtractor())
            log.info("triage ready")
        except Exception as e:
            raise RuntimeError(
                "Triage stage failed to initialize (asset gazetteer or language "
                f"detector). Fix the named component and restart: {e}"
            ) from e

        log.info("analyzers ready in %.1fs", time.time() - t0)

    def submit(self, fn, *args, **kwargs):
        return self._executor.submit(fn, *args, **kwargs)

    def analyze_tweet(self, text: str) -> Optional[dict]:
        if not text:
            return None
        c = self.tweet_analyzer.classify_post(text)
        return c.to_dict() if c is not None else None

    def analyze_telegram(self, messages: List[dict], asset_id: Optional[int] = None) -> Optional[dict]:
        c = self.telegram_analyzer.classify_message_group(messages, asset_id=asset_id)
        return c.to_dict() if c is not None else None

    def analyze_article(self, payload: dict) -> Optional[dict]:
        """Match neurons/miner.py 3.5.0 article path (mandatory triage)."""
        title = payload.get("title") or ""
        content = payload.get("content") or ""

        triage_rec, proof, _ = self.triage_stage.evaluate(title, content)

        # Titleless / irrelevant → claim + proof only (keeps batch complete).
        if not title or triage_rec.get("label") == "irrelevant":
            return triage_only_analysis_dict(triage_rec, proof)

        # Relevant and borderline get full analysis; borderline is flagged after.
        intel = self.article_intel.analyze(
            article_id=payload.get("article_id"),
            url=payload.get("url"),
            title=title,
            source=payload.get("source"),
            published=payload.get("published"),
            summary=payload.get("summary"),
            content=content,
            raw_html=payload.get("raw_html"),
            miner_hotkey=payload.get("miner_hotkey"),
        )
        if intel is None:
            return triage_only_analysis_dict(triage_rec, proof)

        return intel_to_analysis_dict(intel, triage_rec=triage_rec, proof=proof)


_ENGINE: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    global _ENGINE
    if _ENGINE is None:
        raise RuntimeError("engine not initialized")
    return _ENGINE


def _json_response(handler: BaseHTTPRequestHandler, code: int, body: dict):
    data = json.dumps(body, default=str).encode("utf-8")
    try:
        handler.send_response(code)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
        # Thin miner timed out / restarted while we were still analyzing.
        # Work is done; don't escalate into a secondary 500 write attempt.
        log.warning("client disconnected during response: %s", e)


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


class PoolHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silence default access log
        return

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/health", "/"):
            eng = get_engine()
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "workers": eng.workers,
                    "article_intel": eng.article_intel is not None,
                    "triage": eng.triage_stage is not None,
                },
            )
            return
        _json_response(self, 404, {"ok": False, "error": "not_found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = _read_json(self)
        except Exception as e:
            _json_response(self, 400, {"ok": False, "error": f"bad_json:{e}"})
            return
        try:
            if path == "/v1/tweet":
                self._tweet(body)
            elif path == "/v1/tweets/batch":
                self._tweets_batch(body)
            elif path == "/v1/telegram":
                self._telegram(body)
            elif path == "/v1/telegram/batch":
                self._telegram_batch(body)
            elif path == "/v1/article":
                self._article(body)
            elif path == "/v1/articles/batch":
                self._articles_batch(body)
            else:
                _json_response(self, 404, {"ok": False, "error": "not_found"})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            log.warning("client disconnected path=%s: %s", path, e)
        except Exception as e:
            log.error("handler error path=%s: %s", path, e)
            _json_response(self, 500, {"ok": False, "error": str(e)})

    def _tweet(self, body: dict):
        eng = get_engine()
        t0 = time.time()
        fut = eng.submit(eng.analyze_tweet, body.get("text") or "")
        analysis = fut.result()
        ms = int((time.time() - t0) * 1000)
        log.info("tweet ok=%s ms=%s", analysis is not None, ms)
        _json_response(self, 200, {"ok": analysis is not None, "analysis": analysis, "ms": ms})

    def _tweets_batch(self, body: dict):
        eng = get_engine()
        items = body.get("items") or []
        t0 = time.time()
        futs = {
            eng.submit(eng.analyze_tweet, (it.get("text") or "")): it
            for it in items
        }
        results = []
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                analysis = fut.result()
            except Exception as e:
                log.error("tweet job failed: %s", e)
                analysis = None
            results.append({"id": it.get("id"), "analysis": analysis})
        by_id = {r["id"]: r for r in results}
        ordered = [
            {"id": it.get("id"), "analysis": (by_id.get(it.get("id")) or {}).get("analysis")}
            for it in items
        ]
        ms = int((time.time() - t0) * 1000)
        ok_n = sum(1 for r in ordered if r.get("analysis") is not None)
        log.info("tweets_batch n=%s ok=%s ms=%s", len(items), ok_n, ms)
        _json_response(self, 200, {"ok": True, "results": ordered, "ms": ms})

    def _telegram(self, body: dict):
        eng = get_engine()
        t0 = time.time()
        fut = eng.submit(
            eng.analyze_telegram,
            body.get("messages") or [],
            body.get("asset_id"),
        )
        analysis = fut.result()
        ms = int((time.time() - t0) * 1000)
        log.info("telegram ok=%s ms=%s", analysis is not None, ms)
        _json_response(self, 200, {"ok": analysis is not None, "analysis": analysis, "ms": ms})

    def _telegram_batch(self, body: dict):
        eng = get_engine()
        items = body.get("items") or []
        t0 = time.time()
        futs = {}
        for it in items:
            fut = eng.submit(
                eng.analyze_telegram,
                it.get("messages") or [],
                it.get("asset_id"),
            )
            futs[fut] = it
        results = []
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                analysis = fut.result()
            except Exception as e:
                log.error("telegram job failed: %s", e)
                analysis = None
            results.append({"id": it.get("id"), "analysis": analysis})
        by_id = {r["id"]: r for r in results}
        ordered = [
            {"id": it.get("id"), "analysis": (by_id.get(it.get("id")) or {}).get("analysis")}
            for it in items
        ]
        ms = int((time.time() - t0) * 1000)
        ok_n = sum(1 for r in ordered if r.get("analysis") is not None)
        log.info("telegram_batch n=%s ok=%s ms=%s", len(items), ok_n, ms)
        _json_response(self, 200, {"ok": True, "results": ordered, "ms": ms})

    def _article(self, body: dict):
        eng = get_engine()
        t0 = time.time()
        fut = eng.submit(eng.analyze_article, body)
        analysis = fut.result()
        ms = int((time.time() - t0) * 1000)
        aid = body.get("article_id")
        log.info("article id=%s ok=%s ms=%s", aid, analysis is not None, ms)
        _json_response(self, 200, {"ok": analysis is not None, "analysis": analysis, "ms": ms})

    def _articles_batch(self, body: dict):
        eng = get_engine()
        articles = body.get("articles") or []
        miner_hotkey = body.get("miner_hotkey")
        t0 = time.time()
        futs = {}
        for art in articles:
            payload = dict(art)
            payload.setdefault("miner_hotkey", miner_hotkey)
            futs[eng.submit(eng.analyze_article, payload)] = art
        results = []
        for fut in as_completed(futs):
            art = futs[fut]
            try:
                analysis = fut.result()
            except Exception as e:
                log.error("article job id=%s failed: %s", art.get("article_id"), e)
                analysis = None
            results.append({"id": art.get("article_id"), "analysis": analysis})
        by_id = {r["id"]: r for r in results}
        ordered = [
            {
                "id": a.get("article_id"),
                "analysis": (by_id.get(a.get("article_id")) or {}).get("analysis"),
            }
            for a in articles
        ]
        ms = int((time.time() - t0) * 1000)
        ok_n = sum(1 for r in ordered if r.get("analysis") is not None)
        log.info("articles_batch n=%s ok=%s ms=%s", len(articles), ok_n, ms)
        _json_response(self, 200, {"ok": True, "results": ordered, "ms": ms})


def _quiet_third_party_loggers():
    for name in (
        "urllib3",
        "httpx",
        "openai",
        "httpcore",
        "transformers",
        "torch",
        "sentence_transformers",
        "flair",
        "ner_fusion",
        "refined",
        "spacy",
        "asyncio",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def run_server(host: str = "127.0.0.1", port: int = 30000, workers: int = 4):
    global _ENGINE
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _quiet_third_party_loggers()
    _ENGINE = InferenceEngine(workers=workers)
    httpd = ThreadingHTTPServer((host, port), PoolHandler)
    log.info("listening on http://%s:%s", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        httpd.server_close()
        _ENGINE._executor.shutdown(wait=False, cancel_futures=True)
