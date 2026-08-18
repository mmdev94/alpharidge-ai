# Baked-in launch env (set BEFORE bittensor/torch import so operators don't have to set them):
#   BT_NO_PARSE_CLI_ARGS   — bittensor 10.4 ignores CLI args (--netuid/--wallet) without this
#   CUBLAS_WORKSPACE_CONFIG — deterministic cuBLAS for cross-host consensus parity (must precede CUDA init)
import os
os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import time
import typing
import threading
import copy
import asyncio
import logging
from datetime import datetime, timezone
import bittensor as bt

# Bittensor Miner Template:
import alpharidge_ai

# import base miner class which takes care of most of the boilerplate
from alpharidge_ai.base.miner import BaseMinerNeuron
from alpharidge_ai import config as ar_config
from alpharidge_ai.utils.api_models import TweetAnalysisBase, TelegramMessageAnalysis, NewsArticleAnalysisBase

_THIN = bool(getattr(ar_config, "INFERENCE_POOL_URL", "") or "")
_ALLOW_THIN = ("[ARTICLE]",)


def _simple_reason(err, fallback: str = "unknown") -> str:
    """One-line failure reason for PM2 logs (no newlines / huge traces)."""
    if err is None:
        return fallback
    if isinstance(err, BaseException):
        text = f"{type(err).__name__}: {err}"
    else:
        text = str(err).strip() or fallback
    text = " ".join(text.split())
    if len(text) > 180:
        text = text[:177] + "..."
    return text


def _article_log(event: str, **fields) -> None:
    """Emit concise article lifecycle logs regardless of Bittensor log level."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Keep failure reasons first and always present for *failed* / *timeout* events.
    if "failed" in event or "timeout" in event or event.endswith("unavailable"):
        reason = fields.pop("reason", None)
        fields = {"reason": _simple_reason(reason), **fields}
    details = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    line = f"{timestamp} [ARTICLE] {event}"
    if details:
        line = f"{line} {details}"
    print(line, flush=True)


def _fmt_step_s(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def _summarize_solve_steps(payload: dict, count: int) -> str:
    """Compact bottleneck summary from pool article-batch timings."""
    steps = payload.get("steps") or []
    if not steps:
        for item in payload.get("results") or []:
            timings = item.get("timings") or {}
            if timings:
                steps.append({"id": item.get("id"), **timings})
    if not steps:
        return "no_steps"
    causes = {}
    queue_vals, ner_vals, llm_vals = [], [], []
    for row in steps:
        cause = row.get("cause") or "unknown"
        causes[cause] = causes.get(cause, 0) + 1
        if row.get("queue") is not None:
            queue_vals.append(float(row["queue"]))
        if row.get("ner") is not None:
            ner_vals.append(float(row["ner"]))
        llm = float(row.get("llm1") or 0) + float(row.get("llm2") or 0)
        if llm:
            llm_vals.append(llm)
    top_cause = max(causes.items(), key=lambda kv: kv[1])[0]
    parts = [f"bottleneck={top_cause}"]
    if queue_vals:
        parts.append(f"queue_max={max(queue_vals):.1f}s")
    if ner_vals:
        parts.append(f"ner_max={max(ner_vals):.1f}s")
    if llm_vals:
        parts.append(f"llm_max={max(llm_vals):.1f}s")
    parts.append("causes=" + ",".join(f"{k}:{v}" for k, v in sorted(causes.items())))
    slow = sorted(
        steps,
        key=lambda r: float(r.get("total") or 0) + float(r.get("queue") or 0),
        reverse=True,
    )[:3]
    if slow:
        slow_txt = ";".join(
            f"id={r.get('id')} q={_fmt_step_s(r.get('queue'))} ner={_fmt_step_s(r.get('ner'))} "
            f"llm1={_fmt_step_s(r.get('llm1'))} llm2={_fmt_step_s(r.get('llm2'))} cause={r.get('cause')}"
            for r in slow
        )
        parts.append(f"slowest[{slow_txt}]")
    return " ".join(parts)


def _quiet_thin_logging():
    """Suppress framework chatter; article lifecycle uses always-on stdout logs."""

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
            except Exception:
                return False
            return any(a in msg for a in _ALLOW_THIN)

    flt = _Filter()
    root = logging.getLogger()
    for h in list(root.handlers):
        h.addFilter(flt)
    for name in ("bittensor", "alpharidge_ai", "asyncio", "urllib3", "httpx"):
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            h.addFilter(flt)

    # bt.logging often bypasses stdlib filters. Article lifecycle does not use it.
    def _wrap(fn):
        def _inner(msg, *args, **kwargs):
            s = str(msg)
            if any(a in s for a in _ALLOW_THIN):
                return fn(msg, *args, **kwargs)
            return None

        return _inner

    for attr in ("info", "success", "debug", "trace", "warning", "error", "critical"):
        if hasattr(bt.logging, attr):
            setattr(bt.logging, attr, _wrap(getattr(bt.logging, attr)))


if _THIN:
    _quiet_thin_logging()


if not _THIN:
    from alpharidge_ai.analyzer.aspect_sentiment import install_meta_init_guard
    from alpharidge_ai.analyzer import setup_analyzer
    from alpharidge_ai.analyzer import setup_telegram_analyzer
    from alpharidge_ai.analyzer import setup_news_analyzer
    from alpharidge_ai.analyzer import setup_article_intelligence_analyzer
    from alpharidge_ai.models.article_intelligence import SCHEMA_VERSION
    from alpharidge_ai.triage import (
        FLAG_DISCARD, FLAG_VALUABLE, TRIAGE_SCHEMA_VERSION, analysis_indicates_value)

    # Must precede every model load — see install_meta_init_guard. A miner that loses the
    # race returns nothing for the affected articles and earns no points for them.
    install_meta_init_guard()


class Miner(BaseMinerNeuron):
    """
    V3 Miner: Processes TweetBatch / TelegramBatch / ArticleBatch from validators.

    When INFERENCE_POOL_URL is set, this process stays thin (no local NER stack)
    and forwards analysis to the shared pool on localhost.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        self.pool = None
        self.analyzer = None
        self.telegram_analyzer = None
        self.news_analyzer = None
        self.article_intel_analyzer = None
        self.triage_stage = None

        if _THIN:
            from alpharidge_ai.utils.inference_client import InferencePoolClient

            self.pool = InferencePoolClient()
            try:
                health = self.pool.health()
                bt.logging.info(
                    f"[Miner] Thin mode → pool {ar_config.INFERENCE_POOL_URL} health={health.get('ok')}"
                )
            except Exception as e:
                _article_log(
                    "pool_unavailable",
                    reason=e,
                    url=ar_config.INFERENCE_POOL_URL,
                )
                raise
        else:
            # Initialize analyzer for tweet classification
            bt.logging.info("[Miner] Initializing analyzer...")
            self.analyzer = setup_analyzer()
            self.telegram_analyzer = setup_telegram_analyzer()
            self.news_analyzer = setup_news_analyzer()
            bt.logging.info("[Miner] News analyzer initialized")

            # The V2 analyzer and the triage stage are required; fail startup
            # rather than fall back silently.
            try:
                self.article_intel_analyzer = setup_article_intelligence_analyzer()
                bt.logging.info("[Miner] ArticleIntelligence analyzer initialized")
            except Exception as e:
                raise RuntimeError(
                    "[Miner] ArticleIntelligence (V2) analyzer failed to initialize; "
                    "the subnet requires triage and triage requires V2. "
                    f"Fix the analyzer setup and restart: {e}") from e

            try:
                from alpharidge_ai.analyzer.asset_extractor import AssetExtractor
                from alpharidge_ai.analyzer.triage_stage import TriageStage
                self.triage_stage = TriageStage(AssetExtractor())
                bt.logging.info("[Miner] Article triage stage initialized")
            except Exception as e:
                raise RuntimeError(
                    "[Miner] Triage stage failed to initialize (asset gazetteer or "
                    "language detector). The subnet requires triage; fix the named "
                    f"component and restart: {e}") from e
            bt.logging.info("[Miner] Analyzer initialized")

        # NOTE: we intentionally do NOT reuse a single bt.Dendrite across threads/event-loops.
        # Miner responses are sent back to validators from a background thread with its own event loop.

        # IMPORTANT: Register a concrete TweetBatch handler on the axon.
        # Bittensor routes requests by synapse class name; attaching only `forward(self, bt.Synapse)`
        # registers the generic `Synapse` endpoint and does *not* register `TweetBatch`.
        self.axon.attach(
            forward_fn=self.forward_tweets,
            blacklist_fn=self.blacklist_tweet_batch,
            priority_fn=self.priority_tweet_batch,
        )
        
        # Register TelegramBatch handler
        self.axon.attach(
            forward_fn=self.forward_telegram_messages,
            blacklist_fn=self.blacklist_telegram_batch,
            priority_fn=self.priority_telegram_batch,
        )

        # Register ArticleBatch handler
        self.axon.attach(
            forward_fn=self.forward_articles,
            blacklist_fn=self.blacklist_article_batch,
            priority_fn=self.priority_article_batch,
        )

        hotkey = self.wallet.hotkey.ss58_address
        if not _THIN:
            bt.logging.info(f"[Miner] V3 miner started with hotkey: {hotkey}")

    async def blacklist_tweet_batch(
        self, synapse: alpharidge_ai.protocol.TweetBatch
    ) -> typing.Tuple[bool, str]:
        """Typed wrapper so bittensor's axon signature checks pass for TweetBatch."""
        return await self.blacklist(synapse)

    async def priority_tweet_batch(self, synapse: alpharidge_ai.protocol.TweetBatch) -> float:
        """Typed wrapper so bittensor's axon signature checks pass for TweetBatch."""
        return await self.priority(synapse)

    async def blacklist_telegram_batch(
        self, synapse: alpharidge_ai.protocol.TelegramBatch
    ) -> typing.Tuple[bool, str]:
        """Typed wrapper so bittensor's axon signature checks pass for TelegramBatch."""
        return await self.blacklist(synapse)

    async def priority_telegram_batch(self, synapse: alpharidge_ai.protocol.TelegramBatch) -> float:
        """Typed wrapper so bittensor's axon signature checks pass for TelegramBatch."""
        return await self.priority(synapse)

    async def blacklist_article_batch(
        self, synapse: alpharidge_ai.protocol.ArticleBatch
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_article_batch(self, synapse: alpharidge_ai.protocol.ArticleBatch) -> float:
        return await self.priority(synapse)

    async def forward_is_alive(self, synapse: alpharidge_ai.protocol.IsAlive) -> alpharidge_ai.protocol.IsAlive:
        """
        Processes incoming IsAlive synapses from validators.
        """
        synapse.is_alive = True
        return synapse
    
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        """
        Processes incoming synapses. Routes TweetBatch requests to forward_tweets.
        
        Args:
            synapse (bt.Synapse): The incoming synapse request.
            
        Returns:
            bt.Synapse: The processed synapse response.
        """
        if isinstance(synapse, alpharidge_ai.protocol.TweetBatch):
            return await self.forward_tweets(synapse)
        
        bt.logging.warning(f"Received synapse type: {type(synapse).__name__}, but no handler implemented")
        return synapse

    async def forward_tweets(self, synapse: alpharidge_ai.protocol.TweetBatch) -> alpharidge_ai.protocol.TweetBatch:
        """
        Processes TweetBatch requests from validators.
        
        Spawns a background thread to analyze tweets and send results back to the validator.
        Returns immediately to avoid blocking the axon.
        
        Args:
            synapse: TweetBatch containing list of tweets to analyze
            
        Returns:
            TweetBatch (returns immediately, processing happens in background)
        """
        validator_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        bt.logging.info(f"[Miner] Received TweetBatch with {len(synapse.tweet_batch)} tweet(s) from validator {validator_hotkey}")
        
        if not validator_hotkey:
            bt.logging.warning("[Miner] No validator hotkey found in synapse, cannot send response back")
            return synapse
        
        # Make a deep copy of the synapse for background processing
        synapse_copy = copy.deepcopy(synapse)
        
        # Start background thread for processing and sending response
        thread = threading.Thread(
            target=self._process_and_send_tweets,
            args=(synapse_copy, validator_hotkey),
            daemon=True
        )
        thread.start()
        
        bt.logging.info(f"[Miner] Started background processing for TweetBatch, returning immediately")
        return synapse

    async def forward_telegram_messages(self, synapse: alpharidge_ai.protocol.TelegramBatch) -> alpharidge_ai.protocol.TelegramBatch:
        """
        Processes TelegramBatch requests from validators.
        
        Spawns a background thread to analyze telegram messages and send results back to the validator.
        Returns immediately to avoid blocking the axon.
        
        Args:
            synapse: TelegramBatch containing list of messages to analyze
            
        Returns:
            TelegramBatch (returns immediately, processing happens in background)
        """
        validator_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        bt.logging.info(f"[Miner] Received TelegramBatch with {len(synapse.message_batch)} message(s) from validator {validator_hotkey}")
        
        if not validator_hotkey:
            bt.logging.warning("[Miner] No validator hotkey found in synapse, cannot send response back")
            return synapse
        
        # Make a deep copy of the synapse for background processing
        synapse_copy = copy.deepcopy(synapse)
        
        # Start background thread for processing and sending response
        thread = threading.Thread(
            target=self._process_and_send_telegram_messages,
            args=(synapse_copy, validator_hotkey),
            daemon=True
        )
        thread.start()
        
        bt.logging.info(f"[Miner] Started background processing for TelegramBatch, returning immediately")
        return synapse

    async def forward_articles(self, synapse: alpharidge_ai.protocol.ArticleBatch) -> alpharidge_ai.protocol.ArticleBatch:
        validator_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        _article_log(
            "received",
            count=len(synapse.article_batch),
            validator=validator_hotkey,
        )

        if not validator_hotkey:
            bt.logging.warning("[Miner] No validator hotkey found in synapse, cannot send response back")
            return synapse

        synapse_copy = copy.deepcopy(synapse)

        thread = threading.Thread(
            target=self._process_and_send_articles,
            args=(synapse_copy, validator_hotkey),
            daemon=True
        )
        thread.start()

        bt.logging.info(f"[Miner] Started background processing for ArticleBatch, returning immediately")
        return synapse

    def _send_synapse(self, synapse, validator_hotkey: str, label: str) -> bool:
        try:
            validator_uid = self.metagraph.hotkeys.index(validator_hotkey)
        except ValueError:
            if label == "ArticleBatch":
                _article_log(
                    "submit_failed",
                    validator=validator_hotkey,
                    reason="validator_not_in_metagraph",
                )
            else:
                bt.logging.error(f"[Miner] Validator hotkey {validator_hotkey} not found in metagraph")
            return False

        validator_axon = self.metagraph.axons[validator_uid]
        if label != "ArticleBatch":
            bt.logging.info(
                f"[Miner] Background: Found validator UID {validator_uid}, sending {label} via dendrite"
            )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        dendrite = None
        ok = False
        try:
            dendrite = bt.Dendrite(wallet=self.wallet)
            responses = loop.run_until_complete(
                dendrite.forward(
                    axons=[validator_axon],
                    synapse=synapse,
                    timeout=30.0,
                    deserialize=True,
                )
            )
            try:
                status_code = responses[0].dendrite.status_code if responses and responses[0].dendrite else None
                status_msg = responses[0].dendrite.status_message if responses and responses[0].dendrite else None
            except Exception:
                status_code, status_msg = None, None
            if status_code != 200:
                if label == "ArticleBatch":
                    _article_log(
                        "submit_failed",
                        reason=status_msg or f"validator_status_{status_code}",
                        validator=validator_hotkey,
                        status=status_code,
                    )
                else:
                    bt.logging.error(
                        f"[Miner] Background: Validator response failed (status={status_code}): {status_msg}"
                    )
            else:
                if label == "ArticleBatch":
                    _article_log("submitted", validator=validator_hotkey, status=status_code)
                else:
                    bt.logging.info(
                        f"[Miner] Background: Successfully sent processed {label} back to validator {validator_hotkey}"
                    )
                ok = True
        except Exception as e:
            if label == "ArticleBatch":
                name = type(e).__name__
                event = (
                    "submit_timeout"
                    if "timeout" in name.lower() or "timed out" in str(e).lower()
                    else "submit_failed"
                )
                _article_log(event, reason=e, validator=validator_hotkey)
            else:
                bt.logging.error(f"[Miner] Background: Failed to send response to validator: {e}")
        finally:
            try:
                if dendrite is not None:
                    if hasattr(dendrite, "aclose_session"):
                        loop.run_until_complete(dendrite.aclose_session())
                    elif hasattr(dendrite, "close_session"):
                        dendrite.close_session()
            except Exception:
                pass
            loop.close()
        return ok

    def _process_and_send_tweets(self, synapse: alpharidge_ai.protocol.TweetBatch, validator_hotkey: str):
        """
        Background thread function to process tweets and send results back to validator.
        """
        try:
            bt.logging.info(f"[Miner] Background: Processing {len(synapse.tweet_batch)} tweets")

            if self.pool is not None:
                items = [{"id": t.id, "text": t.text or ""} for t in synapse.tweet_batch if t.text]
                results = self.pool.analyze_tweets_batch(items)
                by_id = {r.get("id"): r.get("analysis") for r in results}
                for tweet in synapse.tweet_batch:
                    analysis = by_id.get(tweet.id)
                    if not analysis:
                        continue
                    tweet.analysis = TweetAnalysisBase(
                        sentiment=analysis["sentiment"],
                        asset_id=analysis["asset_id"],
                        asset_symbol=analysis["asset_symbol"],
                        content_type=analysis["content_type"],
                        technical_quality=analysis["technical_quality"],
                        market_analysis=analysis["market_analysis"],
                        impact_potential=analysis["impact_potential"],
                    )
            else:
                for tweet in synapse.tweet_batch:
                    if not tweet.text:
                        bt.logging.warning(f"[Miner] Skipping tweet {tweet.id} - no text content")
                        continue

                    classification = self.analyzer.classify_post(tweet.text)

                    if classification is None:
                        bt.logging.warning(f"[Miner] Failed to classify tweet {tweet.id}")
                        continue

                    tweet.analysis = TweetAnalysisBase(
                        sentiment=classification.sentiment.value,
                        asset_id=classification.asset_id,
                        asset_symbol=classification.asset_symbol,
                        content_type=classification.content_type.value,
                        technical_quality=classification.technical_quality.value,
                        market_analysis=classification.market_analysis.value,
                        impact_potential=classification.impact_potential.value,
                    )

            bt.logging.info(f"[Miner] Background: Finished processing, sending back to validator {validator_hotkey}")

            from alpharidge_ai.utils.miner_signing import sign_items
            miner_signatures, nonces = sign_items(self.wallet.hotkey, synapse.tweet_batch, id_attr="id")
            synapse.miner_signatures = miner_signatures
            synapse.nonces = nonces
            self._send_synapse(synapse, validator_hotkey, "TweetBatch")

        except Exception as e:
            bt.logging.error(f"[Miner] Background: Error processing tweets: {e}")

    def _process_and_send_telegram_messages(self, synapse: alpharidge_ai.protocol.TelegramBatch, validator_hotkey: str):
        """
        Background thread function to process telegram messages and send results back to validator.
        """
        try:
            bt.logging.info(f"[Miner] Background: Processing {len(synapse.message_batch)} telegram messages")

            if self.pool is not None:
                items = []
                for msg in synapse.message_batch:
                    if not msg.content:
                        continue
                    messages_for_analysis = [{
                        'message_id': msg.id,
                        'username': msg.sender_username or msg.sender_name,
                        'content': msg.content,
                    }]
                    if msg.context_messages:
                        for ctx in msg.context_messages:
                            messages_for_analysis.insert(0, {
                                'message_id': ctx.id,
                                'username': ctx.sender_username or ctx.sender_name,
                                'content': ctx.content,
                            })
                    items.append({
                        "id": msg.id,
                        "messages": messages_for_analysis,
                        "asset_id": msg.inherited_asset_id,
                    })
                results = self.pool.analyze_telegram_batch(items)
                by_id = {r.get("id"): r.get("analysis") for r in results}
                from datetime import datetime
                for msg in synapse.message_batch:
                    analysis = by_id.get(msg.id)
                    if not analysis:
                        continue
                    msg.analysis = TelegramMessageAnalysis(
                        id=0,
                        message_id=msg.id,
                        sentiment=analysis["sentiment"],
                        asset_id=analysis["asset_id"],
                        asset_symbol=analysis["asset_symbol"],
                        content_type=analysis["content_type"],
                        technical_quality=analysis["technical_quality"],
                        market_analysis=analysis["market_analysis"],
                        impact_potential=analysis["impact_potential"],
                        relevance_confidence=analysis["relevance_confidence"],
                        analyzed_at=datetime.now().isoformat(),
                    )
            else:
                for msg in synapse.message_batch:
                    if not msg.content:
                        bt.logging.warning(f"[Miner] Skipping telegram message {msg.id} - no content")
                        continue

                    messages_for_analysis = [{
                        'message_id': msg.id,
                        'username': msg.sender_username or msg.sender_name,
                        'content': msg.content,
                    }]

                    if msg.context_messages:
                        for ctx in msg.context_messages:
                            messages_for_analysis.insert(0, {
                                'message_id': ctx.id,
                                'username': ctx.sender_username or ctx.sender_name,
                                'content': ctx.content,
                            })

                    inherited_asset_id = msg.inherited_asset_id

                    classification = self.telegram_analyzer.classify_message_group(
                        messages_for_analysis,
                        asset_id=inherited_asset_id
                    )

                    if classification is None:
                        bt.logging.warning(f"[Miner] Failed to classify telegram message {msg.id}")
                        continue

                    from datetime import datetime
                    msg.analysis = TelegramMessageAnalysis(
                        id=0,
                        message_id=msg.id,
                        sentiment=classification.sentiment.value,
                        asset_id=classification.asset_id,
                        asset_symbol=classification.asset_symbol,
                        content_type=classification.content_type.value,
                        technical_quality=classification.technical_quality.value,
                        market_analysis=classification.market_analysis.value,
                        impact_potential=classification.impact_potential.value,
                        relevance_confidence=classification.relevance_confidence,
                        analyzed_at=datetime.now().isoformat(),
                    )

            bt.logging.info(f"[Miner] Background: Finished processing telegram messages, sending back to validator {validator_hotkey}")

            from alpharidge_ai.utils.miner_signing import sign_items
            miner_signatures, nonces = sign_items(self.wallet.hotkey, synapse.message_batch, id_attr="id")
            synapse.miner_signatures = miner_signatures
            synapse.nonces = nonces
            self._send_synapse(synapse, validator_hotkey, "TelegramBatch")

        except Exception as e:
            bt.logging.error(f"[Miner] Background: Error processing telegram messages: {e}")

    def _apply_article_analysis(self, article, analysis: dict):
        if not analysis:
            return
        # V1 fallback shape uses snake_case sector_id / content_type
        if "analysisData" in analysis or "sectorId" in analysis or "contentType" in analysis:
            article.analysis = NewsArticleAnalysisBase(**analysis)
            return
        article.analysis = NewsArticleAnalysisBase(
            sentiment=analysis["sentiment"],
            sector_id=analysis.get("sector_id"),
            sector_symbol=analysis.get("sector_symbol"),
            content_type=analysis.get("content_type"),
            technical_quality=analysis.get("technical_quality"),
            market_analysis=analysis.get("market_analysis"),
            impact_potential=analysis.get("impact_potential"),
            relevance_confidence=analysis.get("relevance_confidence"),
        )

    def _process_and_send_articles(self, synapse: alpharidge_ai.protocol.ArticleBatch, validator_hotkey: str):
        started_at = time.monotonic()
        try:
            bt.logging.info(f"[Miner] Background: Processing {len(synapse.article_batch)} articles")

            if self.pool is not None:
                articles = []
                for article in synapse.article_batch:
                    articles.append(
                        {
                            "article_id": article.id,
                            "url": article.url,
                            "title": article.title,
                            "source": article.source,
                            "published": article.published,
                            "summary": article.summary,
                            "content": article.content,
                            "raw_html": getattr(article, "raw_html", None),
                        }
                    )
                stop_wait = threading.Event()

                def _wait_heartbeat():
                    while not stop_wait.wait(30):
                        elapsed = time.monotonic() - started_at
                        _article_log(
                            "solving",
                            count=len(articles),
                            seconds=f"{elapsed:.0f}",
                            validator=validator_hotkey,
                            stage="waiting_pool",
                        )

                waiter = threading.Thread(target=_wait_heartbeat, daemon=True)
                waiter.start()
                try:
                    payload = self.pool.analyze_articles_batch(
                        articles,
                        miner_hotkey=self.wallet.hotkey.ss58_address if self.wallet else None,
                    )
                finally:
                    stop_wait.set()
                results = payload.get("results") or []
                by_id = {r.get("id"): r.get("analysis") for r in results}
                for article in synapse.article_batch:
                    self._apply_article_analysis(article, by_id.get(article.id))
                _article_log(
                    "solve_steps",
                    count=len(articles),
                    validator=validator_hotkey,
                    detail=_summarize_solve_steps(payload, len(articles)),
                )
            else:
                for article in synapse.article_batch:
                    if not article.title:
                        # Never skip: an absent response fails the size check. A
                        # titleless article still gets a claim and a proof.
                        rec, proof, _ = self.triage_stage.evaluate("", article.content or "")
                        article.analysis = NewsArticleAnalysisBase(
                            sentiment="neutral",
                            analysisData={
                                "schema_version": SCHEMA_VERSION,
                                "triage_schema_version": TRIAGE_SCHEMA_VERSION,
                                "triage": rec,
                                "proof_of_read": proof,
                            },
                        )
                        continue

                    # Label every article. Relevant and borderline get the full
                    # analysis; borderline is then flagged from the analysis result.
                    triage_rec, proof, _ = self.triage_stage.evaluate(
                        article.title, article.content or "")
                    if triage_rec["label"] == "irrelevant":
                        article.analysis = NewsArticleAnalysisBase(
                            sentiment="neutral",
                            analysisData={
                                "schema_version": SCHEMA_VERSION,
                                "triage_schema_version": TRIAGE_SCHEMA_VERSION,
                                "triage": triage_rec,
                                "proof_of_read": proof,
                            },
                        )
                        continue

                    # V2: Full ArticleIntelligence analysis
                    if self.article_intel_analyzer is not None:
                        intel = self.article_intel_analyzer.analyze(
                            article_id=article.id,
                            url=article.url,
                            title=article.title,
                            source=article.source,
                            published=article.published,
                            summary=article.summary,
                            content=article.content,
                            raw_html=getattr(article, "raw_html", None),
                            miner_hotkey=self.wallet.hotkey.ss58_address if self.wallet else None,
                        )
                        if intel is None:
                            # Analysis unavailable: send the claim + proof so
                            # the batch stays complete.
                            article.analysis = NewsArticleAnalysisBase(
                                sentiment="neutral",
                                analysisData={
                                    "schema_version": SCHEMA_VERSION,
                                    "triage_schema_version": TRIAGE_SCHEMA_VERSION,
                                    "triage": triage_rec,
                                    "proof_of_read": proof,
                                },
                            )
                            continue
                        if intel is not None:
                            analysis_blob = intel.model_dump()
                            if triage_rec["label"] == "borderline":
                                triage_rec["flag"] = (
                                    FLAG_VALUABLE if analysis_indicates_value(analysis_blob)
                                    else FLAG_DISCARD)
                            analysis_blob["triage_schema_version"] = TRIAGE_SCHEMA_VERSION
                            analysis_blob["triage"] = triage_rec
                            analysis_blob["proof_of_read"] = proof
                            article.analysis = NewsArticleAnalysisBase(
                                sentiment=intel.overall_sentiment.value,
                                sectorId=intel.topic_signature.primary_sector_id,
                                sectorSymbol=intel.topic_signature.primary_sector_symbol,
                                contentType=intel.content_type.value,
                                technicalQuality=intel.technical_quality if isinstance(intel.technical_quality, str) else intel.technical_quality.value if hasattr(intel.technical_quality, 'value') else str(intel.technical_quality),
                                marketAnalysis=intel.market_analysis_type.value,
                                impactPotential=intel.impact_potential.value,
                                relevanceConfidence="high" if intel.assets else "low",
                                overallSentimentScore=intel.overall_sentiment_score,
                                sentimentDirection=intel.sentiment_direction.value,
                                urgency=intel.urgency.value,
                                temporalFocus=intel.temporal_focus.value,
                                factualConfidence=intel.factual_confidence.value,
                                positioningSignal=intel.positioning_signal.value,
                                targetAudience=intel.target_audience.value,
                                credibilityFlag=intel.credibility_flag.value,
                                primaryGeo=intel.primary_geo.value,
                                marketSession=intel.market_session.value,
                                detectedLanguage=intel.detected_language,
                                stalenessFlag=intel.staleness_flag.value,
                                forwardEventType=intel.forward_event_type.value,
                                assets=[a.model_dump() for a in intel.assets],
                                entities=[e.model_dump() for e in intel.entities],
                                economicData=[d.model_dump() for d in intel.economic_data],
                                numericClaims=[c.model_dump() for c in intel.numeric_claims],
                                quotes=[q.model_dump() for q in intel.quotes],
                                contagionLinks=[l.model_dump() for l in intel.contagion_links],
                                chartSummary=intel.chart_summary.model_dump(),
                                eventFingerprint=intel.event_fingerprint.model_dump(),
                                narrativeKeywords=intel.narrative_keywords,
                                topicSignature=intel.topic_signature.model_dump(),
                                textStats=intel.text_stats.model_dump(),
                                inferredImpacts=[i.model_dump() for i in intel.inferred_impacts] if intel.inferred_impacts else None,
                                analysisData=analysis_blob,
                            )
                            bt.logging.info(f"[Miner] V2 analysis complete for article {article.id}: "
                                            f"{len(intel.assets)} assets, {len(intel.entities)} entities, "
                                            f"{len(intel.contagion_links)} contagion links")
                            continue

                    # V1 fallback: original NewsRelevanceAnalyzer
                    classification = self.news_analyzer.classify_article(article.title, article.summary, article.content)

                    if classification is None:
                        bt.logging.warning(f"[Miner] Failed to classify article {article.id}")
                        continue

                    article.analysis = NewsArticleAnalysisBase(
                        sentiment=classification.sentiment.value,
                        sector_id=classification.sector_id,
                        sector_symbol=classification.sector_symbol,
                        content_type=classification.content_type.value,
                        technical_quality=classification.technical_quality.value,
                        market_analysis=classification.market_analysis.value,
                        impact_potential=classification.impact_potential.value,
                        relevance_confidence=classification.relevance_confidence,
                    )

            solved = sum(1 for article in synapse.article_batch if article.analysis is not None)
            total = len(synapse.article_batch)
            _article_log(
                "solved",
                count=solved,
                total=total,
                seconds=f"{time.monotonic() - started_at:.1f}",
                validator=validator_hotkey,
            )
            if total and solved < total:
                _article_log(
                    "solve_partial",
                    missing=f"{total - solved}/{total}",
                    count=solved,
                    total=total,
                    validator=validator_hotkey,
                )

            bt.logging.info(f"[Miner] Background: Finished processing articles, sending back to validator {validator_hotkey}")

            from alpharidge_ai.utils.miner_signing import sign_items
            miner_signatures, nonces = sign_items(self.wallet.hotkey, synapse.article_batch, id_attr="id")
            synapse.miner_signatures = miner_signatures
            synapse.nonces = nonces
            self._send_synapse(synapse, validator_hotkey, "ArticleBatch")

        except Exception as e:
            name = type(e).__name__
            event = (
                "solve_timeout"
                if "timeout" in name.lower() or "timed out" in str(e).lower()
                else "solve_failed"
            )
            _article_log(
                event,
                reason=e,
                count=len(synapse.article_batch),
                seconds=f"{time.monotonic() - started_at:.1f}",
                validator=validator_hotkey,
                hint="check alpha-pool article_step logs (queue=pool_busy ner=vps_busy llm1/llm2=llm_slow)",
            )

    async def forward_score(self, synapse: alpharidge_ai.protocol.Score) -> alpharidge_ai.protocol.Score:
        """
        Processes incoming Score synapses from validators.
        
        Receives the score that the validator has given this hotkey for a 100-block interval.
        """
        if _THIN:
            return synapse
        block_window_start = synapse.block_window_start
        block_window_end = synapse.block_window_end
        score = synapse.score
        rewards = synapse.rewards
        penalties = synapse.penalties
        validator_hotkey = synapse.validator_hotkey
        bt.logging.info(
            f"[Score] Epoch blocks {block_window_start}-{block_window_end}: {score:.0f} points"
            f" ({rewards} rewards, {penalties} penalties) from validator {validator_hotkey}"
        )
        return synapse

    async def forward_validation_result(self, synapse: alpharidge_ai.protocol.ValidationResult) -> alpharidge_ai.protocol.ValidationResult:
        """
        Processes incoming ValidationResult synapses from validators.
        
        Receives validation results for a specific post, including whether it passed or failed and why.
        """
        if _THIN:
            return synapse
        validation_id = synapse.validation_id
        post_id = synapse.post_id
        success = synapse.success
        validator_hotkey = synapse.validator_hotkey
        failure_reason = synapse.failure_reason
        
        if success:
            bt.logging.info(
                f"[ValidationResult] ✓ Post {post_id} PASSED validation from validator {validator_hotkey} "
                f"(validation_id: {validation_id})"
            )
        else:
            failure_code = failure_reason.get("code", "unknown") if failure_reason else "unknown"
            failure_message = failure_reason.get("message", "Unknown error") if failure_reason else "Unknown error"
            bt.logging.warning(
                f"[ValidationResult] ✗ Post {post_id} FAILED validation from validator {validator_hotkey} "
                f"(validation_id: {validation_id}): {failure_code} - {failure_message}"
            )
        
        return synapse

    async def blacklist(
        self, synapse: bt.Synapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contracted via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (bt.Synapse): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        # Check if hotkey is registered BEFORE trying to get its index
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        # Only get uid if hotkey is in metagraph (to avoid IndexError)
        try:
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        except ValueError:
            # Hotkey not found in metagraph (shouldn't happen if check above passed, but be safe)
            bt.logging.warning(f"Hotkey {synapse.dendrite.hotkey} not found in metagraph")
            return True, "Hotkey not in metagraph"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        if not _THIN:
            bt.logging.trace(
                f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
            )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: bt.Synapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (bt.Synapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        if not _THIN:
            bt.logging.trace(
                f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
            )
        return priority


    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up when miner exits."""
        super().__exit__(exc_type, exc_value, traceback)
        return False


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            if not _THIN:
                bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
