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
    if "failed" in event or "timeout" in event or "partial" in event or event.endswith("unavailable"):
        reason = fields.pop("reason", None)
        fields = {"reason": _simple_reason(reason), **fields}
    details = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    line = f"{timestamp} [ARTICLE] {event}"
    if details:
        line = f"{line} {details}"
    print(line, flush=True)


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
    from alpharidge_ai.inference_pool.mapping import intel_to_analysis_dict, triage_only_analysis_dict

    # Must precede every model load — see install_meta_init_guard.
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
            bt.logging.info("[Miner] Initializing analyzer...")
            self.analyzer = setup_analyzer()
            self.telegram_analyzer = setup_telegram_analyzer()
            self.news_analyzer = setup_news_analyzer()
            bt.logging.info("[Miner] News analyzer initialized")
            try:
                self.article_intel_analyzer = setup_article_intelligence_analyzer()
                bt.logging.info("[Miner] ArticleIntelligence analyzer initialized")
            except Exception as e:
                bt.logging.warning(
                    f"[Miner] ArticleIntelligence analyzer init failed, falling back to V1: {e}"
                )
                self.article_intel_analyzer = None

            if (
                getattr(ar_config, "MINER_TRIAGE_ENABLED", False)
                and self.article_intel_analyzer is not None
            ):
                try:
                    from alpharidge_ai.analyzer.asset_extractor import AssetExtractor
                    from alpharidge_ai.analyzer.triage_stage import TriageStage

                    self.triage_stage = TriageStage(AssetExtractor())
                    bt.logging.info("[Miner] Article triage stage enabled")
                except Exception as e:
                    bt.logging.warning(f"[Miner] Triage stage init failed, triage disabled: {e}")
            bt.logging.info("[Miner] Analyzer initialized")

        self.axon.attach(
            forward_fn=self.forward_tweets,
            blacklist_fn=self.blacklist_tweet_batch,
            priority_fn=self.priority_tweet_batch,
        )
        self.axon.attach(
            forward_fn=self.forward_telegram_messages,
            blacklist_fn=self.blacklist_telegram_batch,
            priority_fn=self.priority_telegram_batch,
        )
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
        return await self.blacklist(synapse)

    async def priority_tweet_batch(self, synapse: alpharidge_ai.protocol.TweetBatch) -> float:
        return await self.priority(synapse)

    async def blacklist_telegram_batch(
        self, synapse: alpharidge_ai.protocol.TelegramBatch
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_telegram_batch(self, synapse: alpharidge_ai.protocol.TelegramBatch) -> float:
        return await self.priority(synapse)

    async def blacklist_article_batch(
        self, synapse: alpharidge_ai.protocol.ArticleBatch
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_article_batch(self, synapse: alpharidge_ai.protocol.ArticleBatch) -> float:
        return await self.priority(synapse)

    async def forward_is_alive(self, synapse: alpharidge_ai.protocol.IsAlive) -> alpharidge_ai.protocol.IsAlive:
        synapse.is_alive = True
        return synapse

    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        if isinstance(synapse, alpharidge_ai.protocol.TweetBatch):
            return await self.forward_tweets(synapse)
        if not _THIN:
            bt.logging.warning(
                f"Received synapse type: {type(synapse).__name__}, but no handler implemented"
            )
        return synapse

    async def forward_tweets(self, synapse: alpharidge_ai.protocol.TweetBatch) -> alpharidge_ai.protocol.TweetBatch:
        validator_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        bt.logging.info(
            f"[Miner] Received TweetBatch with {len(synapse.tweet_batch)} tweet(s) from validator {validator_hotkey}"
        )
        if not validator_hotkey:
            return synapse
        synapse_copy = copy.deepcopy(synapse)
        threading.Thread(
            target=self._process_and_send_tweets,
            args=(synapse_copy, validator_hotkey),
            daemon=True,
        ).start()
        return synapse

    async def forward_telegram_messages(
        self, synapse: alpharidge_ai.protocol.TelegramBatch
    ) -> alpharidge_ai.protocol.TelegramBatch:
        validator_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        bt.logging.info(
            f"[Miner] Received TelegramBatch with {len(synapse.message_batch)} message(s) from validator {validator_hotkey}"
        )
        if not validator_hotkey:
            return synapse
        synapse_copy = copy.deepcopy(synapse)
        threading.Thread(
            target=self._process_and_send_telegram_messages,
            args=(synapse_copy, validator_hotkey),
            daemon=True,
        ).start()
        return synapse

    async def forward_articles(
        self, synapse: alpharidge_ai.protocol.ArticleBatch
    ) -> alpharidge_ai.protocol.ArticleBatch:
        validator_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        _article_log(
            "received",
            count=len(synapse.article_batch),
            validator=validator_hotkey,
        )
        if not validator_hotkey:
            return synapse
        synapse_copy = copy.deepcopy(synapse)
        threading.Thread(
            target=self._process_and_send_articles,
            args=(synapse_copy, validator_hotkey),
            daemon=True,
        ).start()
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
                bt.logging.error(
                    f"[Miner] Validator hotkey {validator_hotkey} not found in metagraph"
                )
            return False
        validator_axon = self.metagraph.axons[validator_uid]
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
                status_code = (
                    responses[0].dendrite.status_code if responses and responses[0].dendrite else None
                )
                status_msg = (
                    responses[0].dendrite.status_message
                    if responses and responses[0].dendrite
                    else None
                )
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
                        f"[Miner] Validator response failed (status={status_code}): {status_msg}"
                    )
            else:
                if label == "ArticleBatch":
                    _article_log("submitted", validator=validator_hotkey, status=status_code)
                else:
                    bt.logging.info(
                        f"[Miner] Submitted successfully {label} to validator {validator_hotkey}"
                    )
                ok = True
        except Exception as e:
            if label == "ArticleBatch":
                name = type(e).__name__
                event = "submit_timeout" if "timeout" in name.lower() or "timed out" in str(e).lower() else "submit_failed"
                _article_log(event, reason=e, validator=validator_hotkey)
            else:
                bt.logging.error(f"[Miner] Failed to send {label} to validator: {e}")
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
        try:
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
                        continue
                    classification = self.analyzer.classify_post(tweet.text)
                    if classification is None:
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

            from alpharidge_ai.utils.miner_signing import sign_items

            miner_signatures, nonces = sign_items(self.wallet.hotkey, synapse.tweet_batch, id_attr="id")
            synapse.miner_signatures = miner_signatures
            synapse.nonces = nonces
            self._send_synapse(synapse, validator_hotkey, "TweetBatch")
        except Exception as e:
            bt.logging.error(f"[Miner] Error processing tweets: {e}")

    def _process_and_send_telegram_messages(
        self, synapse: alpharidge_ai.protocol.TelegramBatch, validator_hotkey: str
    ):
        try:
            if self.pool is not None:
                items = []
                for msg in synapse.message_batch:
                    if not msg.content:
                        continue
                    messages_for_analysis = [
                        {
                            "message_id": msg.id,
                            "username": msg.sender_username or msg.sender_name,
                            "content": msg.content,
                        }
                    ]
                    if msg.context_messages:
                        for ctx in msg.context_messages:
                            messages_for_analysis.insert(
                                0,
                                {
                                    "message_id": ctx.id,
                                    "username": ctx.sender_username or ctx.sender_name,
                                    "content": ctx.content,
                                },
                            )
                    items.append(
                        {
                            "id": msg.id,
                            "messages": messages_for_analysis,
                            "asset_id": msg.inherited_asset_id,
                        }
                    )
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
                from datetime import datetime

                for msg in synapse.message_batch:
                    if not msg.content:
                        continue
                    messages_for_analysis = [
                        {
                            "message_id": msg.id,
                            "username": msg.sender_username or msg.sender_name,
                            "content": msg.content,
                        }
                    ]
                    if msg.context_messages:
                        for ctx in msg.context_messages:
                            messages_for_analysis.insert(
                                0,
                                {
                                    "message_id": ctx.id,
                                    "username": ctx.sender_username or ctx.sender_name,
                                    "content": ctx.content,
                                },
                            )
                    classification = self.telegram_analyzer.classify_message_group(
                        messages_for_analysis, asset_id=msg.inherited_asset_id
                    )
                    if classification is None:
                        continue
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

            from alpharidge_ai.utils.miner_signing import sign_items

            miner_signatures, nonces = sign_items(
                self.wallet.hotkey, synapse.message_batch, id_attr="id"
            )
            synapse.miner_signatures = miner_signatures
            synapse.nonces = nonces
            self._send_synapse(synapse, validator_hotkey, "TelegramBatch")
        except Exception as e:
            bt.logging.error(f"[Miner] Error processing telegram messages: {e}")

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

    def _process_and_send_articles(
        self, synapse: alpharidge_ai.protocol.ArticleBatch, validator_hotkey: str
    ):
        started_at = time.monotonic()
        try:
            if self.pool is not None:
                articles = []
                for article in synapse.article_batch:
                    if not article.title:
                        continue
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
                results = self.pool.analyze_articles_batch(
                    articles,
                    miner_hotkey=self.wallet.hotkey.ss58_address if self.wallet else None,
                    triage_enabled=bool(getattr(ar_config, "MINER_TRIAGE_ENABLED", False)),
                )
                by_id = {r.get("id"): r.get("analysis") for r in results}
                for article in synapse.article_batch:
                    self._apply_article_analysis(article, by_id.get(article.id))
            else:
                for article in synapse.article_batch:
                    if not article.title:
                        continue
                    triage_rec = proof = None
                    if self.triage_stage is not None:
                        triage_rec, proof, _ = self.triage_stage.evaluate(
                            article.title, article.content or ""
                        )
                        if triage_rec["label"] != "relevant":
                            article.analysis = NewsArticleAnalysisBase(
                                **triage_only_analysis_dict(triage_rec, proof)
                            )
                            continue

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
                        if intel is not None:
                            article.analysis = NewsArticleAnalysisBase(
                                **intel_to_analysis_dict(
                                    intel, triage_rec=triage_rec, proof=proof
                                )
                            )
                            bt.logging.info(
                                f"[Miner] V2 analysis complete for article {article.id}: "
                                f"{len(intel.assets)} assets, {len(intel.entities)} entities, "
                                f"{len(intel.contagion_links)} contagion links"
                            )
                            continue

                    classification = self.news_analyzer.classify_article(
                        article.title, article.summary, article.content
                    )
                    if classification is None:
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
                    reason=f"missing_analysis {total - solved}/{total}",
                    count=solved,
                    total=total,
                    validator=validator_hotkey,
                )

            from alpharidge_ai.utils.miner_signing import sign_items

            miner_signatures, nonces = sign_items(
                self.wallet.hotkey, synapse.article_batch, id_attr="id"
            )
            synapse.miner_signatures = miner_signatures
            synapse.nonces = nonces
            self._send_synapse(synapse, validator_hotkey, "ArticleBatch")
        except Exception as e:
            name = type(e).__name__
            event = "solve_timeout" if "timeout" in name.lower() or "timed out" in str(e).lower() else "solve_failed"
            _article_log(
                event,
                reason=e,
                count=len(synapse.article_batch),
                seconds=f"{time.monotonic() - started_at:.1f}",
                validator=validator_hotkey,
            )

    async def forward_score(self, synapse: alpharidge_ai.protocol.Score) -> alpharidge_ai.protocol.Score:
        if _THIN:
            return synapse
        validator_hotkey = synapse.validator_hotkey
        block_window_start = synapse.block_window_start
        block_window_end = synapse.block_window_end
        score = synapse.score
        rewards = synapse.rewards
        penalties = synapse.penalties
        bt.logging.info(
            f"[Score] Epoch blocks {block_window_start}-{block_window_end}: {score:.0f} points"
            f" ({rewards} rewards, {penalties} penalties) from validator {validator_hotkey}"
        )
        return synapse

    async def forward_validation_result(
        self, synapse: alpharidge_ai.protocol.ValidationResult
    ) -> alpharidge_ai.protocol.ValidationResult:
        return synapse

    async def blacklist(self, synapse: bt.Synapse) -> typing.Tuple[bool, str]:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            bt.logging.warning(f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"
        try:
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        except ValueError:
            bt.logging.warning(f"Hotkey {synapse.dendrite.hotkey} not found in metagraph")
            return True, "Hotkey not in metagraph"
        if self.config.blacklist.force_validator_permit:
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"
        if not _THIN:
            bt.logging.trace(f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, "Hotkey recognized!"

    async def priority(self, synapse: bt.Synapse) -> float:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[caller_uid])
        return priority

    def save_state(self):
        return False


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            if not _THIN:
                bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
