"""Map analyzer outputs to the dict shapes miners attach onto synapse items."""

from __future__ import annotations

from typing import Any, Dict, Optional

from alpharidge_ai.models.article_intelligence import SCHEMA_VERSION
from alpharidge_ai.triage import TRIAGE_SCHEMA_VERSION


def intel_to_analysis_dict(
    intel,
    *,
    triage_rec: Optional[dict] = None,
    proof: Optional[dict] = None,
) -> Dict[str, Any]:
    """Build NewsArticleAnalysisBase kwargs (plus analysisData) from ArticleIntelligence."""
    analysis_blob = intel.model_dump()
    if triage_rec is not None:
        analysis_blob["triage_schema_version"] = TRIAGE_SCHEMA_VERSION
        analysis_blob["triage"] = triage_rec
        analysis_blob["proof_of_read"] = proof
    return {
        "sentiment": intel.overall_sentiment.value,
        "sectorId": intel.topic_signature.primary_sector_id,
        "sectorSymbol": intel.topic_signature.primary_sector_symbol,
        "contentType": intel.content_type.value,
        "technicalQuality": (
            intel.technical_quality
            if isinstance(intel.technical_quality, str)
            else intel.technical_quality.value
            if hasattr(intel.technical_quality, "value")
            else str(intel.technical_quality)
        ),
        "marketAnalysis": intel.market_analysis_type.value,
        "impactPotential": intel.impact_potential.value,
        "relevanceConfidence": "high" if intel.assets else "low",
        "overallSentimentScore": intel.overall_sentiment_score,
        "sentimentDirection": intel.sentiment_direction.value,
        "urgency": intel.urgency.value,
        "temporalFocus": intel.temporal_focus.value,
        "factualConfidence": intel.factual_confidence.value,
        "positioningSignal": intel.positioning_signal.value,
        "targetAudience": intel.target_audience.value,
        "credibilityFlag": intel.credibility_flag.value,
        "primaryGeo": intel.primary_geo.value,
        "marketSession": intel.market_session.value,
        "detectedLanguage": intel.detected_language,
        "stalenessFlag": intel.staleness_flag.value,
        "forwardEventType": intel.forward_event_type.value,
        "assets": [a.model_dump() for a in intel.assets],
        "entities": [e.model_dump() for e in intel.entities],
        "economicData": [d.model_dump() for d in intel.economic_data],
        "numericClaims": [c.model_dump() for c in intel.numeric_claims],
        "quotes": [q.model_dump() for q in intel.quotes],
        "contagionLinks": [l.model_dump() for l in intel.contagion_links],
        "chartSummary": intel.chart_summary.model_dump(),
        "eventFingerprint": intel.event_fingerprint.model_dump(),
        "narrativeKeywords": intel.narrative_keywords,
        "topicSignature": intel.topic_signature.model_dump(),
        "textStats": intel.text_stats.model_dump(),
        "inferredImpacts": (
            [i.model_dump() for i in intel.inferred_impacts] if intel.inferred_impacts else None
        ),
        "analysisData": analysis_blob,
    }


def triage_only_analysis_dict(triage_rec: dict, proof: Optional[dict]) -> Dict[str, Any]:
    return {
        "sentiment": "neutral",
        "analysisData": {
            "schema_version": SCHEMA_VERSION,
            "triage_schema_version": TRIAGE_SCHEMA_VERSION,
            "triage": triage_rec,
            "proof_of_read": proof,
        },
    }
