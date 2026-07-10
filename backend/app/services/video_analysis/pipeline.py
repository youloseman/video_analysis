"""Minimal pipeline constants extracted from Motus.

Milestone 1 intentionally does NOT port the full Motus
``VideoAnalysisPipeline`` (database, LLM recommendations, video
overlay, Cloudflare-R2 storage, thumbnails). The standalone
side-view driver in ``backend/scripts/analyze_local.py`` reproduces
only the minimal analysis path.

This module exists solely so the copied biomechanics core that
imports ``app.services.video_analysis.pipeline`` keeps working
unchanged. Today that is ``landmark_stabilizer`` importing
``SPORT_SAMPLE_RATES`` for its Butterworth effective-fps calculation.
The values below are copied verbatim from the Motus pipeline so the
behaviour is identical.
"""

# Default frame sampling: process every Nth frame (Motus default for
# sports without a specific rate).
FRAME_SAMPLE_RATE = 3

# Sport-specific sample rates (lower = more frames = better temporal
# resolution). run/bike analyze every frame; the driver's adaptive
# sampling raises the effective rate for long clips.
SPORT_SAMPLE_RATES = {
    "bike": 1,
    "run": 1,
    "swim": 1,
}

__all__ = ["FRAME_SAMPLE_RATE", "SPORT_SAMPLE_RATES"]
