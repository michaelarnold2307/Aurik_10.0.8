"""
SVG icon loader for Aurik phase overlay icons.
Uses QSvgRenderer to render vector icons into QPixmap at any requested size.
Thread-safe singleton cache keyed by (icon_name, size).
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

_cache: dict[tuple[str, int], object] = {}
_lock = threading.Lock()

# Absolute path to the phase_icons directory shipped with the application.
_ICON_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "phase_icons")

# Absolute path to the carrier_icons directory.
_CARRIER_ICON_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "carrier_icons")
_DEFECT_ICON_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "defect_icons")

# Mapping from emoji/symbol strings (used as keys in _STAGE_VISUALS) to SVG filenames.
# Each value is the filename stem (without .svg) in the phase_icons directory.
EMOJI_TO_SVG: dict[str, str] = {
    # Init / generic controls
    "⚙": "init",
    "🔗": "signal_chain",
    "💿": "medium",
    "🔎": "defect_scan",
    "🔍": "defect_scan",
    "📋": "plan",
    "🧭": "strategy",
    # Repair / DSP
    "⚡": "dsp_repair",
    "🔧": "dropout_repair",
    "〰": "hum_dc",
    # Timing / pitch
    "🎵": "pitch_timing",
    # Analysis
    "📊": "analysis",
    # Noise
    "🔇": "noise_reduce",
    # Spectral / frequency
    "📶": "spectral",
    # EQ / mastering
    "🎛": "eq",
    "🎚": "dynamics",
    # Vocal
    "🎤": "vocal",
    # Stereo / spatial
    "🎧": "stereo",
    "🌐": "spatial",
    # Instruments
    "🎸": "guitar",
    "🎺": "brass",
    "🥁": "drums",
    "🎹": "piano",
    # Exciter / warmth
    "✨": "exciter",
    # Quality / export
    "✓": "quality_check",
    "💾": "export",
    # Status icons (referenced by name, not emoji)
    "status_fixed": "status_fixed",
    "status_high": "status_high",
    "status_medium": "status_medium",
    "status_low": "status_low",
    "status_clean": "status_clean",
    # Explicit SVG-stem keys (used by differentiated _STAGE_VISUALS entries)
    "reverb": "reverb",
    "transient": "transient",
    "tape_vintage": "tape_vintage",
    "mastering": "mastering",
    "lufs": "lufs",
    "bass": "bass",
    "air": "air",
    "harmonic": "harmonic",
    "declip": "declip",
    # ML model icons
    "🧠": "ml_brain",  # DeepFilterNet / neural AI
    "🔬": "ml_microscope",  # MDX23C analysis
    "🌊": "ml_wave",  # SGMSE+ diffusion
    "🚀": "ml_rocket",  # Apollo
    "🎶": "ml_notes",  # CREPE pitch
    "🎼": "ml_score",  # FCPE
    "📡": "ml_satellite",  # AudioSR bandwidth
    "💎": "ml_gem",  # Vocos neural vocoder
    "🔊": "ml_speaker_loud",  # BigVGAN
    "👂": "ml_ear",  # PANNs audio tagging
    "🌀": "ml_swirl",  # Flow-Matching / CQTdiff+
    "🔮": "ml_crystal",  # CQTdiff+ crystal
    "🔉": "ml_deesser",  # ML-DeEsser
    # DSP algorithm icons
    "🏛": "dsp_columns",  # WPE reverb
    "🔽": "dsp_lowpass",  # Hochpass/Lowpass filter
    "📉": "dsp_trending",  # Noise profiling
    "🚪": "dsp_gate",  # Noise gate
    "🔄": "dsp_rotate",  # Phase align
    "📐": "ruler",  # LMS-Adaptive / TruePeak / LUFS
    "📏": "ruler",  # LUFS-Norm (same ruler SVG)
    "🛡": "dsp_shield",  # Transient guard
    "⚔": "dsp_transient",  # Transient shaper
    "🌈": "dsp_spectrum",  # Spectral coherence
    "➖": "dsp_dc",  # DC offset
}


def _make_pixmap(svg_path: str, size: int) -> object:
    """
    Render an SVG file into a transparent QPixmap of the given square size.
    Returns a fallback 1x1 transparent pixmap if the file is missing or QSvgRenderer unavailable.
    """
    try:
        from PyQt5.QtCore import QRectF, Qt
        from PyQt5.QtGui import QPainter, QPixmap
        from PyQt5.QtSvg import QSvgRenderer

        if not os.path.isfile(svg_path):
            logger.warning("aurik_icons: SVG file not found: %s", svg_path)
            px = QPixmap(size, size)
            px.fill(Qt.transparent)
            return px

        renderer = QSvgRenderer(svg_path)
        px = QPixmap(size, size)
        px.fill(Qt.transparent)
        p = QPainter(px)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        renderer.render(p, QRectF(0, 0, size, size))
        p.end()
        return px

    except ImportError:
        logger.debug("aurik_icons: PyQt5.QtSvg not available — returning fallback pixmap")
        try:
            from PyQt5.QtCore import Qt
            from PyQt5.QtGui import QPixmap

            px = QPixmap(size, size)
            px.fill(Qt.transparent)
            return px
        except Exception:
            return None


def get_icon(key: str, size: int = 24) -> object | None:
    """
    Return a QPixmap for the given icon key (emoji or SVG name) at the requested size.
    Result is cached per (key, size). Returns None if PyQt5 is unavailable.

    Args:
        key:  Emoji character or explicit SVG name stem (e.g. "status_fixed").
        size: Square pixel dimension for the rendered pixmap (default 24).

    Returns:
        QPixmap or None.
    """
    cache_key = (key, size)
    if cache_key in _cache:
        return _cache[cache_key]

    with _lock:
        # Double-checked under lock
        if cache_key in _cache:
            return _cache[cache_key]

        svg_stem = EMOJI_TO_SVG.get(key, key)
        svg_path = os.path.join(_ICON_DIR, f"{svg_stem}.svg")
        px = _make_pixmap(svg_path, size)
        _cache[cache_key] = px
        return px


def get_carrier_icon(key: str, size: int = 22) -> object | None:
    """
    Return a QPixmap for the given carrier/medium type key at the requested size.
    SVG files live in resources/carrier_icons/. Result is cached per (key, size).
    Returns None if PyQt5 is unavailable.

    Args:
        key:  Medium type string, e.g. "vinyl", "shellac", "cd_digital".
        size: Square pixel dimension (default 22, matching the HTML label display size).

    Returns:
        QPixmap or None.
    """
    cache_key = (f"__carrier__{key}", size)
    if cache_key in _cache:
        return _cache[cache_key]

    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]

        svg_path = os.path.join(_CARRIER_ICON_DIR, f"{key}.svg")
        px = _make_pixmap(svg_path, size)
        _cache[cache_key] = px
        return px


def get_defect_icon(key: str, size: int = 16) -> object | None:
    """
    Return a QPixmap for the given defect type key at the requested size.
    SVG files live in resources/defect_icons/. Falls back to .png if no .svg exists.
    Result is cached per (key, size). Returns None if PyQt5 is unavailable.

    Args:
        key:  Defect type string, e.g. "clicks", "hum", "dropout".
        size: Square pixel dimension (default 16 for chip-row display).

    Returns:
        QPixmap or None.
    """
    cache_key = (f"__defect__{key}", size)
    if cache_key in _cache:
        return _cache[cache_key]

    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]

        svg_path = os.path.join(_DEFECT_ICON_DIR, f"{key}.svg")
        if not os.path.isfile(svg_path):
            # Fall back to PNG if SVG not present
            svg_path = os.path.join(_DEFECT_ICON_DIR, f"{key}.png")
        px = _make_pixmap(svg_path, size)
        _cache[cache_key] = px
        return px


def clear_cache() -> None:
    """Flush all cached pixmaps (e.g. after DPI change)."""
    with _lock:
        _cache.clear()
