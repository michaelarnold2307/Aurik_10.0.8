# Ermöglicht Plugin-Importe als aurik6.plugins.*

# ---------------------------------------------------------------------------
# SOTA-Plugins (v9.9.x) — kein Docker, kein CUDA, CPU-only
# ---------------------------------------------------------------------------

from .apollo_plugin import (  # noqa: F401
    ApolloPlugin,
    CodecRepairResult,
    get_apollo,
    repair_codec_artifacts,
)
from .bigvgan_v2_plugin import (  # noqa: F401
    BigVGANv2Plugin,
    VocoderResult,
    get_bigvgan_v2,
    synthesize_audio,
)
from .bs_roformer_plugin import (  # noqa: F401
    BSRoFormerPlugin,
    StemSeparationResult,
    get_bs_roformer,
    separate_stems,
)
from .cqtdiff_plus_plugin import (  # noqa: F401
    CQTdiffPlusPlugin,
    InpaintingResult,
    get_cqtdiff_plus,
    inpaint_gap,
)
from .laion_clap_plugin import (  # noqa: F401
    AudioTaggingResult,
    LAIONCLAPPlugin,
    get_laion_clap,
    tag_audio,
)
from .utmos_plugin import (  # noqa: F401
    MOSResult,
    UTMOSPlugin,
    estimate_mos,
    get_utmos,
)
from .vocos_plugin import (  # noqa: F401
    VocosPlugin,
    VocosResult,
    get_vocos_plugin,
    vocode_mel,
)

__all__ = [
    # BS-RoFormer — Stem Separation
    "BSRoFormerPlugin",
    "StemSeparationResult",
    "separate_stems",
    "get_bs_roformer",
    # CQTdiff+ — Diffusions-Inpainting (Lücken ≥ 50 ms)
    "CQTdiffPlusPlugin",
    "InpaintingResult",
    "inpaint_gap",
    "get_cqtdiff_plus",
    # Apollo — Codec-Artefakt-Entfernung (MP3/AAC/ATRAC)
    "ApolloPlugin",
    "CodecRepairResult",
    "repair_codec_artifacts",
    "get_apollo",
    # Vocos — Primärer Vocoder (MIT, 8× schneller als BigVGAN-v2 auf CPU)
    "VocosPlugin",
    "VocosResult",
    "vocode_mel",
    "get_vocos_plugin",
    # BigVGAN-v2 — Sekundärer Vocoder (optional, Apache 2.0)
    "BigVGANv2Plugin",
    "VocoderResult",
    "synthesize_audio",
    "get_bigvgan_v2",
    # LAION-CLAP — Audio-Tagging Instrumente/Genre/Material
    "LAIONCLAPPlugin",
    "AudioTaggingResult",
    "tag_audio",
    "get_laion_clap",
    # UTMOS — MOS-Schätzung ohne Referenz (Musik-orientiert)
    "UTMOSPlugin",
    "MOSResult",
    "estimate_mos",
    "get_utmos",
]
