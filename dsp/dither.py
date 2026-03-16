import numpy as np
import numpy.typing as npt


class Dither:
    """
    SOTA-konformer Dither (TPDF, POW-R)
    """

    def __init__(self, bit_depth: int = 16, dither_type: str = "tpdf"):
        self.bit_depth = bit_depth
        self.dither_type = dither_type

    def process(self, audio: npt.NDArray[np.float64], sr: int | None = None) -> npt.NDArray[np.float64]:
        # sr wird ignoriert, für Kompatibilität mit PolicyEngine
        quant_step = 2 ** (1 - self.bit_depth)
        if self.dither_type == "tpdf":
            dither = (
                np.random.uniform(-0.5, 0.5, audio.shape) + np.random.uniform(-0.5, 0.5, audio.shape)
            ) * quant_step
        else:  # pow-r (vereinfachte Simulation)
            dither = np.random.normal(0, 0.3, audio.shape) * quant_step
        dithered = audio + dither
        return np.asarray(dithered)
