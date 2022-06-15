import torch
from styleformer import Styleformer


class FormalityStyleformerTransformer:

    def __init__(self, to_formal: bool = False, to_casual: bool = False, quality_filter: float = 0.8):
        if (to_formal and to_casual) or (not to_formal and not to_casual):
            raise ValueError("you must choose exactly one option")

        style = 0 if to_formal else 1
        self.styleformer = Styleformer(style=style)

        self.inference_on = 0 if torch.cuda.is_available() else -1
        self.quality_filter = quality_filter

    def transfer(self, text: str) -> str:
        result = self.styleformer.transfer(text, inference_on=self.inference_on, quality_filter=self.quality_filter)
        if result is None:
            raise ValueError("no good quality transfers available :(")
        return result
