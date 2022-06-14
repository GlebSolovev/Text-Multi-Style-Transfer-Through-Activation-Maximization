from abc import ABC, abstractmethod
from typing import List


class StyleClassifier(ABC):

    Token = str
    TokenizedText = List[Token]
    Batch = List[TokenizedText]

    @abstractmethod
    def transform_to_batches(self, texts: List[str]) -> List[Batch]:
        raise NotImplementedError()

    @abstractmethod
    def classify(self, batch) -> List[float]:
        raise NotImplementedError()
