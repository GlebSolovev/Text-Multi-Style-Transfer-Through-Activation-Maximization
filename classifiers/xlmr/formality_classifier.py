import torch
from torch import tensor
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
from typing import List, Tuple


class FormalityXLMRClassifier:

    def __init__(self):
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained('SkolkovoInstitute/xlmr_formality_classifier')
        self.model = XLMRobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/xlmr_formality_classifier')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

    def encode(self, text: str) -> tensor:
        return self.tokenizer.encode(text, return_tensors='pt').to(self.device)

    def classify(self, batch: tensor) -> Tuple[float, float]:
        result = self.model(batch)
        values = result.logits.data.tolist()[0]
        return values[0], values[1]
