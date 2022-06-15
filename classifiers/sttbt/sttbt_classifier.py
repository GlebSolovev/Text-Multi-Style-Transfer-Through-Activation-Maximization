from abc import ABC, abstractmethod
from typing import List
import torch
from torch.autograd import Variable
from nltk.tokenize import word_tokenize

from onmt import CNNModels, Constants


class STTBTClassifier(ABC):
    Token = str
    TokenizedText = List[Token]
    Batch = List[TokenizedText]

    @classmethod
    @abstractmethod
    def get_model_checkpoint_path(cls) -> str:
        raise NotImplementedError()

    def __init__(self, batch_size: int = 64, max_text_length_in_tokens: int = 50):
        self.batch_size = batch_size
        self.max_text_length_in_tokens = max_text_length_in_tokens
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_checkpoint_path = self.get_model_checkpoint_path()
        checkpoint = torch.load(model_checkpoint_path)
        model_opt = checkpoint['opt']
        self.src_dict = checkpoint['dicts']['src']

        model = CNNModels.ConvNet(model_opt, self.src_dict)
        model.load_state_dict(checkpoint['model'])
        model.to(self.device)

        self.model = model
        self.model.eval()

    def transform_to_batches(self, texts: List[str]) -> List[Batch]:
        batches = []
        batch = []

        for text in texts:
            tokens = word_tokenize(text)
            batch += [tokens[:self.max_text_length_in_tokens]]
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        return batches

    def classify(self, batch: Batch) -> List[float]:
        encoded_batch = self.__to_indexes_and_one_hot_encode(batch)
        outputs = self.model(encoded_batch)
        outputs = Variable(outputs.data, requires_grad=False)
        return list(map(lambda packed_res: packed_res[0], outputs.data.tolist()))

    def __to_indexes_and_one_hot_encode(self, batch: Batch):
        data = [self.src_dict.convertToIdx(b, Constants.UNK_WORD, padding=True) for b in batch]
        srcBatch, lengths = STTBTClassifier.__batchify(data, align_right=False)

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        batch = zip(indices, srcBatch)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        indices, srcBatch = zip(*batch)

        srcBatch = torch.stack(srcBatch, 0).t().contiguous()
        srcBatch.to(self.device)

        # replace volatile=self.volatile with requires_grad=False:
        # first is deprecated and is disabled in modern torch versions
        # b = Variable(b, volatile=self.volatile)
        src = Variable(srcBatch, requires_grad=False)

        # translate Batch
        inp = src % self.src_dict.size()
        inp_ = torch.unsqueeze(inp, 2).to(self.device)

        float_tensor = torch.cuda.FloatTensor if self.device.type != "cpu" else torch.FloatTensor
        one_hot = Variable(float_tensor(src.size(0), src.size(1), self.src_dict.size()).zero_())
        one_hot_scattered = one_hot.scatter_(2, inp_, 1)

        return one_hot_scattered

    @staticmethod
    def __batchify(data, align_right=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        return out, lengths
