import argparse
import os
from typing import List, NoReturn, Dict, Any

import nltk
import torch

from hpay.internal.model import make_model
from hpay.internal.data import prepare_data, get_cuda, non_pair_data_loader, id2text_sentence


class SentimentHPAYTransformer:

    def __init__(self, to_positive: bool = False, to_negative: bool = False, max_sequence_length: int = 60):
        if (to_positive and to_negative) or (not to_positive and not to_negative):
            raise ValueError("you must choose exactly one option")

        self.input_style_label = 0 if to_positive else 1
        self.max_sequence_length = max_sequence_length
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.data_path = 'hpay/internal/data/amazon/processed_files/'
        self.word_to_id_file = self.data_path + 'word_to_id.txt'
        self.model_checkpoint_file = 'hpay/internal/save/amazon_ae_model_params.pkl'

    def __read_word_to_id_dict(self) -> Dict[str, int]:
        word_dict = {}
        num = 0
        with open(self.word_to_id_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip()
                word = item.split('\t')[0]
                word_dict[word] = num
                num += 1
        return word_dict

    def index_texts_to_file(self, texts: List[str], output_file: str) -> NoReturn:
        word_dict = self.__read_word_to_id_dict()

        id_texts = []
        for text in texts:
            word_list = nltk.word_tokenize(text)
            id_text = []
            for word in word_list:
                word = word.lower()
                id_text.append(word_dict[word])
            id_texts.append(id_text)

        with open(output_file, 'w') as f:
            for id_text in id_texts:
                f.write("%s\n" % (' '.join([str(k) for k in id_text])))

    def transfer_from_index(self, index_text_file: str) -> List[str]:
        args = self.__build_args()

        args.id_to_word, args.vocab_size, args.train_file_list, args.train_label_list = prepare_data(
            data_path=args.data_path, max_num=args.word_dict_max_num, task_type='yelp')

        ae_model = get_cuda(make_model(d_vocab=args.vocab_size,
                                       N=args.num_layers_AE,
                                       d_model=args.transformer_model_size,
                                       latent_size=args.latent_size,
                                       gpu=args.gpu,
                                       d_ff=args.transformer_ff_size), args.gpu)

        ae_model.load_state_dict(torch.load(self.model_checkpoint_file, map_location=self.device))

        eval_data_loader = non_pair_data_loader(
            batch_size=1, id_bos=args.id_bos,
            id_eos=args.id_eos, id_unk=args.id_unk,
            max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size,
            gpu=args.gpu
        )

        eval_file_list = [index_text_file]
        eval_label_list = [[self.input_style_label]]

        eval_data_loader.create_batches(eval_file_list, eval_label_list, if_shuffle=False)
        ae_model.eval()

        transferred_text = []
        for it in range(eval_data_loader.num_batch):
            batch_sentences, tensor_labels, \
            tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, \
            tensor_tgt_mask, tensor_ntokens = eval_data_loader.next_batch()

            latent = ae_model.getLatent(tensor_src, tensor_src_mask)
            style, similarity = ae_model.getSim(latent)
            t_sign = 2 * (1 - tensor_labels.long()) - 1

            trans_emb = style.clone()[torch.arange(style.size(0)), (1 - tensor_labels).long().item()]
            own_emb = style.clone()[torch.arange(style.size(0)), tensor_labels.long().item()]
            w = args.weight
            out_1 = ae_model.beam_decode(latent + t_sign * w * (own_emb + trans_emb), args.beam_size,
                                         args.max_sequence_length, args.id_bos)
            style_1 = id2text_sentence(out_1[0], args.id_to_word)
            transferred_text.append(style_1)

        return transferred_text

    def transfer(self, texts: List[str]) -> List[str]:
        index_texts_file = "sentiment_hpay_transfer_tmp.txt"
        self.index_texts_to_file(texts, index_texts_file)
        result = self.transfer_from_index(index_texts_file)
        os.remove(index_texts_file)
        return result

    def __build_args(self) -> Any:
        parser = argparse.ArgumentParser()
        parser.add_argument('--id_pad', type=int, default=0, help='')
        parser.add_argument('--id_unk', type=int, default=1, help='')
        parser.add_argument('--id_bos', type=int, default=2, help='')
        parser.add_argument('--id_eos', type=int, default=3, help='')

        ######################################################################################
        #  File parameters
        ######################################################################################
        parser.add_argument('--task', type=str, default='amazon', help='Specify datasets.')
        parser.add_argument('--word_to_id_file', type=str, default='', help='')
        parser.add_argument('--data_path', type=str, default=self.data_path, help='')
        parser.add_argument('--name', type=str, default='He')
        parser.add_argument('--beam_size', type=int, default=10)
        parser.add_argument('--epoch', type=int, default=108)

        ######################################################################################
        #  Model parameters
        ######################################################################################
        parser.add_argument('--word_dict_max_num', type=int, default=5, help='')
        parser.add_argument('--batch_size', type=int, default=128, help='')
        parser.add_argument('--max_sequence_length', type=int, default=self.max_sequence_length)
        parser.add_argument('--num_layers_AE', type=int, default=2)
        parser.add_argument('--transformer_model_size', type=int, default=256)
        parser.add_argument('--transformer_ff_size', type=int, default=1024)

        parser.add_argument('--latent_size', type=int, default=256)
        parser.add_argument('--word_dropout', type=float, default=1.0)
        parser.add_argument('--embedding_dropout', type=float, default=0.5)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--label_size', type=int, default=1)

        parser.add_argument('--gpu', type=int, default=0)
        parser.add_argument('--weight', type=float, default=9)
        parser.add_argument('--mode', type=str, default='add')

        parser.add_argument('--if_load_from_checkpoint', type=bool)
        args = parser.parse_args()
        return args
