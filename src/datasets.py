from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel

from src import *
from src import PAD_TOKEN, MASK_TOKEN
from src.eraser_utils import *
from src.vocabulary import *
from src.utils import *
from argparse import ArgumentParser
from tqdm import tqdm
from copy import deepcopy

from bisect import bisect_left, bisect_right
import shutil


class BaseLightningDataModule(LightningDataModule):
    '''
    基础的pytorch lightning数据集类，包含基础参数和基础操作。需要实现self.setup获得dataset.
    '''

    def __init__(self,
                 data_path,
                 tokenizer,
                 max_len=512,
                 batch_size=16,
                 num_workers=0,
                 aspect=-1,
                 shuffle_train=True,
                 collate_fn=None,
                 loss_fn='mse',
                 balance=False,
                 reset_data=False,
                 val_is_test=False,
                 unlimit_test_length=False,
                 **kwargs):
        super().__init__()

        self.tokenizer = tokenizer
        self.aspect = aspect
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.max_len = max_len
        self.shuffle_train = shuffle_train
        self.is_setup = False
        self.loss_fn = loss_fn
        self.balance = balance
        self.reset_data = reset_data
        self.val_is_test = val_is_test
        self.unlimit_test_length = unlimit_test_length

        self.train_dataset = self.val_dataset = self.test_dataset = None

        self.save_hyperparameters(ignore=['tokenizer', 'collate_fn'])

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        parser.add_argument('--data_path', type=str,
                            help='path to trained file.')
        parser.add_argument('--max_len', type=int, default=512,
                            help='maximum document length.')
        parser.add_argument('--batch_size', type=int, default=16,
                            help='batch size.')
        parser.add_argument('--num_workers', type=int, default=0,
                            help='number of workers in dataloader.')
        parser.add_argument('--shuffle_train', action='store_true',
                            help='shuffle training data.')
        parser.add_argument('--mask_rationale_epoch', type=int, default=0,
                            help='# epochs that rationales are masked during training.')
        parser.add_argument('--balance', action='store_true', help='whether to balance the pos and neg samples.')
        parser.add_argument('--reset_data', action='store_true',
                            help='whether to reset processed data saved on your desk.')
        parser.add_argument('--val_is_test', action='store_true',
                            help='set test set as validation set to check the tendency of fscore.')
        parser.add_argument('--unlimited_test_length', action='store_true')

        return parser

    def prepare_data_per_node(self):
        pass

    def prepare_data(self):
        pass

    def setup(self, stage: str) -> None:
        raise NotImplementedError

    def train_dataloader(self):
        assert self.train_dataset is not None, f'call self.setup("fit") first.'
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_train, collate_fn=self.collate_fn)

    def val_dataloader(self):
        if self.val_is_test:
            return self.test_dataloader()

        assert self.val_dataset is not None, f'call self.setup("val") first.'
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        assert self.test_dataset is not None, f'call self.setup("test") first.'
        collate_fn = deepcopy(self.collate_fn)
        collate_fn.max_len = 10000
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, collate_fn=collate_fn)

    def predict_dataloader(self):
        assert self.test_dataset is not None, f'call self.setup("test") first.'
        collate_fn = self.collate_fn
        if self.unlimit_test_length:
            collate_fn = deepcopy(self.collate_fn)
            collate_fn.max_len = 10000
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, collate_fn=collate_fn)


class EraserLightning(BaseLightningDataModule):
    def __init__(self,
                 data_path,
                 tokenizer,
                 max_sent_len=30,
                 max_len=512,
                 max_num_sent=50,
                 batch_size=16,
                 shuffle_train=True,
                 num_workers=0,
                 max_query_len=30,
                 sentence_level=True,
                 reset_data=False,
                 collate_fn=None,
                 **kwargs):
        self.max_sent_len = max_sent_len
        self.max_num_sent = max_num_sent
        self.max_query_len = max_query_len
        self.sentence_level = sentence_level
        super().__init__(data_path=data_path, tokenizer=tokenizer,
                         max_len=max_len, batch_size=batch_size,
                         num_workers=num_workers, shuffle_train=shuffle_train,
                         collate_fn=collate_fn, reset_data=reset_data)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        '''add task specific params'''
        p = parser.add_argument_group('dataset')
        p = super().add_argparse_args(p)
        p.add_argument('--flat_doc', action='store_true',
                       help='whether to flat document or not.')
        p.add_argument('--max_sent_len', type=int, default=30,
                       help='maximum sentence length.')
        p.add_argument('--max_num_sent', type=int, default=50,
                       help='maximum # sentences.')
        p.add_argument('--concat_query', action='store_true',
                       help='whether to concat query at the beginning of sentences.')

        return parser

    def setup(self, stage: str) -> None:
        if self.is_setup:
            return
        self.train_dataset = Eraser(self.data_path, self.tokenizer, 'train',
                                    max_sent_len=self.max_sent_len,
                                    max_num_sent=self.max_num_sent,
                                    max_len=self.max_len,
                                    max_query_len=self.max_query_len,
                                    sentence_level=self.sentence_level,
                                    balance=self.balance, reset_data=self.reset_data)
        self.val_dataset = Eraser(self.data_path, self.tokenizer, 'val',
                                  max_sent_len=self.max_sent_len,
                                  max_num_sent=self.max_num_sent,
                                  max_len=self.max_len,
                                  max_query_len=self.max_query_len,
                                  sentence_level=self.sentence_level,
                                  balance=False, reset_data=self.reset_data)
        self.test_dataset = Eraser(self.data_path, self.tokenizer, 'test',
                                   max_sent_len=self.max_sent_len,
                                   max_num_sent=self.max_num_sent,
                                   max_len=self.max_len,
                                   max_query_len=self.max_query_len,
                                   sentence_level=self.sentence_level,
                                   balance=False, reset_data=self.reset_data)

        self.is_setup = True


class BeerLightning(BaseLightningDataModule):
    def setup(self, stage: str) -> None:
        if self.is_setup:
            return
        self.train_dataset = Beer(self.data_path, self.tokenizer, 'train',
                                  aspect=self.aspect, loss_fn=self.loss_fn,
                                  balance=self.balance, reset_data=self.reset_data)

        self.val_dataset = Beer(self.data_path, self.tokenizer, 'val',
                                aspect=self.aspect, loss_fn=self.loss_fn,
                                balance=False, reset_data=self.reset_data)

        self.test_dataset = Beer(self.data_path, self.tokenizer, 'test',
                                 aspect=self.aspect, loss_fn=self.loss_fn,
                                 balance=False, reset_data=self.reset_data)

        self.is_setup = True

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        p = parser.add_argument_group('dataset')
        p = super().add_argparse_args(p)
        p.add_argument('--aspect', type=int, default=-1,
                       help='which aspect to use.')

        return parser


class FRBeerLightning(BaseLightningDataModule):
    def setup(self, stage: str) -> None:
        if self.is_setup:
            return
        self.train_dataset = BeerData(self.data_path,
                                      aspect=self.aspect,
                                      mode='train',
                                      word2idx=self.tokenizer.w2i,
                                      balance=self.balance,
                                      max_length=self.max_len)

        self.val_dataset = BeerData(self.data_path,
                                      aspect=self.aspect,
                                      mode='dev',
                                      word2idx=self.tokenizer.w2i,
                                      balance=False,
                                      max_length=self.max_len)

        self.test_dataset = BeerAnnotation(os.path.join(self.data_path, 'annotations.json'),
                                      aspect=self.aspect,
                                      word2idx=self.tokenizer.w2i,
                                      max_length=self.max_len)

        self.is_setup = True

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        p = parser.add_argument_group('dataset')
        p = super().add_argparse_args(p)
        p.add_argument('--aspect', type=int, default=-1,
                       help='which aspect to use.')

        return parser


class HotelLightning(BaseLightningDataModule):
    def setup(self, stage: str) -> None:
        if self.is_setup:
            return
        self.train_dataset = Hotel(self.data_path, self.tokenizer, 'train',
                                   aspect=self.aspect, balance=self.balance,
                                   reset_data=self.reset_data)
        self.val_dataset = Hotel(self.data_path, self.tokenizer, 'val',
                                 aspect=self.aspect, balance=False,
                                 reset_data=self.reset_data)
        self.test_dataset = Hotel(self.data_path, self.tokenizer, 'test',
                                  aspect=self.aspect, balance=False,
                                  reset_data=self.reset_data)
        self.is_setup = True

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        p = parser.add_argument_group('dataset')
        p = super().add_argparse_args(p)
        p.add_argument('--aspect', type=int, default=0,
                       help='which aspect to use.')

        return parser


class BaseDataset(Dataset):
    '''基础Dataset类，实现读取数据的逻辑，需要实现setup方法'''

    def __init__(self,
                 data_path,
                 tokenizer,
                 part='train',
                 aspect=-1,
                 balance=False,
                 reset_data=False, ):
        super().__init__()

        self.data_path = data_path
        self.part = part
        self.aspect = aspect
        self.data = None
        self.balance = balance
        self.tokenizer = tokenizer
        self.reset_data = reset_data
        self.use_plms = isinstance(self.tokenizer, PreTrainedTokenizerBase)

        self.setup()

    def save_processed_data(self, saved_file):
        with open(saved_file, 'wb') as f:
            pickle.dump(self.data, f)
        print(f'Processed data saved at {saved_file}.')

    def load_processed_data(self, saved_file):
        with open(saved_file, 'rb') as f:
            self.data = pickle.load(f)
        print(f'Processed data loaded from {saved_file}.')

    def get_save_path(self, basename):
        if self.balance:
            basename = 'balanced_' + basename
        if hasattr(self.tokenizer, 'name_or_path'):
            basename = get_final_path(self.tokenizer.name_or_path) + '_' + basename

        return os.path.join(self.data_path, basename)

    def tokenize(self, word):
        return self.tokenizer(word, add_special_tokens=False,
                              return_attention_mask=False,
                              return_token_type_ids=False)['input_ids']

    def get_ids_and_offsets(self, text):
        '''
        :param text: List[str]
        :return:
        '''
        ids = []
        token_offsets = []
        cur = 0
        for word in text:
            idx = self.tokenize(word)
            cur += len(idx)
            token_offsets.append(cur - 1)
            ids += idx
        ids = np.array(ids, dtype=np.int64)
        token_offsets = np.array(token_offsets, dtype=np.int64)
        return ids, token_offsets

    def setup(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def balance_data(self, data):
        pos_samples = []
        neg_samples = []
        for sample in tqdm(data, desc=f'balancing {self.part} set...'):
            if sample[1] > 0:
                pos_samples.append(sample)
            else:
                neg_samples.append(sample)

        print(f'before balancing: {len(pos_samples)} pos samples, {len(neg_samples)} neg samples.')

        min_num_samples = min(len(pos_samples), len(neg_samples))
        pos_samples = random.sample(pos_samples, min_num_samples)
        neg_samples = random.sample(neg_samples, min_num_samples)

        print(f'after balancing: {len(pos_samples)} pos samples, {len(neg_samples)} neg samples.')

        return pos_samples + neg_samples

    def create_rationale_by_list(self, l, length):
        rationale = np.zeros((length,), dtype=np.int64)
        for e in l:
            rationale[e[0]:e[1]] = 1
        return rationale


class BeerData(Dataset):
    def __init__(self, data_dir, aspect, mode, word2idx, balance=False, max_length=256, neg_thres=0.4, pos_thres=0.6,
                 stem='reviews.aspect{}.{}.txt'):
        super().__init__()
        self.mode_to_name = {'train': 'train', 'dev': 'heldout'}
        self.mode = mode
        self.neg_thres = neg_thres
        self.pos_thres = pos_thres
        self.input_file = os.path.join(data_dir, stem.format(str(aspect), self.mode_to_name[mode]))
        self.inputs = []
        self.masks = []
        self.labels = []
        self._convert_examples_to_arrays(
            self._create_examples(aspect, balance), max_length, word2idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs, masks, labels = self.inputs[item], self.masks[item], self.labels[item]
        return inputs, masks, labels

    def _create_examples(self, aspect, balance=False):
        examples = []
        with open(self.input_file, "r") as f:
            lines = f.readlines()
            for (i, line) in enumerate(lines):
                labels, text = line.split('\t')
                labels = [float(v) for v in labels.split()]
                if labels[aspect] <= self.neg_thres:
                    label = 0
                elif labels[aspect] >= self.pos_thres:
                    label = 1
                else:
                    continue
                examples.append({'text': text, "label": label})
        print('Dataset: Beer Review')
        print('{} samples has {}'.format(self.mode_to_name[self.mode], len(examples)))

        pos_examples = [example for example in examples if example['label'] == 1]
        neg_examples = [example for example in examples if example['label'] == 0]

        print('%s data: %d positive examples, %d negative examples.' %
              (self.mode_to_name[self.mode], len(pos_examples), len(neg_examples)))

        if balance:

            random.seed(12252018)

            print('Make the Training dataset class balanced.')

            min_examples = min(len(pos_examples), len(neg_examples))

            if len(pos_examples) > min_examples:
                pos_examples = random.sample(pos_examples, min_examples)

            if len(neg_examples) > min_examples:
                neg_examples = random.sample(neg_examples, min_examples)

            assert (len(pos_examples) == len(neg_examples))
            examples = pos_examples + neg_examples
            print(
                'After balance training data: %d positive examples, %d negative examples.'
                % (len(pos_examples), len(neg_examples)))
        return examples

    def _convert_single_text(self, text, max_length, word2idx):
        """
        Converts a single text into a list of ids with mask.
        """
        input_ids = []

        text_ = text.strip().split(" ")

        if len(text_) > max_length:
            text_ = text_[0:max_length]

        for word in text_:
            word = word.strip()
            try:
                input_ids.append(word2idx[word])
            except:
                # if the word is not exist in word2idx, use <unknown> token
                input_ids.append(0)

        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1] * len(input_ids)

        # zero-pad up to the max_seq_length.
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_length
        assert len(input_mask) == max_length

        return input_ids, input_mask

    def _convert_examples_to_arrays(self, examples, max_length, word2idx):
        """
        Convert a set of train/dev examples numpy arrays.
        Outputs:
            data -- (num_examples, max_seq_length).
            masks -- (num_examples, max_seq_length).
            labels -- (num_examples, num_classes) in a one-hot format.
        """

        data = []
        labels = []
        masks = []
        for example in examples:
            input_ids, input_mask = self._convert_single_text(example["text"],
                                                              max_length, word2idx)

            data.append(input_ids)
            masks.append(input_mask)
            labels.append(example["label"])

        self.inputs = torch.from_numpy(np.array(data))
        self.masks = torch.from_numpy(np.array(masks))
        self.labels = torch.from_numpy(np.array(labels)).long()


class BeerAnnotation(Dataset):

    def __init__(self, annotation_path, aspect, word2idx, max_length=256, neg_thres=0.4, pos_thres=0.6):
        super().__init__()
        self.inputs = []
        self.masks = []
        self.labels = []
        self.rationales = []
        self._create_example(annotation_path, aspect, word2idx, max_length, pos_thres, neg_thres)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs, masks, labels, rationales = self.inputs[item], self.masks[item], self.labels[item], self.rationales[
            item]
        return inputs, masks, labels, rationales

    def _create_example(self, annotation_path, aspect, word2idx, max_length, pos_thres, neg_thres):
        data = []
        masks = []
        labels = []
        rationales = []

        print('Dataset: Beer Review')

        with open(annotation_path, "r", encoding='utf-8') as fin:
            for counter, line in tqdm(enumerate(fin)):
                item = json.loads(line)

                # obtain the data
                text_ = item["x"]
                y = item["y"][aspect]
                rationale = item[str(aspect)]

                # check if the rationale is all zero
                if len(rationale) == 0:
                    # no rationale for this aspect
                    continue

                # process the label
                if float(y) >= pos_thres:
                    y = 1
                elif float(y) <= neg_thres:
                    y = 0
                else:
                    continue

                # process the text
                input_ids = []
                if len(text_) > max_length:
                    text_ = text_[0:max_length]

                for word in text_:
                    word = word.strip()
                    try:
                        input_ids.append(word2idx[word])
                    except:
                        # word is not exist in word2idx, use <unknown> token
                        input_ids.append(0)

                # process mask
                # The mask has 1 for real word and 0 for padding tokens.
                input_mask = [1] * len(input_ids)

                # zero-pad up to the max_seq_length.
                while len(input_ids) < max_length:
                    input_ids.append(0)
                    input_mask.append(0)

                assert (len(input_ids) == max_length)
                assert (len(input_mask) == max_length)

                # construct rationale
                binary_rationale = [0] * len(input_ids)
                for zs in rationale:
                    start = zs[0]
                    end = zs[1]
                    if start >= max_length:
                        continue
                    if end >= max_length:
                        end = max_length

                    for idx in range(start, end):
                        binary_rationale[idx] = 1

                data.append(input_ids)
                labels.append(y)
                masks.append(input_mask)
                rationales.append(binary_rationale)

        self.inputs = torch.from_numpy(np.array(data))
        self.labels = torch.from_numpy(np.array(labels)).long()
        self.masks = torch.from_numpy(np.array(masks))
        self.rationales = torch.from_numpy(np.array(rationales))
        tot = self.labels.shape[0]
        print('annotation samples has {}'.format(tot))
        pos = torch.sum(self.labels)
        neg = tot - pos
        print('annotation data: %d positive examples, %d negative examples.' %
              (pos, neg))


class Beer(BaseDataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 part='train',
                 aspect=0,
                 loss_fn='mse',
                 balance=False,
                 reset_data=False,
                 neg_thres=0.4,
                 pos_thres=0.6, ):
        part = 'heldout' if part in ['dev', 'val'] else part
        self.loss_fn = loss_fn
        self.neg_thres = neg_thres
        self.pos_thres = pos_thres
        super().__init__(data_path, tokenizer, part, aspect, balance=balance, reset_data=reset_data)

    def setup(self):
        prev_tag = '' if self.loss_fn == 'mse' else 'binarized.'
        self.data = []

        saved_file = self.get_save_path(f'{prev_tag}beer_aspect{self.aspect}_{self.part}.cache')

        if os.path.exists(saved_file) and not self.reset_data:
            self.load_processed_data(saved_file)
            return

        if self.part in ['train', 'heldout']:
            filename = os.path.join(self.data_path, 'reviews.aspect%s.%s.txt' % (self.aspect, self.part))
            with open(filename, encoding='utf8') as f:
                for line in tqdm(f, desc=f'reading {self.part} set'):
                    splited_line = line.strip().split()
                    text = splited_line[5:]

                    ids, token_offsets = self.get_ids_and_offsets(text)

                    scores = splited_line[:5]
                    scores = float(scores[self.aspect])
                    if self.loss_fn == 'ce':
                        scores = 1 if scores >= self.pos_thres else 0 if scores <= self.neg_thres else -1
                    if scores == -1:
                        continue

                    self.data.append((text, scores, None, ids, token_offsets))

            if self.loss_fn == 'ce' and self.balance and self.part == 'train':
                self.data = self.balance_data(self.data)

        else:
            filename = os.path.join(self.data_path, 'annotations.json')
            with open(filename, encoding='utf8') as f:
                for line in tqdm(f, desc=f'reading {self.part} set'):
                    each = json.loads(line.strip())

                    text = each['x']
                    ids, token_offsets = self.get_ids_and_offsets(text)
                    scores = each['y']

                    scores = float(scores[self.aspect])
                    rationales = self.create_rationale_by_list(each['%d' % self.aspect], len(ids))
                    if sum(rationales) == 0:
                        continue
                    assert rationales.shape == ids.shape, f'{rationales.shape}, {ids.shape}'

                    if self.loss_fn == 'ce':
                        scores = 1 if scores >= self.pos_thres else 0 if scores <= self.neg_thres else -1
                    if scores == -1:
                        continue

                    self.data.append((text, scores, rationales, ids, token_offsets))

        self.save_processed_data(saved_file)


class Hotel(BaseDataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 part='train',
                 aspect=-1,
                 balance=False,
                 reset_data=False):
        part = 'dev' if part in ['heldout', 'val'] else part
        self.hotel_map = ['hotel_Location', 'hotel_Service', 'hotel_Cleanliness']
        super().__init__(data_path, tokenizer, part, aspect, balance=balance, reset_data=reset_data)

    def setup(self):
        saved_file = self.get_save_path(f'hotel_aspect{self.aspect}_{self.part}.cache')

        if os.path.exists(saved_file) and not self.reset_data:
            self.load_processed_data(saved_file)
            return

        self.data = []
        if self.part in ['train', 'dev']:
            filename = os.path.join(self.data_path, 'hotel%d/%s.tsv' % (self.aspect, self.part))
            with open(filename, encoding='utf8') as f:
                next(f)
                for line in tqdm(f, desc=f'reading {self.part} set'):
                    splited_line = line.strip().split('\t')
                    text = splited_line[-1].split()
                    ids, token_offsets = self.get_ids_and_offsets(text)
                    label = splited_line[1]

                    self.data.append((text, int(label), None, ids, token_offsets))

            if self.balance and self.part == 'train':
                self.data = self.balance_data(self.data)

        else:
            filename = os.path.join(self.data_path, 'annoated/%s.train' % (self.hotel_map[int(self.aspect)]))
            with open(filename, encoding='utf8') as f:
                next(f)
                for line in tqdm(f, desc=f'reading {self.part} set'):
                    splited_line = line.strip().split('\t')

                    text = splited_line[2].split()
                    ids, token_offsets = self.get_ids_and_offsets(text)
                    scores = splited_line[1]
                    rationales = splited_line[3].split()
                    rationales = np.array(rationales, dtype=np.int64)
                    assert rationales.shape == ids.shape, f'{rationales.shape} != {ids.shape}'
                    # spans = []
                    # s = e = -1
                    # for i, rat in enumerate(rationales.split()):
                    #     if float(rat)>0:
                    #         if s==-1:
                    #             s=i
                    #         e=i+1
                    #     else:
                    #         if s!=-1:
                    #             spans.append((s, e))
                    #         s=e=-1
                    # if not spans:
                    #     spans.append((-1, -1))

                    self.data.append((text, int(scores), rationales, ids, token_offsets))

        self.save_processed_data(saved_file)


class Eraser(BaseDataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 part='train',
                 max_sent_len=30,
                 max_num_sent=50,
                 max_len=512,
                 max_query_len=30,
                 sentence_level=True,
                 balance=False,
                 reset_data=False):
        self.max_sent_len = max_sent_len
        self.max_num_sent = max_num_sent
        self.max_len = max_len
        self.max_query_len = max_query_len
        self.sentence_level = sentence_level

        # 在Eraser Benchmark里面的各种标签映射。
        self.label_map = {'POS': 1, 'NEG': 0, 'REFUTES': 0, 'SUPPORTS': 1,
                          'False': 0, 'True': 1, False: 0, True: 1}

        super().__init__(data_path, tokenizer, part, None, balance=balance, reset_data=reset_data)

    def setup(self):
        '''
        read dataset from curtain path.
        :return:
        '''
        saved_file = self.get_save_path(f'{self.part}_{self.max_len}_{self.max_query_len}_{self.sentence_level}.cache')

        if os.path.exists(saved_file) and not self.reset_data:
            self.load_processed_data(saved_file)
            return

        self.data = []

        # 获取数据路径与标注数据。
        path = os.path.join(self.data_path, self.part + '.jsonl')
        datas = annotations_from_jsonl(path)
        datas = self.clean(datas)
        doc_ids = set(e.docid.strip('.') for e in
                      chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(datas)))))
        docs = load_documents(self.data_path, doc_ids)

        for data in tqdm(datas):
            # 读取当前标注数据对应的文档。
            # assert len(data.docids) == 1, f'{data.docids}'
            docid = data.docids[0].strip('.') if bool(data.docids) else data.annotation_id
            doc = docs[docid]
            flat_doc = list(chain.from_iterable(doc))

            # 将query分开成list，并且截断。
            query = data.query.split()
            if not isalpha(query[-1][-1]):
                query[-1], sym = query[-1][:-1], query[-1][-1]
                query.append(sym)
            query = query[:self.max_query_len]
            query_ids, query_offsets = self.get_ids_and_offsets(query)

            # 将所有的句子转化为子词，并且生成对应的offset
            ids = []
            token_offsets = []
            cur = 0
            for sent in doc:
                idx = []
                offset = []
                for each in sent:
                    idx.append(self.tokenize(each))
                    cur += len(idx[-1])
                    offset.append(cur - 1)
                token_offsets.append(offset)
                ids.append(list(chain.from_iterable(idx)))

            # 获取每个句子对应的token span。
            sentence_spans = []
            s = e = 0
            for sent in ids:
                e += len(sent)
                sentence_spans.append((s, e))
                # assert [flat_doc[s:e]] == sent, f'{flat_doc[s:e]}, {sent}'
                s = e
            assert len(doc) == len(sentence_spans)

            # 找到句子最大长度对应的那个span, 512 -> (a, b), a < 512, b >= 512.
            max_len = self.max_len - len(query) - 1 - (2 if self.use_plms else 0)
            idx = bisect_left([each[1] for each in sentence_spans], max_len)
            idx = min(idx, len(sentence_spans) - 1)  # 最大长度小于max_len
            valid_num_sent = idx + 1

            # 就算句子最大长度小于max_len很多，这个写法也OK。
            num_less_word = max_len - sentence_spans[idx][0]
            text = list(chain.from_iterable(ids[:idx])) + ids[idx][:num_less_word]

            # 截断offset到最后一个token。如果不是一个完整的token，则截断。
            token_offsets = list(chain.from_iterable(token_offsets))
            token_offset_idx = bisect_left(token_offsets, len(text))
            token_offsets = token_offsets[:token_offset_idx + 1]
            token_offsets[-1] = len(text) - 1

            # 将sentence_span也截断。
            sentence_spans = sentence_spans[:valid_num_sent]
            if sentence_spans[-1][-1] > max_len:
                sentence_spans[-1] = (sentence_spans[-1][0], max_len)

            # 将没有超过max_len的rationale拿出来。
            rationales = []
            for e in data.all_evidences():
                if self.sentence_level:
                    if e.start_sentence > idx:  # 如果超过，则跳过。
                        continue
                    if e.end_sentence < idx:
                        end_sentence = e.end_sentence + 1  # 转换成左闭右开区间
                    else:
                        end_sentence = idx + 1
                    rationales.append((e.start_sentence, end_sentence))

                    assert list(chain.from_iterable(doc[e.start_sentence:e.end_sentence])) == e.text.split(), \
                        f"{list(chain.from_iterable(doc[e.start_sentence:e.end_sentence]))}, {e.text.split()}"
                else:
                    final_idx = sentence_spans[-1][-1]
                    if e.start_token > final_idx:
                        continue
                    if e.end_token <= final_idx:
                        end_token = e.end_token
                    else:
                        end_token = len(text)
                    rationales.append((e.start_token, end_token))

                    # TODO: delete this. check maybe wrong.
                    # assert flat_doc[rationales[-1][0]:rationales[-1][1]] == e.text.split(), \
                    #     f"{flat_doc[rationales[-1][0]:rationales[-1][1]]}, {e.text.split()}"
            # TODO: 将rationale转化为list
            # rationales = self.create_rationale_by_list(rationales, )
            # print(self.max_len)
            # don't check this, todo: later
            # assert len(list(chain.from_iterable(ids))) <= self.max_len, f'{len(list(chain.from_iterable(ids)))}, {self.max_len}'
            self.data.append((text, int(self.label_map[data.classification]), rationales, query, sentence_spans, ids,
                              query_ids, token_offsets))

        if self.balance and self.part == 'train':
            self.data = self.balance_data(self.data)

        self.save_processed_data(saved_file)

    def clean(self, data):
        '''清除非空数据'''
        new_data = []
        for each in data:
            if bool(each.all_evidences()):
                new_data.append(each)

        return new_data


class BaseCollator:
    '''Collator基类，定义了一些基础的参数和函数。比如截断rationale，掩盖rationale，解码张量和填充等。'''

    def __init__(self, vocab, max_len: int = 10000,
                 mask_rationale_epoch=0, sentence_level=False, **kwargs):
        self.vocab = vocab
        self.pad_id = vocab.pad_token_id
        self.unk_id = vocab.unk_token_id
        self.mask_id = vocab.mask_token_id
        self.sep_id = vocab.sep_token_id
        self.max_len = max_len
        self.mask_rationale_epoch = mask_rationale_epoch
        self.sentence_level = sentence_level
        self.use_plms = isinstance(vocab, PreTrainedTokenizerBase)

    def chunk_rationales(self, text, rationales, rationale_is_list):
        '''截断rationale，将rationale截断到文本最大长度的位置。'''
        if rationale_is_list:
            return [[(r[0], min(r[1], len(t))) for r in each if r[0] < len(t)] for each, t in zip(rationales, text)]
        return rationales

    def mask_rationales(self, ids, rationales, rationale_is_list, sentence_spans=None):
        '''
        掩盖rationale。测试rationale的合理性。
        分为rationale是否是列表，和是不是句子级的rationale的情况。
        '''
        if rationale_is_list:
            # span level rationale，一段一段填。
            for t, r in zip(ids, rationales):
                for each in r:
                    # 句子级的rationale只影响开始位置和结束位置。
                    if self.sentence_level:
                        s = sentence_spans[each[0]][0]
                        e = sentence_spans[each[1]][1]
                    else:
                        s = each[0]
                        e = each[1]

                    t[s:e] = [MASK_TOKEN] * (e - s)
        else:
            # token level rationale，一个一个填。
            for t, r in zip(ids, rationales):
                for i in range(len(t)):
                    t[i] = MASK_TOKEN if r[i] else t[i]

    def decode(self, tensor):
        '''
        解码单个张量。以<pad>为结束标志。
        :param tensor: (L, ), torch.tensor.
        :return: 解码后的list。
        '''
        text = []
        for each in tensor:
            e = each
            if e == self.pad_id:
                return text
            text.append(self.vocab.i2w[e])

        return text

    def find_and_pad_maxlen(self, xs, pad_content):
        '''
        找到tokens里面的最大长度，并且将整个列表都填充到指定长度。
        :param xs: list, 不同长度的列表
        :param pad_content: 填充的内容，和tokens[0][0]的类型一致
        :return: 填充到同一长度过后的列表。
        '''
        length = [len(x) for x in xs]
        mx_len = max(length)
        padded_tokens = [self.pad(each, mx_len, pad_content) for each in xs]

        return np.array(padded_tokens)

    def pad(self, x, length, pad_content):
        '''
        填充函数，将token使用pad_content填充到max_len
        :param x: list，待填充的内容。
        :param length: 指定长度。
        :param pad_content: 填充内容。
        :return: 填充后的列表
        '''
        if len(x) >= length:
            return x[:length]

        return np.concatenate([x, np.full((length - len(x)), pad_content)], axis=0)

    def rectified_offset(self, ids):
        token_offsets = []
        for idx in ids:
            offsets = []
            cur = 0
            for each in idx:
                cur += len(each)
                offsets.append(cur)
            token_offsets.append(offsets)

        return np.array(token_offsets, dtype=np.long)

    def to_device(self, d, device):
        nd = {}
        for k, v in d.items():
            nd[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        return nd


class FRCollator(BaseCollator):
    def __call__(self, batch):
        '''
        基础Collator，将文本和rationale填充到对应长度，分别组成张量。
        对于Beer和Hotel数据集可用。
        :return Dict
        '''
        rationales = None
        if len(batch[0]) == 4:
            ids, masks, labels, rationales = zip(*batch)
        else:
            ids, masks, labels = zip(*batch)

        masks = torch.stack(masks, dim=0)
        lengths = masks.sum(-1)
        new_idx = torch.argsort(lengths, descending=True)
        masks = masks[new_idx]
        lengths = lengths[new_idx]

        ids = torch.stack(ids, dim=0)[new_idx]

        labels = torch.stack(labels, dim=0)[new_idx]
        if rationales is not None: rationales = torch.stack(rationales, dim=0)[new_idx]


        return {
            'tensors': ids,
            'mask': masks,
            'labels': labels,
            'rationales': rationales,
            'text': '',
            'lengths': lengths,
        }


class Collator(BaseCollator):
    def __init__(self, vocab, max_len: int = 10000, mask_rationale_epoch=0, **kwargs):
        super().__init__(vocab=vocab,
                         max_len=max_len, mask_rationale_epoch=mask_rationale_epoch,
                         sentence_level=False)
        self.rationale_is_list = None

    def __call__(self, batch):
        '''
        基础Collator，将文本和rationale填充到对应长度，分别组成张量。
        对于Beer和Hotel数据集可用。
        :return Dict
        '''

        text, label, rationales, ids, token_offsets = zip(*batch)
        has_rationale = any([each is not None for each in rationales])

        # print(rationales)
        # print(self.rationale_is_list)
        if has_rationale:
            # 将token_level的rationale对应到正确的位置。左闭右开。
            offset_rationale = []
            for idx, (rat, offset) in enumerate(zip(rationales, token_offsets)):
                assert len(rat) == len(offset), f'{len(rat)}, {len(offset)}'
                cur_rationale = np.zeros((offset[-1] + 1,), dtype=np.int64)
                for i, each in enumerate(rat):
                    if i == len(rat) - 1:
                        cur_rationale[offset[i]:] = each
                    else:
                        cur_rationale[offset[i]: offset[i + 1]] = each
                offset_rationale.append(cur_rationale)
            rationales = offset_rationale

        # we don't want to chunk rationales, especially on test set.
        # rationales = self.chunk_rationales(text, rationales, rationale_is_list)
        # TODO: 新版改正。
        # if self.mask_rationale_epoch > 0 and has_ratioale:
        #     self.mask_rationales(ids, rationales, self.rationale_is_list)

        # token to idx.

        tokens = [each[:self.max_len] for each in ids]
        if self.use_plms:
            tokens = [([self.vocab.cls_token_id] + each[:self.max_len - 2] + [self.vocab.sep_token_id])
                      for each in tokens]
        # pad
        padded_tokens = self.find_and_pad_maxlen(tokens, self.pad_id)
        # list to tensor.
        tensors = torch.tensor(padded_tokens, dtype=torch.long)
        mask = tensors != self.pad_id
        label = np.array(label)
        if isinstance(label[0], float):
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(-1)
        else:
            label = torch.tensor(label, dtype=torch.long)

        tensor_rationales = None
        # pad rationales and convert them to tensors.
        if has_rationale:
            padded_rationales = self.find_and_pad_maxlen(rationales, 0)
            tensor_rationales = torch.tensor(padded_rationales, dtype=torch.long)

        return {
            'tensors': tensors,
            'mask': mask,
            'labels': label,
            'rationales': tensor_rationales,
            'text': text,
            'lengths': mask.sum(-1)
        }


# TODO: 使用ids动态构建句子，而不是text
class CollatorWithQuery(BaseCollator):
    def __init__(self, vocab, max_len: int = 10000,
                 mask_rationale_epoch=0, sentence_level=True, **kwargs):
        super().__init__(vocab=vocab,
                         max_len=max_len, mask_rationale_epoch=mask_rationale_epoch,
                         sentence_level=sentence_level)

    def __call__(self, batch):
        '''
        Eraser benchmark使用的collator
        :return: tensor Dict
        '''
        text, label, rationales, query, sentence_spans, ids, query_ids, token_offsets = zip(*batch)
        # print([len(list(chain.from_iterable(e))) for e in ids])
        ###### 预处理 ######
        new_sentence_spans = []
        new_rationales = []
        token_type_ids = []
        new_ids = []
        for idx, (txt, que, rat, ss, to) in enumerate(zip(text, query_ids, rationales,
                                                          sentence_spans, token_offsets)):
            # 将query拼接到token之前。
            # query 和 text 之间有一个 <sep> .
            # 拼接之前展开子词
            # txt=list(chain.from_iterable(txt))
            txt = np.concatenate([que, [self.sep_id], txt], axis=0)
            type_ids = [0] * (len(que) + 1) + [1] * len(txt)
            if self.use_plms:
                txt = [self.vocab.cls_token_id] + txt[:self.max_len - 2] + [self.vocab.sep_token_id]
                type_ids = [0] + type_ids[:self.max_len - 2] + [1]

            new_ids.append(txt)
            token_type_ids.append(type_ids)

            # 句子span的范围，在最前面加上一个0到query的长度+1，后面的都加上query的长度+1
            n_s = [(0, len(que) + 1)]
            for s in ss:
                n_s.append((to[s[0]] + len(que) + 1, to[s[1] - 1] + len(que) + 1 + 1))
            new_sentence_spans.append(n_s)

            # 根据是不是句子级别，rationale的内容不一样
            # 是句子级别则返回开始和结束的句子idx, [s, e).
            # token级别则返回开始和结束的token idx, [s, e).
            n_rat = []
            for r in rat:
                if self.sentence_level:
                    # 如果是句子级的，则+1，需要营造右边开区间。
                    ans = (r[0] + 1, r[1] + 1)
                else:
                    # 否则加上query长度再+1，因为还有个<sep>符号。
                    ans = (to[r[0]] + len(que) + 1, to[r[1] - 1] + len(que) + 1 + 1)
                n_rat.append(ans)
            new_rationales.append(n_rat)
        ids = new_ids
        # print([len(e) for e in ids])
        sentence_spans = new_sentence_spans
        rationales = new_rationales
        ###### 预处理结束 ######

        ###### 句子级处理 ######
        sentence_mask = None
        sentence_type_ids = None
        if self.sentence_level:
            # 生成sentence mask和sentence type ids。
            sentence_type_ids = []
            sentence_mask = []
            for ss in sentence_spans:
                # sentence mask。用来指示每个样本的最大有效句子长度
                cur_mask = [1] * len(ss)

                # sentence type ids, 指示每一句话的范围。从0开始。
                cur_sent_id = 0
                type_ids = []
                for s in ss:
                    type_ids += [cur_sent_id] * (s[1] - s[0])
                    cur_sent_id += 1

                sentence_type_ids.append(type_ids)
                sentence_mask.append(cur_mask)

            # pad
            padded_sentence_mask = self.find_and_pad_maxlen(sentence_mask, 0)
            padded_sentence_type_ids = self.find_and_pad_maxlen(sentence_type_ids, 0)
            # to tensor.
            sentence_mask = torch.tensor(padded_sentence_mask, dtype=torch.bool)
            sentence_type_ids = torch.tensor(padded_sentence_type_ids, dtype=torch.long)
        ###### 句子级处理结束 ######

        # 不截断。
        # rationales = self.chunk_rationales(text, rationales)
        if self.mask_rationale_epoch > 0:
            self.mask_rationales(text, rationales)

        # token to idx.
        tokens = ids
        # pad.
        padded_tokens = self.find_and_pad_maxlen(tokens, self.pad_id)
        padded_rationales = self.find_and_pad_maxlen(rationales, (-1, -1))
        padded_sentence_spans = self.find_and_pad_maxlen(sentence_spans, (0, 0))
        padded_token_type_ids = self.find_and_pad_maxlen(token_type_ids, -1)

        # to tensor.
        tensors = torch.tensor(padded_tokens, dtype=torch.long)
        mask = tensors != self.pad_id
        label = torch.tensor(label, dtype=torch.long)  # [B, ]
        tensor_rationales = torch.tensor(padded_rationales, dtype=torch.long)
        tensor_sentence_spans = torch.tensor(padded_sentence_spans, dtype=torch.long)
        tensor_token_type_ids = torch.tensor(padded_token_type_ids, dtype=torch.long)
        # query_tensors=None
        # if not self.concat_query:
        #     query_tokens = [[self.vocab.w2i.get(each, self.unk_id) for each in e] for e in query]
        #     padded_query_tokens = self.find_and_pad_maxlen(query_tokens, self.pad_id)
        #     query_tensors = torch.tensor(padded_query_tokens, dtype=torch.long)

        return {
            'tensors': tensors,
            'mask': mask,
            'token_type_ids': tensor_token_type_ids,
            'labels': label,
            'rationales': tensor_rationales,
            'sentence_spans': tensor_sentence_spans,
            # 'query':query_tensors,
            'sentence_mask': sentence_mask,
            'sentence_type_ids': sentence_type_ids,
            'text': text,
        }

    def chunk_rationales(self, text, rationales):
        return super().chunk_rationales(text, rationales, True)

    def mask_rationales(self, text, rationales, sentence_spans):
        return super().mask_rationales(text, rationales, True, sentence_spans)


def get_dataset_cls(dataset_name):
    if dataset_name == 'beer':
        return FRBeerLightning
    elif dataset_name == 'hotel':
        return HotelLightning
    elif dataset_name in eraser:
        return EraserLightning
    else:
        raise ValueError(f'Unknown dataset name {dataset_name}')


def get_collator_cls(dataset_name):
    if dataset_name not in eraser:
        return FRCollator
    else:
        return CollatorWithQuery


datasets = ['boolq', 'multirc', 'movies', 'fever']
if __name__ == '__main__':
    vocab = Vocabulary()
    vectors = load_embeddings('../pretrained/glove.6B.50d.txt', vocab, 50)

    collate_fn = Collator(vocab, max_len=1000, sentence_level=False)

    dataset1 = Beer('../data/beer', vocab, 'test', 2, loss_fn='ce', )
    dataset1.setup()

    # dataset2 = BeerData('../data/beer', 0, 'train', vocab.w2i, max_length=1000)
    dataset2 = BeerAnnotation('../data/beer/annotations.json', 2, vocab.w2i, 1000)
    assert len(dataset1) == len(dataset2), f'{len(dataset1)}, {len(dataset2)}'
    for a, b in tqdm(zip(dataset1, dataset2)):
        # assert len(a[2]) == b[-1].shape[0], f'{len(a[2])}, {b[-1].shape[0]}'
        assert b[-1].sum() == sum(a[2]), f'{b[-1].sum()}, {sum(a[2])}'
        assert all(x == y for x, y in zip(a[2], b[-1])), f'{a[2]}, {b[-1]}'
        # assert all(x==y for x, y in zip(a[-2], b[0])), f'{a[-2]}, {b[0]}'


    exit(0)


    def run_check(dataset, ds, partition):
        datas = annotations_from_jsonl(f'../data/{ds}/{partition}.jsonl')
        datas = Eraser.clean(None, datas)
        doc_ids = set(e.docid.strip('.') for e in
                      chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(datas)))))
        docs = load_documents(f'../data/{ds}', doc_ids)

        gold = []
        for data in tqdm(datas):
            for e in data.all_evidences():
                assert len(
                    e.text.split()) == e.end_token - e.start_token, f'{len(e.text.split())}, {e.end_token}, {e.start_token}'
            gold.append(' '.join([e.text.lower() for e in sorted(data.all_evidences(), key=lambda x: x.start_token)]))

        # gold = []
        # with open('../data/hotel/annoated/hotel_Cleanliness.train', encoding='utf8') as f:
        #     next(f)
        #     for line in tqdm(f):
        #         line = line.strip().split('\t')
        #         label = line[1]
        #         text = line[2].split()
        #         rationale = line[3].split()
        #
        #         # assert len(text)==len(rationale), f'{len(text)}, {len(rationale)}'
        #
        #         ans = ''
        #         for t, r in zip(text, rationale):
        #             if int(r)==1:
        #                 ans += '**' + t + '** '
        #             else:
        #                 ans += t + ' '
        #         gold.append(ans)
        #         if len(text) != len(rationale):
        #             print(len(text)-len(rationale))
        #             print(ans)
        #             print(sum([float(each) for each in rationale]))
        # dataset = HotelLightning('../data/hotel', tokenizer=vocab, collate_fn=collate_fn, batch_size=1,
        #                          aspect=2, reset_data=True)
        dataset.setup(None)
        rats = []
        generate = []
        cur = 0
        for idx, each in enumerate(tqdm(eval(f'dataset.{partition}_dataloader()'))):
            text = [[vocab.decode(e.item()) for e in ee] for ee in each['tensors']]

            for rat, tex in zip(each['rationales'], text):
                ans = ''
                for i, t in enumerate(tex):
                    if torch.bitwise_and(i >= rat[:, 0], i < rat[:, 1]).any():
                        ans += t + ' '
                    # else:
                    #     ans += t + ' '
                generate.append(ans)
                rats.append(rat)

        for idx, (i, j) in enumerate(zip(generate, gold)):
            for a, b in zip(i.split(), j.split()):
                if a != '[UNK]' and a != b:
                    cur += 1
                    print(i)
                    print(j)
                    print(datas[idx])
                    break
        print(f'{ds}: {partition} -> {cur}')


    vocab = Vocabulary()
    vectors = load_embeddings('../../pretrained/glove.6B.50d.txt', vocab, 50)

    collate_fn = CollatorWithQuery(vocab, max_len=1000, sentence_level=False)
    for ds in datasets:
        dataset = EraserLightning(f'../data/{ds}', tokenizer=vocab, collate_fn=collate_fn,
                                  sentence_level=False, reset_data=False, batch_size=1,
                                  max_len=1000000, shuffle_train=False)
        run_check(dataset, ds, 'test')
        # for partition in ['train', 'val', 'test']:
        #     run_check(dataset, ds, partition)
    # dataset.setup(None)
    # for e in tqdm(dataset.train_dataloader()):
    #     # print(e)
    #     # exit(0)
    #     pass
    # for e in tqdm(dataset.val_dataloader()):
    #     # print(e)
    #     # exit(0)
    #     pass
    # for e in tqdm(dataset.test_dataloader()):
    #     # print(e)
    #     # exit(0)
    #     pass

    # exit(0)
    # 数据集统计
    # for dataset in datasets:
    #     data = annotations_from_jsonl('../../data/%s/train.jsonl' % dataset)
    #
    #     tokens = np.zeros((100000,))
    #     sentence = np.zeros((100000,))
    #     mx_tk = -1
    #     mx_s = -1
    #     for d in data:
    #         for e in d.all_evidences():
    #             tokens[e.start_token:e.end_token] += 1
    #             sentence[e.start_sentence:e.end_sentence + 1] += 1
    #
    #             mx_tk = max(mx_tk, e.end_token)
    #             mx_s = max(mx_s, e.end_sentence + 1)
    #     plt.bar(np.arange(mx_tk), tokens[:mx_tk])
    #     plt.title(f'{dataset}')
    #     plt.show()
    #     plt.bar(np.arange(mx_s), sentence[:mx_s])
    #     plt.title(f'{dataset}')
    #     plt.show()
    #     continue
    #
    #     doc_ids = set([each.docids[0] if bool(each.docids) else each.annotation_id for each in data])
    #     document = load_documents('../../data/%s' % dataset, doc_ids)
    #     doc = load_flattened_documents('../../data/%s' % dataset, doc_ids)
    #     print(len(document))
    #     print(len(data))
    #
    #     # exit(0)
    #
    #     cls = Counter([each['classification'] for each in data])
    #     # docids_len = Counter([len(each['docids']) if bool(each['docids']) else 0 for each in data])
    #     evidences_len = Counter([len(each.all_evidences()) for each in data])
    #
    #     total_rationale_len = []
    #     for each in data:
    #         total_length = 0
    #         for e in each.all_evidences():
    #             total_length += e.end_token - e.start_token
    #         total_rationale_len.append(total_length)
    #     tr_len = Counter(total_rationale_len)
    #     total_rationale_len = np.array(total_rationale_len)
    #
    #     temp = []
    #     for each in data:
    #         for e in each.all_evidences():
    #             # print(e)
    #             temp.append(int(e['end_token']) - int(e['start_token']))
    #     rationale_len = Counter(temp)
    #
    #     query_len = Counter([len(each['query'].split()) for each in data])
    #     # sentence_len = Counter([len(each) for each in chain.from_iterable(document.values())])
    #     # doc_len = Counter([len(each) for each in doc.values()])
    #
    #     sent_len = [len(each) for each in chain.from_iterable(document.values())]
    #     doc_len = np.array([len(each) for each in doc.values()])
    #     e_len = [len(each.all_evidences()) for each in data]
    #     q_len = [len(each['query'].split()) for each in data]
    #
    #     plot_bar(cls, '%s label dist.' % dataset, 'label', '#', True)
    #
    #     # print(f'sent_len: max: {max(sent_len)}, avg: {np.mean(sent_len)}')
    #     # print(f'doc_len: max: {max(doc_len)}, avg: {np.mean(doc_len)}')
    #     # print(f'e_len: max: {max(e_len)}, avg: {np.mean(e_len)}')
    #     # print(f'r_len: max: {max(temp)}, avg: {np.mean(temp)}')
    #     # print(f'total_r_len: max: {max(total_rationale_len)}, avg: {np.mean(total_rationale_len)}')
    #     # print(f'total_r_len_1: max: {max(total_rationale_len/doc_len)}, avg: {np.mean(total_rationale_len/doc_len)}')
    #     # print(f'q_len: max: {max(q_len)}, avg: {np.mean(q_len)}')
    #
    #     # print(cls)
    #     # print(docids_len) # 1
    #     # print(evidences_len)
    #     # print(rationale_len)
    #     # print(query_len)
    #     # print(sentence_len)
    #     # print(doc_len)
    #
    #     # plot_bar(cls, '%s label dist.'%dataset, 'label', '#', True)
    #     # # plot_bar(docids_len, '%s doc length dist.'%dataset, 'length', '#', True)
    #     # plot_bar(evidences_len, '%s evidence length dist.'%dataset, 'length', '#', True)
    #     # plot_bar(rationale_len, '%s rationale length dist.'%dataset, 'length', '#tokens', False)
    #     # plot_bar(query_len, '%s query length dist.'%dataset, 'length', '#tokens', False)
    #     # plot_bar(tr_len, '%s total rationale length dist.'%dataset, 'length', '#tokens', False)
    #     # plot_bar(sentence_len, '%s sentence length dist.' % dataset, 'length', '#tokens', False)
    #     # plot_bar(doc_len, '%s document length dist.' % dataset, 'length', '#tokens', False)
