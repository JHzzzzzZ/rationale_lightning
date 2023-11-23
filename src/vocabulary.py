from collections import Counter, OrderedDict
from src import *


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first seen"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__,
                           OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""

    def __init__(self):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []

    def count_token(self, t):
        self.freqs[t] += 1

    def add_token(self, t):
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)

    def build(self, min_freq=0):
        self.add_token(UNK_TOKEN)  # reserve 0 for <unk>
        self.add_token(PAD_TOKEN)  # reserve 1 for <pad> (discussed later)

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)

    def decode(self, x, **kwargs):
        return self.i2w[x]

    def __call__(self, x, **kwargs):
        return {'input_ids': [self.w2i.get(x.lower(), self.w2i[UNK_TOKEN])]}

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_token_id(self):
        return self.w2i[PAD_TOKEN]

    @property
    def sep_token_id(self):
        return self.w2i[SEP_TOKEN]

    @property
    def unk_token_id(self):
        return self.w2i[UNK_TOKEN]

    @property
    def mask_token_id(self):
        return self.w2i[MASK_TOKEN]
