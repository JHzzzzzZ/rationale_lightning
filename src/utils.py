import os
import random
from torch.nn.init import _calculate_fan_in_and_fan_out
from src import *
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def decorate_token(t, z_):
    dec = "**" if z_ == 1 else "__" if z_ > 0 else ""
    return dec + t + dec


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def find_ckpt_in_directory(path):
    for f in os.listdir(os.path.join(path, "")):
        if f.startswith('model'):
            return os.path.join(path, f)
    print("Could not find ckpt in {}".format(path))


def filereader(path):
    """read SST lines"""
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\", "")


def print_parameters(model):
    """Prints model parameters"""
    total = 0
    total_wo_embed = 0
    total_w_grad = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        total_wo_embed += np.prod(p.shape) if "embed" not in name else 0
        total_w_grad += np.prod(p.shape) if p.requires_grad else 0
        print("{:30s} {:14s} requires_grad={}".format(name, str(list(p.shape)),
                                                      p.requires_grad))
    print("\nTotal parameters: {}".format(total))
    print("Total parameters (w/o embed): {}".format(total_wo_embed))
    print('Total trainable parameters (w grad):{}\n'.format(total_w_grad))


def to_device(tensors, device):
    a = []
    for tensor in tensors:
        if tensor is not None:
            a.append(tensor.to(device))
        else:
            a.append(None)

    return a


def load_embeddings(path, vocab, dim):
    """
    Load word embeddings and update vocab.
    :param path: path to word embedding file
    :param vocab:
    :param dim: dimensionality of the pre-trained embeddings
    :return:
    """
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot find the word embedding in {path}")
    vectors = []
    w2i = {}
    i2w = []

    def maybe_insert_special_token(dim):
        if len(vectors):
            return
        # Random embedding vector for unknown words
        vectors.append(np.random.uniform(
            -0.05, 0.05, dim).astype(np.float32))
        w2i[UNK_TOKEN] = 0
        i2w.append(UNK_TOKEN)

        # Zero vector for padding and mask
        vectors.append(np.zeros(dim).astype(np.float32))
        w2i[PAD_TOKEN] = 1
        i2w.append(PAD_TOKEN)

        vectors.append(np.zeros(dim).astype(np.float32))
        w2i[MASK_TOKEN] = 2
        i2w.append(MASK_TOKEN)

        # Random embedding vector for unknown words
        vectors.append(np.random.uniform(
            -0.05, 0.05, dim).astype(np.float32))
        w2i[SEP_TOKEN] = 3
        i2w.append(SEP_TOKEN)

    maybe_insert_special_token(dim)

    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            word, vec = line.split(' ', 1)
            w2i[word] = len(vectors)
            i2w.append(word)
            v = np.array(vec.split(), dtype=np.float32)
            vectors.append(v)
            assert len(v) == dim, 'dim mismatch get %d v.s. %d' % (len(v), dim)

    vocab.w2i = w2i
    vocab.i2w = i2w

    return np.stack(vectors)


def get_minibatch(data, batch_size=256, shuffle=False):
    """Return minibatches, optional shuffling"""

    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch

    batch = []

    # yield minibatches
    for example in data:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch
            batch = []

    # in case there is something left
    if len(batch) > 0:
        yield batch


def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))


def prepare_minibatch(mb, vocab, device=None, sort=True):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    # batch_size = len(mb)
    lengths = np.array([len(ex.tokens) for ex in mb])
    maxlen = lengths.max()
    reverse_map = None

    # vocab returns 0 if the word is not there
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]
    y = [ex.scores for ex in mb]

    x = np.array(x)
    y = np.array(y, dtype=np.float32)

    if sort:  # required for LSTM
        sort_idx = np.argsort(lengths)[::-1]
        x = x[sort_idx]
        y = y[sort_idx]

        # create reverse map
        reverse_map = np.zeros(len(lengths), dtype=np.int32)
        for i, j in enumerate(sort_idx):
            reverse_map[j] = i

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)

    return x, y, reverse_map


def xavier_uniform_n_(w, gain=1., n=4):
    """
    Xavier initializer for parameters that combine multiple matrices in one
    parameter for efficiency. This is e.g. used for GRU and LSTM parameters,
    where e.g. all gates are computed at the same time by 1 big matrix.
    :param w:
    :param gain:
    :param n:
    :return:
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out = fan_out // n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)


def initialize_model_(model):
    """
    Model initialization.

    :param model:
    :return:
    """
    print("Glorot init")
    for name, p in model.named_parameters():
        if name.startswith("embed") or "lagrange" in name:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))
        elif "lstm" in name and len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier_n", name, p.shape))
            xavier_uniform_n_(p)
        elif len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier", name, p.shape))
            torch.nn.init.xavier_uniform_(p)
        elif "bias" in name:
            print("{:10s} {:20s} {}".format("zeros", name, p.shape))
            torch.nn.init.constant_(p, 0.)
        else:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))


def make_kv_string(d):
    out = []
    for k, v in d.items():
        if isinstance(v, float):
            out.append("{} {:.4f}".format(k, v))
        else:
            out.append("{} {}".format(k, v))

    return " ".join(out)


def get_z_stats(z=None, mask=None):
    """
    Computes statistics about how many zs are
    exactly 0, continuous (between 0 and 1), or exactly 1.

    :param z:
    :param mask: mask in [B, T]
    :return:
    """

    z = torch.where(mask, z, z.new_full([1], 1e2))

    mask = mask.expand_as(z)

    num_0 = (z == 0.).sum().item()
    num_c = ((z > 0.) & (z < 1.)).sum().item()
    num_1 = (z == 1.).sum().item()

    total = num_0 + num_c + num_1
    mask_total = mask.sum().item()

    assert total == mask_total, "total mismatch"
    return num_0, num_c, num_1, mask_total


def isalpha(x):
    return x.islower() or x.isupper()


def contain_alpha(x):
    for each in x:
        if isalpha(each):
            return True
    return False


def plot_bar(length_dist, title='', xtitle='', ytitle='', need_text=False):
    import matplotlib.pyplot as plt
    sorted_x, sorted_y = zip(*sorted(length_dist.items(), key=lambda x: x[0]))
    # print(sorted_cls_x)
    plt.bar(sorted_x, sorted_y)
    if need_text:
        if isinstance(sorted_x[0], str):
            for x, y in enumerate(sorted_y):
                plt.text(x + 0.05, y + 0.05, '%d' % y, ha='center',
                         va='bottom')
        else:
            for x, y in zip(sorted_x, sorted_y):
                plt.text(x + 0.05, y + 0.05, '%d' % y, ha='center',
                         va='bottom')

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title)

    plt.savefig(title + '.jpg')

    plt.show()


def get_final_path(x):
    if x[-1] == '/':
        x = x[:-1]
    return os.path.basename(x)


def plot_and_save(x, title='', xtitle='', ytitle=''):
    import matplotlib.pyplot as plt
    if isinstance(x, dict):
        for k, v in x.items():
            if k not in draw:
                continue
            for i, each in enumerate(v):
                if i!=0 and each==-1:
                    v[i] = v[i-1]
            for i, each in enumerate(reversed(v)):
                if i!=0 and each==-1:
                    v[len(v)-i-1] = v[len(v)-i]
            # print(v)
            plt.plot(np.arange(len(v)), v, label=k)
        plt.legend()
    elif isinstance(x[0], list):
        for i, v in enumerate(x):
            for i, each in enumerate(v):
                if i!=0 and each==-1:
                    v[i] = v[i-1]
            for i, each in enumerate(reversed(v)):
                if i!=0 and each==-1:
                    v[len(v)-i-1] = v[len(v)-i]
            plt.plot(np.arange(len(v)), v, label=i)
        plt.legend()
    else:
        plt.plot(np.arange(len(x)), x)

    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    plt.savefig(title+'.jpg')

    plt.show()

    return x