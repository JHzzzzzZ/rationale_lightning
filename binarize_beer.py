import os
from tqdm import tqdm
import json
from argparse import ArgumentParser

def binarize_func(x, pos_thres, neg_thres):
    if float(x) >= pos_thres:
        return 1
    elif float(x) <= neg_thres:
        return 0

    return

def convert_train_or_heldout(filepath, aspect):
    data = []
    with open(filepath, encoding='utf8') as f:
        for line in tqdm(f, desc='loading data...'):
            splited_line = line.split()

            scores = splited_line[:5]
            text_list = splited_line[5:]

            scores[aspect] = binarize_func(scores[aspect], 0.6, 0.4)
            if scores[aspect] is not None:
                data.append((scores, text_list))

    return data

def save_list(data, aimpath):
    with open(aimpath, 'w', encoding='utf8') as f:
        for scores, text_list in tqdm(data, desc='saving data...'):
            score_str = ' '.join(map(str, scores))
            text_str = ' '.join(text_list)
            f.write('%s\t%s\n'%(score_str, text_str))

def convert_and_save_test(path, aspect):
    test_path = os.path.join(path, 'annotations.json')
    save_path = os.path.join(path, 'binarized.annotations.aspect%d.json'%(aspect))
    data = []
    with open(test_path, encoding='utf8') as f:
        for line in tqdm(f, desc='loading test data...'):
            d = json.loads(line.strip())
            d['y'][aspect] = binarize_func(d['y'][aspect], 0.6, 0.4)
            if d['y'][aspect] is not None:
                data.append(d)

    with open(save_path, 'w', encoding='utf8') as f:
        for d in tqdm(data, desc='saving test data...'):
            f.write(json.dumps(d, ensure_ascii=False)+'\n')

    return data

def get_args():
    parser = ArgumentParser()

    parser.add_argument('--path', type=str, default='./data/beer')
    parser.add_argument('--aspect', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    train_path = os.path.join(args.path, 'reviews.aspect%d.train.txt'%(args.aspect))
    eval_path = os.path.join(args.path, 'reviews.aspect%d.heldout.txt' % (args.aspect))


    train_save_path = os.path.join(args.path, 'binarized.reviews.aspect%d.train.txt' % (args.aspect))
    eval_save_path = os.path.join(args.path, 'binarized.reviews.aspect%d.heldout.txt' % (args.aspect))


    data = convert_train_or_heldout(train_path, args.aspect)
    save_list(data, train_save_path)
    data = convert_train_or_heldout(eval_path, args.aspect)
    save_list(data, eval_save_path)

    convert_and_save_test(args.path, args.aspect)

