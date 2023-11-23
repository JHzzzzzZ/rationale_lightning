from src.datasets import *
from collections import defaultdict

dataset_name = 'beer'
if __name__ == '__main__':
    vocab = Vocabulary()
    _ = load_embeddings('./pretrained/glove.6B.50d.txt', vocab, 50)
    collate_fn = get_collator_cls(dataset_name)(vocab=vocab,max_len=10000000)
    datacls = get_dataset_cls(dataset_name)('./data/%s'%dataset_name, tokenizer=vocab, max_len=10000000, balance=True,
                                      batch_size=1, aspect=2, collate_fn=collate_fn, loss_fn='ce', sentence_level=False
                                      )
    datacls.setup(None)
    c1=25
    c2=10
    total_length = 2000
    length = [0 for _ in range(total_length//c1)]
    lengths = []
    rat_lengths = []
    l=0
    import numpy as np
    rat_dist = np.zeros((total_length//c2, ))
    for e in datacls.test_dataloader():
        ids = e['tensors'][0]
        text = e['text'][0]

        assert len(ids)==len(text), f'{len(ids)}, {len(text)}'
        if e.get('rationales') is not None:
            rationales = e['rationales'][0]

            for i, r in enumerate(rationales):
                if r == 1:
                    rat_dist[i//c2] += 1
            rat_lengths.append(sum(rationales))
        lengths.append(len(ids))
        length[len(ids) // c1] += 1

    print('#: ', len(lengths))
    print('avg: ', sum(lengths) / len(lengths))
    print('max: ', max(lengths))
    if rat_lengths:
        print('avg_rat: ', sum(rat_lengths) / len(rat_lengths))
        print('max_rat: ', max(rat_lengths))
        print('rat_percent: ', sum(rat_lengths) / sum(lengths))

    sstart = -1
    eend = total_length//c1
    for i in range(total_length//c1):
        if length[i] > 0 and sstart == -1: sstart = i
        if length[i] > 0: eend = i

    print(sstart, eend)
    plt.bar(range(sstart*c1, (eend+1)*c1, c1), length[sstart:eend+1], width=c1*0.7)
    # plt.xticks(range(sstart, eend+1, 2))
    plt.show()

    start = -1
    end = total_length // c2
    for i in range(total_length // c2):
        if rat_dist[i] > 0 and start == -1: start = i
        if rat_dist[i] > 0: end = i

    if start != -1:
        print(start, end)
        plt.bar(range(start*c2, (end+1)*c2, c2), rat_dist[start:end+1], width=c2*0.7)
        # plt.xticks(range(start, end + 1, 2))
        plt.show()

