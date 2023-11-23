from src.datasets import *

path = './data/beer'
if __name__ == '__main__':
    dataset = Beer(path, part='test', aspect=0)

    golden_list = []
    for sample in dataset:
        text_list = sample[0]
        rationales = sample[-1]

        ans_list = []
        for idx, word in enumerate(text_list):
            flag=False
            for rationale in rationales:
                if rationale[0]<=idx<rationale[1]:
                    flag=True
                    break
            tag = '**' if flag else ''
            ans_list.append(f'{tag}{word}{tag}')

        golden_list.append(' '.join(ans_list))

    with open('golden.txt', 'w', encoding='utf8') as f:
        for line in golden_list:
            f.write(line+'\n')
