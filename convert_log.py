import os
from src.utils import plot

from transformers.models.bart import BartModel

path = './metrics.csv'

if __name__ == '__main__':
    logs= {}
    with open(path, encoding='utf8') as f:
        keys = next(f).strip().split(",")
        for each in keys:
            logs.setdefault(each, [])
        # print(keys)

        for line in f:
            line = line.strip().split(',')
            for i, value in enumerate(line):
                logs[keys[i]].append(float(value) if value else -1)


    print(plot(logs, title='aaa', xtitle='steps', ytitle=''))
