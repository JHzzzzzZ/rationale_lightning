import os
from src.utils import plot_and_save

path = '/home/duanjw/workspace/jh/lightning-rationale/output/kuma_beer_grow-fgm1/lightning_logs/version_0/metrics.csv'
title='metrics'

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


    plot_and_save(logs, title=title, xtitle='steps', ytitle='')
