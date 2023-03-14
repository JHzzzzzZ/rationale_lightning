#!/usr/bin/env python

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"
INIT_TOKEN = "@@NULL@@"
SEP_TOKEN = '[SEP]'

show_in_bar = ['zmean', 'obj', 'p1', 'selected', 'precision',
               'recall', 'fscore', 'acc', 'margin']

# not_in_draw = ['step', 'epoch',
#                'test_obj', 'test_loss'
#                ]

draw = ['val_precision', 'val_recall', 'val_fscore']

eraser = ['boolq', 'esnli', 'fever', 'movies', 'multirc', 'scifact']
