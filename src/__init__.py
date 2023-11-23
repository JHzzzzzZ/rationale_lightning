#!/usr/bin/env python
# TODO: BERT forward with token reduction and rationale extraction. -> sparisity & continuity
# TODO: 后期关闭auxiliary，因为encoder已经够好了
# TODO: 添加language model的auxilary task
# TODO：对比学习 & ranking combo，自己和自己的rationale是正例，自己和其他是负例

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"
INIT_TOKEN = "@@NULL@@"
SEP_TOKEN = '[SEP]'

show_in_bar = ['sparsity', 'precision', 'full_acc',
               'recall', 'f1', 'acc', 'margin', 'kl_loss']

# not_in_draw = ['step', 'epoch',
#                'test_obj', 'test_loss'
#                ]

draw = ['val_precision', 'val_recall', 'val_fscore']

eraser = ['boolq', 'esnli', 'fever', 'movies', 'multirc', 'scifact']