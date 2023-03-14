from src.models import *
from src.datasets import *
from argparse import ArgumentParser
from pytorch_lightning import Trainer
# from collections import namedtuple

def get_test_args():
    parser = ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, default='./output/lightning_logs/version_1/checkpoints/last.ckpt')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--accelerator', type=str, choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--devices', type=str, default="0,")

    return parser.parse_args()

path = './pretrained/glove.6B.300d.txt'
if __name__ == '__main__':
    args = get_test_args()
    check_point = torch.load(args.ckpt_path)
    state_dict = check_point['state_dict']
    args = vars(args)

    vocab = Vocabulary()
    _ = load_embeddings(path, vocab, 300)

    temp = check_point['hyper_parameters']
    temp.update(args)
    args = temp
    args['vectors'] = None
    args['vocab'] = vocab

    model = get_model_cls(args['model']).load_from_checkpoint(args['ckpt_path'], strict=False, vocab=args['vocab'],
                                                              vectors=args['vectors'])

    if args['dataset'] not in eraser:
        collate_fn = Collator(model.vocab, max_len=args['max_len'], mask_rationale_epoch=args['mask_rationale_epoch'])
    else:
        collate_fn = CollatorWithQuery

    collate_fn = get_collator_cls(args['dataset'])(model.vocab, max_len=args['max_len'],
                                                   mask_rationale_epoch=args['mask_rationale_epoch'])

    dataset = get_dataset_cls(args['dataset'])(collate_fn=collate_fn,
                                               **args)

    trainer = Trainer(accelerator=args['accelerator'], devices=args['devices'], )

    model._log_hyperparams = False
    dataset._log_hyperparams = False

    trainer.validate(model, datamodule=dataset)
    trainer.test(model, datamodule=dataset)
