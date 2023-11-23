from src.models import *
from src.datasets import *
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from src.vocabulary import Vocabulary
# from collections import namedtuple

def get_test_args():
    parser = ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, default='./output/lightning_logs/version_1/checkpoints/last.ckpt')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--accelerator', type=str, choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--devices', type=str, default="0,")
    parser.add_argument('--embedding_path', type=str)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--save_decoded_answer', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_test_args()
    check_point = torch.load(args.ckpt_path)
    state_dict = check_point['state_dict']
    args = vars(args)
    temp = check_point['hyper_parameters']
    temp.update(args)
    args = temp
    path = args['embedding_path']
    print(args)

    if not os.path.exists(os.path.dirname(path)):
        path = os.path.join('/shared_data/pretrained_models/',
                             os.path.basename(path))
        # if args['model_path'] is not None:
        #     args['model_path'] = os.path.join('/shared_data/pretrained_models/',
        #                         os.path.basename(args['model_path']))

    if args['model_path'] is not None:
        vocab = AutoTokenizer.from_pretrained(args['model_path'])
    else:
        vocab = Vocabulary()
        _ = load_embeddings(path, vocab, args['embed_size'])

    print(args['ckpt_path'])
    model = get_model_cls(args['model']).load_from_checkpoint(args['ckpt_path'], strict=False, vocab=vocab,
                                                              vectors=None, save_decoded_answer=args['save_decoded_answer'])
    model.auxiliary_full_pred = True
    collate_fn = get_collator_cls(args['dataset'])(vocab, max_len=args['max_len'],
                                                   mask_rationale_epoch=args['mask_rationale_epoch'])

    dataset = get_dataset_cls(args['dataset'])(collate_fn=collate_fn,
                                               tokenizer=vocab,
                                               **args)
    dataset.val_is_test=False

    model._log_hyperparams=False
    dataset._log_hyperparams=False

    trainer = Trainer(accelerator=args['accelerator'], devices=args['devices'], )

    if args['eval']:
        trainer.validate(model, datamodule=dataset)
    trainer.test(model, datamodule=dataset)

