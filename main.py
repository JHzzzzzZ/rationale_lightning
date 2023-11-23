from src.models import *
from src.datasets import *
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, RichProgressBar
from torch.optim.lr_scheduler import ExponentialLR, \
    MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import seed_everything

import os
import shutil


def get_optimizer_params(args):
    optimizer_params = {'type': args.optimizer,
                        'lr': args.lr,
                        'weight_decay': args.weight_decay}
    if optimizer_params['type'] == 'sgd':
        optimizer_params.update({'momentum': args.momentum})

    return optimizer_params


def get_lr_scheduler_params(args):
    lr_scheduler_params = {
        'type': args.lr_scheduler,
        'gamma': args.gamma,
    }
    if lr_scheduler_params['type'] == 'multi_step':
        lr_scheduler_params.update({'milestones': args.milestones})
    elif lr_scheduler_params['type'] == 'null':
        return {}

    return lr_scheduler_params


def check_and_clean_dir(directory):
    if os.path.exists(directory):
        flag = False
        for e in os.listdir(directory):
            if e.endswith('.ckpt'):
                flag = True
                break

        if not flag or not contain_alpha(os.path.basename(directory)):
            shutil.rmtree(directory)
            os.mkdir(directory)
        elif bool(os.listdir(directory)):
            choice = None
            while choice is None:
                choice = input(f'{directory} is not empty, clean it?[y/N]:').lower()
                if choice not in 'yn':  # '' in 'yn' -> True
                    # if choice not in ['y', 'n']:
                    choice = None

            if choice == 'y':
                shutil.rmtree(directory)
                os.mkdir(directory)



    else:
        os.mkdir(directory)

    print(f'model will be saved at {directory}')


def main(args):
    debug = args.debug
    dict_args = vars(args)
    seed_everything(dict_args['seed'])

    if dict_args['loss_fn'] == 'ce' and dict_args['output_size'] == 1:
        dict_args['output_size'] = 2

    use_plms = dict_args['model_path'] is not None

    if debug:
        dict_args['embedding_path'] = os.path.join(dict_args['embedding_path'].rsplit('/', 1)[0],
                                                   'glove.6B.50d.txt')
        dict_args['embed_size'] = 50

    check_and_clean_dir(args.output_dir)

    vectors=None
    if not use_plms:
        if not os.path.exists(os.path.dirname(dict_args['embedding_path'])):
            dict_args['embedding_path'] = os.path.join('/shared_data/pretrained_models/',
                                                       os.path.basename(dict_args['embedding_path']))
            if use_plms:
                dict_args['embedding_path'] = os.path.join('/shared_data/pretrained_models/',
                                                           os.path.basename(dict_args['model_path']))
        print("Loading pre-trained word embeddings")
        vocab = Vocabulary()
        # print(embedding_path, args.embedding_path)
        vectors = load_embeddings(dict_args['embedding_path'], vocab, dict_args['embed_size'])
    else:
        print("Loading pre-trained language model tokenizer")
        vocab = AutoTokenizer.from_pretrained(dict_args['model_path'])

    optimizer_params = get_optimizer_params(args)
    lr_scheduler_params = get_lr_scheduler_params(args)

    collate_fn = get_collator_cls(args.dataset)(vocab=vocab,
                                                max_len=args.max_len,
                                                mask_rationale_epoch=args.mask_rationale_epoch,
                                                concat_query=args.concat_query if hasattr(args, 'concat_query') else None,
                                                sentence_level=args.sentence_level)

    model_name = args.model
    dataset_name = args.dataset

    model = get_model_cls(model_name)(vocab=vocab,
                                      vectors=vectors,
                                      optimizer_params=optimizer_params,
                                      lr_scheduler_params=lr_scheduler_params,
                                      **dict_args)
    # windows 暂时不支持 linux 报错.
    # if torch.__version__ >= '2.0.0' and os.name == 'posix':
    #     import torch._dynamo as td
    #     td.config.verbose = True
    #     model = torch.compile(model, )

    dataset = get_dataset_cls(dataset_name)(tokenizer=vocab,
                                            collate_fn=collate_fn,
                                            **dict_args, )

    callbacks = []
    save_callback = ModelCheckpoint(dirpath=args.output_dir, filename='best', monitor='val_%s'%args.monitor,
                                    save_last=True, mode=args.monitor_mode)
    callbacks.append(save_callback)
    # if model_name == 'baseline':
    early_stop = EarlyStopping(monitor='val_%s'%args.monitor, patience=args.patience,
                               mode=args.monitor_mode, min_delta=1e-4, )
    callbacks.append(early_stop)

    resume_from_checkpoint=None
    if bool(os.listdir(args.output_dir)):
        resume_from_checkpoint=os.path.join(args.output_dir, 'last.ckpt')

    trainer = Trainer.from_argparse_args(args,
                                         callbacks=callbacks,
                                         default_root_dir=args.output_dir)

    trainer.fit(model, datamodule=dataset, ckpt_path=resume_from_checkpoint,)

    trainer.test(model, datamodule=dataset, ckpt_path='last')
    trainer.test(model, datamodule=dataset, ckpt_path='best')
    # trainer.test(model, datamodule=dataset, ckpt_path='last')


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=3687)

    parser.add_argument('--model', type=str, choices=['rationale', 'baseline', 'rationale_rev'])
    parser.add_argument('--dataset', type=str, choices=['beer', 'hotel'] + eraser)
    temper_args = parser.parse_known_args()[0]

    parser = get_model_cls(temper_args.model).add_argparse_args(parser)
    parser = get_dataset_cls(temper_args.dataset).add_argparse_args(parser)

    parser.add_argument('--embedding_path', type=str, default='../pretrained/glove.6B.50d.txt',
                        help='path to pretrained word embedding.')
    parser.add_argument('--embed_size', type=int, default=50,
                        help='pretrained word embedding dim.')
    parser.add_argument('--model_path', type=str, default=None,
                        help='path to pretrained language model.')

    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'],
                        default='adam')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='sgd momentum.')

    parser.add_argument('--lr_scheduler', type=str, choices=['exp', 'multi_step', 'null'],
                        default='exp')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='decay rate.')
    parser.add_argument('--milestones', type=int, nargs='+', default=[15, 50, 80],
                        help='multi step scheduler milestones.')

    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--sentence_level', action='store_true')
    parser.add_argument('--monitor', type=str, default='loss')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--monitor_mode', type=str, choices=['min', 'max'], default='min')

    parser = Trainer.add_argparse_args(parser)

    return parser.parse_args()


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    args = get_args()
    print(args)
    # exit(0)
    main(args)
