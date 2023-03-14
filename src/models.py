#!/usr/bin/env python

import torch.nn
from pytorch_lightning.core.module import LightningModule
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR

from src.datasets import *
from src.metrics import *
from src.modules import Classifier, Generator
from src.modules import GumbelGate, KumaGate


class BaseModule(LightningModule):
    '''
    pytorch lightning的基础模型，定义了基础参数，与基本的训练步骤等。
    需要实现get_loss函数，返回指标字典。
    '''

    def __init__(self,
                 vocab,
                 vectors=None,
                 model_path: str = None,
                 embed_size: int = 300,
                 fix_emb: bool = True,
                 hidden_size: int = 200,
                 output_size: int = 1,
                 dropout: float = 0.1,
                 layer: str = 'rcnn',
                 loss_fn: str = 'ce',
                 optimizer_params: dict = {},
                 lr_scheduler_params: dict = {},
                 show_per_epoch: int = 5,
                 show_num_samples: int = 3,
                 save_decoded_answer: bool = True,
                 fgm: bool = False,
                 fgm_epsilon: float = 1,
                 **kwargs):
        '''

        :param vocab: Vocabulary
        :param vectors: np.array, (#embeddings, embed_size). pretrained word embeddings.
        :param embed_size: int. embedding dims of pretrained embedding.
        :param fix_emb: bool, whether freeze pretrained word embeddings or not.
        :param hidden_size: int, the hidden size of LSTM/RCNN layers.
        :param output_size: int, #labels.
        :param dropout: float, dropout rate.
        :param layer: str, ['lstm', 'rcnn', 'cnn'], which backbone to use.
        :param loss_fn: str, ['bce', 'mse'], 'mse' <-> regression tasks.
        :param optimizer_params: dict, the configuration of optimizer.
                                ['adam', 'adamw', 'sgd'] are supported.
                                for example:
                                {
                                    'type': 'adam',
                                    'lr': 1e-4,
                                    'weight_decay': 2e-6,
                                    # 'momentum': 0.9,
                                }
        :param lr_scheduler_params: dict, the configuration of learning rate scheduler.
                                    ['exp', 'multi_step'] are supported.
                                    for example:
                                    {
                                        'type': 'exp',
                                        'gamma': 0.97,
                                        # 'milestones': [10, 20, 30]
                                    }
        :param kwargs:
        '''
        super().__init__()
        self.vocab = vocab
        # load pretrained embeddings.
        embed = nn.Embedding(self.vocab.vocab_size, embed_size, padding_idx=1)
        if vectors is not None:
            with torch.no_grad():
                embed.weight.data.copy_(torch.from_numpy(vectors))
                # I need compute gradient for embedding layer while using FGM.
                # put the freeze operation to the optimizer.
                # print("Embeddings fixed: {}".format(fix_emb))
                # embed.weight.requires_grad = not fix_emb
        self.embed = nn.Sequential(embed, nn.Dropout(dropout))
        self.fix_emb = fix_emb

        self.encoder = Classifier(embed_size=embed_size, model_path=model_path,
                                  hidden_size=hidden_size, output_size=output_size,
                                  dropout=dropout, layer=layer)

        self.model_path = model_path
        if model_path is not None:
            del self.embed
            self.embed = self.encoder.enc_layer.bert.embeddings

        self.layer = layer
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        self.optimizer_params = optimizer_params
        self.lr_scheduler_params = lr_scheduler_params

        self.show_per_epoch = show_per_epoch
        self.show_num_samples = show_num_samples
        self.show_queue = []
        self.save_queue = []
        self.save_decoded_answer = save_decoded_answer

        self.fgm = fgm
        self.fgm_epsilon = fgm_epsilon

        # ce or mse loss.
        if loss_fn == 'ce':
            self.criterion = nn.NLLLoss(reduction='none')
        elif loss_fn == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        else:
            raise ValueError('Unknown loss function.')

        # we need some metrics for performing model selection.
        self.metrics = {}
        self.metrics['acc'] = Accuracy('multiclass', num_classes=output_size)
        self.metrics['span_f1'] = SpanFscore()

        self.num_samples = None
        self.temper_num_samples = None

        # this line is very slow since the vocabulary is very large (up to 400,000 individual words).
        self.save_hyperparameters(ignore=['vocab', 'vectors'])

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        '''
        加入最基础的命令行设置.
        :param parser: ArgumentParser.
        :return: parser with base configs.
        '''
        parser.add_argument('--hidden_size', type=int, default=200)
        parser.add_argument('--output_size', type=int, default=1)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--fix_emb', action='store_true')

        parser.add_argument('--layer', type=str, choices=['rcnn', 'lstm', 'bert', 'gru', 'rnn'], default='lstm')
        parser.add_argument('--loss_fn', type=str, choices=['mse', 'ce'], default='ce')

        parser.add_argument('--save_decoded_answer', action='store_true')

        parser.add_argument('--fgm', action='store_true')
        parser.add_argument('--fgm_epsilon', type=float, default=1.0)


        return parser

    def get_loss(self, *args, **kwargs) -> dict:
        '''!!! override this !!! and return a metrics dict. '''
        raise NotImplementedError(f'You should override get_loss function.')

    def get_fgm_loss(self, *args, **kwargs) -> dict:
        loss_dict = self.get_loss(*args, **kwargs)
        loss_dict['loss'].backward()

        embeddings = {}
        for n, p in self.embed.named_parameters():
            if not isinstance(self.embed, nn.Sequential) and 'word' not in n:
                continue
            embeddings[n] = p.data.clone()
            norm = torch.norm(p.grad)
            if norm != 0 and not torch.isnan(norm):
                theta = self.fgm_epsilon * p.grad / norm
                p.data.add_(theta)

        adv_loss_dict = self.get_loss(*args, **kwargs)

        for n, p in self.embed.named_parameters():
            if n in embeddings:
                p.data = embeddings[n]

        return adv_loss_dict

    def decode_text(self, batch, pred_z=None, num_decode=-1):
        '''
        解码文本和z.
        如果z==1则对应token两侧用**包围。
        如果0<z<1则对应token两侧用__包围。
        否则对应token两侧没有特殊标记。
        :param batch: 模型输入，应该需要有'text'键。
        :param num_decode: 解码的文本数量，num_decode = min(batch_size, num_decode)。
        :return: list，解码后的字符串列表。
        '''
        ans = []
        # 有generator类才需要解码。
        if pred_z is not None:
            batch_text = batch['text']
            batch_z = pred_z.cpu()
            for i, (text, z) in enumerate(zip(batch_text, batch_z)):
                if num_decode == i:
                    break

                single_ans = []
                for token, rat in zip(text, z):
                    rat = rat.item()
                    tag = ''
                    if rat == 1:
                        tag = '**'
                    elif rat > 0:
                        tag = '__'
                    single_ans.append(f'{tag}{token}{tag}')

                ans.append(' '.join(single_ans))

        return ans

    def show_text(self, batch, pred_z=None):
        if pred_z is not None:
            print(f'\nShowing decoded text...')
            ans = self.decode_text(batch, pred_z, self.show_num_samples)
            for each in ans:
                print(each)
            print()

    def base_on_start(self):
        self.temper_num_samples = self.num_samples
        if hasattr(self, 'generator'):
            self.generator.num_samples = 1
        self.num_samples = 1

    def base_step(self, batch, batch_idx, prefix='train'):
        '''
        base step. both train, eval and test step are based on this function.
        :param prefix: which part is this in.
        :return: dict containing loss and metrics.
        '''
        # get metrics dict.
        if self.fgm and self.training:
            loss_optional = self.get_fgm_loss(batch)
        else:
            loss_optional = self.get_loss(batch)

        # logging metrics
        batch_size = batch['labels'].size(0)
        # show_in_bar is a global configuration in __init__ file,
        # controlling which metrics will show in the console.
        bar_dict = {f'{prefix}_' + k: loss_optional[k] for k in show_in_bar if k in loss_optional}
        other_dict = {f'{prefix}_' + k: loss_optional[k] for k in loss_optional if k not in show_in_bar}
        self.log_dict(bar_dict,
                      prog_bar=True, batch_size=batch_size, logger=True)
        self.log_dict(other_dict,
                      prog_bar=False, batch_size=batch_size, logger=True)

        return loss_optional

    def base_on_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int):
        outputs['text'] = batch['text']
        outputs['pred_z'] = self.generator.z if hasattr(self, 'generator') else None

        return outputs

    def base_epoch_end(self, outputs, prefix='train'):
        '''
        base epoch end operations.
        :param outputs: list, epoch outputs from each step.
        :param prefix: which part ?
        :return: None
        '''
        # get the first batch as example.
        optional = outputs[0]
        ret = {}

        # check whether 'acc' and 'fscore' in returned dict.
        if optional.get(f'acc', None) is not None:
            ans = self.metrics['acc'].compute()
            ret[f'{prefix}_acc'] = ans
            self.log(f'{prefix}_acc', ans, logger=True)

        if optional.get(f'fscore', None) is not None:
            ans = self.metrics['span_f1'].compute()
            ans = {f'{prefix}_{k}': v for k, v in ans.items()}
            ret.update(ans)
            self.log_dict(ans, logger=True)

        # print metrics if they exist in every epoch end.
        # if ret:
        #     print(f'{prefix} epoch end: ')
        #     for k, v in ret.items():
        #         print(f'\t{k}: {v}')

        # reset every metric classes in dict(self.metrics).
        for k, v in self.metrics.items():
            v.reset()

    def base_on_end(self):
        self.num_samples = self.temper_num_samples
        if hasattr(self, 'generator'):
            self.generator.num_samples = self.num_samples

    def training_step(self, batch, batch_idx):
        self.log('lr', self.optimizers().param_groups[0]['lr'], logger=True)
        return self.base_step(batch, batch_idx, prefix='train')

    def training_epoch_end(self, outputs) -> None:
        self.base_epoch_end(outputs, prefix='train')

    def on_validation_start(self) -> None:
        self.base_on_start()

    def validation_step(self, batch, batch_idx):
        return self.base_step(batch, batch_idx, prefix='val')

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int):
        return self.base_on_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def validation_epoch_end(self, outputs) -> None:
        if (self.current_epoch + 1) % self.show_per_epoch == 0:
            self.show_text(outputs[0], outputs[0]['pred_z'])
        print()
        return self.base_epoch_end(outputs, prefix='val')

    def on_validation_end(self) -> None:
        self.base_on_end()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        pass

    def on_test_start(self) -> None:
        self.base_on_start()

    def test_step(self, batch, batch_idx):
        return self.base_step(batch, batch_idx, prefix='test')

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int):
        outputs = self.base_on_batch_end(outputs, batch, batch_idx, dataloader_idx)

        if self.save_decoded_answer:
            ans = self.decode_text(batch, outputs['pred_z'])
            self.save_queue.extend(ans)

        return outputs

    def test_epoch_end(self, outputs) -> None:
        self.show_text(outputs[0], outputs[0]['pred_z'])

        return self.base_epoch_end(outputs, prefix='test')

    def on_test_end(self) -> None:
        self.base_on_end()

        filename = os.path.join(self.trainer.default_root_dir, 'test_output.txt')
        if bool(self.save_queue):
            with open(filename, 'w', encoding='utf8') as f:
                for string in self.save_queue:
                    f.write(string + '\n')
            print(f'Test outputs have written in {filename}.')

    def configure_optimizers(self):
        '''
        configurate optimizer and lr_scheduler based on corresponding dict.
        :return: optimizer or {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        '''
        # freeze the embedding layer or not .
        param_except_embed = filter(lambda x: x[1].requires_grad and (not self.fix_emb or 'embed' not in x[0]),
                                    self.named_parameters())
        param_except_embed = list(map(lambda x: x[1], param_except_embed))  # map 在遍历一遍后就没有了。
        print(f'total trainable params: {sum([each.numel() for each in param_except_embed])}.')
        lr = self.optimizer_params.pop('lr', 1e-3)
        opt = self.optimizer_params.pop('type', 'adam')
        if opt == 'adam':
            optimizer = optim.Adam(param_except_embed,
                                   lr=lr, **self.optimizer_params)
        elif opt == 'adamw':
            optimizer = optim.AdamW(param_except_embed,
                                    lr=lr, **self.optimizer_params)
        elif opt == 'sgd':
            optimizer = optim.SGD(param_except_embed,
                                  lr=lr, **self.optimizer_params)
        else:
            raise ValueError(f'cannot find the optimizer {opt}.')

        if not bool(self.lr_scheduler_params):
            return optimizer

        opt = self.lr_scheduler_params['type']
        gamma = self.lr_scheduler_params['gamma']
        if opt == 'exp':
            lr_scheduler = ExponentialLR(optimizer, gamma)
        elif opt == 'multi_step':
            lr_scheduler = MultiStepLR(optimizer, milestones=self.lr_scheduler_params['milestones'],
                                       gamma=gamma)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class BaseRationaleModel(BaseModule):
    def __init__(self,
                 vocab,
                 vectors=None,
                 model_path: str = None,
                 embed_size: int = 300,
                 fix_emb: bool = True,
                 hidden_size: int = 400,
                 output_size: int = 1,
                 dropout: float = 0.1,
                 gate: str = 'kuma',
                 temperature: float = 0.5,
                 min_temperature: float = 5e-3,
                 layer: str = 'lstm',
                 pretrained_cls_ckpt_path: str = None,
                 freeze_cls: bool = False,
                 sparsity: float = 0.0003,
                 coherence: float = 2.,
                 selection: float = 0.13,
                 lasso: float = 0.02,
                 loss_fn: str = 'ce',
                 decay: float = 1e-5,
                 num_samples: int = 1,
                 ranking: str = None,
                 margin: float = 1e-2,
                 margin_weight: float = 1e-2,
                 sentence_level: bool = False,
                 optimizer_params: dict = {},
                 lr_scheduler_params: dict = {},
                 **kwargs
                 ):
        super().__init__(vocab=vocab, vectors=vectors, model_path=model_path,
                         fix_emb=fix_emb, embed_size=embed_size,
                         loss_fn=loss_fn, hidden_size=hidden_size,
                         output_size=output_size, dropout=dropout,
                         layer=layer, optimizer_params=optimizer_params,
                         lr_scheduler_params=lr_scheduler_params,
                         **kwargs)

        self.pretrained_classifier = pretrained_cls_ckpt_path is not None
        self.freeze_cls = freeze_cls
        if self.pretrained_classifier is None and self.freeze_cls:
            print('No pretrained classifier loaded, random initialized classifier will not be froze.')

        self.sparsity = sparsity
        self.coherence = coherence
        self.selection = selection
        self.lasso = lasso
        self.optimizer_params = optimizer_params

        self.ranking = ranking
        self.num_samples = num_samples
        self.margin = margin
        self.margin_weight = margin_weight

        self.sentence_level = sentence_level

        self.decay = decay

        self.ranking_fn = self._ranking_loss_fn

        if self.pretrained_classifier:
            print(f'Loading pretrained classifier from {pretrained_cls_ckpt_path}.')
            ckpt = torch.load(pretrained_cls_ckpt_path)
            state_dict = {k.split('encoder.', 1)[-1]: v for k, v in ckpt['state_dict'].items() if 'encoder' in k}
            self.encoder.load_state_dict(state_dict)
            self.encoder.requires_grad_(not self.freeze_cls)

        self.generator = Generator(embed_size=embed_size, hidden_size=hidden_size, model_path=model_path,
                                   dropout=dropout, layer=layer, gate=gate, temperature=temperature,
                                   min_temperature=min_temperature, num_samples=num_samples,
                                   sentence_level=sentence_level)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        '''
        Add model specific params.
        '''
        p = parser.add_argument_group('model')
        p = super().add_argparse_args(p)

        p.add_argument('--gate', type=str, choices=['gumbel', 'kuma', 'straight', 'reinforce'], default='reinforce')
        p.add_argument('--temperature', type=float, default=0.5)
        p.add_argument('--min_temperature', type=float, default=5e-3)
        p.add_argument('--decay', type=float, default=1e-5)

        p.add_argument('--pretrained_cls_ckpt_path', type=str, default=None,
                       help='path to pretrained classifier checkpoint.')
        p.add_argument('--freeze_cls', action='store_true')

        p.add_argument('--margin', type=float, default=1e-2)
        p.add_argument('--margin_weight', type=float, default=1e-2)

        p.add_argument('--sparsity', type=float, default=0.0003)
        p.add_argument('--coherence', type=float, default=2)
        p.add_argument('--selection', type=float, default=0.2)
        p.add_argument('--lasso', type=float, default=0.02)

        p.add_argument('--num_samples', type=int, default=1)
        p.add_argument('--ranking', type=str, default='null', choices=['pairwise', 'null', 'margin', 'margin_pair'])

        return parser

    def _ranking_loss_fn(self, x, margin):
        return torch.maximum(x + margin, x.new_zeros(()))

    def get_ranking_loss(self, ranked_value):
        margin_loss = 0
        if self.ranking == 'margin':
            '''
            loss = max(xi-xj+(j-i)*margin, 0)
            margin大小和当前两个向量的排名差距有关。
            '''
            # TODO: 其实也是pairwise ranking loss
            # margin-based ranking triplet loss.
            margin = torch.linspace(1, 2, steps=ranked_value.shape[0], device=ranked_value.device).unsqueeze(
                -1) * self.margin  # (s, B)
            margin_loss = self.ranking_fn(ranked_value, margin)
        elif self.ranking == 'pairwise':
            '''
            (full - good1) - (full - good2) = good2 - good1
            [0, g2-g1, g3-g1, g4-g1]
            [g1-g2, 0, g3-g2, g4-g2]
            [g1-g3, g3-g2, 0, g4-g3]
            [g1-g4, g2-g4, g3-g4, 0]
            '''

            # 计算基于排序的pairwise margin loss矩阵。取上三角。
            margin_matrix = ranked_value.unsqueeze(1) - ranked_value.unsqueeze(0)  # (s, s, B)
            margin_loss = self.ranking_fn(margin_matrix, self.margin)
            margin_vec = torch.triu(margin_loss.permute(2, 0, 1), diagonal=1)  # (B, s, s) 不包括对角线

            # 计算基于full的margin loss
            full_margin_loss = self.ranking_fn(ranked_value, self.margin)

            # 组合上述两个损失。
            margin_loss = torch.cat([margin_vec.permute(1, 2, 0), full_margin_loss.unsqueeze(0)], dim=0)  # (s+1, s, B)
        elif self.ranking == 'margin_pair':
            '''
            margin 和 pairwise 的组合。
            其中rationale之间的margin控制在[0, 1]*self.margin，而全文和翻转之间的margin控制在[1, 2]*self.margin.
            '''

            margin = torch.linspace(0, 1, steps=ranked_value.shape[0], device=ranked_value.device).unsqueeze(
                -1) * self.margin  # (s, B)
            margin_mat = margin.unsqueeze(1) - margin.unsqueeze(0)  # (s, s, B)

            # 计算基于排序的pairwise margin loss矩阵。取上三角。
            margin_matrix = ranked_value.unsqueeze(1) - ranked_value.unsqueeze(0)  # (s, s, B)
            margin_loss = self.ranking_fn(margin_matrix, margin_mat)
            margin_vec = torch.triu(margin_loss.permute(2, 0, 1), diagonal=1)  # (B, s, s) 不包括对角线

            larger_margin = torch.linspace(0, 1, steps=ranked_value.shape[0], device=ranked_value.device).unsqueeze(
                -1) * self.margin
            # 计算基于full的margin loss
            full_margin_loss = self.ranking_fn(ranked_value, larger_margin)

            # 组合上述两个损失。
            margin_loss = torch.cat([margin_vec.permute(1, 2, 0), full_margin_loss.unsqueeze(0)], dim=0)  # (s+1, s, B)

        return margin_loss

    def get_main_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_loss(self, batch):
        """
        This computes the loss for the whole model.
        forward -> get main loss -> get z loss.

        :param batch:
        :return:
        """
        targets = batch['labels']
        full_preds, preds = self(**batch)
        rationales = batch['rationales']
        batch_size = targets.shape[0]
        mask = batch['mask']
        # print(preds.shape, targets.shape)
        optional = {}
        sparsity = self.sparsity
        coherence = self.coherence
        # num_samples != 1 maybe.
        loss_mat = self.criterion(preds, targets.repeat(self.num_samples, *targets.shape[1:]))  # [B, 1]
        # main loss for p(y | x,z)
        loss_vec = loss_mat.squeeze(-1)  # [s, B]
        loss_vec = loss_vec.reshape(self.num_samples,
                                    loss_vec.shape[0] // self.num_samples,
                                    *loss_vec.shape[1:])
        loss_full_vec = None
        if full_preds is not None:
            loss_full_mat = self.criterion(full_preds, targets)
            loss_full_vec = loss_full_mat.mean(-1)  # [B]

        main_loss = self.get_main_loss(loss_vec=loss_vec, loss_full_vec=loss_full_vec, optional=optional,
                                       full_preds=full_preds, preds=preds, targets=targets)

        # print(obj.shape)
        # coherency is 2*sparsity (after modifying sparsity rate)
        coherent_factor = sparsity * coherence

        z = self.generator.z.squeeze(-1)  # [B, T]
        gen_dict = self.generator.get_loss(z=z, dist=self.generator.z_dists,
                                           sparsity=sparsity, coherent_factor=coherent_factor,
                                           loss_vec=main_loss, selection=self.selection,
                                           lasso=self.lasso, **batch)

        gate_cost = gen_dict['gate_cost']

        optional.update(gen_dict)

        num_0, num_c, num_1, total = get_z_stats(z, mask)
        optional["p0"] = num_0 / float(total)
        optional["pc"] = num_c / float(total)
        optional["p1"] = num_1 / float(total)
        optional["selected"] = optional['pc'] + optional['p1']

        main_loss = main_loss + gate_cost

        if rationales is not None:
            if self.sentence_level:
                sentence_rationale = batch['sentence_rationale']
                sentence_rationale = sentence_rationale[
                    torch.arange(batch_size).unsqueeze(-1), batch['sentence_type_ids']]
                optional.update(self.metrics['span_f1'](z.detach().cpu(), sentence_rationale.cpu(), mask=mask.cpu()))
            else:
                optional.update(self.metrics['span_f1'](z.detach().cpu(), rationales.cpu(), mask=mask.cpu()))

        optional['loss'] = main_loss

        return optional

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        # 如果gate是gumbel，则将温度逐渐降低。 max(exp(-decay*step)*t, min_t).
        if isinstance(self.generator.z_layer, GumbelGate):
            temperature = self.generator.z_layer.temperature
            min_temperature = self.generator.z_layer.min_temperature
            temperature = max(temperature * np.exp(-self.decay), min_temperature)
            self.log('gumbel_temperature', temperature, logger=True)
            self.generator.z_layer.temperature = temperature

        # 如果gate是kuma，则将sparsity强度逐渐升高。min(exp(decay*step)*s, 1)
        if isinstance(self.generator.z_layer, KumaGate):
            self.sparsity = self.sparsity * np.exp(self.decay)
            self.sparsity = min(self.sparsity, 1.)
            self.log('sparsity_weight', self.sparsity, logger=True, prog_bar=True)


class RationaleModel(BaseRationaleModel):
    '''
    基础的gen-enc框架，根据传入参数不同，可以选择强化学习、Straight Through和不同重参数化技巧。
    '''
    def forward(self, tensors, mask, **kwargs):
        """
        Generate a sequence of zs with the Generator.
        Then predict with sentence x (zeroed out with z) using Encoder.

        :param x: [B, T] (that is, batch-major is assumed)
        :return:
        """
        emb = self.embed(tensors)
        z = self.generator(emb, mask, **kwargs)
        # print(kwargs.get('labels'))
        y_ = self.encoder(emb, mask, z, **kwargs)

        y = None if self.ranking == 'null' else self.encoder(emb, mask, **kwargs)[0]

        return y, y_[0]

    def get_main_loss(self, loss_vec, loss_full_vec=None, optional={}, preds=None, full_preds=None, targets=None,
                      **kwargs):
        obj = loss_full_vec.mean() if loss_full_vec is not None else loss_vec.mean()  # [1]
        optional['obj'] = obj

        margin_loss = 0
        if self.ranking != 'null':
            margin_value = loss_full_vec - loss_vec  # [s, B]
            ranked_value, _ = margin_value.sort(dim=0, descending=True)

            margin_loss = self.get_ranking_loss(ranked_value)
            margin_loss = torch.mean(margin_loss)

        if isinstance(self.criterion, nn.NLLLoss):
            optional['acc'] = self.metrics['acc'](full_preds.cpu() if full_preds is not None else preds.cpu(),
                                                  targets.cpu())

        return obj + self.margin_weight * margin_loss


class RationaleRevMarginModel(BaseRationaleModel):
    def forward(self, tensors, mask, **kwargs):
        """
        用generator生成z，将z反转后传入encoder生成掩盖rationale后的预测。
        最大化原始预测和掩盖后预测的损失差值。

        :param x: [B, T] (that is, batch-major is assumed)
        :return:
        """
        emb = self.embed(tensors)
        z = self.generator(emb, mask, **kwargs)
        y = self.encoder(emb, mask, )
        y_ = self.encoder(emb, mask, (1 - z))
        return y[0], y_[0]

    def get_main_loss(self, loss_vec, loss_full_vec=None, optional={}, full_preds=None, targets=None,
                      **kwargs):
        obj = loss_full_vec.mean()
        optional['obj'] = obj

        margin_value = loss_full_vec - loss_vec  # [s, B]
        if self.ranking == 'null':
            margin_loss = self.ranking_fn(margin_value, self.margin)
        else:
            ranked_value, _ = margin_value.sort(dim=0, descending=False)
            margin_loss = self.get_ranking_loss(ranked_value)

        if isinstance(self.criterion, nn.NLLLoss):
            optional['acc'] = self.metrics['acc'](full_preds.cpu(), targets.cpu())

        return obj + self.margin_weight * margin_loss.mean()


class BaselineModel(BaseModule):
    def forward(self, tensors, mask, **kwargs):
        """
        baseline model

        :param x: [B, T] (that is, batch-major is assumed)
        :return:
        """
        emb = self.embed(tensors)
        y = self.encoder(emb, mask)
        return y[0]

    def get_loss(self, batch):
        """
        This computes the loss for the whole model.

        :param preds:
        :param targets:
        :param mask:
        :param iter_i:
        :return:
        """
        targets = batch['labels']
        preds = self(**batch)

        optional = {}
        loss_mat = self.criterion(preds, targets)  # [B, T]

        # main loss for p(y | x,z)
        loss_vec = loss_mat.squeeze(-1)  # [B]
        cost_e = loss_vec.mean()  # [1]

        optional['obj'] = cost_e

        if isinstance(self.criterion, nn.NLLLoss):
            optional['acc'] = self.metrics['acc'](preds.cpu(), targets.cpu())

        optional['loss'] = cost_e

        return optional


def get_model_cls(model_name):
    if model_name == 'rationale':
        return RationaleModel
    elif model_name == 'baseline':
        return BaselineModel
    elif model_name == 'rationale_rev':
        return RationaleRevMarginModel
    else:
        raise ValueError(f'Unknown model name {model_name}')
