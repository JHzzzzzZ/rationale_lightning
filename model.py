from torch import nn
import torch
from pytorch_lightning.core.module import LightningModule


class RationaleModel(LightningModule):
    '''
    基础的gen-enc框架，根据传入参数不同，可以选择强化学习、Straight Through和不同重参数化技巧。
    '''
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
                 share: bool = False,
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
                 straight_with_gumbel: bool = False,
                 loss_weight: float = 1.,
                 optimizer_params: dict = {},
                 lr_scheduler_params: dict = {},
                 show_per_epoch: int = 33,
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
        self.embed_size = embed_size
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
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        elif loss_fn == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        else:
            raise ValueError('Unknown loss function.')

        # we need some metrics for performing model selection.
        self.metrics = torch.nn.ModuleDict()
        if loss_fn == 'ce':
            self.metrics['acc'] = MultiClassAccuracy()
            self.metrics['full_acc'] = MultiClassAccuracy()
            # self.metrics['acc'] = Accuracy()
            # self.metrics['full_acc'] = MultiClassAccuracy()
        self.metrics['token_f1'] = TokenF1()
        self.metrics['sparsity'] = SparsityMetrics()

        self.num_samples = None
        self.temper_num_samples = None

        self.sparsity = sparsity
        self.coherence = coherence
        self.selection = selection
        self.lasso = lasso
        self.optimizer_params = optimizer_params

        self.ranking = ranking
        self.num_samples = num_samples
        self.margin = margin
        self.margin_weight = margin_weight
        self.loss_weight = loss_weight

        self.output_size = output_size

        self.share = share
        self.decay = decay

        self.ranking_fn = self._ranking_loss_fn

        ####### model definition #######
        enc_size = hidden_size*2 if not self.pretrained_classifier else self.encoder.config.bert.config.hidden_size
        self.encoder = get_encoder(layer, embed_size, hidden_size, model_path=model_path, share=share)
        self.dropout = nn.Dropout(dropout)
        self.pred_head = nn.Linear(enc_size, output_size)

        self.gen_encoder = get_encoder(layer, embed_size, hidden_size, model_path=model_path, share=share)
        self.gate = get_gate_cls(gate)(enc_size, temperature=temperature,
                                       min_temperature=min_temperature,
                                       use_gumbel=straight_with_gumbel)



        # this line is very slow since the vocabulary is very large (up to 400,000 individual words).
        self.save_hyperparameters(ignore=['vocab', 'vectors'])


    def forward(self, tensors, mask, **kwargs):
        """
        Generate a sequence of zs with the Generator.
        Then predict with sentence x (zeroed out with z) using Encoder.

        :param x: [B, T] (that is, batch-major is assumed)
        :return:
        """
        batch_size, sequence_length = tensors.shape
        emb = self.embed(tensors) * mask[..., None]
        ####### Generator #######
        h, _ = self.gen_encoder(emb, mask, **kwargs)
        assert h.shape == (batch_size, sequence_length, self.hidden_size*2), f'{h.shape}'
        
        # generate mask
        z = self.gate(self.dropout(h), self.num_samples)  # (ns, B, T, 1)
        original_z = z
        assert z.shape == (self.num_samples, batch_size, sequence_length, 1), f'{z.shape}, {emb.shape}'
        #########################

        ####### Classifier #######
        masked_emb = emb * z  # (ns, B, T, h) zero-out operation
        assert masked_emb.shape == (self.num_samples, batch_size, sequence_length, self.embed_size), f'{masked_emb.shape}'

        h_, maxpooled_h_ = self.encoder(masked_emb.squeeze(0), mask, **kwargs)
        
        y_ = self.pred_head(self.dropout(maxpooled_h_))  
        assert y_.shape == (self.num_samples * batch_size, self.output_size), f'{y_.shape}'
        #########################

        return {'preds': y_, 'original_z': original_z, 'z': z, 'full_preds': y}

    def get_main_loss(self, preds, targets, optional=None, **kwargs):
        loss_mat = self.criterion(preds, targets.repeat(self.num_samples, *targets.shape[1:]))  # [B, 1]
        # main loss for p(y | x,z)
        loss_vec = loss_mat.squeeze(-1)  # [s, B]
        loss_vec = loss_vec.reshape(self.num_samples,
                                    loss_vec.shape[0] // self.num_samples,
                                    *loss_vec.shape[1:])

        obj = loss_vec.mean()  # [1]
        optional['obj'] = obj
        # print(obj)

        if self.auxiliary_full_pred:
            full_loss = self.criterion(kwargs['full_preds'], targets).mean()
            optional['full_obj'] = full_loss
            obj += full_loss

        margin_loss = 0
        if self.ranking != 'null' and self.num_samples > 1:
            margin_value = loss_vec  # [s, B]
            ranked_value, _ = margin_value.sort(dim=0, descending=False) # good -> bad

            margin_loss = self.get_ranking_loss(ranked_value)
            margin_loss = torch.mean(margin_loss)
            optional['margin_loss'] = margin_loss

        return self.loss_weight * obj + self.margin_weight * margin_loss


    def get_loss(self, batch):
        """
        This computes the loss for the whole model.
        forward -> get main loss -> get z loss.

        :param batch:
        :return:
        """
        targets = batch['labels']
        outputs = self(**batch)
        rationales = batch['rationales']
        mask = batch['mask']

        optional = {}
        sparsity = self.sparsity
        coherence = self.coherence
        # num_samples != 1 maybe.

        ########## Main Loss ##########
        main_loss = self.get_main_loss(full_preds=outputs.get('full_preds'), preds=outputs.get('preds'), optional=optional,
                                       targets=targets)
        optional['main_loss'] = main_loss

        ########## Sparsity ##########
        coherent_factor = sparsity * coherence
        z = outputs.get('original_z', outputs['z']).squeeze(-1)
        gen_dict = self.gate.get_loss(z=z, sparsity_factor=sparsity, coherent_factor=coherent_factor,
                                       loss_vec=main_loss, selection=self.selection,
                                       lasso=self.lasso, **batch)

        gate_cost = gen_dict['gate_cost']
        optional.update(gen_dict)
        ##############################

        main_loss = main_loss + gate_cost
        optional['loss'] = main_loss

        ########## Metrics ##########
        optional.update(self.metrics['sparsity'](z.detach(), mask=batch['sentence_mask'] if self.sentence_level else mask))
        if rationales is not None:
            optional.update(self.metrics['token_f1'](z.detach(), rationales, mask=mask))

        if not isinstance(self.criterion, nn.MSELoss):
            optional.update(self.metrics['acc'](outputs.get('preds'),
                                                  targets.repeat(self.num_samples)))

        return optional


class RNNEncoder(nn.Module):
    """
    This module encodes a sequence into a single vector using an LSTM.
    """

    def __init__(self,
                 in_features,
                 hidden_size: int = 200,
                 batch_first: bool = True,
                 bidirectional: bool = True,
                 layer: str = 'lstm',
                 attention: bool = False, ):
        """
        :param in_features:
        :param hidden_size:
        :param batch_first:
        :param bidirectional:
        """
        super(RNNEncoder, self).__init__()
        if layer == 'lstm':
            self.layer = nn.LSTM(in_features, hidden_size, batch_first=batch_first,
                                 bidirectional=bidirectional)
        elif layer == 'gru':
            self.layer = nn.GRU(in_features, hidden_size, batch_first=batch_first,
                                bidirectional=bidirectional)
        elif layer == 'rnn':
            self.layer = nn.RNN(in_features, hidden_size, batch_first=batch_first,
                                bidirectional=bidirectional)
        else:
            raise ValueError(f'{layer} should be one of [lstm, gru, rnn]')
        self.layernorm = nn.LayerNorm(hidden_size * (1 + bidirectional))
        self.attention = attention
        if self.attention:
            dim = hidden_size * (1 + bidirectional)
            self.linear = nn.Linear(dim, dim)
            self.le = nn.Linear(dim, 1, bias=False)

    def forward(self, x, mask, lengths, **kwargs):
        """
        Encode sentence x
        :param x: sequence of word embeddings, shape [B, T, E]
        :param mask: byte mask that is 0 for invalid positions, shape [B, T]
        :param lengths: the lengths of each input sequence [B]
        :return:
        """
        # packed_sequence = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        # outputs, (hx, cx) = self.layer(packed_sequence)
        # outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=x.shape[1])
        outputs, _ = self.layer(x)
        outputs = self.layernorm(outputs)

        mask = mask.bool().unsqueeze(-1)

        max_out, _ = torch.max(outputs * mask + (~mask) * -1e6, dim=-2)

        # # classify from concatenation of final states
        # if self.lstm.bidirectional:
        #     final = torch.cat([hx[-2], hx[-1]], dim=-1)
        # else:  # classify from final state
        #     final = hx[-1]
        #
        # if self.attention:
        #     x = self.linear(outputs).tanh()
        #     score = self.le(x).softmax(-2)
        #     final = (outputs * score).sum(-2)

        return outputs, max_out


global_encoders = []
def get_encoder(layer, in_features, hidden_size, bidirectional=True, model_path=None, share=False):
    """Returns the requested layer."""
    if share and global_encoders:
        return global_encoders[-1]

    if layer in ("lstm", 'gru', 'rnn'):
        model = RNNEncoder(in_features, hidden_size, batch_first=True,
                          bidirectional=bidirectional, layer=layer)
    elif layer == "rcnn":
        model = RCNNEncoder(in_features, hidden_size,
                           bidirectional=bidirectional)
    elif layer == "bow":
        model = BOWEncoder()
    elif layer == "cnn":
        model = CNNEncoder(
            embedding_size=in_features, hidden_size=hidden_size,
            kernel_size=5)
    elif layer == 'bert':
        assert model_path is not None
        model = BERTEncoder(model_path)
    else:
        raise ValueError("Unknown layer")
    
    global_encoders.append(model)
    return model



class HardGumbelDist(nn.Module):
    def __init__(self, logits, temperature=0.5, **kwargs):
        super().__init__()
        self.logits=logits
        self.temperature=temperature

    def sample(self, shape):
        logits = self.logits
        target_shape = shape + logits.shape
        logits = logits.unsqueeze(0).expand(target_shape)
        sampled = torch.nn.functional.gumbel_softmax(logits, tau=self.temperature, hard=True)

        return sampled

    def rsample(self, shape):
        logits = self.logits
        target_shape = shape + logits.shape
        logits = logits.unsqueeze(0).expand(target_shape)
        sampled = torch.nn.functional.gumbel_softmax(logits, tau=self.temperature, hard=False)

        return sampled



class StraightThroughGate(nn.Module):
    """
    Computes a Bernoulli Gate
    Assigns a 0 or a 1 to each input word.
    """

    def __init__(self, in_features, 
                 out_features=1, 
                 use_gumbel=False, 
                 temperature=0.5,
                 min_temperature=0.05, **kwargs):
        super().__init__()
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.layer = nn.Linear(in_features, 1 if not use_gumbel else 2)

    def forward(self, x, num_samples=1):
        """
        Compute Binomial gate
        :param x: word represenatations [B, T, D]
        :return: gate distribution
        """
        logits = self.layer(x)  # [B, T, 1]
        if not self.use_gumbel:
            dist = torch.distributions.Bernoulli(logits=logits)
            z = dist.sample((num_samples, ))[..., -1:] # (ns, B, T, 1)
            z = dist.probs - dist.probs.detach() + z.detach()
        else:
            dist = HardGumbelDist(logits=logits, temperature=self.temperature, min_temperature=self.min_temperature)
            z = dist.sample((num_samples, ))[..., -1:] # (ns, B, T, 1)

        return z


    def get_loss(self, z, mask,
                 sparsity_factor, coherent_factor,
                 selection=0, lasso=0,
                 **kwargs):
        mask = mask.unsqueeze(0).expand_as(z)
        # compute generator loss
        optional = {}
        assert mask.shape == z.shape, f'{mask.shape} != {z.shape}'

        # sparsity regularization
        zsum = (z * mask).sum() / mask.sum()  # [B]
        optional['zmean'] = zsum

        zsum = (zsum - selection).abs()
        optional['spr_loss'] = zsum

        zdiff = (z[..., 1:] - z[..., :-1]).abs() * mask[..., 1:]
        zdiff = (zdiff - lasso).mean()
        optional['coherent_loss'] = zdiff

        cost_vec = zsum * sparsity_factor + zdiff * coherent_factor
        optional['gate_cost'] = cost_vec.mean()
        # print(f'gate_cost: {cost_vec.mean()}')
        return optional


def get_gate_cls(gate_name):
    if gate_name == 'gumbel':
        return GumbelGate
    elif gate_name == 'gumbel_max':
        return GumbelMaxGate
    elif gate_name == 'reinforce':
        return BernoulliGate
    elif gate_name == 'kuma':
        return KumaGate
    elif gate_name == 'straight':
        return StraightThroughGate
    else:
        raise ValueError(f'Unknown gate name {gate_name}.')



