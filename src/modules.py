import copy

from torch import nn
import numpy as np

from transformers import AutoModel
import math
import torch
from torch.nn import Linear, Sequential
from torch.distributions import RelaxedOneHotCategorical, RelaxedBernoulli
from torch.distributions.bernoulli import Bernoulli
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.kumaraswamy import Kumaraswamy, AffineTransform, TransformedDistribution
import torch.nn.functional as F

class Classifier(nn.Module):
    """
    The Encoder takes an input text (and rationale z) and computes p(y|x,z)

    Supports a sigmoid on the final result (for regression)
    If not sigmoid, will assume cross-entropy loss (for classification)

    """

    def __init__(self,
                 embed_size,
                 hidden_size: int = 200,
                 output_size: int = 1,
                 dropout: float = 0.1,
                 layer: str = "rcnn",
                 model_path: str = None,
                 ):
        super().__init__()

        self.enc_layer = get_encoder(layer, embed_size, hidden_size, model_path=model_path)

        if hasattr(self.enc_layer, "cnn"):
            enc_size = self.enc_layer.cnn.out_channels
        elif isinstance(self.enc_layer, BERTEncoder):
            enc_size = self.enc_layer.bert.config.hidden_size
        else:
            enc_size = hidden_size * 2

        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(enc_size, output_size)
        )

    def forward(self, emb, mask, **kwargs):
        rnn_mask = mask
        # apply z to main inputs
        # z is also used to control when the encoder layer is active
        lengths = mask.long().sum(1)

        # print(lengths)
        hidden, final = self.enc_layer(emb, rnn_mask, lengths, **kwargs)
        # predict sentiment from final state(s)
        # y = self.output_layer(final)
        mask = mask.bool().unsqueeze(-1)
        hidden = hidden * mask + (~mask) * (-1e6)
        y = self.output_layer(hidden.max(-2)[0]) # for max pooling

        return y, final


class RationaleClassifier(nn.Module):
    """
    The Encoder takes an input text (and rationale z) and computes p(y|x,z)

    Supports a sigmoid on the final result (for regression)
    If not sigmoid, will assume cross-entropy loss (for classification)

    """

    def __init__(self,
                 embed_size,
                 hidden_size: int = 200,
                 output_size: int = 1,
                 dropout: float = 0.1,
                 layer: str = "rcnn",
                 model_path: str = None,
                 ):
        super().__init__()

        self.enc_layer = get_encoder(layer, embed_size, hidden_size, model_path=model_path)

        if hasattr(self.enc_layer, "cnn"):
            enc_size = self.enc_layer.cnn.out_channels
        elif isinstance(self.enc_layer, BERTEncoder):
            enc_size = self.enc_layer.bert.config.hidden_size
        else:
            enc_size = hidden_size * 2

        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(enc_size, output_size)
        )

    def forward(self, emb, mask, z=None, **kwargs):
        has_z = z is not None
        num_samples = None if not has_z else z.shape[0]
        rnn_mask = mask
        # apply z to main inputs
        # z is also used to control when the encoder layer is active
        lengths = mask.long().sum(1)

        # print(lengths)
        hidden, final = self.enc_layer(emb, rnn_mask, lengths, **kwargs)
        # predict sentiment from final state(s)
        # y = self.output_layer(final)
        mask = mask.bool().unsqueeze(-1).repeat(num_samples, 1, 1)

        hidden = hidden * mask + (~mask) * (-1e6)
        y = self.output_layer(hidden.max(-2)[0]) # for max pooling

        return y, final


class RCNNCell(nn.Module):
    """
    RCNN Cell
    Used in "Rationalizing Neural Predictions" (Lei et al., 2016)
    This is the bigram version of the cell.
    """

    def __init__(self, input_size, hidden_size):
        """
        Initializer.
        :param input_size:
        :param hidden_size:
        """
        super(RCNNCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # project input for λt, ct1, ct2
        self.ih_layer = nn.Linear(input_size, 3 * hidden_size, bias=False)

        # project previous state for λt (and add bias)
        self.hh_layer = nn.Linear(hidden_size, hidden_size, bias=True)

        # final output bias
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """This is PyTorch's default initialization method"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, prev, mask=None):
        """
        input is (batch, input_size)
        hx is ((batch, hidden_size), (batch, hidden_size))
        """
        prev_h, prev_c1, prev_c2 = prev

        # project input x and previous state h
        ih_combined = self.ih_layer(input_)
        wlx, w1x, w2x = torch.chunk(ih_combined, 3, dim=-1)
        ulh = self.hh_layer(prev_h)

        # main RCNN computation
        lambda_ = (wlx + ulh).sigmoid()
        c1 = lambda_ * prev_c1 + (1 - lambda_) * w1x
        c2 = lambda_ * prev_c2 + (1 - lambda_) * (prev_c1 + w2x)

        h = (c2 + self.bias).tanh()

        return h, c1, c2

    def __repr__(self):
        return "{}({:d}, {:d})".format(
            self.__class__.__name__, self.input_size, self.hidden_size)


class RCNN(nn.Module):
    """
    Encodes sentence with an RCNN
    Assumes batch-major tensors.
    """

    def __init__(self, in_features, hidden_size, bidirectional=False):
        super(RCNN, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.cell = RCNNCell(in_features, hidden_size)

        if bidirectional:
            self.cell_rev = RCNNCell(in_features, hidden_size)
        else:
            self.cell_rev = None

    @staticmethod
    def empty_state(batch_size, hidden_size, device):
        """
        Returns an initial empty state.
        :param batch_size:
        :param hidden_size:
        :param device:
        :return: tuple of (h, c1, c2)
        """
        h_prev = torch.zeros(batch_size, hidden_size, device=device)
        c1_prev = torch.zeros(batch_size, hidden_size, device=device)
        c2_prev = torch.zeros(batch_size, hidden_size, device=device)
        state = (h_prev, c1_prev, c2_prev)
        return state

    @staticmethod
    def _step(x_t, cell, state: tuple, mask_t):
        """
        Take a single step.
        :param x: the input for this time step [B, D]
        :param state: tuple of (h, c1, c2)
        :param mask_t: mask for this time step [B]
        :return:
        """
        h_prev, c1_prev, c2_prev = state
        mask_t = mask_t.unsqueeze(-1)

        h, c1, c2 = cell(x_t, state)  # step

        h_prev = mask_t * h + (1 - mask_t) * h_prev
        c1_prev = mask_t * c1 + (1 - mask_t) * c1_prev
        c2_prev = mask_t * c2 + (1 - mask_t) * c2_prev

        state = (h_prev, c1_prev, c2_prev)
        return state

    @staticmethod
    def _unroll(x, cell, mask,
                state: tuple = None):

        batch_size, time, emb_size = x.size()
        assert mask.size(1) == time, "time mask mismatch"

        # initial state
        if state is None:
            state = RCNN.empty_state(
                batch_size, cell.hidden_size, device=x.device)

        # process this time-major
        x = x.transpose(0, 1).contiguous()  # [T, B, D]
        mask = mask.transpose(0, 1).contiguous().float()  # time-major: [T, B]

        # process input x one time step at a time
        outputs = []

        for x_t, mask_t in zip(x, mask):
            # only update if mask active (skip zeroed words)
            state = RCNN._step(x_t, cell, state, mask_t)
            outputs.append(state[0])

        # return batch-major
        outputs = torch.stack(outputs, dim=1)  # [batch_size, time, D]

        return outputs

    def forward(self, x, mask, lengths=None, state: tuple = None):
        """
        :param x: input sequence [B, T, D] (batch-major)
        :param mask: mask with 0s for invalid positions
        :param lengths:
        :param state: take a step from this state, or None to start from zeros
        :return:
        """
        assert lengths is not None, "provide lengths"

        # forward pass
        outputs = RCNN._unroll(x, self.cell, mask, state=state)

        # only if this is a full unroll (full sequence, e.g. for encoder)
        # extract final states from forward outputs
        batch_size, time, dim = outputs.size()

        final_indices = torch.arange(batch_size).to(x.device)
        final_indices = (final_indices * time) + lengths - 1
        final_indices = final_indices.long()

        final = outputs.view([-1, dim]).index_select(0, final_indices)

        if self.bidirectional:
            assert state is None, \
                "can only provide state for unidirectional RCNN"

            # backward pass
            idx_rev = torch.arange(x.size(1) - 1, -1, -1)
            mask_rev = mask[:, idx_rev]  # fix for pytorch 1.2
            x_rev = x[:, idx_rev]
            outputs_rev = RCNN._unroll(x_rev, self.cell_rev, mask_rev)
            final_rev = outputs_rev[:, -1, :].squeeze(1)
            # outputs_rev = outputs_rev.flip(1)
            outputs_rev = outputs_rev[:, idx_rev]  # back into original order

            # concatenate with forward pass
            final = torch.cat([final, final_rev], dim=-1)
            outputs = torch.cat([outputs, outputs_rev], dim=-1)

        # mask out invalid positions
        outputs = torch.where(mask.unsqueeze(2), outputs, x.new_zeros([1]))

        return outputs, final


class GumbelDist(nn.Module):
    def __init__(self, logits, temperature=0.5, **kwargs):
        super().__init__()
        if logits.size(-1) == 1:
            base_cls = RelaxedBernoulli
        else:
            base_cls = RelaxedOneHotCategorical

        self.base_dist = base_cls(temperature=temperature, logits=logits)
        self.logits = self.base_dist.logits

    def sample(self, shape):
        # k = self.k
        # if self.threshold is not None:
        #     k = self.threshold * self.logits.size(-2)
        # _, sampled_idx = self.logits.topk(k, dim=-2) # (1, B, k)
        # ans = torch.zeros_like(self.logits)
        # # print(ans.shape, sampled_idx.shape)
        # ans.scatter_(-2, sampled_idx, 1)

        return self.base_dist.sample(shape)[..., -1:]

    def rsample(self, shape):
        sampled = self.base_dist.rsample(shape)

        return sampled[..., -1:]


class HardGumbelDist(nn.Module):
    def __init__(self, logits, temperature=0.5, **kwargs):
        super().__init__()
        self.logits=logits
        self.temperature=temperature

    def sample(self, shape):
        # k = self.k
        # if self.threshold is not None:
        #     k = self.threshold * self.logits.size(-2)
        # _, sampled_idx = self.logits.topk(k, dim=-2) # (1, B, k)
        # ans = torch.zeros_like(self.logits)
        # # print(ans.shape, sampled_idx.shape)
        # ans.scatter_(-2, sampled_idx, 1)
        logits = self.logits
        target_shape = shape + logits.shape
        logits = logits.unsqueeze(0).expand(target_shape)
        sampled = F.gumbel_softmax(logits, tau=self.temperature, hard=True)

        return sampled

    def rsample(self, shape):
        logits = self.logits
        target_shape = shape + logits.shape
        logits = logits.unsqueeze(0).expand(target_shape)
        sampled = F.gumbel_softmax(logits, tau=self.temperature, hard=False)

        return sampled[..., -1:]


class GumbelMaxDist(nn.Module):
    def __init__(self, logits, temperature=0.5, k=None, threshold=None):
        super().__init__()
        if logits.size(-1) == 1:
            base_cls = RelaxedBernoulli
        else:
            base_cls = RelaxedOneHotCategorical

        self.base_dist = base_cls(temperature=temperature, logits=logits)
        self.logits = self.base_dist.logits
        self.k = k or 1
        self.threshold = threshold

    def sample(self, shape):
        k = self.k
        if self.threshold is not None:
            k = self.threshold * self.logits.size(-2)
        _, sampled_idx = self.logits.topk(k, dim=-2)  # (1, B, k)
        ans = torch.zeros_like(self.logits)
        # print(ans.shape, sampled_idx.shape)
        ans.scatter_(-2, sampled_idx, 1)

        return ans[..., -1:]

    def rsample(self, shape):
        k = self.k
        if self.threshold is not None:
            k = 1
        shape = [k] + list(shape)

        sampled = self.base_dist.rsample(shape)
        sampled = sampled.max(dim=0)[0]

        return sampled[..., -1:]

    def prior(self, var_size, device):
        if self.threshold is None:
            ones = torch.ones(var_size, dtype=torch.float, device=device)
            ones = ones / var_size[-2]
            return ones
        thres = torch.tensor([self.threshold], dtype=torch.float, device=device)
        return thres.expand(var_size)


class GumbelGate(nn.Module):
    """
    Computes a Bernoulli Gate
    Assigns a 0 or a 1 to each input word.
    """

    def __init__(self, in_features, out_features=2, temperature=0.5, min_temperature=5e-3, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.min_temperature = min_temperature

        self.layer = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features // 2, 1),
            nn.LogSigmoid()
        )

    def forward(self, x, **kwargs):
        """
        Compute Binomial gate
        :param x: word represenatations [B, T, D]
        :return: gate distribution
        """
        logits = self.layer(x)  # [B, T, 1]
        dist = GumbelDist(logits=logits, temperature=self.temperature)

        return dist

    def get_loss(self, z, mask, dist, sparsity, coherent_factor, selection=0, lasso=0, beta=1,
                 sentence_mask=None, **kwargs):
        '''
        :param z: (B, L)
        :param mask: (B, L)
        :param kwargs:
        :return:
        '''
        optional = {}
        lengths = mask.sum(-1)

        # sparsity
        spr_loss = z.sum(-1) / (lengths + 1e-12)
        optional['zmean'] = spr_loss.mean()
        spr_loss = (spr_loss - selection).abs()
        optional['spr_loss'] = spr_loss.mean()
        # spr_loss = torch.maximum(spr_loss - selection, spr_loss.new_zeros(()))
        #
        # # continuous
        cont_loss = (z[..., 1:] - z[..., :-1]).abs().sum(-1) / (lengths - 1 + 1e-12)
        cont_loss = (cont_loss - lasso).abs()
        # cont_loss = torch.maximum(cont_loss - lasso, cont_loss.new_zeros(()))

        cost = sparsity * spr_loss.mean() + coherent_factor * cont_loss.mean()
        optional['gate_cost'] = cost

        return optional


class GumbelMaxGate(nn.Module):

    def __init__(self, in_features, out_features=2, temperature=0.5, min_temperature=5e-3, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.min_temperature = min_temperature

        self.layer = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features // 2, 1),
            nn.LogSigmoid()
        )

    def forward(self, x, k=1, threshold=None):
        """
        Compute Binomial gate
        :param x: word represenatations [B, T, D]
        :return: gate distribution
        """
        logits = self.layer(x)  # [B, T, 1]
        dist = GumbelMaxDist(logits=logits, temperature=self.temperature, k=k, threshold=threshold)

        return dist

    def get_loss(self, z, mask, dist, sparsity, coherent_factor, selection=0, lasso=0, beta=1,
                 sentence_mask=None, **kwargs):
        '''
        :param z: (B, L)
        :param mask: (B, L)
        :param kwargs:
        :return:
        '''
        optional = {}
        lengths = mask.sum(-1)

        logits = dist.base_dist.logits
        if dist.threshold is not None:
            d1 = torch.distributions.Bernoulli(logits=logits)
            d2 = torch.distributions.Bernoulli(probs=dist.prior(logits.size(), device=logits.device))
            kl_loss = dist.k * torch.distributions.kl_divergence(d1, d2)
            kl_loss = torch.where(sentence_mask or mask, kl_loss, kl_loss.new_zeros(()))
            kl_loss = kl_loss.sum(-1) / ((sentence_mask or mask).sum(-1) + 1e-12) / mask.shape[0]
        else:
            prior = dist.prior(logits.size(), device=logits.device)

            kl_loss = torch.nn.KLDivLoss(reduction='mean', log_target=True)(prior.log(), logits)
        optional['kl_loss'] = kl_loss.mean()

        # sparsity
        spr_loss = z.sum(-1) / (lengths + 1e-12)
        optional['zmean'] = spr_loss.mean()
        spr_loss = (spr_loss - dist.k / lengths).abs()
        optional['spr_loss'] = spr_loss.mean()
        # spr_loss = torch.maximum(spr_loss - selection, spr_loss.new_zeros(()))
        #
        # # continuous
        # cont_loss = (z[..., 1:] - z[..., :-1]).abs().sum(-1) / (lengths - 1 + 1e-12)
        # cont_loss = (cont_loss - lasso).abs()
        # cont_loss = torch.maximum(cont_loss - lasso, cont_loss.new_zeros(()))

        cost = beta * kl_loss.mean() + sparsity * spr_loss.mean()
        optional['gate_cost'] = cost

        return optional


class BernoulliGate(nn.Module):
    """
    Computes a Bernoulli Gate
    Assigns a 0 or a 1 to each input word.
    """

    def __init__(self, in_features, out_features=1, **kwargs):
        super(BernoulliGate, self).__init__()

        self.layer = Sequential(
            Linear(in_features, out_features, bias=True)
        )

        self.baselines = []

    def forward(self, x):
        """
        Compute Binomial gate
        :param x: word represenatations [B, T, D]
        :return: gate distribution
        """
        logits = self.layer(x)  # [B, T, 1]
        dist = Bernoulli(logits=logits)
        dist.rsample = dist.sample
        return dist

    def get_loss(self, z, mask, dist, loss_vec,
                 sparsity, coherent_factor,
                 selection=0, lasso=0,
                 **kwargs):
        optional = {}
        # compute generator loss
        lengths = mask.sum(-1)
        baseline = torch.tensor(self.baselines, device=lengths.device).mean()

        logp_z0 = dist.log_prob(z.new_zeros([1])).squeeze(-1)  # [B,T], log P(z = 0 | x)
        logp_z1 = dist.log_prob(z.new_ones([1])).squeeze(-1)  # [B,T], log P(z = 1 | x)

        # compute log p(z|x) for each case (z==0 and z==1) and mask
        logpz = torch.where(z == 0, logp_z0, logp_z1)
        logpz = torch.where(mask, logpz, logpz.new_zeros([1]))

        # sparsity regularization
        zsum = z.sum(-1) / (lengths + 1e-12)  # [B]
        optional['zmean'] = zsum.mean()
        zsum = (zsum - selection).abs()

        zdiff = (z[..., 1:] - z[..., :-1])
        zdiff = zdiff.abs().sum(-1) / (lengths - 1 + 1e-12)  # [B]
        zdiff = (zdiff - lasso).abs()

        cost_vec = loss_vec + zsum * sparsity + zdiff * coherent_factor
        # print(loss_vec.shape, zsum.shape, zdiff.shape, logpz.shape, mask.shape)
        cost_logpz = ((cost_vec - baseline).unsqueeze(-1) * logpz * mask).sum()  # cost_vec is neg reward

        self.baselines.append(cost_vec.mean().cpu().item())
        self.baselines = self.baselines[:1024]

        optional['gate_cost'] = cost_logpz

        return optional


class StraightThroughGate(nn.Module):
    """
    Computes a Bernoulli Gate
    Assigns a 0 or a 1 to each input word.
    """

    def __init__(self, in_features, out_features=1, use_gumbel=False, temperature=0.5,
                 min_temperature=0.05, **kwargs):
        super().__init__()
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.layer = Linear(in_features, 1 if not use_gumbel else 2)

    def forward(self, x, num_samples=1):
        """
        Compute Binomial gate
        :param x: word represenatations [B, T, D]
        :return: gate distribution
        """
        logits = self.layer(x)  # [B, T, 1]
        if not self.use_gumbel:
            dist = Bernoulli(logits=logits)
            z = dist.sample((num_samples, ))[..., -1:] # (ns, B, T, 1)
            z = dist.probs - dist.probs.detach() + z.detach()
        else:
            dist = HardGumbelDist(logits=logits, temperature=self.temperature, min_temperature=self.min_temperature)
            z = dist.sample((num_samples, ))[..., -1:] # (ns, B, T, 1)

        return z


    def get_loss(self, z, mask,
                 sparsity, coherent_factor,
                 selection=0, lasso=0,
                 **kwargs):
        mask = mask.unsqueeze(0).expand_as(z)
        # compute generator loss
        optional = {}
        assert mask.shape == z.shape, f'{mask.shape} != {z.shape}'
        # print(z.shape, mask.shape, lengths.shape)
        # sparsity regularization
        zsum = (z * mask).sum() / mask.sum()  # [B]
        # model.log('zmean', zsum.mean(), prog_bar=True, logger=True)
        optional['zmean'] = zsum
        zsum = (zsum - selection).abs()
        optional['spr_loss'] = zsum

        zdiff = (z[..., 1:] - z[..., :-1]).abs() * mask[..., 1:]
        zdiff = (zdiff - lasso).mean()
        optional['coherent_loss'] = zdiff
        # print(z.shape)
        # print(f'zmean: {zsum}', f'zdiff: {zdiff}', f'sparsity: {sparsity}', f'coherent_factor: {coherent_factor}')
        cost_vec = zsum * sparsity + zdiff * coherent_factor
        optional['gate_cost'] = cost_vec.mean()
        # print(f'gate_cost: {cost_vec.mean()}')
        return optional


class HardKumaDist(TransformedDistribution):
    def __init__(self, a, b, l=-0.1, r=1.1, validate_args=None):
        self.a = a
        self.b = b
        self.l = l
        self.r = r
        self.base_dist = Kumaraswamy(a, b, validate_args=validate_args)
        transforms = AffineTransform(loc=l, scale=(r - l))

        super().__init__(self.base_dist, transforms, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        sampled = super().rsample(sample_shape)

        return sampled.clamp(0, 1)

    def sample(self, sample_shape=torch.Size()):
        sampled = super().sample(sample_shape)

        return sampled.clamp(0, 1)


class KumaGate(nn.Module):
    """
    Computes a Hard Kumaraswamy Gate
    """

    def __init__(self, in_features, out_features=1, support=(-0.1, 1.1),
                 dist_type="hardkuma", **kwargs):
        super(KumaGate, self).__init__()

        self.dist_type = dist_type

        self.layer_a = Sequential(
            Linear(in_features, out_features),
            nn.Softplus()
        )
        self.layer_b = Sequential(
            Linear(in_features, out_features),
            nn.Softplus()
        )

        # support must be Tensors
        self.support = [support[0], support[1]]

        self.a = None
        self.b = None

    def forward(self, x, mask=None):
        """
        Compute latent gate
        :param x: word represenatations [B, T, D]
        :param mask: [B, T]
        :return: gate distribution
        """

        a = self.layer_a(x)
        b = self.layer_b(x)

        a = a.clamp(1e-6, 100.)  # extreme values could result in NaNs
        b = b.clamp(1e-6, 100.)  # extreme values could result in NaNs

        self.a = a
        self.b = b

        # we return a distribution (from which we can sample if we want)
        if self.dist_type == "kuma":
            dist = Kumaraswamy(a, b)
        elif self.dist_type == "hardkuma":
            dist = HardKumaDist(a, b, self.support[0], self.support[1])
        else:
            raise ValueError("unknown dist")

        return dist

    def get_loss(self, z, mask, dist, sparsity, coherent_factor,
                 selection=0, lasso=0, sentence_type_ids=None, sentence_mask=None,
                 **kwargs):
        '''
        :param z: (B, L)
        :param mask: (B, L)
        :param kwargs:
        :return:
        '''
        lengths = mask.sum(-1)
        optional = {}

        optional['a'] = dist.a.mean()
        optional['b'] = dist.b.mean()

        # (-dist.l) / (dist.r - dist.l)
        pdf0 = dist.cdf(z.new_zeros([]))
        pdf0 = pdf0.squeeze(-1)

        if sentence_type_ids is not None and pdf0.shape != z.shape:
            shape = list(z.shape)
            shape[-1] = 1
            ones = z.new_ones(()).expand(shape)
            pdf0 = torch.cat([ones, pdf0], dim=-1)
            pdf0 = pdf0 * sentence_mask
            pdf0 = pdf0[..., torch.arange(z.shape[-2]).unsqueeze(-1), sentence_type_ids]

        pdf0 = torch.where(mask, pdf0, pdf0.new_zeros([1]))  # [B, T]
        # pdf0=pdf0[0]
        pdf_nonzero = 1. - pdf0  # [B, T]
        pdf_nonzero = torch.where(mask, pdf_nonzero, pdf_nonzero.new_zeros([1]))

        # sparsity
        l0 = pdf_nonzero.sum(-1) / (lengths + 1e-12)  # [B]
        optional['zmean'] = (z.sum(-1) / (lengths + 1e-12)).mean()
        spr_loss = torch.maximum(l0 - selection, l0.new_zeros(())).mean()

        # spr_loss = (l0 - selection).abs().mean()

        # cost z_t = 0, z_{t+1} = non-zero
        zt_zero = pdf0[..., :-1]
        ztp1_nonzero = pdf_nonzero[..., 1:]

        # cost z_t = non-zero, z_{t+1} = zero
        zt_nonzero = pdf_nonzero[..., :-1]
        ztp1_zero = pdf0[..., 1:]

        # number of transitions per sentence normalized by length
        lasso_cost = zt_zero * ztp1_nonzero + zt_nonzero * ztp1_zero
        lasso_cost = lasso_cost * mask.float()[..., :-1]
        lasso_cost = lasso_cost.sum(-1) / (lengths + 1e-12)  # [B]

        # continuous
        cont_loss = torch.maximum(lasso_cost - lasso, lasso_cost.new_zeros(())).mean()
        # cont_loss = (lasso_cost - lasso).abs().mean()

        z_loss = (optional['zmean'] - selection).abs().mean()

        optional['gate_cost'] = sparsity * spr_loss + coherent_factor * cont_loss + sparsity * z_loss

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


class Generator(nn.Module):
    """
    The Generator takes an input text and returns samples from p(z|x)
    """

    def __init__(self,
                 embed_size,
                 hidden_size: int = 200,
                 dropout: float = 0.1,
                 layer: str = "rcnn",
                 share: bool = False,
                 gate: str = 'gumbel',
                 temperature: float = 0.5,
                 min_temperature: float = 5e-3,
                 num_samples: int = 1,
                 sentence_level: bool = False,
                 model_path: str = None,
                 k=None,
                 threshold=None,
                 straight_with_gumbel: bool = False,
                 ):

        super().__init__()
        self.num_samples = num_samples
        self.enc_layer = get_encoder(layer, embed_size, hidden_size, model_path=model_path, share=share)
        self.k = k
        self.threshold = threshold

        if isinstance(self.enc_layer, BERTEncoder):
            enc_size = self.enc_layer.bert.config.hidden_size
        else:
            enc_size = hidden_size * 2

        self.sentence_level = sentence_level
        if self.sentence_level:
            self.sentence_map_layer = nn.Sequential(
                nn.Linear(enc_size * 2, enc_size),
                nn.Dropout(dropout),
                nn.ReLU()
            )

        self.dropout = nn.Dropout(dropout)

        self.gate = gate

        self.z_layer = get_gate_cls(gate)(enc_size, temperature=temperature, min_temperature=min_temperature,
                                          k=self.k, threshold=self.threshold, use_gumbel=straight_with_gumbel)

        self.z = None  # z samples
        self.z_dists = None  # z distribution(s)

    @property
    def get_loss(self):
        return self.z_layer.get_loss

    def forward(self,
                emb,
                mask,
                sentence_spans=None,
                sentence_mask=None,
                sentence_type_ids=None,
                **kwargs):

        # encode sentence
        lengths = mask.long().sum(1)
        batch_size = emb.shape[0]
        h, _ = self.enc_layer(emb, mask, lengths, **kwargs)
        h = self.dropout(h)
        if self.sentence_level:
            # 拿到每个句子start和end的向量，拼接起来后映射到原始维度。
            shape = [batch_size, sentence_spans.shape[1]] + list(h.shape[2:])
            # print(sentence_spans.shape, h.shape)

            # sentence_representation_start = h[torch.arange(batch_size).unsqueeze(-1), sentence_spans[..., 0]] # # this is slow
            sentence_representation_start = h.gather(1, sentence_spans[..., 0:1].expand(shape))
            sentence_representation_start = sentence_representation_start[:, 1:]
            # sentence_representation_end = h[torch.arange(batch_size).unsqueeze(-1), sentence_spans[..., 1] - 1] # this is slow
            sentence_representation_end = h.gather(1, torch.maximum(sentence_spans[..., 1:] - 1, sentence_spans.new_zeros(())).expand(shape))
            sentence_representation_end = sentence_representation_end[:, 1:]
            assert sentence_representation_start.shape == (batch_size, sentence_spans.shape[1] - 1, h.shape[-1])

            sentence_representation = torch.cat([sentence_representation_start, sentence_representation_end],
                                                dim=-1)  # (B, num_sentence, 2*h)

            h = self.sentence_map_layer(sentence_representation)  # (B, num_sentence, h)
            # print(sentence_mask.shape, h.shape)
            h = torch.where(sentence_mask[..., 1:, None], h, h.new_zeros(()))

        # compute parameters for Bernoulli p(z|x)
        z_dist = self.z_layer(h)

        if self.training:  # sample or re-parameterized sample.
            z = z_dist.rsample((self.num_samples,))  # [B, T, 1]
        else:  # deterministic
            z = z_dist.sample((self.num_samples,))  # [B, T, 1]

        self.original_z = z
        # 第一句话(query)永远是rationale的一部分。
        if self.sentence_level:
            # get ones
            shape = list(z.shape)  # (B, n_s, 1)
            shape[-2] = 1  # (B, 1, 1)
            ones = z.new_ones(()).expand(shape)

            # 和z拼接，在句子级的情况下表示第一句话是rationale。
            z = torch.cat([ones, z], dim=-2)  # (B, n_s+1, 1)

            self.original_z = z

            # 句子级别的z转换成token级别，方便和embedding相乘。
            z = z * sentence_mask.unsqueeze(-1)

            # z = z[..., torch.arange(batch_size).unsqueeze(-1), sentence_type_ids, :] # 这样非常慢
            # print(z.shape, emb.shape, sentence_type_ids.shape)
            shape = list(z.shape)
            shape[-2] = sentence_type_ids.shape[-1]
            z = z.gather(index=sentence_type_ids.unsqueeze(-1).expand(shape), dim=-2)  # (B, n_t, 1)
            assert z.shape == (self.num_samples, batch_size, emb.shape[1], 1), f'{z.shape}, {emb.shape}'

        if self.num_samples == 1:
            z = z[0]

        z = z.squeeze(-1)  # [B, T, 1]  -> [B, T]
        z = torch.where(mask, z, z.new_zeros([1]))

        self.z = z
        self.z_dists = z_dist
        return z


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


class BERTEncoder(nn.Module):
    """
    This module encodes a sequence into a single vector using an LSTM.
    """

    def __init__(self,
                 model_path: str, ):
        """
        :param in_features:
        :param hidden_size:
        :param batch_first:
        :param bidirectional:
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)

    def forward(self, emb, mask, length, token_type_ids=None, **kwargs):
        """
        Encode sentence x
        :param x: sequence of word embeddings, shape [B, T, E]
        :param mask: byte mask that is 0 for invalid positions, shape [B, T]
        :param lengths: the lengths of each input sequence [B]
        :return:
        """
        outputs = self.bert(inputs_embeds=emb, attention_mask=mask, token_type_ids=token_type_ids)

        return outputs[0], outputs[1]


class RCNNEncoder(nn.Module):
    """
    This module encodes a sequence into a single vector using an LSTM.
    """

    def __init__(self, in_features, hidden_size, batch_first: bool = True,
                 bidirectional: bool = True):
        super(RCNNEncoder, self).__init__()
        assert batch_first, "only batch_first=True supported"
        self.rcnn = RCNN(in_features, hidden_size, bidirectional=bidirectional)

    def forward(self, x, mask, lengths, **kwargs):
        """

        :param x: sequence of word embeddings, shape [B, T, E]
        :param mask: byte mask that is 0 for invalid positions, shape [B, T]
        :param lengths: the lengths of each input sequence [B]
        :return:
        """
        return self.rcnn(x, mask, lengths)


class BOWEncoder(nn.Module):
    """
    Returns a bag-of-words for a sequence of word embeddings.
    Ignores masked-out positions.
    """

    def __init__(self):
        super(BOWEncoder, self).__init__()

    def forward(self, x, mask, lengths, **kwargs):
        """

        :param x: sequence of word embeddings, shape [B, T, E]
        :param mask: byte mask that is 0 for invalid positions, shape [B, T]
        :param lengths: the lengths of each input sequence [B]
        :return:
        """
        bow = x * mask.unsqueeze(-1).float()
        bow = bow.sum(1)  # sum over time to get [B, E]
        bow = bow / lengths.unsqueeze(-1).float()  # normalize by sent length
        return None, bow


class CNNEncoder(nn.Module):
    """
    Returns a bag-of-words for a sequence of word embeddings.
    Ignores masked-out positions.
    """

    def __init__(self,
                 embedding_size: int = 300,
                 hidden_size: int = 200,
                 kernel_size: int = 5):
        super(CNNEncoder, self).__init__()
        padding = kernel_size // 2
        self.cnn = nn.Conv1d(embedding_size, hidden_size, kernel_size,
                             padding=padding, bias=True)

    def forward(self, x, mask, lengths, **kwargs):
        """

        :param x: sequence of word embeddings, shape [B, T, E]
        :param mask: byte mask that is 0 for invalid positions, shape [B, T]
        :param lengths: the lengths of each input sequence [B]
        :return:
        """
        # Conv1d Input:  (N, embedding_size E, T)
        # Conv1d Output: (N, hidden_size D,    T)
        x = x.transpose(1, 2)  # make [B, E, T]

        x = self.cnn(x)

        x = x.transpose(1, 2)  # make [B, T, D]
        x = x * mask.unsqueeze(-1).float()  # mask out padding
        x = x.sum(1) / lengths.unsqueeze(-1).float()  # normalize by sent length

        return None, x

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

if __name__ == '__main__':
    a = HardKumaDist(1, 1)
    print(a.sample((10, 10)))
