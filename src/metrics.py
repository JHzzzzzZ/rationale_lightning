from torchmetrics import Metric, Accuracy, MeanMetric
import torch


class SpanFscore(Metric):
    def __init__(self, rationale_is_list=True, eps=1e-10):
        super().__init__()
        self.rationale_is_list = rationale_is_list
        self.eps = eps
        self.reset()

    def reset(self):
        self.match = 0
        self.total_p = 0
        self.total_r = 0

    def compute(self):
        P = self.match / (self.total_p + self.eps)
        R = self.match / (self.total_r + self.eps)
        F = 2*P*R/(P+R+self.eps)
        ans = {'precision': P, 'recall': R, 'fscore': F}
        return ans

    def update(self, match, non_zero, gt):
        self.match += match
        self.total_p += non_zero.sum()
        self.total_r += gt.sum()

    def forward(self, pred, span, mask):
        '''
        :param pred: (B, L, 1) probability
        :param span: list: (B, maxlen, 2) or (B, maxlen)
        :return:
        '''
        num_samples = 1
        if len(pred.shape) == 4 :
            num_samples, B, L = pred.shape[0], pred.shape[1], pred.shape[2]
        else:
            B, L = pred.shape[0], pred.shape[1]
        self.rationale_is_list = span.shape[-1] == 2
        if self.rationale_is_list:
            temp = torch.arange(L, device=pred.device).view(1, 1, 1, -1).expand(num_samples, B, -1, -1)  # (ns, B, L, 1)

            gt = torch.bitwise_and(torch.ge(temp, span[..., 0:1]),
                                   torch.le(temp, span[..., 1:2]))  # (ns, B, maxlen, L)
            gt = torch.any(gt, dim=2)
        else:
            assert span.shape==(B, L), f'{span.shape}, {pred.shape}'
            gt = span

        non_zero = (pred > 0).squeeze(-1)  # (B, L)
        non_zero = non_zero * mask

        gt = gt * mask

        match = torch.bitwise_and(non_zero, gt).sum()

        self.update(match, non_zero, gt)

        return self.compute()


class SparsityMetrics(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("prediction_positive", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state('num_words', default=torch.zeros(1), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, mask: torch.Tensor):
        mask = mask.long()
        if mask.dim() < preds.dim():
            mask = mask.unsqueeze(0).expand_as(preds)
        assert preds.shape == mask.shape, f'{preds.shape}, {mask.shape}'  # (B, L)

        predictions = mask * (preds>0.5).long()
        self.prediction_positive += predictions.sum()
        self.num_words += mask.sum()

        assert self.num_words >= self.prediction_positive, f'{self.num_words}, {self.prediction_positive}'

        return self.compute()

    def compute(self):
        sparsity = self.prediction_positive / self.num_words

        return {'sparsity': sparsity}


class TokenF1(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("true_positives", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("prediction_positive", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("target_positive", default=torch.zeros(1), dist_reduce_fx="sum")


    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        mask = mask.long()
        if mask.dim() < preds.dim():
            mask = mask.unsqueeze(0).expand_as(preds)
        if target.dim() < preds.dim():
            target = target.unsqueeze(0).expand_as(preds)
        assert preds.shape == target.shape == mask.shape, f'{preds.shape}, {target.shape}, {mask.shape}'
        predictions = mask * (preds>0.5).long()
        target = mask * target.long()

        true_positives = predictions * target

        self.true_positives += true_positives.sum()
        self.prediction_positive += predictions.sum()
        self.target_positive += target.sum()
        assert self.prediction_positive >= self.true_positives, f'{self.prediction_positive}, {self.true_positives}'
        assert self.target_positive >= self.true_positives, f'{self.target_positive}, {self.true_positives}'


    def compute(self):
        precision = self.true_positives / (self.prediction_positive + 1e-10)
        recall = self.true_positives / (self.target_positive + 1e-10)
        f1 = 2 * precision * recall / (precision + recall)

        return {'precision': precision, 'recall': recall, 'f1': f1}


class MultiClassAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("num_corrects", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.argmax(dim=-1)
        assert preds.shape == target.shape
        num_corrects = (preds == target).sum()
        self.num_corrects += num_corrects
        self.total += target.numel()

        return self.compute()

    def compute(self):
        return {'acc': self.num_corrects / self.total}

# # TODO: 改成与预测兼容的。
# class TokenF1(Metric):
#     def __init__(self, num_classes: int):
#         super().__init__()
#         self.num_classes = num_classes
#         self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
#         self.add_state("false_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
#         self.add_state("false_negatives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
#
#     def update(self, preds: torch.Tensor, target: torch.Tensor):
#         assert preds.shape == target.shape
#         for i in range(self.num_classes):
#             true_positives = (preds == i) & (target == i)
#             false_positives = (preds == i) & (target != i)
#             false_negatives = (preds != i) & (target == i)
#
#             self.true_positives[i] += true_positives.sum()
#             self.false_positives[i] += false_positives.sum()
#             self.false_negatives[i] += false_negatives.sum()
#
#     def compute(self):
#         precision = self.true_positives / (self.true_positives + self.false_positives)
#         recall = self.true_positives / (self.true_positives + self.false_negatives)
#         f1 = 2 * precision * recall / (precision + recall)
#
#         weights = self.true_positives + self.false_negatives
#         weights /= weights.sum()
#
#         return (f1 * weights).sum()