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
        self.match += match.cpu().item()
        self.total_p += non_zero.sum().cpu().item()
        self.total_r += gt.sum().cpu().item()

    def forward(self, pred, span, mask):
        '''
        :param pred: (B, L, 1) probability
        :param span: list: (B, maxlen, 2) or (B, maxlen)
        :return:
        '''
        B, L = pred.shape[0], pred.shape[1]
        self.rationale_is_list = span.shape[-1] == 2
        if self.rationale_is_list:
            temp = torch.arange(L, device=pred.device).view(1, 1, -1).expand(B, -1, -1)  # (B, L, 1)

            gt = torch.bitwise_and(torch.ge(temp, span[..., 0:1]),
                                   torch.le(temp, span[..., 1:2]))  # (B, maxlen, L)
            gt = torch.any(gt, dim=1)
        else:
            assert span.shape==(B, L), f'{span.shape}, {pred.shape}'
            gt = span

        non_zero = (pred > 0).squeeze(-1)  # (B, L)
        non_zero = non_zero * mask

        gt = gt * mask

        match = torch.bitwise_and(non_zero, gt).sum()

        self.update(match, non_zero, gt)

        return self.compute()



