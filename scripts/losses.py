import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def mse_loss(recon, target):
    bs = target.size(0)
    return torch.sum(torch.square(recon - target)) / bs

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, tau, device, cosine_similarity=False):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = tau
        self.device = device
        self.similarity_fn = self._cosine_similarity if cosine_similarity else self._dot_similarity
        mask = self._build_neg_mask().type(torch.bool)
        self.register_buffer("neg_mask", mask)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _build_neg_mask(self):
        bs = self.batch_size
        diagonal = np.eye(2 * bs)
        q1 = np.eye(2 * bs, 2 * bs, k=bs)
        q3 = np.eye(2 * bs, 2 * bs, k=-bs)
        mask = torch.from_numpy((diagonal + q1 + q3))
        return (1 - mask).type(torch.bool)

    @staticmethod
    def _dot_similarity(x, y):
        x = x.unsqueeze(1)
        y = y.T.unsqueeze(0)
        return torch.tensordot(x, y, dims=2)

    @staticmethod
    def _cosine_similarity(x, y):
        sim = nn.CosineSimilarity(dim=-1)
        return sim(x.unsqueeze(1), y.unsqueeze(0))

    def forward(self, representation):
        bs = self.batch_size
        sim = self.similarity_fn(representation, representation)
        l_pos = torch.diag(sim, bs)
        r_pos = torch.diag(sim, -bs)
        positives = torch.cat([l_pos, r_pos]).view(2 * bs, 1)
        negatives = sim[self.neg_mask.to(sim.device)].view(2 * bs, -1)
        logits = torch.cat((positives, negatives), dim=1) / self.temperature
        labels = torch.zeros(2 * bs, device=sim.device, dtype=torch.long)
        loss = self.criterion(logits, labels)
        return loss / (2 * bs)

class JointLoss(nn.Module):
    def __init__(self, batch_size, tau, device, contrastive_loss, distance_loss,
                 use_reconstruction, cosine_similarity):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.distance_loss = distance_loss
        self.use_reconstruction = use_reconstruction
        self.batch_size = batch_size
        self.ntxent = NTXentLoss(batch_size, tau, device, cosine_similarity)

    def forward(self, representation, xrecon, xorig):
        recon = mse_loss(xrecon, xorig) if self.use_reconstruction else xrecon.new_zeros(())
        closs = recon
        zloss = recon
        loss = recon

        if self.contrastive_loss:
            closs = self.ntxent(representation)
            loss = loss + closs

        if self.distance_loss:
            zi, zj = torch.split(representation, self.batch_size)
            zloss = mse_loss(zi, zj)
            loss = loss + zloss

        return loss, closs, recon, zloss

