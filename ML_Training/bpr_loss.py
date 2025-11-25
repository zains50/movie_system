import torch
import torch.nn as nn
class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, user_emb, pos_emb, neg_emb):
        # user_emb, pos_emb, neg_emb have shape [batch_size, embedding_dim]

        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)

        diff = pos_scores - neg_scores
        loss = -torch.mean(torch.log(torch.sigmoid(diff) + 1e-8))

        return loss
