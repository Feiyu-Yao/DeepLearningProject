import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# every operation should be pytorch
'''
Group text and mask as a pair insteads of concatnating them.
Text need to be upsampled to the same length as mask.
if want to implement cross-video comparison, need to use NTXentLoss:
    https://github.com/KevinMusgrave/pytorch-metric-learning/issues/179
'''

class TMContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(TMContrastiveLoss, self).__init__()
        self.total_loss = 0
        self.temperature = temperature
        
    def forward(self, prediction_mask, positive_encoded_txt, negative_encoded_txt, device):
        '''
        "pred_masks": Tensor of dim [time, batch_size, num_queries, H, W] with the predicted masks logits
        make pairs
        positive pairs: (mask1, word1), (mask2, word2)
        negative pairs: all other pairs
        '''
        positive_pairs, negative_pairs = [], []
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(len(prediction_mask)):
            for j in range(len(positive_encoded_txt)):
                if (i == j):
                    # mark as positive pairs
                    mask, ptext, ntext = prediction_mask[i], positive_encoded_txt[j], negative_encoded_txt[j]
                    positive_pairs.append(torch.exp(cos(mask, ptext)/self.temperature))
                    negative_pairs.append(torch.exp(cos(mask, ntext)/self.temperature))
                else:
                    mask, ntext = prediction_mask[i], negative_encoded_txt[i]
                    negative_pairs.append(torch.exp(cos(mask, ntext)/self.temperature))
                    
        total = torch.stack([torch.stack(negative_pairs).sum(dim=0), torch.stack(positive_pairs).sum(dim=0)]).sum(dim=0)
            
        # compute loss
        for p in positive_pairs:
            self.total_loss += - torch.log(torch.divide(p, total))
        
        return self.total_loss
