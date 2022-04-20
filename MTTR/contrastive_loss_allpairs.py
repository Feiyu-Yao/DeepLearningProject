import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# every operation should be pytorch
'''
Negative pairs also include sentence from different mask+original mask
if want to implement cross-video comparison, need to use NTXentLoss:
    https://github.com/KevinMusgrave/pytorch-metric-learning/issues/179
'''

class AllPairsContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(AllPairsContrastiveLoss, self).__init__()
        self.total_loss = 0
        self.temperature = temperature
        
    def forward(self, prediction_mask, positive_encoded_txt, negative_encoded_txt, device):
        '''
        "pred_masks": Tensor of dim [time, batch_size, num_queries, H, W] with the predicted masks logits
        make pairs
        for mask in outputs:
            for word in original:
                make positive pairs
            for word in inverse:
                make negative pairs
        for each mask in outputs:
            compute contrastive loss
        '''
        positive, negative = [], []
        for i in range(len(prediction_mask)):
            for j in range(len(positive_encoded_txt)):
                if (i == j):
                    # mark as positive pairs
                    mask, ptext, ntext = prediction_mask[i], positive_encoded_txt[j], negative_encoded_txt[j]
                    mask_text = torch.cat((mask, ptext), dim=0)
                    positive.append(mask_text)
                    
                    mask_text = torch.cat((mask, ntext), dim=0)
                    negative.append(mask_text)
                else:
                    mask, ntext = prediction_mask[i], negative_encoded_txt[i]
                    mask_text = torch.cat((mask, ntext), dim=0)
                    negative.append(mask_text)
                    
        positive_pairs, negative_pairs = [], []
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        # group and duplicate all positive pairs
        for i in range(len(positive)):
            p, n = torch.flatten(positive[i]), torch.flatten(negative[i])
            
            # pad p or n in case they are not the same length
            mask = torch.zeros(abs(p.shape[0] - n.shape[0])).to(device)
            if (p.shape[0] > n.shape[0]):
                n = torch.cat((n, mask), dim=0)
            else:
                p = torch.cat((p, mask), dim=0)
            
            # import pdb; pdb.set_trace()     
            positive_pairs.append(torch.exp(cos(p, p)/self.temperature))
            negative_pairs.append(torch.exp(cos(p, n)/self.temperature))
            
        total = torch.stack([torch.stack(negative_pairs).sum(dim=0), torch.stack(positive_pairs).sum(dim=0)]).sum(dim=0)
            
        # compute loss
        for p in positive_pairs:
            self.total_loss += - torch.log(torch.divide(p, total))
        
        return self.total_loss
