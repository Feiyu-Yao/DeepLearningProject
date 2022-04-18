import torch
from torch import nn
import numpy as np

# every operation should be pytorch
'''
Don't group pairs from different video together,
if want to implement cross-video comparison, need to use NTXentLoss:
    https://github.com/KevinMusgrave/pytorch-metric-learning/issues/179
'''

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.total_loss = 0
        self.temperature = temperature
        
    def forward(self, prediction_mask, positive_encoded_txt, negative_encoded_txt):
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
        import pdb; pdb.set_trace()
        positive, negative = [], []
        for i in len(prediction_mask):
            for j in len(positive_encoded_txt):
                if (i == j):
                    mask, ptext, ntext = prediction_mask[i], positive_encoded_txt[j], negative_encoded_txt[j]
                    mask_text = torch.cat((mask, ptext), dim=0)
                    positive.append(mask_text)
                    
                    mask_text = torch.cat((mask, ntext), dim=0)
                    negative.append(mask_text)
                    
        positive_pairs, negative_pairs = [], []
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # group and duplicate all positive pairs
        for i in range(len(positive)):
            positive_pairs.append(np.exp(cos(positive[i], positive[i])/self.temperature))
            negative_pairs.append(np.exp(cos(positive[i], negative[i])/self.temperature))
            
        total = np.sum([np.sum(negative_pairs), np.sum(positive_pairs)])
            
        # compute loss
        for p in positive_pairs:
            self.total_loss += - np.log(p/total)
        
        return self.total_loss
