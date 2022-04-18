import torch
from torch import nn
import numpy as np

# every operation should be pytorch
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.outputs = outputs
        self.targets = targets
        self.text_queries_original = text_queries_original
        self.text_queries_inverse = text_queries_inverse
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
        
        return self.total_loss
