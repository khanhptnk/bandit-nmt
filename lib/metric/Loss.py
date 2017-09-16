from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weighted_xent_loss(logits, targets, weights):
    log_dist = F.log_softmax(logits)
    losses = -log_dist.gather(1, targets.unsqueeze(1)).squeeze(1)
    losses = losses * weights
    return losses.sum()

def weighted_mse(logits, targets, weights):
    losses = (logits - targets)**2
    losses = losses * weights
    return losses.sum()
