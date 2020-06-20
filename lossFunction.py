import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torch.autograd import Variable
from math import pi

margin = 1

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=False, reduction='elementwise_mean', pos_weight=None):

    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)

    if pos_weight is None:
        ce_loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    else:
        log_weight = 1 + (pos_weight - 1) * target
        ce_loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())
    
    #    if math.isnan(ce_loss.mean()):
    #       ce_loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp() + 0.1).log())
     #   print(input.grad)
    #    return torch.Tensor([0]) 
            
    if weight is not None:
        ce_loss = ce_loss * weight

    if reduction == False:
        return ce_loss
    elif reduction == 'elementwise_mean':
        return ce_loss.mean()
    else:
        return ce_loss.sum()

class Loss_classi(nn.Module):
    def __init__(self):
        super(Loss_classi, self).__init__()

    def loss_classi(self, output, label):

        pos = torch.sum(label)
        pos_num = F.relu(pos - 1) + 1
        total = torch.numel(label)
        neg_num = F.relu(total - pos - 1) + 1
        pos_w = neg_num / pos_num

        classi_loss = binary_cross_entropy_with_logits(output, label, pos_weight=pos_w, reduce=True)
        
        return classi_loss

    def forward(self, output, label):

        loss = self.loss_classi(output, label)
        return loss


class localizationLoss(nn.Module):
    def __init__(self):
        super(localizationLoss, self).__init__()

    def forward(self, rvec, rvecgt, tvec, tvecgt):
        #print (quaternion)
        #qDot = quaternion * quaterniongt
        #qDot = qDot.sum(1)
        #theta = (torch.acos(qDot) * 180 / pi)[0]
        #print('theta(rotation loss):', theta)
        rl2norm = torch.sqrt(torch.sum(rvec ** 2))
        #rvec = torch.div(rvec, rl2norm)
        rl2normgt = torch.sqrt(torch.sum(rvecgt ** 2))
        #rvecgt = torch.div(rvecgt, rl2normgt)
        tl2norm = torch.sqrt(torch.sum(tvec ** 2))
        #tvec = torch.div(tvec, tl2norm)
        tl2normgt = torch.sqrt(torch.sum(tvecgt ** 2))
        #tvecgt = torch.div(tvecgt, tl2normgt)
        rloss = torch.dist(rvec, rvecgt, 2)
        tloss = torch.dist(tvec, tvecgt, 2)
        #print ('tLoss:', tloss)
        #print ('rLoss: ', rloss)
        #print ('r: ', rvec)
        #print ('rgt: ', rvecgt)
        #print ('t: ', tvec)
        #print ('tgt: ',tvecgt)
        #print ('')
        #print ('tLoss: ', tl2norm, '\n------------------------------\n')
        lossValue = (rloss * 100.0  + tloss).float()
        #print (theta, l2norm, lossValue)

        return lossValue, rloss, tloss


