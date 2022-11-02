#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: dice_loss.py
# description:
# implementation of dice loss for NLP tasks.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for MRC framework of NER.

    Alpha: This helps push down the weight given to easy examples. This helps in distinguishing hard negative and hard positive examples.
    Smoothing Factor: This is added to the numerator and the denominator of the dice coefficent.
                      This is done so that the negative examples with {2 * p_i * y_i = 0} also contribute a little to the loss due to the smoothing factor.
    Loss Reduction used is mean reduction.

    The Dice Loss can be applied to other tasks as well with a few changes.
    """
    def __init__(self,
                 alpha: float = 0.0,
                 smoothing_factor: float  = 1):
        super(DiceLoss, self).__init__()
        self.smoothing_factor = smoothing_factor
        self.alpha = alpha
    def forward(self, input_tensor: Tensor, target: Tensor) -> Tensor:
        #print(input_tensor.shape, target.shape)
          input_tensor = input_tensor.view(-1)
          input_tensor = torch.sigmoid(input_tensor)
          target_tensor = target.view(-1).float()
          adaptive = ((1 - input_tensor) ** self.alpha) * input_tensor
          numerator = 2 * (torch.sum(adaptive * target_tensor, -1)) + self.smoothing_factor
          denominator = torch.sum(adaptive) + torch.sum(target_tensor) + self.smoothing_factor
          loss = 1 - numerator/denominator
          return loss.mean()

        #logits_size = input.shape[-1]
        #print(input.shape, target.shape)


        #loss = self.calculate_loss(input_tensor, target_tensor, logits_size)


    def __str__(self):
        return f"Dice Loss smoothing_factor :{self.smoothing_factor}, alpha: {self.alpha}"

    def __repr__(self):
        return str(self)
