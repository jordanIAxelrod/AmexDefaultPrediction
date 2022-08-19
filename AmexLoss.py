import torch
import torch.nn as nn
from config import *

class AmexLoss(nn.Module):

    def __init__(self, trailing_size: int, loss: nn.Module):
        self.trailing_pred = []
        self.trailing_ground = []
        self.trailing_size = trailing_size
        self.standard_loss = loss

    def forward(self, prediction: torch.Tensor, ground: torch.Tensor):
        if len(self.trailing_loss) < self.trailing_size:
            self.trailing_pred.append(prediction.cpu())
            self.trailing_ground.append(ground.cpu())
            return self.loss(prediction, ground)
        else:
            preds, index = torch.cat(
                self.trailing_pred
            ).sort(
                dim=0, descending=False
            ).to(device)
            weight = (20 - 19 * torch.cat(self.trailing_ground)[index].to(device))
            cutoff = 0.04 * weight.sum()
            weight = torch.nonzero(weight.cumsum() > cutoff)[-1][0]
            preds = preds[weight]  # top five percent

            fltr = torch.logical_and(prediction > preds, ground == 0)

            bce = ground * torch.log(prediction) + (1 - ground) * torch.log(1 - prediction)

            loss = fltr * bce * 20 + torch.logical_not(fltr) * bce
            self.trailing_pred.pop(0)
            self.trailing_ground.pop(0)
            self.trailing_pred.append(prediction.cpu())
            self.trailing_ground.append(ground.cpu())
            return loss.sum() / loss.shape[0]
