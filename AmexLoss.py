import torch
import torch.nn as nn


class AmexLoss(nn.Module):

    def __init__(self, trailing_size: int, loss: nn.Module):
        self.trailing_loss = []
        self.trailing_size = trailing_size
        self.standard_loss = loss

    def forward(self, prediction: torch.Tensor, ground: torch.Tensor):
        if len(self.trailing_loss) < self.trailing_size:
            return self.loss(prediction, ground)
        else:
            cutoff = torch.cat(
                self.trailing_loss
            ).sort(
                dim=0, descending=True
            )[self.trailing_loss[0].shape[0] * self.trailing_size // 20]  # top five percent

            fltr = torch.logical_and(prediction > cutoff, ground == 1)

            bce = ground * torch.log(prediction) + (1 - ground) * torch.log(1 - prediction)

            loss = fltr * bce * 20 + torch.logical_not(fltr) * bce
            return loss.sum() / loss.shape[0]
