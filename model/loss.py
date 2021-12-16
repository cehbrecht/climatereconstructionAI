import torch
import torch.nn as nn


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, mask, output, gt, discr_output):
        loss_dict = {}
        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)
        loss_dict['prc'] = self.l1(gt, output)# + self.l1(gt, output)
        loss_dict['tv'] = total_variation_loss(output)
        loss_dict['gan'] = - torch.mean(torch.log(discr_output))
        return loss_dict


class DiscriminatorLoss(nn.Module):
    def forward(self, discr_gt, discr_output):
        loss = -torch.mean(discr_gt) - torch.mean(torch.log(1 - discr_output))
        return loss
