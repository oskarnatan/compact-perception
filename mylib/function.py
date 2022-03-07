from torch import nn



class IOUScore(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Yp, Yt):
        output_ = Yp > 0.5 
        target_ = Yt > 0.5 
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = intersection / union
        return iou


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Yp, Yt, smooth=1e-7):
        num = Yt.size(0)
        Yp = Yp.view(num, -1)
        Yt = Yt.view(num, -1)
        bce = nn.functional.binary_cross_entropy(Yp, Yt)
        intersection = (Yp * Yt).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / (Yp.sum() + Yt.sum() + smooth))
        bce_dice_loss = bce + dice_loss
        return bce_dice_loss


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Yp, Yt):
        num = Yt.size(0)
        Yp = Yp.view(num, -1)
        Yt = Yt.view(num, -1)
        loss = nn.functional.l1_loss(Yp, Yt)
        return loss


class HuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Yp, Yt, b=0.5):
        num = Yt.size(0)
        Yp = Yp.view(num, -1)
        Yt = Yt.view(num, -1)
        loss = nn.functional.smooth_l1_loss(Yp, Yt, beta=b)

        return loss

