import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()

        for x in range(12):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        return h_relu1

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.ab = ablation
        self.down_sample_4 = nn.Upsample(scale_factor=1 / 4, mode='bilinear')
    def forward(self, restore, sharp, blur):
        B, C, H, W = restore.size()
        restore_vgg, sharp_vgg, blur_vgg = self.vgg(restore), self.vgg(sharp), self.vgg(blur)

        # filter out sharp regions
        threshold = 0.01
        mask = torch.mean(torch.abs(sharp-blur), dim=1).view(B, 1, H, W)
        mask[mask <= threshold] = 0
        mask[mask > threshold] = 1
        mask = self.down_sample_4(mask)
        d_ap = torch.mean(torch.abs((restore_vgg - sharp_vgg.detach())), dim=1).view(B, 1, H//4, W//4)
        d_an = torch.mean(torch.abs((restore_vgg - blur_vgg.detach())), dim=1).view(B, 1, H//4, W//4)
        mask_size = torch.sum(mask)
        contrastive = torch.sum((d_ap / (d_an + 1e-7)) * mask) / mask_size

        return contrastive


class ContrastLoss_Ori(nn.Module):
    def __init__(self, ablation=False):
        super(ContrastLoss_Ori, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.ab = ablation

    def forward(self, restore, sharp, blur):

        restore_vgg, sharp_vgg, blur_vgg = self.vgg(restore), self.vgg(sharp), self.vgg(blur)
        d_ap = self.l1(restore_vgg, sharp_vgg.detach())
        d_an = self.l1(restore_vgg, blur_vgg.detach())
        contrastive_loss = d_ap / (d_an + 1e-7)
        
        return contrastive_loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        # x = torch.clamp(x + 0.5, min = 0,max = 1)
        # y = torch.clamp(y + 0.5, min = 0,max = 1)
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class Stripformer_Loss(nn.Module):

    def __init__(self, ):
        super(Stripformer_Loss, self).__init__()

        self.char = CharbonnierLoss()
        self.edge = EdgeLoss()
        self.contrastive = ContrastLoss()

    def forward(self, restore, sharp, blur):
        char = self.char(restore, sharp)
        edge = 0.05 * self.edge(restore, sharp)
        contrastive = 0.0005 * self.contrastive(restore, sharp, blur)
        loss = char + edge + contrastive
        return loss


def get_loss(model):
    if model['content_loss'] == 'Stripformer_Loss':
        content_loss = Stripformer_Loss()
    else:
        raise ValueError("ContentLoss [%s] not recognized." % model['content_loss'])
    return content_loss