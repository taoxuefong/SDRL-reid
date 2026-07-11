from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

from .ssdm import PatchGenerator, SSDM

__all__ = ['ResNetPart', 'sdrl_resnet18', 'sdrl_resnet34', 'sdrl_resnet50', 'sdrl_resnet101',
           'sdrl_resnet152']


class ResNetPart(nn.Module):
    """ ResNet with part features by uniform partitioning. Optional SSDM for semantic patches. """
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, num_parts=3, num_classes=0, use_ssdm=False, ddim_steps=50,
                 ssdm_train_refine=False, ssdm_refine_t_max=100, ssdm_mask_alpha=1.0):
        super(ResNetPart, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.use_ssdm = use_ssdm
        self.ddim_steps = ddim_steps
        self.ssdm_train_refine = ssdm_train_refine
        self.ssdm_refine_t_max = ssdm_refine_t_max
        # Soft mask blend: U = x*((1-a)+a*m); a<1 keeps some background to reduce accuracy drop from mask/aug misalignment
        self.ssdm_mask_alpha = ssdm_mask_alpha
        # Construct base (pretrained) resnet
        if depth not in ResNetPart.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNetPart.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        self.num_parts = num_parts
        self.num_classes = num_classes

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.rap = nn.AdaptiveAvgPool2d((self.num_parts, 1))

        # global feature classifiers
        self.bnneck = nn.BatchNorm1d(2048)
        init.constant_(self.bnneck.weight, 1)
        init.constant_(self.bnneck.bias, 0)
        self.bnneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        init.normal_(self.classifier.weight, std=0.001)

        # part feature classifiers
        for i in range(self.num_parts):
            name = 'bnneck' + str(i)
            setattr(self, name, nn.BatchNorm1d(2048))
            init.constant_(getattr(self, name).weight, 1)
            init.constant_(getattr(self, name).bias, 0)
            getattr(self, name).bias.requires_grad_(False)

            name = 'classifier' + str(i)
            setattr(self, name, nn.Linear(2048, self.num_classes, bias=False))

        if use_ssdm:
            self.patch_proposal = PatchGenerator()
            self.ssdm = SSDM(num_steps=1000, ddim_steps=ddim_steps)

        if not pretrained:
            self.reset_params()

    def _parts_from_theta(self, x, theta):
        """Sample part features from feature map x using theta [B, 3, 2, 3]."""
        f_p = []
        for i in range(self.num_parts):
            stripe = theta[:, i, :, :].float()
            grid = F.affine_grid(stripe, x.size(), align_corners=False)
            f_p.append(F.grid_sample(x, grid, align_corners=False))
        return f_p

    def _sdc_loss(self, x, theta_t, theta_tm1, tau=0.07):
        """
        SDC: decoupled contrastive on (theta_{t-1}, theta_t) patch features.
        L_sdc = (1/(2P)) * sum_i (l^{t-1}_i + l^t_i).
        Positive: (F^{t-1}_i, F^t_i); negatives: all other 2P-1 patch features.
        Similarity = -L2 distance; InfoNCE over 2P candidates.
        """
        P = self.num_parts
        B = x.size(0)
        # SDC constrains geometry (theta) only, avoiding perturbation of backbone discriminative features
        x_sdc = x.detach()
        f_t_list = self._parts_from_theta(x_sdc, theta_t)
        f_tm1_list = self._parts_from_theta(x_sdc, theta_tm1)
        F_t = []
        F_tm1 = []
        for i in range(P):
            # SDC does not reuse bnneck, avoiding noisy patches polluting BN running stats
            ft_i = self.gap(f_t_list[i]).view(B, -1)
            ftm1_i = self.gap(f_tm1_list[i]).view(B, -1)
            ft_i = F.normalize(ft_i, p=2, dim=1)
            ftm1_i = F.normalize(ftm1_i, p=2, dim=1)
            F_t.append(ft_i)
            F_tm1.append(ftm1_i)
        F_t = torch.stack(F_t, dim=-1)
        F_tm1 = torch.stack(F_tm1, dim=-1)

        def d(a, b):
            return (a - b).norm(dim=1)

        loss_sdc = 0.0
        # All 2P features: [F_tm1_0..F_tm1_{P-1}, F_t_0..F_t_{P-1}]
        for i in range(P):
            # l^{t-1}_i: anchor F^{t-1}_i, positive F^t_i (index P+i in 2P list)
            anchor = F_tm1[:, :, i]
            logits = []
            for j in range(P):
                logits.append(-d(anchor, F_tm1[:, :, j]).unsqueeze(1))
            for j in range(P):
                logits.append(-d(anchor, F_t[:, :, j]).unsqueeze(1))
            logits = torch.cat(logits, dim=1) / tau
            labels = torch.full((B,), P + i, device=x.device, dtype=torch.long)
            loss_sdc = loss_sdc + F.cross_entropy(logits, labels)
            # l^t_i: anchor F^t_i, positive F^{t-1}_i (index i in 2P list)
            anchor = F_t[:, :, i]
            logits = []
            for j in range(P):
                logits.append(-d(anchor, F_tm1[:, :, j]).unsqueeze(1))
            for j in range(P):
                logits.append(-d(anchor, F_t[:, :, j]).unsqueeze(1))
            logits = torch.cat(logits, dim=1) / tau
            labels = torch.full((B,), i, device=x.device, dtype=torch.long)
            loss_sdc = loss_sdc + F.cross_entropy(logits, labels)
        return loss_sdc / (2 * P)

    def forward(self, x, mask=None):
        x = self.base(x)

        f_g = self.gap(x)
        f_g = f_g.view(x.size(0), -1)
        f_g = self.bnneck(f_g)

        if self.training is False:
            f_g = F.normalize(f_g)
            return f_g

        logits_g = self.classifier(f_g)

        if self.use_ssdm:
            # Eq.19/24: U = F(x) ⊗ m (mask downsampled to feature map size, then element-wise multiply).
            # When mask=None, U=x (original behavior). Global feature f_g is unaffected by mask.
            if mask is not None:
                m_ds = F.interpolate(mask.float(), size=x.shape[-2:], mode='nearest')
                a = self.ssdm_mask_alpha
                U = x * ((1.0 - a) + a * m_ds)
            else:
                U = x
            theta0 = self.patch_proposal(U)
            if self.training:
                if self.ssdm_train_refine:
                    loss_ssdm, theta_t, theta_tm1, theta_parts = self.ssdm.training_loss(
                        theta0, f_g, return_refined=True, refine_t_max=self.ssdm_refine_t_max)
                else:
                    loss_ssdm, theta_t, theta_tm1 = self.ssdm.training_loss(
                        theta0, f_g, return_refined=False)
                    theta_parts = theta0
                f_p_list = self._parts_from_theta(U, theta_parts)
                loss_sdc = self._sdc_loss(U, theta_t, theta_tm1)
            else:
                theta = self.ssdm.ddim_sample(f_g, theta_init=theta0, num_steps=self.ddim_steps)
                f_p_list = self._parts_from_theta(U, theta)
        else:
            f_p = self.rap(x)
            f_p = f_p.view(f_p.size(0), f_p.size(1), -1)
            f_p_list = [f_p[:, :, i] for i in range(self.num_parts)]

        logits_p = []
        fs_p = []
        for i in range(self.num_parts):
            if self.use_ssdm:
                f_p_i = self.gap(f_p_list[i]).view(x.size(0), -1)
            else:
                f_p_i = f_p_list[i]
            f_p_i = getattr(self, 'bnneck' + str(i))(f_p_i)
            logits_p_i = getattr(self, 'classifier' + str(i))(f_p_i)
            logits_p.append(logits_p_i)
            fs_p.append(f_p_i)

        fs_p = torch.stack(fs_p, dim=-1)
        logits_p = torch.stack(logits_p, dim=-1)

        if self.use_ssdm and self.training:
            return f_g, fs_p, logits_g, logits_p, loss_ssdm, loss_sdc
        return f_g, fs_p, logits_g, logits_p

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def extract_all_features(self, x):
        x = self.base(x)

        f_g = self.gap(x)
        f_g = f_g.view(x.size(0), -1)
        f_g = self.bnneck(f_g)
        f_g = F.normalize(f_g)

        if self.use_ssdm:
            theta0 = self.patch_proposal(x)
            # For pseudo-label clustering, keep part extraction stable with theta0.
            # DDIM-refined theta can be noisy in early epochs and harm clustering quality.
            f_p_list = self._parts_from_theta(x, theta0)
            fs_p = []
            for i in range(self.num_parts):
                f_p_i = self.gap(f_p_list[i]).view(x.size(0), -1)
                f_p_i = getattr(self, 'bnneck' + str(i))(f_p_i)
                f_p_i = F.normalize(f_p_i)
                fs_p.append(f_p_i)
            fs_p = torch.stack(fs_p, dim=-1)
        else:
            f_p = self.rap(x)
            f_p = f_p.view(f_p.size(0), f_p.size(1), -1)
            fs_p = []
            for i in range(self.num_parts):
                f_p_i = f_p[:, :, i]
                f_p_i = getattr(self, 'bnneck' + str(i))(f_p_i)
                f_p_i = F.normalize(f_p_i)
                fs_p.append(f_p_i)
            fs_p = torch.stack(fs_p, dim=-1)

        return f_g, fs_p


def sdrl_resnet18(**kwargs):
    return ResNetPart(18, **kwargs)


def sdrl_resnet34(**kwargs):
    return ResNetPart(34, **kwargs)


def sdrl_resnet50(**kwargs):
    return ResNetPart(50, **kwargs)


def sdrl_resnet101(**kwargs):
    return ResNetPart(101, **kwargs)


def sdrl_resnet152(**kwargs):
    return ResNetPart(152, **kwargs)
