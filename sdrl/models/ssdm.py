# coding: utf-8
"""
SSDM: Semantic Spatial Diffusion Model.
STN localization network + modern diffusion (cosine schedule, DDIM) to refine
spatial transformer parameters for semantic part patches.
"""
from __future__ import absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

THETA_DIM = 3 * 2 * 3  # 3 parts, 2x3 affine per part


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule (Nichol & Dhariwal)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.999)


class PatchGenerator(nn.Module):
    """STN localization network: predicts theta from feature map U (optionally masked)."""
    def __init__(self):
        super(PatchGenerator, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, THETA_DIM),
        )
        # init for 3 horizontal stripes (1/3 each)
        init_bias = [1, 1/3, 1/3] * 6
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(init_bias, dtype=torch.float32))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        """
        x: [B, 2048, H, W]. Optional mask: [B, 1, Hm, Wm] (auto downsampled to x spatial size), then U = x * mask.
        Returns theta: [B, 3, 2, 3].
        """
        if mask is not None:
            if mask.shape[-2:] != x.shape[-2:]:
                mask = F.interpolate(mask, size=x.shape[-2:], mode='nearest')
            x = x * mask
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        return theta.view(-1, 3, 2, 3)


class SSDM(nn.Module):
    """
    Semantic Spatial Diffusion Model: refines theta via diffusion.
    - Training: predict noise epsilon, loss = MSE(epsilon_pred, epsilon).
    - Inference: DDIM with few steps (e.g. 50), start from STN theta_init.
    """
    def __init__(self, num_steps=1000, ddim_steps=50, time_emb_dim=128):
        super(SSDM, self).__init__()
        self.num_steps = num_steps
        self.ddim_steps = ddim_steps
        self.time_emb_dim = time_emb_dim
        # Schedules
        betas = cosine_beta_schedule(num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sigma_t', torch.sqrt(betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        # DDIM: use a subset of timesteps
        self.ddim_timesteps = torch.linspace(num_steps - 1, 0, ddim_steps).long()
        # Denoiser: (noisy_theta_flat + global_feat + t_emb) -> noise
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.denoiser = nn.Sequential(
            nn.Linear(2048 + THETA_DIM + time_emb_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, THETA_DIM),
        )
        for m in self.denoiser.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def time_embed(self, t):
        """Sinusoidal time embedding."""
        half = self.time_emb_dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.float()
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: x_t = sqrt(alpha_cumprod_t) * x0 + sqrt(1-alpha_cumprod_t) * eps."""
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device, dtype=x_start.dtype)
        b = x_start.shape[0]
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(b, *([1] * (x_start.dim() - 1)))
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(b, *([1] * (x_start.dim() - 1)))
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def predict_x0_from_noise(self, x_t, t, eps):
        """x0 = (x_t - sqrt(1-alpha_cumprod_t)*eps) / sqrt(alpha_cumprod_t)."""
        b = x_t.shape[0]
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(b, *([1] * (x_t.dim() - 1)))
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(b, *([1] * (x_t.dim() - 1)))
        return (x_t - sqrt_one_minus * eps) / sqrt_alpha.clamp(min=1e-8)

    def forward_denoise(self, x_t, t, global_feat):
        """Predict noise epsilon given x_t and t. global_feat: [B, 2048]."""
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        if x_t.dim() == 4:
            x_flat = x_t.view(x_t.size(0), -1)
        else:
            x_flat = x_t
        inp = torch.cat([global_feat, x_flat, t_emb], dim=1)
        return self.denoiser(inp)

    def training_loss(self, theta0, global_feat, return_refined=False, refine_t_max=100):
        """
        Sample t (from 1 to T-1 so t-1 exists), add noise, predict noise.
        Return: loss_ssdm (MSE), theta_t, theta_{t-1} for SDC; optionally theta_refined.

        When return_refined=True: additionally use smaller t (<= refine_t_max) for single-step x0 prediction for part sampling.
        At large t, Θ_t is near pure noise; using it directly for part classification pollutes global CE via L_ces and can crash training.
        theta0: [B, 3, 2, 3], global_feat: [B, 2048].
        """
        b = theta0.size(0)
        theta0_flat = theta0.view(b, -1)
        t = torch.randint(1, self.num_steps, (b,), device=theta0.device).long()
        noise = torch.randn_like(theta0_flat, device=theta0.device, dtype=theta0.dtype)
        theta_t_flat = self.q_sample(theta0_flat, t, noise)
        eps_pred = self.forward_denoise(theta_t_flat, t, global_feat)
        loss_ssdm = F.mse_loss(eps_pred, noise)
        theta_t = theta_t_flat.view(b, 3, 2, 3)
        t_m1 = (t - 1).clamp(min=0)
        theta_tm1_flat = self.q_sample(theta0_flat, t_m1, noise)
        theta_tm1 = theta_tm1_flat.view(b, 3, 2, 3)
        if return_refined:
            t_cap = min(int(refine_t_max), self.num_steps - 1)
            t_r = torch.randint(1, t_cap + 1, (b,), device=theta0.device).long()
            noise_r = torch.randn_like(theta0_flat, device=theta0.device, dtype=theta0.dtype)
            theta_tr_flat = self.q_sample(theta0_flat, t_r, noise_r)
            eps_r = self.forward_denoise(theta_tr_flat, t_r, global_feat)
            theta_refined = self.predict_x0_from_noise(
                theta_tr_flat, t_r, eps_r).view(b, 3, 2, 3)
            return loss_ssdm, theta_t, theta_tm1, theta_refined
        return loss_ssdm, theta_t, theta_tm1

    @torch.no_grad()
    def ddim_sample(self, global_feat, theta_init=None, num_steps=None):
        """
        DDIM sampling. Start from theta_init (from STN) or random.
        num_steps: default self.ddim_steps (e.g. 50).
        Returns theta: [B, 3, 2, 3].
        """
        num_steps = num_steps or self.ddim_steps
        b = global_feat.size(0)
        device = global_feat.device
        # Timestep sequence for DDIM (e.g. [999, 979, ..., 0])
        step_indices = torch.linspace(self.num_steps - 1, 0, num_steps + 1).long()
        if theta_init is not None:
            x = theta_init.view(b, -1)
        else:
            x = torch.randn(b, THETA_DIM, device=device, dtype=global_feat.dtype)
        for i in range(num_steps):
            t_cur = step_indices[i]
            t_batch = torch.full((b,), t_cur, device=device, dtype=torch.long)
            eps = self.forward_denoise(x, t_batch, global_feat)
            alpha = self.alphas_cumprod[t_cur]
            if i < num_steps:
                t_next = step_indices[i + 1]
                alpha_next = self.alphas_cumprod[t_next]
                # DDIM: x_{t-1} = sqrt(alpha_next)*x0_pred + sqrt(1-alpha_next)*eps
                x0_pred = self.predict_x0_from_noise(x, t_batch, eps)
                x = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1.0 - alpha_next) * eps
            else:
                x0_pred = self.predict_x0_from_noise(x, t_batch, eps)
                x = x0_pred
        return x.view(b, 3, 2, 3)
