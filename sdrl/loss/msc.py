# coding: utf-8
"""
MSC (Multi-view Similarity Consistency) Loss.
Aligns distance distributions and instance similarity between source and DAM-enhanced views.
Only applied when batch has 32 samples (16 source + 16 enhanced).
"""
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F


def mmd_rbf_1d(x, y, sigma=None):
    """MMD with RBF kernel between two 1d tensors (e.g. distance distributions)."""
    if x.numel() == 0 or y.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    x = x.unsqueeze(1).float()
    y = y.unsqueeze(1).float()
    if sigma is None:
        xx = (x - x.t()).pow(2)
        yy = (y - y.t()).pow(2)
        xy = (x - y.t()).pow(2)
        sigma = (xx.sum() + yy.sum() + xy.sum()) / (xx.numel() + yy.numel() + xy.numel() + 1e-8)
        sigma = sigma.clamp(min=1e-4).sqrt()
    k_xx = torch.exp(-(x - x.t()).pow(2) / (2 * sigma ** 2)).mean()
    k_yy = torch.exp(-(y - y.t()).pow(2) / (2 * sigma ** 2)).mean()
    k_xy = torch.exp(-(x - y.t()).pow(2) / (2 * sigma ** 2)).mean()
    return k_xx + k_yy - 2 * k_xy


class MSCLoss(nn.Module):
    """
    Multi-view Similarity Consistency: distribution constraint (MMD) + instance constraint (in-batch).
    Expects batch of 32: emb_g[:16] = source, emb_g[16:32] = enhanced; same for pids, cids.
    """

    def __init__(self, tau=0.1, lam_dis=1.0, lam_ins=1.0, use_instance=True):
        super(MSCLoss, self).__init__()
        self.tau = tau
        self.lam_dis = lam_dis
        self.lam_ins = lam_ins
        # When use_instance=False, only distribution constraint (Eq.8-12); instance constraint delegated to MSCMemory (Eq.13-16)
        self.use_instance = use_instance

    def forward(self, emb_g, pids, cids):
        """
        emb_g: (2H, D), pids: (2H,), cids: (2H,)
        First half = source, last half = enhanced (DAM).
        """
        B = emb_g.size(0)
        if B < 4 or B % 2 != 0:
            return torch.tensor(0.0, device=emb_g.device)
        half = B // 2
        f_s = F.normalize(emb_g[:half], dim=1)
        f_e = F.normalize(emb_g[half:], dim=1)
        pid_s, cid_s = pids[:half], cids[:half]
        pid_e, cid_e = pids[half:], cids[half:]

        # ----- Distribution constraint -----
        # dse_cd: source vs enhanced (same identity, cross-camera by construction)
        dse_cd = (f_s - f_e).norm(dim=1)

        # dss_cd: source vs source, same pid, different cid
        dss_cd_list = []
        for i in range(half):
            for j in range(i + 1, half):
                if pid_s[i] == pid_s[j] and cid_s[i] != cid_s[j]:
                    dss_cd_list.append((f_s[i] - f_s[j]).norm().unsqueeze(0))
        dss_cd = torch.cat(dss_cd_list, dim=0) if dss_cd_list else dse_cd.new_zeros(1)

        # dse_id: source vs enhanced, same pid, same cid (enhanced's cid = background = cam)
        dse_id_list = []
        for i in range(half):
            for j in range(half):
                if pid_s[i] == pid_e[j] and cid_s[i] == cid_e[j]:
                    dse_id_list.append((f_s[i] - f_e[j]).norm().unsqueeze(0))
        dse_id = torch.cat(dse_id_list, dim=0) if dse_id_list else dse_cd.new_zeros(1)

        # dss_id: source vs source, same pid, same cid
        dss_id_list = []
        for i in range(half):
            for j in range(i + 1, half):
                if pid_s[i] == pid_s[j] and cid_s[i] == cid_s[j]:
                    dss_id_list.append((f_s[i] - f_s[j]).norm().unsqueeze(0))
        dss_id = torch.cat(dss_id_list, dim=0) if dss_id_list else dse_cd.new_zeros(1)

        l_dis_cd = mmd_rbf_1d(dse_cd, dss_cd)
        l_dis_id = mmd_rbf_1d(dse_id, dss_id)
        l_dis_msc = l_dis_cd + l_dis_id

        # Distribution constraint only (instance constraint handled by MSCMemory)
        if not self.use_instance:
            return self.lam_dis * l_dis_msc

        # ----- Instance constraint (in-batch) -----
        all_f = torch.cat([f_s, f_e], dim=0)
        all_pid = torch.cat([pid_s, pid_e], dim=0)
        all_cid = torch.cat([cid_s, cid_e], dim=0)
        sim = all_f @ all_f.t() / self.tau

        # Cross-camera: for each enhanced i, positives = same pid & same cid in the set w_c'
        l_cc = torch.tensor(0.0, device=emb_g.device)
        n_cc = 0
        for i in range(half):
            c_prime = cid_e[i]
            in_c_prime = (all_cid == c_prime)
            if in_c_prime.sum() == 0:
                continue
            sim_i = sim[half + i, in_c_prime]
            pos_in_block = (all_pid[in_c_prime] == pid_e[i])
            if pos_in_block.sum() == 0:
                continue
            labels = pos_in_block.float() / pos_in_block.sum().float().clamp(min=1e-8)
            l_cc = l_cc + (-(labels * F.log_softmax(sim_i, dim=0)).sum())
            n_cc += 1
        if n_cc > 0:
            l_cc = l_cc / n_cc

        # Intra-camera: for each source i, positives = same pid & same cid (exclude self)
        l_ic = torch.tensor(0.0, device=emb_g.device)
        n_ic = 0
        for i in range(half):
            c = cid_s[i]
            in_c = (all_cid == c)
            if in_c.sum() == 0:
                continue
            sim_i = sim[i, in_c]
            pos_in_block = (all_pid[in_c] == pid_s[i])
            # exclude self (index i is in the block)
            indices_c = torch.nonzero(in_c, as_tuple=False).squeeze(-1)
            self_loc = (indices_c == i).nonzero(as_tuple=True)[0]
            if len(self_loc) > 0:
                pos_in_block[self_loc[0]] = False
            if pos_in_block.sum() == 0:
                continue
            labels = pos_in_block.float() / pos_in_block.sum().float().clamp(min=1e-8)
            l_ic = l_ic + (-(labels * F.log_softmax(sim_i, dim=0)).sum())
            n_ic += 1
        if n_ic > 0:
            l_ic = l_ic / n_ic

        l_ins_msc = l_cc + l_ic if (n_cc > 0 or n_ic > 0) else emb_g.new_zeros(1)
        if n_cc == 0 and n_ic == 0:
            l_ins_msc = emb_g.new_zeros(1)

        return self.lam_dis * l_dis_msc + self.lam_ins * l_ins_msc


class MSCMemory(nn.Module):
    """Full-gallery memory bank for MSC instance constraint (paper Eq.13-16).

    Maintains source/enhanced banks Ws/We (proxy_s / proxy_e), EMA-updated (Eq.13);
    - Lcc (Eq.15): match enhanced fe via k-NN Ki,c' in camera c' **We**;
    - Lic (Eq.16): match source fs via k-NN Ki,c in camera c **Ws** (excluding self).
    """

    def __init__(self, num_samples, num_features, tau=0.1, momentum=0.8, k_nn=32):
        super(MSCMemory, self).__init__()
        self.num_samples = num_samples
        self.num_features = num_features
        self.tau = tau
        self.momentum = momentum
        self.k_nn = k_nn
        self.register_buffer('proxy_s', torch.zeros(num_samples, num_features))
        self.register_buffer('proxy_e', torch.zeros(num_samples, num_features))
        self.register_buffer('pids', torch.zeros(num_samples).long())
        self.register_buffer('cids', torch.zeros(num_samples).long())

    @torch.no_grad()
    def init_bank(self, features, pids, cids):
        f = F.normalize(features.float(), dim=1)
        self.proxy_s.copy_(f)
        self.proxy_e.copy_(f)
        self.pids.copy_(pids.long())
        self.cids.copy_(cids.long())

    @torch.no_grad()
    def update(self, idx_s, f_s, idx_e, f_e):
        eps = self.momentum
        fs = F.normalize(f_s.detach().float(), dim=1)
        fe = F.normalize(f_e.detach().float(), dim=1)
        for k in range(idx_s.size(0)):
            i = int(idx_s[k].item())
            self.proxy_s[i] = F.normalize(eps * self.proxy_s[i] + (1 - eps) * fs[k], dim=0)
        for k in range(idx_e.size(0)):
            i = int(idx_e[k].item())
            self.proxy_e[i] = F.normalize(eps * self.proxy_e[i] + (1 - eps) * fe[k], dim=0)

    def _infonce_knn(self, feat, cam, pid, bank='s', self_idx=None):
        """k-NN within same-camera bank, then soft InfoNCE over same pid (Eq.15/16)."""
        proxy = self.proxy_e if bank == 'e' else self.proxy_s
        in_cam = (self.cids == cam)
        if in_cam.sum() == 0:
            return None
        cand_idx = torch.nonzero(in_cam, as_tuple=False).squeeze(-1)
        pool = proxy[cand_idx]
        sim_all = pool @ feat
        k = min(self.k_nn, pool.size(0))
        if k <= 0:
            return None
        topk = sim_all.topk(k, largest=True).indices
        pool_k = pool[topk]
        sim = (pool_k @ feat) / self.tau
        cand_global = cand_idx[topk]
        pos = (self.pids[cand_global] == pid)
        if self_idx is not None:
            pos = pos & (cand_global != self_idx)
        if pos.sum() == 0:
            return None
        labels = pos.float() / pos.sum().float().clamp(min=1e-8)
        return -(labels * F.log_softmax(sim, dim=0)).sum()

    def forward(self, f_s, idx_s, cid_s, pid_s, f_e, idx_e, cid_e, pid_e):
        fs = F.normalize(f_s.float(), dim=1)
        fe = F.normalize(f_e.float(), dim=1)
        device = f_s.device

        l_cc, n_cc = torch.zeros((), device=device), 0
        for k in range(fe.size(0)):
            l = self._infonce_knn(fe[k], int(cid_e[k].item()), int(pid_e[k].item()), bank='e')
            if l is not None:
                l_cc = l_cc + l
                n_cc += 1
        if n_cc > 0:
            l_cc = l_cc / n_cc

        l_ic, n_ic = torch.zeros((), device=device), 0
        for k in range(fs.size(0)):
            l = self._infonce_knn(fs[k], int(cid_s[k].item()), int(pid_s[k].item()),
                                  bank='s', self_idx=int(idx_s[k].item()))
            if l is not None:
                l_ic = l_ic + l
                n_ic += 1
        if n_ic > 0:
            l_ic = l_ic / n_ic

        return l_cc + l_ic
