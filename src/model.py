# src/model.py
"""Light-weight controller architectures (ABCD & baselines)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _sin_embed(t: torch.Tensor, dim: int) -> torch.Tensor:  # noqa: D401
    half = dim // 2
    freq = torch.exp(torch.linspace(0, -math.log(10000), half, device=t.device))
    emb = t.float().unsqueeze(-1) * freq
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class MLP(nn.Module):  # noqa: D401
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 32, layers: int = 2):
        super().__init__()
        net = []
        d = in_dim
        for _ in range(layers - 1):
            net += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            d = hidden
        net.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# ABCD – proposed method
# -----------------------------------------------------------------------------
class WeightGen(nn.Module):
    def __init__(self, ctx_dim: int, hidden: int, max_t: int):
        super().__init__()
        self.max_t = max_t
        self.time_dim = hidden
        self.mlp = MLP(ctx_dim + hidden, 1, hidden, layers=2)

    def forward(self, ctx):  # ctx (B, C)
        bsz = ctx.size(0)
        t = torch.arange(self.max_t, device=ctx.device).unsqueeze(0).repeat(bsz, 1)
        t_emb = _sin_embed(t, self.time_dim)  # (B, T, D)
        ctx_e = ctx.unsqueeze(1).expand(-1, self.max_t, -1)
        w = self.mlp(torch.cat([t_emb, ctx_e], dim=-1)).squeeze(-1)  # (B, T)
        return F.softplus(w)


class ABCDController(nn.Module):  # noqa: D401
    def __init__(self, cfg):
        super().__init__()
        ctx_dim = 3
        hidden = int(cfg.model.weight_generator.hidden_size)
        max_t = int(cfg.training.max_timesteps)
        self.w_gen = WeightGen(ctx_dim, hidden, max_t)
        self.policy = MLP(ctx_dim + 3, 1, hidden)
        self.c_pred = MLP(ctx_dim + 3, 1, hidden)
        self.cfg = cfg

    def compute_losses(self, imgs, budgets_ms, quality_targets):  # noqa: D401
        bsz = imgs.size(0)
        logB = torch.log1p(budgets_ms) / 10.0
        logF = torch.zeros_like(logB)
        rho = quality_targets.float()
        ctx = torch.stack([logB, logF, rho], dim=1)
        weights = self.w_gen(ctx)
        w_mean = weights.mean(dim=1)
        mse_img = imgs.pow(2).mean(dim=(1, 2, 3))
        h_final = torch.cat([torch.zeros(bsz, 1, device=imgs.device), mse_img.unsqueeze(1), torch.ones(bsz, 1, device=imgs.device), ctx], dim=1)
        stop_p = torch.sigmoid(self.policy(h_final)).squeeze(1)
        c_pred = F.softplus(self.c_pred(h_final)).squeeze(1)  # seconds

        l_eps = F.mse_loss(w_mean, mse_img.detach())
        kl = F.kl_div((weights / weights.sum(dim=1, keepdim=True)).log(), torch.full_like(weights, 1 / weights.size(1)), reduction="batchmean")
        l_budget = F.relu(c_pred - budgets_ms / 1000.0).pow(2).mean()
        q_proxy = 1.0 / (1.0 + mse_img)
        l_quality = ((stop_p - q_proxy.detach()).pow(2)).mean()

        lamb = self.cfg.training.lambdas
        total = (
            lamb.lambda_eps * l_eps
            + lamb.lambda_1 * kl
            + lamb.lambda_2 * l_budget
            + lamb.lambda_3 * l_quality
        )
        success = (c_pred <= budgets_ms / 1000.0).float().mean()
        metrics = {
            "L_eps": l_eps.detach(),
            "KL_align": kl.detach(),
            "L_budget": l_budget.detach(),
            "L_quality": l_quality.detach(),
            "QUB": q_proxy.mean().detach(),
            "success": success.detach(),
        }
        return {"total": total}, metrics


# -----------------------------------------------------------------------------
# BuCo-MWG baseline (weights only, no stop policy)
# -----------------------------------------------------------------------------
class BuCoController(nn.Module):  # noqa: D401
    def __init__(self, cfg):
        super().__init__()
        ctx_dim = 3
        hidden = int(cfg.model.weight_generator.hidden_size)
        max_t = int(cfg.training.max_timesteps)
        self.w_gen = WeightGen(ctx_dim, hidden, max_t)
        self.cfg = cfg

    def compute_losses(self, imgs, budgets_ms, quality_targets):  # noqa: D401
        logB = torch.log1p(budgets_ms) / 10.0
        ctx = torch.stack([logB, torch.zeros_like(logB), quality_targets], dim=1)
        weights = self.w_gen(ctx)
        w_mean = weights.mean(dim=1)
        mse_img = imgs.pow(2).mean(dim=(1, 2, 3))
        l_eps = F.mse_loss(w_mean, mse_img.detach())
        kl = F.kl_div((weights / weights.sum(dim=1, keepdim=True)).log(), torch.full_like(weights, 1 / weights.size(1)), reduction="batchmean")
        total = l_eps + 0.5 * kl
        # Fake cost = steps * 1ms
        cost_pred = weights.size(1) * 0.001
        success = (cost_pred <= budgets_ms / 1000.0).float().mean()
        metrics = {
            "L_eps": l_eps.detach(),
            "KL_align": kl.detach(),
            "QUB": (1 / (1 + mse_img)).mean().detach(),
            "success": success.detach(),
        }
        return {"total": total}, metrics


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def build_model(cfg):  # noqa: D401
    method = str(cfg.method).lower()
    if any(tok in method for tok in ["abcd", "proposed"]):
        return ABCDController(cfg)
    elif any(tok in method for tok in ["buco", "baseline", "comparative"]):
        return BuCoController(cfg)
    raise ValueError(f"Unknown method '{cfg.method}'.")
