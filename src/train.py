# src/train.py
"""Single-run training entry-point (executed via ``src.main``).

Key features (spec-compliant)
-----------------------------
1. Hydra configuration (@hydra.main, ``config_path='../config'``).
2. Trial vs Full mode handled *before* any heavy-weight work starts.
3. Handles Optuna hyper-parameter optimisation (offline, no WandB logging).
4. Logs *every* batch/epoch metric to WandB when enabled.
5. Writes required keys (incl. "Quality-Under-Budget (QUB)") to
   ``wandb.summary`` and prints the run URL.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import optuna
import torch
from omegaconf import OmegaConf
from torch import nn, optim
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Make repository root importable *after* Hydra changed the CWD.
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import model as models  # noqa: E402  pylint: disable=wrong-import-position
from src import preprocess  # noqa: E402  pylint: disable=wrong-import-position

import wandb  # noqa: E402  pylint: disable=wrong-import-position

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:  # noqa: D401
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Single training run (optionally with HP overrides from Optuna)
# -----------------------------------------------------------------------------

def _train_once(cfg, hp_override: Dict[str, Any] | None = None) -> Dict[str, float]:
    """Train the controller once.

    Parameters
    ----------
    cfg : DictConfig
        *Fully resolved* configuration tree.
    hp_override : dict, optional
        Mapping of dotted-keys → new value (Optuna suggestions).
    """
    if hp_override:
        for dotted_key, value in hp_override.items():
            OmegaConf.update(cfg, dotted_key, value, merge=False)

    _set_seeds(int(cfg.additional_settings.seed))
    device = _device()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader = preprocess.get_data_loaders(cfg, mode=str(cfg.mode))

    # ------------------------------------------------------------------
    # Model + optimisers
    # ------------------------------------------------------------------
    controller: nn.Module = models.build_model(cfg).to(device)

    opt_cls = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
    }[str(cfg.training.optimizer).lower()]

    optimizer = opt_cls(
        controller.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )

    scheduler = None
    if str(cfg.training.scheduler.type).lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.training.epochs))

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.training.mixed_precision))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    best_val_qub = -float("inf")
    best_val_loss = float("inf")

    for epoch in range(1, int(cfg.training.epochs) + 1):
        controller.train()
        for batch_idx, imgs in enumerate(tqdm(train_loader, desc=f"[train] epoch={epoch}"), 1):
            imgs = imgs.to(device, non_blocking=True)
            budgets_ms = torch.tensor(
                random.choices(list(cfg.training.budgets_ms), k=imgs.size(0)),
                dtype=torch.float32,
                device=device,
            )
            quality_targets = torch.tensor(
                random.choices(list(cfg.training.quality_targets), k=imgs.size(0)),
                dtype=torch.float32,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.training.mixed_precision)):
                losses, metrics = controller.compute_losses(imgs, budgets_ms, quality_targets)
                loss = losses["total"]
            scaler.scale(loss).backward()
            if cfg.training.gradient_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(controller.parameters(), float(cfg.training.gradient_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()

            # ---------------- WandB logging ----------------
            global_step += 1
            if wandb.run is not None:  # WandB enabled
                log_dict = {f"train_{k}": float(v.item() if torch.is_tensor(v) else v) for k, v in metrics.items()}
                log_dict.update({"train_loss": float(loss.item()), "epoch": epoch})
                wandb.log(log_dict, step=global_step)

            # Trial-mode: 2 batches max
            if cfg.mode == "trial" and batch_idx >= 2:
                break

        # --------------- validation every epoch ---------------
        controller.eval()
        val_loss_acc = 0.0
        val_qub_acc = 0.0
        val_success_acc = 0.0
        with torch.no_grad():
            for v_batch_idx, imgs in enumerate(val_loader):
                imgs = imgs.to(device, non_blocking=True)
                budgets_ms = torch.tensor(
                    random.choices(list(cfg.training.budgets_ms), k=imgs.size(0)),
                    dtype=torch.float32,
                    device=device,
                )
                quality_targets = torch.tensor(
                    random.choices(list(cfg.training.quality_targets), k=imgs.size(0)),
                    dtype=torch.float32,
                    device=device,
                )
                losses, metrics = controller.compute_losses(imgs, budgets_ms, quality_targets)
                val_loss_acc += float(losses["total"].item())
                val_qub_acc += float(metrics["QUB"].item())
                val_success_acc += float(metrics["success"].item())
                if cfg.mode == "trial":
                    break  # first val batch only
        denom = 1 if cfg.mode == "trial" else len(val_loader)
        val_loss = val_loss_acc / denom
        val_qub = val_qub_acc / denom
        val_success = val_success_acc / denom

        best_val_qub = max(best_val_qub, val_qub)
        best_val_loss = min(best_val_loss, val_loss)

        if wandb.run is not None:
            wandb.log(
                {
                    "val_loss": val_loss,
                    "val_QUB": val_qub,
                    "val_success": val_success,
                    "best_val_QUB": best_val_qub,
                    "epoch": epoch,
                },
                step=global_step,
            )

        if cfg.mode == "trial":
            break  # single epoch only

    # ------------------------------------------------------------------
    # Final metrics + WandB summary
    # ------------------------------------------------------------------
    final = {
        "best_val_QUB": best_val_qub,
        "best_val_loss": best_val_loss,
        "Quality-Under-Budget (QUB)": best_val_qub,
    }
    if wandb.run is not None:
        for k, v in final.items():
            wandb.summary[k] = v
        print(f"[WandB] Run URL: {wandb.run.get_url()}")

    return final


# -----------------------------------------------------------------------------
# Optuna objective builder
# -----------------------------------------------------------------------------

def _optuna_objective(base_cfg):  # noqa: D401
    search_space = base_cfg.optuna.get("search_space", {})

    def objective(trial: optuna.Trial):  # noqa: D401
        hp_override: Dict[str, Any] = {}
        for key, space in search_space.items():
            t = space["type"]
            if t == "loguniform":
                hp_override[key] = trial.suggest_float(key, space["low"], space["high"], log=True)
            elif t == "uniform":
                hp_override[key] = trial.suggest_float(key, space["low"], space["high"], log=False)
            elif t == "categorical":
                hp_override[key] = trial.suggest_categorical(key, space["choices"])
            else:
                raise ValueError(f"Unsupported Optuna space type '{t}'.")
        metrics = _train_once(base_cfg, hp_override)
        # Direction is maximise unless explicitly minimise.
        score = metrics["best_val_QUB"]
        return -score if str(base_cfg.optuna.direction).lower() == "minimize" else score

    return objective


# -----------------------------------------------------------------------------
# Hydra entry-point
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):  # noqa: D401
    # ------------------------------------------------------------------
    # Merge run-specific YAML
    # ------------------------------------------------------------------
    run_yaml = ROOT / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_yaml.exists():
        raise FileNotFoundError(run_yaml)
    # Temporarily disable struct mode to allow new keys during merge
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(run_yaml))
    OmegaConf.set_struct(cfg, True)

    # ------------------------------------------------------------------
    # Trial / Full mode overrides
    # ------------------------------------------------------------------
    if cfg.mode not in {"trial", "full"}:
        raise ValueError("mode must be 'trial' or 'full'")

    if cfg.mode == "trial":
        OmegaConf.update(cfg, "wandb.mode", "disabled", merge=False)
        OmegaConf.update(cfg, "training.epochs", 1, merge=False)
        OmegaConf.update(cfg, "optuna.n_trials", 0, merge=False)
    else:
        OmegaConf.update(cfg, "wandb.mode", "online", merge=False)

    # ------------------------------------------------------------------
    # WandB initialisation (skip entirely when disabled)
    # ------------------------------------------------------------------
    if cfg.wandb.mode != "disabled":
        run_id = cfg.get("run_id", cfg.run)
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=run_id,
            name=run_id,
            resume="allow",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )

    # ------------------------------------------------------------------
    # Hyper-parameter optimisation (offline)
    # ------------------------------------------------------------------
    if int(cfg.optuna.n_trials) > 0:
        study = optuna.create_study(direction=cfg.optuna.direction, study_name=f"{cfg.run}-optuna")
        study.optimize(_optuna_objective(cfg), n_trials=int(cfg.optuna.n_trials))
        best_params = study.best_params
        _train_once(cfg, hp_override=best_params)  # final run with WandB
    else:
        _train_once(cfg)  # plain training


if __name__ == "__main__":
    main()
