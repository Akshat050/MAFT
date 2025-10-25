import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from losses.consistency import symmetric_kl_multi


def schedule_dropout(epoch, total_epochs, p_min=0.05, p_max=0.35):
    """Schedule modality dropout rate from p_min to p_max over training"""
    alpha = min(1.0, epoch / max(1, int(0.6 * total_epochs)))
    return p_min + (p_max - p_min) * alpha


def compute_loss(outputs, batch, lambdas):
    """
    Compute multi-task loss with classification, regression, consistency, and reconstruction.
    
    Args:
        outputs: Model output dictionary
        batch: Batch dictionary with targets
        lambdas: Loss weight dictionary with keys: reg, cons, rec
    
    Returns:
        total_loss: Weighted sum of all losses
        parts: Dictionary of individual loss components
    """
    # Main task losses
    cls_loss = F.cross_entropy(outputs["logits"], batch["classification_targets"])
    
    reg_loss = (
        F.l1_loss(outputs["reg"].squeeze(-1), batch["regression_targets"])
        if "regression_targets" in batch
        else torch.tensor(0.0, device=cls_loss.device)
    )
    
    # Consistency loss (only if at least two modalities present)
    logits_list = []
    if outputs["logits_text"].numel() > 0:
        logits_list.append(outputs["logits_text"])
    if outputs["logits_audio"].numel() > 0:
        logits_list.append(outputs["logits_audio"])
    if outputs["logits_visual"].numel() > 0:
        logits_list.append(outputs["logits_visual"])
    
    cons_loss = symmetric_kl_multi(logits_list, temperature=2.0)
    
    # Reconstruction loss: only for samples where that modality was dropped
    rec_loss = torch.tensor(0.0, device=cls_loss.device)
    z = outputs["rec_audio"]  # using Z directly; simple linear project here
    
    # Create projection layers (in practice, these should be model parameters)
    proj_a = nn.Linear(z.size(-1), outputs["targets_audio_sum"].size(-1)).to(z.device)
    proj_v = nn.Linear(z.size(-1), outputs["targets_visual_sum"].size(-1)).to(z.device)
    
    drops = outputs["drops"]
    
    if drops["drop_a"] is not None and drops["drop_a"].any():
        rec_a = proj_a(z[drops["drop_a"]])
        tgt_a = outputs["targets_audio_sum"][drops["drop_a"]]
        rec_loss = rec_loss + F.l1_loss(rec_a, tgt_a)
    
    if drops["drop_v"] is not None and drops["drop_v"].any():
        rec_v = proj_v(z[drops["drop_v"]])
        tgt_v = outputs["targets_visual_sum"][drops["drop_v"]]
        rec_loss = rec_loss + F.l1_loss(rec_v, tgt_v)
    
    # Total weighted loss
    total = (
        cls_loss
        + lambdas["reg"] * reg_loss
        + lambdas["cons"] * cons_loss
        + lambdas["rec"] * rec_loss
    )
    
    return total, {
        "classification_loss": cls_loss,
        "regression_loss": reg_loss,
        "consistency_loss": cons_loss,
        "reconstruction_loss": rec_loss,
    }


def train_one_epoch(
    model, loader, optimizer, scaler, device, epoch, total_epochs, lambdas, grad_clip=1.0
):
    """
    Train for one epoch with mixed precision and gradient clipping.
    
    Args:
        model: MAFT model
        loader: Training data loader
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        device: Device to train on
        epoch: Current epoch number
        total_epochs: Total number of epochs
        lambdas: Loss weight dictionary
        grad_clip: Gradient clipping threshold
    
    Returns:
        Dictionary of average losses for the epoch
    """
    model.train()
    model.moddrop.p = schedule_dropout(epoch, total_epochs)
    
    logs = dict(cls=0.0, reg=0.0, cons=0.0, rec=0.0, tot=0.0, n=0)
    
    for batch in loader:
        # Move batch to device
        for k in batch:
            batch[k] = batch[k].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(batch)
            loss, parts = compute_loss(outputs, batch, lambdas)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        logs["tot"] += loss.item()
        logs["cls"] += parts["classification_loss"].item()
        logs["reg"] += parts["regression_loss"].item()
        logs["cons"] += parts["consistency_loss"].item()
        logs["rec"] += parts["reconstruction_loss"].item()
        logs["n"] += 1
    
    # Average losses
    for k in list(logs.keys()):
        if k != "n":
            logs[k] = logs[k] / max(logs["n"], 1)
    
    return logs
