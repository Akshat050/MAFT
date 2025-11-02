import torch
import torch.nn.functional as F
from typing import List


def symmetric_kl_multi(logits_list: List[torch.Tensor], temperature: float = 2.0) -> torch.Tensor:
    """
    Symmetric KL divergence between multiple modality logits.
    
    Args:
        logits_list: list of [B, num_classes] tensors (one per modality)
        temperature: softmax temperature for smoothing
    
    Returns:
        scalar loss (average symmetric KL across pairs)
    """
    if not logits_list or len(logits_list) < 2:
        if logits_list:
            return torch.tensor(0.0, device=logits_list[0].device)
        return torch.tensor(0.0)
    
    # soften distributions
    probs = [F.softmax(l / temperature, dim=-1).clamp_min(1e-8) for l in logits_list]
    log_probs = [p.log() for p in probs]
    
    total, count = 0.0, 0
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            # KL(P||Q) + KL(Q||P)
            kl_pq = F.kl_div(log_probs[i], probs[j], reduction="batchmean")
            kl_qp = F.kl_div(log_probs[j], probs[i], reduction="batchmean")
            total += 0.5 * (kl_pq + kl_qp)
            count += 1
    
    return total / max(count, 1)
