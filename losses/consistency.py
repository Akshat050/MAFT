import torch
import torch.nn.functional as F
from typing import List


def symmetric_kl_multi(logits_list: List[torch.Tensor], temperature: float = 2.0) -> torch.Tensor:
    """
    Compute symmetric KL divergence between multiple modality predictions.
    
    Args:
        logits_list: List of logits tensors [B, num_classes] from different modalities
        temperature: Temperature for softening distributions
    
    Returns:
        Average symmetric KL divergence loss
    """
    if len(logits_list) < 2:
        return torch.tensor(0.0, device=logits_list[0].device if logits_list else None)
    
    # Soften all distributions
    probs_list = [F.softmax(logits / temperature, dim=-1) for logits in logits_list]
    
    total_loss = 0.0
    count = 0
    
    # Compute pairwise symmetric KL
    for i in range(len(probs_list)):
        for j in range(i + 1, len(probs_list)):
            p = probs_list[i]
            q = probs_list[j]
            
            # KL(P||Q) + KL(Q||P)
            kl_pq = F.kl_div(q.log(), p, reduction='batchmean')
            kl_qp = F.kl_div(p.log(), q, reduction='batchmean')
            
            total_loss += (kl_pq + kl_qp) / 2.0
            count += 1
    
    return total_loss / count if count > 0 else torch.tensor(0.0, device=logits_list[0].device)
