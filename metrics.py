import torch
from collections import defaultdict

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute overall classification accuracy from logits
    """

    # simple overall accuracy for quick sanity checks
    preds = logits.argmax(dim = 1)
    return (preds == targets).float().mean().item()

def per_class_accuracy(logits: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """
    Compute per-class accuracy as a dict of class index -> score
    """
    
    # reveals which classes are failing even if overall accuracy looks good
    preds = logits.argmax(dim = 1)

    correct = defaultdict(int)
    total = defaultdict(int)

    for p, t in zip(preds, targets):
        total[int(t)] += 1
        if p == t:
            correct[int(t)] += 1

    return {
        c: (correct[c] / total[c]) if total[c] > 0 else 0.0
        for c in range(num_classes)
    }

def confusion_matrix(logits: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """
    Compute confusion matrix (rows = true class, cols = predicted class)
    """
    preds = logits.argmax(dim = 1)

    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for t, p in zip(targets, preds):
        ti = int(t)
        pi = int(p)
        cm[ti][pi] += 1

    return cm