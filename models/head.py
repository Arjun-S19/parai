import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        """
        Build a small MLP classifier for encoder embeddings
        """
        super().__init__()

        # head learns drum classes on top of a pretrained embedding space
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        Map embeddings to class logits
        """
        # x is [b, d] embeddings; output is [b, num_classes] logits for cross entropy
        return self.net(x)