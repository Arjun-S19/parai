import torch.nn as nn
from models.encoders import PannsEncoder
from models.head import MLPHead

class DrumClassifier(nn.Module):
    def __init__(self, encoder: PannsEncoder, num_classes: int):
        """
        Combine a frozen encoder with a trainable classifier head
        """
        super().__init__()

        # swappable encoder
        self.encoder = encoder

        # head adapts automatically to encoder by reading encoder.embedding_dim
        self.head = MLPHead(input_dim = encoder.embedding_dim, num_classes = num_classes)

    def forward(self, waveforms):
        """
        Run encoder + head to produce class logits
        """
        # convert waveforms -> embeddings -> class logits
        emb = self.encoder(waveforms)
        logits = self.head(emb)
        return logits
