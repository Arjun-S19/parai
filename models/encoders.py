import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union
import torch
import torch.nn as nn
from audioset_tagging_cnn.pytorch.models import Cnn6, Cnn10, Cnn14

EncoderName = Literal["cnn6", "cnn10", "cnn14"]

@dataclass(frozen = True)
class PannsParams:
    sample_rate: int = 32000
    window_size: int = 1024
    hop_size: int = 320
    mel_bins: int = 64
    fmin: int = 50
    fmax: int = 14000
    classes_num: int = 527

class PannsEncoder(nn.Module):
    def __init__(self, base_model: nn.Module, embedding_dim: int):
        """
        Wrap a PANNs model and expose a stable embedding interface
        """
        super().__init__()
        self.base_model = base_model
        # embedding_dim is used to auto-configure the classifier head
        self.embedding_dim = embedding_dim

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Convert waveforms into [b, d] embeddings
        """
        # accept [b, 1, t] from dataset or [b, t]
        # PANNs models expect [b, t], so strip the channel dim if present
        if waveforms.dim() == 3:
            if waveforms.size(1) != 1:
                raise ValueError(f"Expected [b, 1, t], got {tuple(waveforms.shape)}")
            x = waveforms[:, 0, :]
        elif waveforms.dim() == 2:
            x = waveforms
        else:
            raise ValueError(f"Expected [b, 1, t] or [b, t], got {tuple(waveforms.shape)}")

        # PANNs forward returns a dict; learned clip embedding is the only thing that matters
        out = self.base_model(x)

        # error if repo interface changes or wired the wrong output
        if not isinstance(out, dict) or "embedding" not in out:
            raise RuntimeError("PANNs output missing 'embedding'")

        emb = out["embedding"]

        # enforce the contract that encoders return [b, d]
        if emb.dim() != 2:
            raise RuntimeError(f"Expected embedding [b, d], got {tuple(emb.shape)}")

        return emb

def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Normalize checkpoint formats into a raw state_dict
    """
    # PANNs checkpoints commonly store weights under the 'model' key
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]

    # some checkpoints are already a raw state_dict (mapping param_name -> tensor)
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt

    raise ValueError("Unsupported checkpoint format (expected state_dict or {'model': state_dict})")

def make_panns_encoder(name: EncoderName,
                       checkpoint_path: Union[str, pathlib.Path],
                       device: Union[str, torch.device] = "cpu",
                       freeze: bool = True,
                       params: Optional[PannsParams] = None,
                       ) -> PannsEncoder:
    """
    Create a PANNs encoder with pretrained weights loaded
    """
    # keep PANNs feature extraction settings in one place for cnn6/cnn10/cnn14
    p = params or PannsParams()

    # embedding_dim is the feature size that classifier head must accept
    if name == "cnn6":
        base = Cnn6(
            sample_rate = p.sample_rate,
            window_size = p.window_size,
            hop_size = p.hop_size,
            mel_bins = p.mel_bins,
            fmin = p.fmin,
            fmax = p.fmax,
            classes_num = p.classes_num,
        )
        embedding_dim = 512

    elif name == "cnn10":
        base = Cnn10(
            sample_rate = p.sample_rate,
            window_size = p.window_size,
            hop_size = p.hop_size,
            mel_bins = p.mel_bins,
            fmin = p.fmin,
            fmax = p.fmax,
            classes_num = p.classes_num,
        )
        embedding_dim = 512

    elif name == "cnn14":
        base = Cnn14(
            sample_rate = p.sample_rate,
            window_size = p.window_size,
            hop_size = p.hop_size,
            mel_bins = p.mel_bins,
            fmin = p.fmin,
            fmax = p.fmax,
            classes_num = p.classes_num,
        )
        embedding_dim = 2048

    else:
        raise ValueError(f"unsupported encoder name: {name}")

    # load pretrained weights so the encoder produces useful embeddings immediately
    ckpt = torch.load(str(checkpoint_path), map_location = "cpu")
    state_dict = _extract_state_dict(ckpt)
    base.load_state_dict(state_dict, strict = True)

    # move encoder to the selected device (cpu for batch indexing/cuda for training speed)
    base.to(device)

    # freezing is the default first training phase: train only the head, not the encoder
    if freeze:
        base.eval()
        for param in base.parameters():
            param.requires_grad = False

    # wrap the PANNs model behind a stable encoder api used everywhere else
    return PannsEncoder(base_model = base, embedding_dim = embedding_dim)