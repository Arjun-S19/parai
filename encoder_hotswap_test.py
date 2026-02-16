import torch
import config
from models.encoders import make_panns_encoder
from models.model import DrumClassifier

def run_once(encoder_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = config.project_root / config.encoders[encoder_name]

    encoder = make_panns_encoder(
        name = encoder_name,
        checkpoint_path = ckpt,
        device = device,
        freeze = True,
    )

    model = DrumClassifier(encoder = encoder, num_classes = config.num_classes).to(device)
    model.eval()

    # generate a dummy batch to validate dimensions
    b = 4
    t = config.num_samples
    x = torch.randn(b, 1, t, device = device)

    with torch.no_grad():
        logits = model(x)

    # embeddings differ in dim between encoders, but logits must match [b, 6]
    print(f"[{encoder_name}] input = {tuple(x.shape)}")
    print(f"[{encoder_name}] embedding_dim = {encoder.embedding_dim}")
    print(f"[{encoder_name}] logits = {tuple(logits.shape)}")
    print("-" * 60)

def main():
    # run all encoders back-to-back to test hotswap
    for name in ["cnn6", "cnn10", "cnn14"]:
        run_once(name)

if __name__ == "__main__":
    main()
