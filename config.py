import pathlib
project_root = pathlib.Path(__file__).resolve().parent

classes = ["808", "clap", "hihat", "kick", "openhat", "snare"]
num_classes = len(classes)

encoders = {"cnn6": "weights/Cnn6_mAP=0.343.pth", "cnn10": "weights/Cnn10_mAP=0.380.pth", "cnn14": "weights/Cnn14_mAP=0.431.pth"}
encoder_name = ["cnn6", "cnn10", "cnn14"][0]
encoder_ckpt_path = project_root / encoders[encoder_name]

sample_rate = 32000
duration = 1.5
num_samples = int(sample_rate * duration)