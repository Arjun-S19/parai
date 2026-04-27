import pathlib
project_root = pathlib.Path(__file__).resolve().parent

classes = ["808", "clap", "hihat", "kick", "openhat", "snare"]

classname_variants = {
    "808": ["808", "808s"],
    "clap": ["clap", "claps"],
    "hihat": ["hihat", "hi hat", "hihats", "hi hats"],
    "kick": ["kick", "kicks"],
    "openhat": ["openhat", "open hat", "openhats", "open hats"],
    "snare": ["snare", "snares"]
}

num_classes = len(classes)

inference_threshold = 0.5

ood_detection = {
	"enabled": True,
	"none_label": "none",
	"temperature": 1.0,
	"energy_threshold": None,
	"per_class_conf_thresholds": {},
	"thresholds_path": "runs/ood_thresholds_cnn6.json",
}

encoders = {"cnn6": "weights/Cnn6_mAP=0.343.pth", "cnn10": "weights/Cnn10_mAP=0.380.pth", "cnn14": "weights/Cnn14_mAP=0.431.pth"}
encoder_name = ["cnn6", "cnn10", "cnn14"][0]
encoder_ckpt_path = project_root / encoders[encoder_name]

sample_rate = 32000
duration = 1.5
num_samples = int(sample_rate * duration)