# parai

Drum sound classification using transfer learning with pretrained PANNs (Pre-Trained Audio Neural Networks) encoders. Given an audio file, the model predicts which of six drum categories it belongs to: 808, clap, hihat, kick, openhat, or snare.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Pretrained PANNs encoder weights (CNN6, CNN10, or CNN14) must be placed in the paths specified in `config.py`.

## Dataset Preparation

Organize raw drum samples with one subdirectory per class:

```
drum_samples/
    808/
    clap/
    hihat/
    kick/
    openhat/
    snare/
```

Then run:

```bash
python build_dataset_csv.py
python audio_normalization.py
```

`build_dataset_csv.py` produces train/validate/test CSV files in `datasets/`.

`audio_normalization.py` resamples and normalizes all audio to 32 kHz, 1.5 seconds, mono, and peak-normalized amplitude, writing output to `normalized_drum_samples/`.

## Training

Train only the classifier head (encoder weights frozen):

```bash
python train_head_only.py --train-csv datasets/1_train.csv --val-csv datasets/2_validate.csv --encoder cnn6
```

Fine-tune the encoder and classifier head jointly:

```bash
python train_finetune.py --train-csv datasets/1_train.csv --val-csv datasets/2_validate.csv --encoder cnn6
```

Supported encoder options: `cnn6`, `cnn10`, `cnn14`. Larger encoders generally yield higher accuracy at the cost of speed.

## Inference

Classify a single file or all audio files in a directory:

```bash
python infer.py --input <audio_file_or_folder> --checkpoint <checkpoint.pth> --encoder cnn6 --out predictions.csv
```

The output CSV contains one row per file with columns: `path`, `pred_label`, `pred_conf`, and `probs` (full per-class probability vector).

## Evaluation

Evaluate a checkpoint on a data split:

```bash
python evaluate_split.py <checkpoint.pth> test --encoder cnn6 --save-cm confusion_matrix.json
```

Split argument accepts `train`, `validate`, or `test`. Reports overall accuracy and per-class accuracy. Optionally saves a confusion matrix to JSON.

## Project Structure

```
parai/
    config.py               # Class labels, sample rate, encoder weight paths
    data.py                 # DrumDataset for loading training data
    metrics.py              # Accuracy, per-class accuracy, confusion matrix
    audio_normalization.py  # Resample, pad, and normalize audio
    build_dataset_csv.py    # Build train/validate/test CSV splits
    train_head_only.py      # Train classifier head with frozen encoder
    train_finetune.py       # Fine-tune encoder and classifier head
    infer.py                # Run inference on audio files
    evaluate_split.py       # Evaluate model on a data split
    models/
        model.py            # DrumClassifier (encoder + head)
        head.py             # MLP classifier head
        encoders.py         # PANNs encoder wrapper
    audioset_tagging_cnn/   # PANNs model definitions (CNN6, CNN10, CNN14)
    requirements.txt
```
