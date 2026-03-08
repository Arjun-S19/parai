# parai

A drum sample classifier built on the [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) pretrained model

## Inference Pipeline

Audio file input → Normalization → PANNs encoder → Embedding → MLP classifier head → Class output

## Training

- Head only: Encoder frozen, only the MLP classifier trained
- Head + Finetune last block: Last convolutional block unfrozen and trained with the head at a lower learning rate

## Upcoming Features

- parai Drumkit Organizer: An application that can organize drum samples in drumkits with a clean interface for producers
- Drum classification extension: Extending the current classifications to more types of drum samples
- Drum class filters: Filtering for certain characteristics in drum classes (ex. distorted 808, wide hihat)
