# ner-train
This repo is used for training purposed for ner classification especially for job similarity matching.

## Dataset
Normally the dataset is not store into github repository, but in most cases, the dataset is given in the readme with the given link. Therefore, i would follow what the community usually done that. The dataset could be download from here: https://drive.google.com/drive/folders/1y_AWyhdZgPsnWZvZY_v5T-pUBAc5G1xu?usp=share_link

## How to use
i will still provide the same requirements with the ner repository, however in order to run this training package it require the mlflow package that is quite long to be written. As i do not focus on writing on this part, i will leave each assessors to follow directly the link of mlflow installation in this link https://www.mlflow.org/docs/latest/gateway/index.html

### Configurations
Each training that are performed under this repository is running by given training configuration that could be found under `ner_train/config/`. Inside the training configuration, you could find these hyper-parameters that could easily adjusted for each parameters that needs to be trained later on. If any training would like to conduct, just add another configuration file.

```yaml
amp:
  enabled: true
benchmark: false
data_parallel: false
hparams:
  batch_size: 32
  iter_per_epoch: 100
  max_iter: 4000
  criterion_cls:
    type: torch.nn.CrossEntropyLoss
  model:
    drop_rate: 0.3
    pretrained_name: 'bert-base-uncased'
    num_classes: 24
    type: ner_train.model.bert.BERTClass
  optimizer:
    lr: 0.001
    type: torch.optim.AdamW
    weight_decay: 0.001
  seed: 0
dataset:
  max_len : 512
  train_dataset:
    train_size : 0.7
    csv_path: "/home/ham/mnt/ssd/data/dataset/resume_data/Resume/Resume-refactor-v3.csv" # path to your trainng csv dataset
deterministic: false
log_every_n_step: 10
mlflow_experiment_name: kupu-oct23-ner # experiments name under mlflow system
mlflow_tags:
  developer: ham
mlflow_tracking_uri: http://localhost:8002/ # where the mlflow host port is runnig
num_workers: 10
use_cpu: false

```

### Training
```bash
cd /path/to/ner-train/
```

```python
python -m ner_train.trainer.hf_trainer \
    -c "/home/ham/mnt/nas/project/kupu/ner-train/ner_train/config/roberta004.yaml" \ #path to your configuration yaml file
    --checkpoint_path "/home/ham/mnt/nas/server/mlflow/experiments/kupu-okt23-ner/" \ #path to save all the training artifacts
    --run_name "roberta-004" #identifier name in mlflow
```