import datetime
import logging
import os
import re
import sys
from argparse import ArgumentParser, Namespace
from collections import namedtuple
from functools import partial
from pathlib import Path
from time import time
from typing import List, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn
import torch.optim
import torch.utils.data.distributed
import torchdata.datapipes.iter as IDP
import yaml
from addict import Dict as Adict

from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchmetrics.classification import accuracy
from torchvision.transforms import Compose

import nltk
from nltk.corpus import stopwords

from ner_train.data.transform.basic import basic_preprocessing, remove_duplicates
from ner_train.utils.general import *
from ner_train.utils.logging import *

def worker_init_fn(worker_id: int, base_seed: int):
    import os  # pylint: disable=reimported,redefined-outer-name

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    import cv2

    cv2.setNumThreads(1)

    worker_seed = worker_id + base_seed
    seed_all(worker_seed)

class hfTrainer:
    start_epoch: int = 0
    current_epoch: int = 0
    global_step: int = 0
    iter_per_epoch: int
    max_iter: int
    max_epoch: int

    _ckpt_name_template = (
        "DIVT-ckpt-{timestamp}-step{global_step}-ep{current_epoch}.pth"
    )

    def __init__(
        self,
        args: Namespace,
        config: Adict,
        checkpoint_dir: Optional[str] = "/home/ham/mnt/nas/",
    ) -> None:
        self.args = args
        self.config = Adict(config)
        self.config.freeze(True)

        self.environment_setup(config=self.config)
        self.set_device(config=self.config)
        self.set_checkpoint_path(checkpoint_dir)
        self.initialize_trainer_variables(config=self.config)
        self.initialize_train_loader(config=self.config)
        self.initialize_val_loaders(config=self.config)
        self.build_model(config=self.config)
        self.configure_optimizer(config=self.config)
        self.configure_scheduler(config=self.config, optimizer=self.optimizer)
        self.load_checkpoint(config=self.config)
        self.set_classes_info(config=self.config)
        self.set_eval_threshold_from(config=self.config)

    def environment_setup(self, config: Adict):
        torch.backends.cudnn.benchmark = config.get("benchmark", False)
        torch.use_deterministic_algorithms(
            mode=config.get("deterministic", False), warn_only=True
        )
        seed_all(config.hparams.seed)

    def initialize_trainer_variables(self, config: Adict):
        self.start_epoch: int = 0
        self.current_epoch: int = self.start_epoch
        self.global_step: int = 0
        self.iter_per_epoch: int = config.hparams.iter_per_epoch
        self.max_iter: int = config.hparams.max_iter
        self.max_epoch: int = config.hparams.get("max_epoch", 0)

    def initialize_train_loader(self):
        self.train_data = pd.read_csv(self.args.train_csv)
        self.train_data.drop(['Resume_html','ID'],axis=1,inplace=True)
        self.train_data['Resume_str'] = self.train_data['Resume_str'].apply(basic_preprocessing)
        self.train_data.rename(columns = {'Resume_str':'resume'}, inplace = True)
        self.train_data['resume'] = self.train_data['resume'].apply(remove_duplicates)

    def set_device(self, config: Adict):
        if config.use_cpu:
            self.device = "cpu"
        else:
            self.device = "cuda"
    
    def set_checkpoint_path(self, checkpoint_dir):
        self.checkpoint_path = str(
            Path(checkpoint_dir).joinpath(self.args.run_name)
        )
    
    def initialize_loggers(self, config: Adict):
        self.timestamp = f"{int(time())}"
        logger = logging.getLogger(__name__)
        self.logger = init_logging(logger, rank=0, stdout=True)

        ckpt_path = self.checkpoint_path
        if ckpt_path:
            self.log_dir: Path = Path(ckpt_path)

            if self.log_dir.exists():
                self.logger.info("Log dir %s already exists", self.log_dir)
            else:
                self.log_dir.mkdir(exist_ok=False, parents=True)
        # else:
        #     self.log_dir: Path = Path(SAGEMAKER_CHECKPOINT_PATH)

        os.environ["MLFLOW_TRACKING_URI"] = config.mlflow_tracking_uri
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

        mlflow_tags = config.mlflow_tags.to_dict()
        mlflow_tags["mlflow.runName"] = str(self.args.run_name)

        self.mlflow_logger = MLFlowLoggerNew(
            rank=0,
            tracking_uri=tracking_uri,
            experiment_name=config.mlflow_experiment_name,
            tags=mlflow_tags,
        )
        self.mlflow_logger.log_hparams(flatten_dict(config.hparams))

        new_config_path = (
            Path(self.checkpoint_path) / Path(self.args.config_path).name
        )
        with open(
            new_config_path, mode="w", encoding="utf-8"
        ) as new_config_file:
            yaml.dump(config.to_dict(), new_config_file, Dumper=yaml.SafeDumper)

        self.mlflow_logger.log_artifact(new_config_path)
    
    def build_model(self, config: Adict):
        # --------------------------- model initialization --------------------------- #
        # note: model must return embedding and logits!
        self.model = object_from_dict(config.hparams.model).to(self.device)
        if config.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        model_ema = config.hparams.get("ema", None)
        self.model_ema = None
        if model_ema:
            self.model_ema = object_from_dict(model_ema, model=self.model)

        self.criterion_cls = object_from_dict(config.hparams.criterion_cls)
        self.criterion_dic = object_from_dict(config.hparams.criterion_dic)

    def configure_optimizer(self, config: Adict):
        # --------------------------------- optimizer -------------------------------- #
        param_groups = [
            {"params": self.model.parameters()},
        ]

        self.optimizer = object_from_dict(
            config.hparams.optimizer, params=param_groups
        )

        self.init_param_lr = []
        for param_group in self.optimizer.param_groups:
            self.init_param_lr.append(param_group["lr"])

        self.grad_scaler = torch.cuda.amp.grad_scaler.GradScaler(**config.amp)

    def configure_scheduler(
        self, config: Adict, optimizer: torch.optim.Optimizer
    ):
        self.scheduler = None
        self.sched_interval = None
        sched_conf = config.hparams.get("scheduler")
        if sched_conf:
            self.scheduler = object_from_dict(sched_conf, optimizer=optimizer)
            self.sched_interval = config.hparams.scheduler_interval
            if self.sched_interval not in ["step", "epoch"]:
                raise ValueError(
                    f"scheduler_interval must be either `step` or `epoch`, got {self.sched_interval}"
                )