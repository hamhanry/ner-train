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

import transformers
from transformers import BertTokenizer,BertModel,BertConfig

from ner_train.data.transform.basic import basic_preprocessing, remove_duplicates
from ner_train.data.datapipe.resume_dataset import ResumeDataset
from ner_train.utils.general import *
from ner_train.utils.logging import *
from ner_train.utils.metric import *

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
        "hf-ckpt-{timestamp}-step{global_step}-ep{current_epoch}.pth"
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
        self.initialized_tokenizer(config=self.config)
        self.initialize_trainer_variables(config=self.config)
        self.initialize_train_test_loader(config=self.config)
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

    def initialize_train_test_loader(self, config: Adict):
        train_params = {'batch_size': config.hparams.batch_size,
                'shuffle': True,
                'num_workers': config.num_workers
                }

        test_params = {'batch_size': config.hparams.batch_size,
                'shuffle': True,
                'num_workers': config.num_workers
                }
        
        train_df = pd.read_csv(config.dataset.train_dataset.csv_path)
        train_dataset = train_df.sample(frac=config.dataset.train_dataset.train_size,random_state=200)
        test_dataset = train_df.drop(train_dataset.index).reset_index(drop=True)
        train_dataset = train_dataset.reset_index(drop=True)

        self.train_dataset = ResumeDataset(train_dataset, self.tokenizer, config.dataset.max_len)
        self.test_dataset = ResumeDataset(test_dataset, self.tokenizer, config.dataset.max_len)
        self.train_loader = DataLoader(self.train_dataset, **train_params)
        self.test_loader = DataLoader(self.test_dataset, **test_params)
        # self.train_data.drop(['Resume_html','ID'],axis=1,inplace=True)
        # self.train_data['Resume_str'] = self.train_data['Resume_str'].apply(basic_preprocessing)
        # self.train_data.rename(columns = {'Resume_str':'resume'}, inplace = True)
        # self.train_data['resume'] = self.train_data['resume'].apply(remove_duplicates)
        
        # stop = stopwords.words('english')
        # self.train_data['resume'] = self.train_data['resume'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


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
        self.model = object_from_dict(config.hparams.model).to(self.device)
        if config.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.criterion_cls = object_from_dict(config.hparams.criterion_cls)

    def initialized_tokenizer(self, config:Adict):
        self.tokenizer = BertTokenizer.from_pretrained(config.hparams.model.pretrained_name)

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
    
    def save_checkpoint(self):
        ckpt_dict = {}
        ckpt_dict["model"] = self.model.state_dict()

        ckpt_dict["optimizer"] = self.optimizer.state_dict()
        ckpt_dict["scheduler"] = (
            self.scheduler.state_dict() if self.scheduler else None
        )
        ckpt_dict["current_epoch"] = self.current_epoch
        ckpt_dict["global_step"] = self.global_step

        save_path = os.path.join(
            str(self.checkpoint_path),
            self._ckpt_name_template.format(
                timestamp=self.timestamp,
                global_step=self.global_step,
                current_epoch=self.current_epoch,
            ),
        )

        self.logger.info("Saving checkpoint to %s", str(save_path))
        torch.save(ckpt_dict, save_path)
        self.mlflow_logger.log_artifact(save_path)

    def set_modules_train(self):
        self.model.train()

    def set_modules_eval(self):
        self.model.eval()

    def train(self, config: Adict):
        self.logger.info("Training started")
        self.set_modules_train()

        train_loss_meter = AverageMeter("train_loss")
        classif_loss_meter = AverageMeter("classif_loss")
        train_acc_meter = AverageMeter("train_acc")
        acc_calculator = accuracy.MulticlassAccuracy(
            num_classes=len(self.data_labels)
        )
        acc_calculator.to(self.device)

        if self.max_epoch:
            max_epoch = self.max_epoch
            self.max_iter = sys.maxsize
        else:
            max_epoch = self.max_iter // self.iter_per_epoch

        train_loader = iter(self.train_loader)

        for epoch in range(self.current_epoch, max_epoch):
            self.current_epoch = epoch
            self.set_modules_train()
            batch_dict: Dict[Tensor, Tensor, Tensor, Tensor]

            self.logger.info("[EPOCH %d]: TRAINING", self.current_epoch)
            for _ in range(self.iter_per_epoch):
                batch_dict = next(train_loader)
                ids = batch_dict['ids'].to(self.device, dtype = torch.long)
                mask = batch_dict['mask'].to(self.device, dtype = torch.long)
                token_type_ids = batch_dict['token_type_ids'].to(self.device, dtype = torch.long)
                targets = batch_dict['targets'].to(self.device, dtype = torch.float)

                self.optimizer.zero_grad()
                with torch.autocast(
                    device_type=self.device, enabled=config.amp.enabled
                ):
                    cls_out: Tensor
                    cls_out = self.model(ids, mask, token_type_ids)

                    classif_loss: Tensor = self.criterion_cls(
                        cls_out, targets
                    )

                    train_loss: Tensor = (
                        classif_loss * config.hparams.lambda_cls
                    )

                    if train_loss.isnan():
                        raise RuntimeError(
                            f"{datetime.datetime.now()}: Training loss is NaN on step "
                            f"{self.global_step}."
                        )

                # ---------------------------- multiclass accuracy --------------------------- #
                train_acc: Tensor = acc_calculator(cls_out, targets)

                # ------------------------- running metric statistics ------------------------ #
                minibatch_size = batch_dict.shape[0]
                classif_loss_meter.update(classif_loss.item(), n=minibatch_size)
                train_loss_meter.update(train_loss.item(), n=minibatch_size)
                train_acc_meter.update(train_acc.item(), n=minibatch_size)

                self.grad_scaler.scale(train_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                grad_norm_to_clip = config.hparams.get("clip_grad_norm", -1)
                if grad_norm_to_clip > 0.0:
                    clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=grad_norm_to_clip,
                    )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                # -------------------------- validation and logging -------------------------- #
                self.global_step += 1
                if self.global_step % config.log_every_n_step == 0:
                    self.logger.info(
                        "[EPOCH %d]: currently on step: %d",
                        self.current_epoch,
                        self.global_step,
                    )

                    step_log_dict = {
                        "train_loss_avg": train_loss_meter.avg,
                        "train_loss": train_loss.item(),
                        "classif_loss_avg": classif_loss_meter.avg,
                        "classif_loss": classif_loss.item(),
                        "train_acc_avg": train_acc_meter.avg,
                        "train_acc_cal": acc_calculator.compute().item(),
                        "train_acc": train_acc.item(),
                        "epoch": self.current_epoch,
                        "optim_lr": self.optimizer.param_groups[0]["lr"],
                    }
                    self.mlflow_logger.log_metrics(
                        step_log_dict, self.global_step
                    )

                if self.sched_interval == "step" and self.scheduler:
                    self.scheduler.step()

            log_dict = {
                "train_loss_avg": train_loss_meter.avg,
                "classif_loss_avg": classif_loss_meter.avg,
                "train_acc_avg": train_acc_meter.avg,
                "train_acc_cal": acc_calculator.compute().item(),
                "epoch": self.current_epoch,
                "optim_lr": self.optimizer.param_groups[0]["lr"],
            }

            self.set_modules_eval()
            # val_log_dict, prediction_dataframes = self.run_validation(
            #     config=config
            # )
            # log_dict.update(val_log_dict)

            self.logger.info(
                "[EPOCH %d]: SAVING CHECKPOINT", self.current_epoch
            )
            self.save_checkpoint()
            self.logger.info(
                "[EPOCH %d]: SAVING CHECKPOINT SUCCESS", self.current_epoch
            )

            self.logger.info(
                "[EPOCH %d]: LOGGING MLFLOW METRICS", self.current_epoch
            )
            self.mlflow_logger.log_metrics(
                log_dict,
                self.global_step,
            )
            # self._save_pred_df(prediction_dataframes)
            self.logger.info(
                "[EPOCH %d]: SUCCESS LOGGING MLFLOW METRICS",
                self.current_epoch,
            )

            self.set_modules_train()
            self.current_epoch += 1

            if self.sched_interval == "epoch" and self.scheduler:
                self.scheduler.step()
    
    def finish(self):
        client = self.mlflow_logger.mlflow_client
        client.set_terminated(self.mlflow_logger.run_id, status="FINISHED")
        self.logger.info("DONE TRAINING!")

    # def validation(self):
    #     fin_targets=[]
    #     fin_outputs=[]
    #     with torch.no_grad():
    #         for _, data in enumerate(self.test_loader):
    #             ids = data['ids'].to(self.device, dtype = torch.long)
    #             mask = data['mask'].to(self.device, dtype = torch.long)
    #             token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
    #             targets = data['targets'].to(self.device, dtype = torch.float)
    #             outputs = model(ids, mask, token_type_ids)
    #             fin_targets.extend(targets.cpu().detach().numpy().tolist())
    #             fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    #     return fin_outputs, fin_targets

def run_training(args, config):
    trainer = hfTrainer(
        args,
        config,
        checkpoint_dir=args.checkpoint_path,
    )

    try:
        trainer.train(config=trainer.config)
    except Exception as e:
        client = trainer.mlflow_logger.mlflow_client
        if isinstance(e, KeyboardInterrupt):
            print(f"{datetime.datetime.now()}: TRAINING INTERRUPTED")
            client.set_terminated(trainer.mlflow_logger.run_id, status="KILLED")
        else:
            print(f"{datetime.datetime.now()}: TRAINING FAILED")
            client.set_terminated(trainer.mlflow_logger.run_id, status="FAILED")
        raise e

    trainer.finish()

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=True,
        help="path to training .yml config file",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help=(
            "Specify checkpoint path. If not given, "
            f"default path is ."
        ),
    )
    parser.add_argument(
        "--run_name", type=str, required=True, help="Run name for MLFlow."
    )
    parser.add_argument(
        "--pass_worker_init_fn",
        action="store_true",
        help="Whether to pass worker init fn to DataLoaders",
    )
    parser.add_argument(
        "--multiplex_multiple_data",
        action="store_true",
        help=(
            "Whether to multiplex data with multiple data args. By default, "
            "multiple data args datapipe are concatenated."
        ),
    )

    args_ = parser.parse_args()

    return args_

def _main():
    _args = parse_args()

    with open(_args.config_path, encoding="utf-8") as f:
        _config = Adict(yaml.load(f, Loader=yaml.SafeLoader))

    _config.mlflow_tags.repo_commit = get_head_commit()
    _config.mlflow_tags.task = "ner"
    _config.mlflow_tags.dl_framework = "pytorch"
    _config.mlflow_tags.dl_framework_version = str(torch.__version__)
    _config.mlflow_tags.cuda_version = str(torch.version.cuda)

    run_training(_args, _config)


if __name__ == "__main__":
    _main()