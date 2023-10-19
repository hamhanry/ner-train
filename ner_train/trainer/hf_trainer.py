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
from torchmetrics import Accuracy
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
        self.initialize_loggers(config=self.config)
        self.initialize_train_test_loader(config=self.config)
        self.build_model(config=self.config)
        self.configure_optimizer(config=self.config)
        self.configure_scheduler(config=self.config, optimizer=self.optimizer)

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
                'shuffle': False,
                'num_workers': config.num_workers
                }
        
        train_df = pd.read_csv(config.dataset.train_dataset.csv_path)
        one_hot = pd.get_dummies(train_df['Category'])
        train_df = pd.concat([train_df, one_hot], axis=1, join='inner')
        train_df['list'] = train_df[train_df.columns[2:]].values.tolist()
        
        train_set = train_df.sample(frac=config.dataset.train_dataset.train_size,random_state=200)
        test_set = train_df.drop(train_set.index).reset_index(drop=True)
        train_set = train_set.reset_index(drop=True)

        self.train_dataset = ResumeDataset(train_set, self.tokenizer, config.dataset.max_len)
        self.test_dataset = ResumeDataset(test_set, self.tokenizer, config.dataset.max_len)
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
        train_top1_acc_meter = AverageMeter("train_top1_acc")
        train_top2_acc_meter = AverageMeter("train_top2_acc")
        train_top5_acc_meter = AverageMeter("train_top5_acc")
        train_top10_acc_meter = AverageMeter("train_top10_acc")
        # acc_calculator = accuracy.MulticlassAccuracy(
        #     num_classes=config.hparams.model.num_classes
        # )
        acc_top1_calculator = Accuracy(
            task="multiclass",
            num_classes=config.hparams.model.num_classes
        )
        acc_top1_calculator.to(self.device)

        acc_top2_calculator = Accuracy(
            task="multiclass",
            num_classes=config.hparams.model.num_classes,
            top_k = 2
        )
        acc_top2_calculator.to(self.device)

        acc_top5_calculator = Accuracy(
            task="multiclass",
            num_classes=config.hparams.model.num_classes,
            top_k = 5
        )
        acc_top5_calculator.to(self.device)

        acc_top10_calculator = Accuracy(
            task="multiclass",
            num_classes=config.hparams.model.num_classes,
            top_k = 10
        )
        acc_top10_calculator.to(self.device)

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
                try:
                    batch_dict = next(train_loader)
                except StopIteration:
                    train_loader = iter(self.train_loader)
                    batch_dict = next(train_loader)

                ids = batch_dict['ids'].to(self.device, non_blocking=True)
                mask = batch_dict['mask'].to(self.device, non_blocking=True)
                token_type_ids = batch_dict['token_type_ids'].to(self.device, non_blocking=True)
                targets = batch_dict['targets'].to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                with torch.autocast(
                    device_type=self.device, enabled=config.amp.enabled
                ):
                    cls_out: Tensor
                    cls_out = self.model(ids, mask, token_type_ids)                    
                    classif_loss: Tensor = self.criterion_cls(
                        cls_out, targets
                    )

                    cls_out = torch.softmax(cls_out, dim=1)
                    train_loss: Tensor = (
                        classif_loss #* config.hparams.lambda_cls
                    )

                    if train_loss.isnan():
                        raise RuntimeError(
                            f"{datetime.datetime.now()}: Training loss is NaN on step "
                            f"{self.global_step}."
                        )

                # ---------------------------- multiclass accuracy --------------------------- #
                # print("\ncls_out\n")
                # print(cls_out.shape, cls_out)
                # print("\ntargets\n")
                # print(targets.shape, targets)
                train_top1_acc: Tensor = acc_top1_calculator(cls_out, targets)
                train_top2_acc: Tensor = acc_top2_calculator(cls_out, targets)
                train_top5_acc: Tensor = acc_top5_calculator(cls_out, targets)
                train_top10_acc: Tensor = acc_top10_calculator(cls_out, targets)

                # ------------------------- running metric statistics ------------------------ #
                minibatch_size = batch_dict['mask'].shape[0]
                classif_loss_meter.update(classif_loss.item(), n=minibatch_size)
                train_loss_meter.update(train_loss.item(), n=minibatch_size)
                train_top1_acc_meter.update(train_top1_acc.item(), n=minibatch_size)
                train_top2_acc_meter.update(train_top2_acc.item(), n=minibatch_size)
                train_top5_acc_meter.update(train_top5_acc.item(), n=minibatch_size)
                train_top10_acc_meter.update(train_top10_acc.item(), n=minibatch_size)

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
                        "train_top1_acc": acc_top1_calculator.compute().item(),
                        "train_top1_acc": train_top1_acc.item(),
                        "train_top2_acc": acc_top2_calculator.compute().item(),
                        "train_top2_acc": train_top2_acc.item(),
                        "train_top5_acc": acc_top5_calculator.compute().item(),
                        "train_top5_acc": train_top5_acc.item(),
                        "train_top10_acc": acc_top10_calculator.compute().item(),
                        "train_top10_acc": train_top10_acc.item(),
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
                "train_acc_top1_avg": train_top1_acc_meter.avg,
                "train_acc_top1": acc_top1_calculator.compute().item(),
                "train_acc_top2_avg": train_top2_acc_meter.avg,
                "train_acc_top2": acc_top2_calculator.compute().item(),
                "train_acc_top5_avg": train_top5_acc_meter.avg,
                "train_acc_top5": acc_top5_calculator.compute().item(),
                "train_acc_top10_avg": train_top5_acc_meter.avg,
                "train_acc_top10": acc_top5_calculator.compute().item(),
                "epoch": self.current_epoch,
                "optim_lr": self.optimizer.param_groups[0]["lr"],
            }

            self.set_modules_eval()
            val_log_dict = self.run_validation(
                config=config
            )
            log_dict.update(val_log_dict)

            # self.logger.info(
            #     "[EPOCH %d]: SAVING CHECKPOINT", self.current_epoch
            # )
            # self.save_checkpoint()
            # self.logger.info(
            #     "[EPOCH %d]: SAVING CHECKPOINT SUCCESS", self.current_epoch
            # )

            self.logger.info(
                "[EPOCH %d]: LOGGING MLFLOW METRICS", self.current_epoch
            )
            self.mlflow_logger.log_metrics(
                log_dict,
                self.global_step,
            )
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

    @torch.no_grad()
    def run_validation(
        self, config: Adict
    ) -> Tuple[Dict[str, float]]:
        all_log_dicts = {}
        val_classif_losses: List[Tensor] = []
        fin_predictions = []
        fin_targets = []
        batch_dict: Dict[Tensor, Tensor, Tensor, Tensor]
        for val_iter, batch_dict in enumerate(self.test_loader):
            if val_iter % config.log_every_n_step == 0:
                self.logger.info(
                    "[EPOCH %d]: Evaluating -- iter %d",
                    self.current_epoch,
                    val_iter,
                )
            
            ids = batch_dict['ids'].to(self.device, non_blocking=True)
            mask = batch_dict['mask'].to(self.device, non_blocking=True)
            token_type_ids = batch_dict['token_type_ids'].to(self.device, non_blocking=True)
            targets = batch_dict['targets'].to(self.device, non_blocking=True)
            with torch.autocast(
                device_type=self.device, enabled=config.amp.enabled
            ):
                outputs = self.model(ids, mask, token_type_ids)
                test_loss = self.criterion_cls(outputs, targets)
                val_classif_losses.append(test_loss.item())
            
            preds = torch.softmax(outputs, dim=1)
            
            fin_predictions.append(preds)
            fin_targets.append(targets)
            # fin_predictions.append(outputs)
        
        # fin_predictions: Tensor = fin_predictions
        # fin_targets: Tensor = fin_targets

        # print(fin_predictions.size)
        # print(fin_targets.size)
        fin_predictions = torch.cat(fin_predictions)
        fin_targets = torch.cat(fin_targets)
    
        val_classif_loss = np.mean(val_classif_losses)
        val_loss = val_classif_loss
        
        val_acc_top1_calculator = Accuracy(
            task="multiclass",
            num_classes=config.hparams.model.num_classes
        )
        val_acc_top1_calculator.to(self.device)
        
        val_acc_top2_calculator = Accuracy(
            task="multiclass",
            num_classes=config.hparams.model.num_classes,
            top_k = 2
        )
        val_acc_top2_calculator.to(self.device)
        
        val_acc_top5_calculator = Accuracy(
            task="multiclass",
            num_classes=config.hparams.model.num_classes,
            top_k = 5
        )
        val_acc_top5_calculator.to(self.device)
        
        val_acc_top10_calculator = Accuracy(
            task="multiclass",
            num_classes=config.hparams.model.num_classes,
            top_k = 10
        )
        val_acc_top10_calculator.to(self.device)

        val_top1_acc: Tensor = val_acc_top1_calculator(fin_predictions, fin_targets)
        val_top2_acc: Tensor = val_acc_top2_calculator(fin_predictions, fin_targets)
        val_top5_acc: Tensor = val_acc_top5_calculator(fin_predictions, fin_targets)
        val_top10_acc: Tensor = val_acc_top10_calculator(fin_predictions, fin_targets)

        all_log_dicts = {
            "val_loss": val_loss.item(),
            "classif_loss": val_classif_loss.item(),
            "val_acc_top1": val_top1_acc.item(),
            "val_acc_top2":val_top2_acc.item(),
            "val_acc_top5": val_top5_acc.item(),
            "val_acc_top10": val_top10_acc.item()
        }
        
        return all_log_dicts

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