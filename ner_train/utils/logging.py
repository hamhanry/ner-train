import os
import sys
import logging
from time import time
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
import torch
from torch.cuda.amp.grad_scaler import GradScaler

from .general import is_homogenous_iterable


class DummyLogger:
    def __init__(self, *_, **__):
        pass

    def __getattr__(self, attr):
        return DummyDispatcher(self, attr)


class DummyDispatcher:
    def __init__(self, caller, name):
        self.caller = caller
        self.name = name

    def __call__(self, *_, **__):
        setattr(self.caller, self.name, self.mock)
        return getattr(self.caller, self.name)(*_, **__)

    @classmethod
    def mock(cls, *_, **__):
        pass


class MlflowExperimentNotFound(Exception):
    pass


class MLFlowLoggerNew:
    def __init__(
        self,
        rank: int,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.rank = rank
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        if self.tracking_uri is None:
            self.tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        if self.experiment_name is None:
            self.experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]

        if self.rank == 0:
            self.mlflow_client = mlflow.tracking.MlflowClient(
                tracking_uri=self.tracking_uri,
            )

            self.experiment = self.mlflow_client.get_experiment_by_name(
                self.experiment_name
            )

            if self.experiment is None:
                raise MlflowExperimentNotFound(
                    f"Experiment {self.experiment_name} does not exist!"
                )

            self.experiment_id = self.experiment.experiment_id

            self.run = self.mlflow_client.create_run(
                experiment_id=self.experiment_id, tags=tags
            )
            self.run_id = self.run.info.run_id

    def log_hparams(self, hparams: Dict[str, Any]):
        if self.rank == 0:
            for k, v in hparams.items():
                self.mlflow_client.log_param(self.run_id, k, v)

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        if self.rank == 0:
            timestamp_ms = int(time() * 1000)

            for k, v in metrics.items():
                self.mlflow_client.log_metric(self.run_id, k, v, timestamp_ms, step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None):
        if self.rank == 0:
            self.mlflow_client.log_artifact(self.run_id, path, artifact_path)

    def log_artifacts(self, path: str, artifact_path: Optional[str] = None):
        if self.rank == 0:
            if not os.path.isdir(path):
                raise NotADirectoryError

            self.mlflow_client.log_artifacts(self.run_id, path, artifact_path)


def init_logging(
    root_logger: logging.Logger,
    rank: int,
    log_path: str = "",
    stdout: bool = False,
    level=logging.INFO,
    enable_on_ranks=None,
):
    """Initialize python loggers.

    Parameters
    ----------
    root_logger : logging.Logger
        Root Logger.
    rank : int
        Global rank of the process.
        Logging will only be initialized on rank 0.
    log_path : str, optional
        Path to save the log file.
    stdout : bool, optional
        Whether to log to stdout, by default False
    """
    if enable_on_ranks is None:
        enable_on_ranks = [0]
    if not is_homogenous_iterable(enable_on_ranks):
        raise ValueError("elements on `enable_on_ranks` must be integers")

    if rank in enable_on_ranks:
        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")

        root_logger.setLevel(level)

        if log_path:
            handler_file = logging.FileHandler(log_path)
            handler_file.setFormatter(formatter)
            root_logger.addHandler(handler_file)

        if stdout:
            handler_stream = logging.StreamHandler(sys.stdout)
            handler_stream.setFormatter(formatter)
            root_logger.addHandler(handler_stream)

        root_logger.propagate = False

        root_logger.info("rank_id: %d" % rank)
        return root_logger

    return DummyLogger()


class CallbackLogging(object):
    def __init__(self, frequency: int, rank: int):
        self.frequency: int = frequency
        self.rank: int = rank

    def __call__(
        self,
        global_step,
        loss: float,
        epoch: int,
        grad_scaler: Optional[GradScaler] = None,
    ):
        if self.rank == 0 and global_step > 0 and global_step % self.frequency == 0:
            if grad_scaler is not None:
                msg = f"Loss {loss:.4f} Epoch: {epoch} Global Step: {global_step} Fp16 Grad Scale: {grad_scaler.get_scale():.2f}"
            else:
                msg = f"Loss {loss:.4f} Epoch: {epoch} Global Step: {global_step}"
            logging.info(msg)


class ObjectLogger:
    def __init__(self, log_dir, name) -> None:
        self.log_dir = log_dir
        self.name = name

    def log_tensor(self, data, step=None, **kwargs):
        if step is None:
            step = int(time())

        save_path = Path(self.log_dir) / f"{self.name}_{step}.pth"
        torch.save(data, save_path, **kwargs)
