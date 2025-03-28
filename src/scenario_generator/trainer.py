import os
import logging
import pickle
import wandb
import numpy as np
import random
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy

from transformers import AutoTokenizer

from model import ScenarioGeneratorModel
from data import make_supervised_data_module

def train(config):
    devices = None
    accelerator = None
    if config.device == -1:
        accelerator = "cpu"
    else:
        accelerator = "gpu"

        temp = config.device.split(",")
        devices = [int(x) for x in temp]

    model_name = config.model_name.split("/")[-1]
    gpu_count = len(devices) if devices is not None else 1

    wandb_logger = WandbLogger(project=config.wandb_project,
                               name=f"{config.task}-{model_name}-batch_size {config.batch_size * config.accumulate_grad_batches}-{config.lr}-{config.lr_scheduler}-{config.memo}")
    logging.info("-"*30 + "Wandb Setting Complete!" + "-"*30)

    seed = config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info("-"*30 + f"Seed {seed} Setting Complete!" + "-"*30)

    model = ScenarioGeneratorModel(config=config)
    logging.info("-"*30 + "Model initialized!" + "-"*30)

    tokenizer = model.tokenizer

    data_module = make_supervised_data_module(tokenizer, config)
    train_dataloader = data_module["train_dataloader"]
    valid_dataloader = data_module["eval_dataloader"]

    logging.info("-"*30 + "Data Loaded!" + "-"*30)



    checkpoint_callback = ModelCheckpoint(
                                          monitor='val_loss',
                                          dirpath=f'{config.checkpoint_path}/{config.task}-{model_name}-batch_size_{config.batch_size * config.accumulate_grad_batches * gpu_count}-seed_{config.seed}-{config.lr}-{config.lr_scheduler}-{config.memo}',
                                          filename= f"{config.task}-{model_name}-batch_size_{config.batch_size * config.accumulate_grad_batches * gpu_count}-seed_{config.seed}-{config.lr}-{config.lr_scheduler}-{config.memo}"+"-{val_loss:.2f}-{epoch}epoch",
                                          save_top_k=5,
                                          save_last=False,
                                          verbose=True,
                                          mode="min"
                                          )


    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=config.early_stop,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    if config.early_stop == 0:
        callback_list = [lr_monitor, checkpoint_callback]
    else:
        callback_list = [checkpoint_callback, lr_monitor, early_stopping]

    # if config.strategy == "deepspeed_stage_3_offload":
    #     config.strategy = DeepSpeedStrategy(
    #         stage=3,
    #         offload_optimizer=True,
    #         offload_parameters=True,
    #     )


    trainer = pl.Trainer(
                         accelerator=accelerator,
                         devices=devices,
                         precision=config.precision,
                         strategy=config.strategy,
                         enable_progress_bar=True,
                         callbacks=callback_list,
                        # max_steps=config.max_steps,
                         max_epochs=config.max_epochs,
                        #  val_check_interval=config.val_check_interval,
                        #  check_val_every_n_epoch=None,
                        #  check_val_every_n_epoch=config.check_val_every_n_epoch,
                         num_sanity_val_steps=config.num_sanity_val_steps,
                         logger=wandb_logger,
                         accumulate_grad_batches=config.accumulate_grad_batches,
                         )

    # logging.info(f"Estimated stepping batches: {trainer.estimated_stepping_batches}")

    logging.info("-"*30 + "Train Start!" + "-"*30)
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader,
                ckpt_path=config.ckpt_path)

    logging.info("-"*30 + "Train Finished!" + "-"*30)

    # logging.info("-"*30 + "Test Start!" + "-"*30)
    # trainer.test(model, test_dataloader, ckpt_path="best")
    # logging.info("-"*30 + "Test Finished!" + "-"*30)
