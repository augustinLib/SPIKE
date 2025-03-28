from typing import Dict, Any, Tuple, List, Union
import faiss
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutput
import logging
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import get_peft_model, LoraConfig, TaskType, LoftQConfig
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, MistralForCausalLM
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule, AdamW, Optimizer, LambdaLR
from collections import OrderedDict
import evaluate


class ScenarioGeneratorModel(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        base_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
            r=config.lora_r,  # LoRA rank
            lora_alpha=config.lora_alpha,  # LoRA alpha scaling
            lora_dropout=config.lora_dropout  # LoRA dropout
        )
        self.model = get_peft_model(base_model, peft_config)

        # pad token id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # tokenizer configuration
        self.tokenizer.padding_side = 'right'

    def forward(
        self,
        batch: Dict[str, torch.Tensor]
        ) -> Union[Tuple, CausalLMOutput]:
        """forward pass

        1. Get input embeddings
        2. Do model forward pass with input embeddings
        3. Return outputs which type is ModelOutput

        Args:
            batch (Dict[str, torch.Tensor]): input dictionary
                input_ids (torch.LongTensor): input ids
                attention_mask (torch.BoolTensor): attention mask
                context_vectors (torch.FloatTensor): context vectors

        Returns:
            outputs (CausalLMOutputWithPast): output dictionary containing loss, logits, and past key values
        """

        input_ids = batch['input_ids']
        outputs = self.model(input_ids=input_ids,
                             attention_mask=batch['attention_mask'],
                             labels=batch['labels'])
        return outputs


    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
        ) -> torch.Tensor:
        """training step

        Args:
            batch (Dict[str, torch.Tensor]): batch dictionary
                input_ids (torch.LongTensor): input ids
                attention_mask (torch.BoolTensor): attention mask
                context_vectors (torch.FloatTensor): context vectors

        Returns:
            loss (torch.Tensor): loss
        """
        # language modeling loss
        outputs = self.forward(batch)
        lm_loss = outputs.loss

        self.log('train_loss', lm_loss, prog_bar=True, sync_dist=True, logger=True, on_step=True, on_epoch=True)
        return lm_loss


    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outputs = self.forward(batch)
            lm_loss = outputs.loss

        self.log('val_loss', lm_loss, prog_bar=True, sync_dist=True, logger=True, on_step=True, on_epoch=True)
        return lm_loss


    def configure_optimizers(
        self
        ) -> Tuple[List[Optimizer], List[LambdaLR]]:
        """define optimizer and scheduler (AdamW, linear warmup)

        Returns:
            Tuple(List[torch.Optimizer], List[transformers.optimization]): optimizer and scheduler
        """

        no_decay = ["bias", "LayerNorm.weight"]


        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.config.strategy == "deepspeed_stage_2_offload":
            optim = DeepSpeedCPUAdam(optimizer_grouped_parameters, lr=self.config.lr, eps=1e-8)
        elif self.config.strategy == "deepspeed_stage_3_offload":
            optim = DeepSpeedCPUAdam(optimizer_grouped_parameters, lr=self.config.lr, eps=1e-8)
        else:
            optim = AdamW(optimizer_grouped_parameters, lr=self.config.lr, eps=1e-8)
        # optim = FusedAdam(self.parameters(), lr=2e-5, eps=1e-8)
        if self.config.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optim,
                num_warmup_steps=self.trainer.estimated_stepping_batches * self.config.num_warmup_steps_ratio,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.config.lr_scheduler == "constant":
            scheduler = get_constant_schedule(optim)

        elif self.config.lr_scheduler == "cosine":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optim,
                num_warmup_steps=self.trainer.estimated_stepping_batches * self.config.num_warmup_steps_ratio,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optim], [scheduler]

