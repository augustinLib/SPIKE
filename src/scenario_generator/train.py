from argparse import ArgumentParser
from trainer import train
import pickle
import logging
import os

def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="None", type=str, help="model name for huggingface model hub")
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--valid_data_path", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--device", default= -1, type=str)
    parser.add_argument("--precision", default= "bf16-mixed")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--memo", type=str, default="None")
    parser.add_argument("--batch_size", default= 64, type=int)
    parser.add_argument("--max_epochs", default=20, type=int)
    parser.add_argument("--num_sanity_val_steps", default= 2, type=int)
    parser.add_argument("--num_warmup_steps_ratio", default= 0.1, type=float)
    parser.add_argument("--lr", default= 2e-5, type=float)
    parser.add_argument("--lr_scheduler", default= "constant", type=str)
    parser.add_argument("--early_stop", default=10, type=int)
    parser.add_argument("--checkpoint_path", default="./checkpoint", type=str)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--lora_r", default=32, type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", default=64, type=int, help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", default=0.1, type=float, help="LoRA dropout")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="weight decay")
    args = parser.parse_args()
    
    return args


def main(config):
    train(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["NCCL_P2P_DISABLE"] = "1"
    config = parse_argument()
    
    main(config)