import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer

IGNORE_INDEX = -100



def generate_document_scenario_prompt(document, dataset=None):
    if dataset == None:
        dataset = ""
    document_prompt1 = f"""You are an advanced language model specializing in knowledge extraction and user need modeling. Your task is to extract hypothetical user scenarios from a given {dataset} document, ensuring that the generated information needs reflect the document's overall insights and knowledge, rather than isolated details.

    Content:
    - Main Topic: Briefly describe the primary subject of the document
    - Key Points: Summarize the core concepts, insights, or knowledge presented
    - Information Needs: Generate a diverse set of possible information needs that can be satisfied by the document
    - Explanation: Explain how the document fulfills that need, ensuring that explanations are generalized and conceptual rather than overly detailed.

    Format:
    - Generate JSON format

    """

    document_prompt2 = """**For Noise Document**
    - If the document consists of ONLY noise, such as sequences of numbers, empty lines, single words, random characters, or unstructured and meaningless fragments, return noise document scenario.

    Noise Document Scenario:
    '{"document_analysis": {"main_topic": "This document is noise document. So document analysis is not exist", "key_points": ["This document is noise document. So key points are not exist"]}, "information_needs": ["This document is noise document. So information needs are not exist"], "explanation": [{"information_need": "This document is noise document. So information needs are not exist.", "explanation": "This document is noise document. So explanation is not exist."}]}'

    <Document>

    """  

    return document_prompt1 + document_prompt2 + document + "\n\n<Answer>\n"



def apply_chat_prompt(tokenizer, source, target):
    messages = [{"role": "user", "content": source},
                {"role": "assistant", "content": target}]
    
    inputs_text = tokenizer.apply_chat_template(messages, tokenize=False)

    return inputs_text

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    # pad token attention mask
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = []

    for source, target in zip(sources, targets):
        examples.append(apply_chat_prompt(tokenizer, source, target))

    sources_template_applied = [i.split("<|start_header_id|>assistant<|end_header_id|>")[:-1][0] for i in examples]

    logging.warning(f"length of examples: {len(examples)}")
    logging.warning(f"examples[0]: {examples[0]}")

    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources_template_applied)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        data = pd.read_parquet(data_path)

        logging.warning("Formatting inputs...")
        documents_content = data["content"]
        dataset_name = data["dataset"]
        content_type = data["type"]

        logging.warning(f"length of documents_content: {len(documents_content)}")
        logging.warning("Generating prompts...")
        sources = []
        for content, name, type in zip(documents_content, dataset_name, content_type):
            sources.append(generate_document_scenario_prompt(content, name))


        targets = data["scenario_list"].tolist()

        logging.warning(f"length of sources: {len(sources)}")
        logging.warning(f"length of targets: {len(targets)}")

        logging.warning(f"sources[0]: {sources[0]}")
        logging.warning(f"targets[0]: {targets[0]}")

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.train_data_path)

    if args.valid_data_path:
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.valid_data_path)
    else:
        eval_dataset = None

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    if eval_dataset:
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)
    else:
        eval_dataloader = None

    return dict(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)
