import faiss
import numpy as np
from vllm import LLM
from argparse import ArgumentParser
from transformers import AutoTokenizer
import pandas as pd
import json
import os
import pickle
from tqdm import tqdm
import logging
import ast

MODEL_DIM_DICT = {
    "intfloat/e5-mistral-7b-instruct": 4096,
    "Salesforce/SFR-Embedding-Mistral": 4096,
    "GritLM/GritLM-7B": 4096,
    "Alibaba-NLP/gte-Qwen1.5-7B-instruct": 4096,
}

MODEL_MAX_LEN_DICT = {
    "intfloat/e5-mistral-7b-instruct": 8192,
    "Salesforce/SFR-Embedding-Mistral": 4090,
    "GritLM/GritLM-7B": 8192,
    "Alibaba-NLP/gte-Qwen1.5-7B-instruct": 16384,
}

COMPONENT_COLNAME_DICT = {
    "I": "information_need",
    "O": "document",
    "M": "main_topic",
    "K": "key_points",
    "E": "explanation",
}


COMPONENT_NAME_DICT = {
    "I": "Information Need",
    "O": "Document",
    "M": "Main Topic",
    "K": "Key Points",
    "E": "Explanation",
}

NO_GENERATOR_COMPONENT_COLNAME_DICT = {
    "O": "content",
}


TASK_MAP = {
'biology': 'Biology',
'earth_science': 'Earth Science',
'economics': 'Economics',
'psychology': 'Psychology',
'robotics': 'Robotics',
'stackoverflow': 'Stack Overflow',
'sustainable_living': 'Sustainable Living',
'pony': 'Pony',
}

def refine_instruction(task,instruction):
    return instruction.format(task=task)

def get_config():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="intfloat/e5-mistral-7b-instruct")
    parser.add_argument("--input_file", type=str, default="data/embedding/input.txt")
    parser.add_argument("--index_file", type=str, default="data/embedding/index.faiss")
    parser.add_argument("--document_col_name", type=str, default="document")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--benchmark", type=str, default="bright")
    parser.add_argument("--dataset", type=str, default="biology")
    parser.add_argument("--version", type=str, default="original_document")
    parser.add_argument("--id_col_name", type=str, default="id")
    parser.add_argument("--index_type", type=str, default="flat")
    parser.add_argument("--generator_type", type=str, default="gpt-4o")
    parser.add_argument("--components", type=str, default="I+O+M+K+E")
    parser.add_argument("--is_multi_content", type=str, default="True")
    args = parser.parse_args()
    return args


def embed_and_index(args):
    component_list = args.components.split("+")
    component_type = args.components
    output_dir = f"./{args.benchmark}/{args.dataset}/{args.generator_type}/{args.version}/{component_type}"
    os.makedirs(output_dir, exist_ok=True)

    model_name = args.model.split("/")[-1]
    
    model = LLM(
        model=args.model,
        device="cuda:0",
        task="embed",
        trust_remote_code=True,
        seed=42
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if args.index_type == "flat":
        index = faiss.IndexIDMap(faiss.IndexFlatIP(4096))
    elif args.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(4096, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        index = faiss.IndexIDMap(index)

    document = pd.read_parquet(args.input_file)

    if args.generator_type == "no_generator":
        # no_generator: only use the original document
        component_colname_dict = NO_GENERATOR_COMPONENT_COLNAME_DICT
        
    else:
        component_colname_dict = COMPONENT_COLNAME_DICT

    column_list = []
    for component in component_colname_dict.keys():
        if component in component_list:
            column_list.append(component_colname_dict[component])
    
    if len(component_list) == 0:
        assert ValueError("No content to embed")


    document[args.document_col_name] = document[column_list].apply(lambda row: '\n'.join(row.values.astype(str)), axis=1)

    if args.is_multi_content == "False":
        document = document.drop_duplicates(subset=[args.id_col_name], keep="first")
        document.reset_index(drop=True, inplace=True)

    elif args.is_multi_content == "True":
        pass

    else:
        assert ValueError("is_multi_content must be True or False")

    with open(f"../data/embedding_instruction/{model_name}/{args.dataset}.json", "r") as f:
        instruction = json.load(f)

    instruction = instruction["instructions"]
    if  "document" in instruction.keys():
        document_instruction = instruction["document"]
    else:
        document_instruction = ""

    if args.dataset in TASK_MAP.keys():
        task = TASK_MAP[args.dataset]
        document_instruction = refine_instruction(task,document_instruction)


    logging.info(f"document_instruction: {document_instruction}")

    content = document[args.document_col_name]
    id_list = document[args.id_col_name]


    logging.info(f"dataset: {args.dataset}")
    logging.info(f"type: {component_type}")
    logging.info(f"is_multi_content: {args.is_multi_content}")
    logging.info(f"{len(content)} documents found")
    logging.info("="*100)

    # build index:id dictionary

    # if f"{output_dir}/index_id_dict.pkl" exists, ignore this step
    if not os.path.exists(f"{output_dir}/index_id_dict.pkl"):
        index_id_dict = {i: id_list[i] for i in range(len(id_list))}

        with open(f"{output_dir}/index_id_dict.pkl", "wb") as f:
            pickle.dump(index_id_dict, f)

        logging.info(f"saving index_id_dict")

    else:
        logging.info(f"index_id_dict.pkl already exists")
    logging.info(f"Embedding documents")

    for i in tqdm(range(0, len(content))):
        c = content[i]
        c = document_instruction + c
        tokenized_c = tokenizer.tokenize(c)
        if len(tokenized_c) > MODEL_MAX_LEN_DICT[args.model]:
            tokenized_c = tokenized_c[:MODEL_MAX_LEN_DICT[args.model]]
            c = tokenizer.convert_tokens_to_string(tokenized_c)

        result = model.encode(c, use_tqdm=False)
        embedding = result[0].outputs.embedding
        embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding)
        index.add_with_ids(embedding, np.array([i], dtype=np.int64))

    faiss.write_index(index, f"{output_dir}/{model_name}_{args.index_type}_index.faiss")
    logging.info(f"Indexing done")

def main():
    args = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    embed_and_index(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
