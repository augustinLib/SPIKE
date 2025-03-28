import pandas as pd
import numpy as np
import faiss
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import argparse
from argparse import ArgumentParser
from transformers import AutoTokenizer
from vllm import LLM
import os
import logging
import pickle

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


def get_config():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="intfloat/e5-mistral-7b-instruct")
    parser.add_argument("--device", type=str, default="0")
    # about benchmark
    parser.add_argument("--benchmark", type=str, default="bright")
    parser.add_argument("--dataset", type=str, default="biology")
    # about id column name
    parser.add_argument("--id_col_name", type=str, default="id")
    # about query type
    parser.add_argument("--query_type", type=str, default="original_query")
    parser.add_argument("--document_alpha", type=float, default=0.5)
    # about document type
    parser.add_argument("--document_type", type=str, default="O")
    parser.add_argument("--is_multi_content", type=str, default="False")
    # about instruction type
    parser.add_argument("--instruction_type", type=str, default="default")
    
    parser.add_argument("--generator_type", type=str, default="False")

    # about version
    parser.add_argument("--version", type=str, default="default")
    # about analysis mode
    parser.add_argument("--analysis_mode", type=bool, default=False)
    # about index type
    parser.add_argument("--index_type", type=str, default="flat")
    # about k
    parser.add_argument("--k", type=int, default=10)


    args = parser.parse_args()
    return args

def refine_instruction(task,instruction):
    return instruction.format(task=task)

    
def calculate_dcg(relevance_scores: List[float], k: int) -> float:
    """Calculate DCG@k"""
    dcg = 0
    for i in range(min(len(relevance_scores), k)):
        dcg += relevance_scores[i] / np.log2(i + 2)
    return dcg

def calculate_ndcg(predicted_ranks: List[int], true_relevance: Dict[str, int], k: int) -> float:
    """Calculate NDCG@k for a single query"""
    # Get relevance scores for predicted ranks
    pred_relevance = [true_relevance.get(str(rank), 0) for rank in predicted_ranks[:k]]
    
    # Calculate ideal DCG
    ideal_relevance = sorted(true_relevance.values(), reverse=True)[:k]
    
    dcg = calculate_dcg(pred_relevance, k)
    idcg = calculate_dcg(ideal_relevance, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def evaluate_dataset(dataset_name: str, 
                    scenario_index_path: str,
                    original_index_path: str,
                    model: LLM,
                    query_path: str,
                    qrel_path: str,
                    scenario_index_id_dict_path: str,
                    original_index_id_dict_path: str,
                    args: argparse.Namespace,
                    k: int = 10,
                    tokenizer: AutoTokenizer = None) -> Dict[str, float]:
    """Evaluate NDCG@k for a dataset using Faiss index"""

    qid_list = []
    top10_docid_list = []
    # original_indices_list = []
    scenario_index_list = []

    # Load Faiss index
    scenario_index = faiss.read_index(scenario_index_path)
    original_index = faiss.read_index(original_index_path)
    

    # Load query 
    logging.info(f"Loading query")
    query_df = pd.read_parquet(query_path)

    # Load index_id_dict
    with open(scenario_index_id_dict_path, "rb") as f:
        scenario_index_id_dict = pickle.load(f)

    with open(original_index_id_dict_path, "rb") as f:
        original_index_id_dict = pickle.load(f)

    model_name = args.model.split("/")[-1]

    with open(f"{data_dir}/embedding_instruction/{model_name}/{args.dataset}.json", "r") as f:
        instruction = json.load(f)

    instruction = instruction["instructions"]
    if  "query" in instruction.keys():
        query_instruction = instruction["query"]
    else:
        query_instruction = ""

    if args.dataset in TASK_MAP.keys():
        task = TASK_MAP[args.dataset]
        query_instruction = refine_instruction(task,query_instruction)

    instruction = query_instruction

    logging.info(f"instruction: {instruction}")

    # extract query embedding
    query_embeddings = []

    logging.info(f"Extracting query embeddings")
    for _, row in tqdm(query_df.iterrows()):            
        query = row['query']
        instructed_query = instruction + query
        tokenized_query = tokenizer.tokenize(instructed_query)
        if len(tokenized_query) > MODEL_MAX_LEN_DICT[args.model]:
            tokenized_query = tokenized_query[:MODEL_MAX_LEN_DICT[args.model]]
            instructed_query = tokenizer.convert_tokens_to_string(tokenized_query)

        query_embedding = model.encode(instructed_query, use_tqdm=False)
        query_embeddings.append(query_embedding[0].outputs.embedding)

    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    
    logging.info(f"query embeddings shape: {query_embeddings.shape}")
    # Load qrels
    qrel_df = pd.read_parquet(qrel_path)
    
    logging.info(f"qrel df loaded")
    
    # Convert qrels to dictionary format {query_id: {doc_id: relevance}}
    qrels_dict = {}
    for _, row in qrel_df.iterrows():
        if row['query-id'] not in qrels_dict:
            qrels_dict[row['query-id']] = {}
        qrels_dict[row['query-id']][str(row['corpus-id'])] = row['score']
    
    # Calculate NDCG@k for each query
    ndcg_scores = {}
    
    for i, query_embedding in tqdm(enumerate(query_embeddings)):
        # Search using Faiss
        # logging.info(f"query_embedding shape: {query_embedding.shape}")
        normalize_query_embedding = query_embedding.reshape(1, -1)
        
        # normalize query embedding for cosine similarity
        # normalize + inner product = cosine similarity
        faiss.normalize_L2(normalize_query_embedding)

        excluded_ids = query_df.loc[i, "excluded_ids"]
        if len(excluded_ids) > 0:
            first_k = k * 50 * 30
        else:
            first_k = k * 30
        # for scenario document, search more than k
        # and remove duplicate, remain only top docid

        SCE_scores, SCE_I = scenario_index.search(normalize_query_embedding, first_k * 10)
        O_scores, O_I = original_index.search(normalize_query_embedding, first_k)

        query_id = query_df.loc[i, "query-id"]
        
        # convert I to corpus-id
        SCE_corpus_ids = [str(scenario_index_id_dict[i]) for i in SCE_I[0]]  
        O_corpus_ids = [str(original_index_id_dict[i]) for i in O_I[0]]  

        # Create lists of (score, corpus_id) tuples
        SCE_corpus_score_id = list(zip(SCE_scores[0], SCE_corpus_ids))
        O_corpus_score_id = list(zip(O_scores[0], O_corpus_ids))

        # 1. Remove duplicates in SCE_corpus_score_id, keeping only the highest score for each id
        SCE_best_scores = {}
        SCE_best_scores_scenario_index= {}
        for i, (score, corpus_id) in enumerate(SCE_corpus_score_id):
            if corpus_id not in SCE_best_scores or score > SCE_best_scores[corpus_id]:
                SCE_best_scores[corpus_id] = score
                SCE_best_scores_scenario_index[corpus_id] = SCE_I[0][i]

        # Convert back to list of tuples
        SCE_deduplicated = [(score, corpus_id, SCE_best_scores_scenario_index[corpus_id]) for corpus_id, score in SCE_best_scores.items()]

        
        # 2. Create a dictionary for O scores for easy lookup
        O_scores_dict = {corpus_id: score for score, corpus_id in O_corpus_score_id}
        
        # Combine scores for overlapping IDs and create the final list
        final_scores = []

        # Process SCE items first
        for score, corpus_id, index in SCE_deduplicated:
            if corpus_id in O_scores_dict.keys():
                combined_score = (score * (1 - args.document_alpha) + O_scores_dict[corpus_id] * args.document_alpha)
                final_scores.append((combined_score, corpus_id, index))
                # Remove from O dictionary to avoid processing twice
                del O_scores_dict[corpus_id]


        if len(final_scores) < k:
            raise ValueError(f"len(final_scores) < k: {len(final_scores)} < {k}")
        
        # 3. Sort by score in descending order
        # in python, when we sort a list of tuples, the sorting is done based on the first element of each tuple by default.
        final_scores.sort(reverse=True)

        corpus_ids_and_index = [(corpus_id, index) for _, corpus_id, index in final_scores]
        
        # 4. Remove excluded IDs
        corpus_ids_and_index = [corpus_id_and_index for corpus_id_and_index in corpus_ids_and_index if corpus_id_and_index[0] not in excluded_ids]
        

        if len(corpus_ids_and_index) < k:
            raise ValueError(f"len(corpus_ids_and_index) < k: {len(corpus_ids_and_index)} < {k}")
        # 5. Truncate to k
        corpus_ids_and_index = corpus_ids_and_index[:k]

        corpus_ids = [corpus_id_and_index[0] for corpus_id_and_index in corpus_ids_and_index]
        scenario_indices = [corpus_id_and_index[1] for corpus_id_and_index in corpus_ids_and_index]


        if args.analysis_mode:
            if args.is_multi_content == "False":
                qid_list.append(query_id)
                top10_docid_list.append(corpus_ids)
                scenario_index_list.append(scenario_indices)

            elif args.is_multi_content == "True":
                qid_list.append(query_id)
                top10_docid_list.append(corpus_ids)
                scenario_index_list.append(scenario_indices)
            else:   
                raise ValueError(f"Invalid is_multi_content: {args.is_multi_content}")

        # Calculate NDCG@k
        if str(query_id) in qrels_dict:
            ndcg = calculate_ndcg(corpus_ids, qrels_dict[str(query_id)], k)
            ndcg_scores[str(query_id)] = ndcg
    
    # Calculate mean NDCG@k
    mean_ndcg = np.mean(list(ndcg_scores.values()))
    
    if args.analysis_mode:
        if args.is_multi_content == "False":
            return {
                'ndcg_scores': ndcg_scores,
                'mean_ndcg': mean_ndcg,
                'qid_list': qid_list,
                'top10_docid_list': top10_docid_list,
                'scenario_indices_list': scenario_index_list
            }
        elif args.is_multi_content == "True":
            return {
                'ndcg_scores': ndcg_scores,
                'mean_ndcg': mean_ndcg,
                'qid_list': qid_list,
                'top10_docid_list': top10_docid_list,
                'scenario_indices_list': scenario_index_list,
            }
        else:
            raise ValueError(f"Invalid is_multi_content: {args.is_multi_content}")
    else:
        return {
            'ndcg_scores': ndcg_scores,
            'mean_ndcg': mean_ndcg,
        }

if __name__ == "__main__":
    args = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    logging.basicConfig(level=logging.INFO)
    model_name = args.model.split("/")[-1]


    result_dir = f"./result/{args.benchmark}/{args.dataset}/{model_name}/{args.query_type}/{args.generator_type}/{args.version}/{args.document_type}/{args.instruction_type}_instruction/{args.document_alpha}"
    os.makedirs(result_dir, exist_ok=True)

    logging.info(f"Loading model")
    model = LLM(
        model=args.model,
        device="cuda:0",
        seed=42,
        trust_remote_code=True,
        # enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Example usage
    benchmark = args.benchmark
    dataset_name = args.dataset  # or any other BEIR dataset
    
    # convert uppercase
    capitalized_benchmark_name = benchmark.upper()
    
    query_type_for_path = "_".join(args.query_type.split("_")[:-1])

    document_type_for_path = args.document_type
    
    version_for_index_path = args.version.split("(")[0]

    data_dir = "../../data"
    
    scenario_index_path = f"{data_dir}/embedding/{benchmark}/{dataset_name}/{args.generator_type}/{version_for_index_path}/{document_type_for_path}/{model_name}_{args.index_type}_index.faiss"
    original_index_path = f"{data_dir}/embedding/{benchmark}/{dataset_name}/no_generator/v0/O/{model_name}_{args.index_type}_index.faiss"
    
    if args.query_type == "original_query":
        query_path = f"{data_dir}/{benchmark}/{capitalized_benchmark_name}/{dataset_name}/query.parquet"
    elif args.query_type == "gpt4_reason":
        query_path = f"{data_dir}/{benchmark}/{capitalized_benchmark_name}/{dataset_name}/gpt4_reason_query.parquet"
    
    qrel_path = f"{data_dir}/{benchmark}/{capitalized_benchmark_name}/{dataset_name}/qrel.parquet"
    scenario_index_id_dict_path = f"{data_dir}/embedding/{benchmark}/{dataset_name}/{args.generator_type}/{version_for_index_path}/{document_type_for_path}/index_id_dict.pkl"
    original_index_id_dict_path = f"{data_dir}/embedding/{benchmark}/{dataset_name}/no_generator/v0/O/index_id_dict.pkl"

    results = evaluate_dataset(
        dataset_name=dataset_name,
        scenario_index_path=scenario_index_path,
        original_index_path=original_index_path,
        model=model,
        query_path=query_path,
        qrel_path=qrel_path,
        scenario_index_id_dict_path=scenario_index_id_dict_path,
        original_index_id_dict_path=original_index_id_dict_path,
        args=args,
        k=args.k,
        tokenizer=tokenizer
    )

    # save ndcg scores with text
    mean_ndcg = results['mean_ndcg']
    with open(f"{result_dir}/ndcg10_scores.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Benchmark: {benchmark}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Query Type: {args.query_type}\n")
        f.write(f"Document Type: {args.document_type}\n")
        f.write(f"Instruction Type: {args.instruction_type}\n")
        f.write(f"Version: {args.version}\n")
        f.write(f"Document Alpha: {args.document_alpha}\n")
        f.write("\n")
        f.write(f"Mean NDCG@10 for {dataset_name}: {mean_ndcg:.4f}\n")

    # save result
    if args.analysis_mode:
        if args.is_multi_content == "True":
            qid_list = results['qid_list']
            top10_docid_list = results['top10_docid_list']
            scenario_indices_list = results['scenario_indices_list']

            analysis_df = pd.DataFrame({
                'query-id': qid_list,
                'top10_docid': top10_docid_list,    
                'scenario_indices': scenario_indices_list
            })

        elif args.is_multi_content == "False":
            qid_list = results['qid_list']
            top10_docid_list = results['top10_docid_list']

            analysis_df = pd.DataFrame({
                'query-id': qid_list,
                'top10_docid': top10_docid_list
            })
            
        else:
            raise ValueError(f"Invalid is_multi_content: {args.is_multi_content}")

        analysis_df.to_parquet(f"{result_dir}/analysis_df.parquet", index=False)
    
    logging.info(f"Mean NDCG@10 for {dataset_name}: {results['mean_ndcg']:.4f}")