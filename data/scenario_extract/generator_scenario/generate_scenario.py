from tooldantic import OpenAiResponseFormatBaseModel as BaseModel
from openai import OpenAI, AsyncOpenAI
import openai
import json
import pandas as pd
from argparse import ArgumentParser
import asyncio
from transformers import AutoTokenizer
from tqdm import tqdm
import logging
import pickle
import os
from concurrent.futures import ProcessPoolExecutor

class Explanation(BaseModel):
    information_need: str
    explanation: str

class DocumentAnalysis(BaseModel):  
    main_topic: str
    key_points: list[str]

class Scenario(BaseModel):
    document_analysis: DocumentAnalysis
    information_needs: list[str]
    explanation: list[Explanation]
    
    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--version", type=str, required=True)
    return parser.parse_args()


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



async def generate_scenario_async(config, client: AsyncOpenAI, document: str, dataset=None) -> str:
    prompt = generate_document_scenario_prompt(document, dataset)
    response = await client.chat.completions.create(
        model=config.model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16384,
        # temperature=0.0,
        timeout=300,
        # response_format=Scenario.model_json_schema(),
    )
    return response.choices[0].message.content

async def process_documents(config, client: AsyncOpenAI, documents: pd.DataFrame) -> list:
    results = []
    semaphore = asyncio.Semaphore(config.batch_size * 2)  
    total = len(documents)
    completed = 0
    progress_bar = tqdm(total=total, desc="Processing documents")
    
    async def process_document(row):
        nonlocal completed
        async with semaphore:
            try:
                scenario = await generate_scenario_async(config, client, row["content"], config.dataset)
                result = {"docid": row["id"], "scenario": scenario}
            except Exception as e:
                tqdm.write(f"Error processing document {row['id']}: {e}")
                result = {"docid": row["id"], "scenario": None, "error": str(e)}
            
            completed += 1
            progress_bar.update(1)
            return result
    
    tasks = [process_document(row) for _, row in documents.iterrows()]
    results = await asyncio.gather(*tasks)
    progress_bar.close()
    return [r for r in results if r is not None]


def truncate_document(tokenizer: AutoTokenizer, content: str) -> str:
    tokenized_document = tokenizer.tokenize(content)
    token_len = len(tokenized_document)
    if token_len > 8192:
        # truncate content
        tokenized_document = tokenized_document[:8192]
        content = tokenizer.convert_tokens_to_string(tokenized_document)

    return content

def main(config):
    openai.log = "none" 
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    
    upper_benchmark = config.benchmark.upper()
    document_path = f"../data/{config.benchmark}/{upper_benchmark}/{config.dataset}/document.parquet"
    document = pd.read_parquet(document_path)
    

    
    logging.info("info test")
    logging.warning(f"Loading tokenizer from {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    document["content"] = document["content"].apply(lambda x: truncate_document(tokenizer, x))

    output_path = f"./{config.version}/{config.benchmark}/{config.dataset}"
    os.makedirs(output_path, exist_ok=True)
    output_file_path = f"{output_path}/generator_scenario.parquet"
    error_file_path = f"{output_path}/error_scenario_docid.parquet"
    raw_file_path = f"{output_path}/raw_scenario.parquet"
    logging.warning(f"Output path: {output_path}")
    

    client = AsyncOpenAI(
        api_key="vllmserverkey",
        base_url=config.base_url
    )

    
    logging.warning(f"Processing documents")
    results = asyncio.run(process_documents(config, client, document))


    logging.warning(f"Processing results")
    result_data = {
        "main_topic": [],
        "key_points": [],
        "information_need": [],
        "explanation": [],
        "doc_id": []
    }
    error_list = []
    raw_docid_list = []
    raw_list = []
    
    with tqdm(total=1, desc="Processing results") as pbar:
        for result in results:
            docid = result["docid"]
            scenario = result["scenario"]
            
            raw_docid_list.append(docid)
            raw_list.append(scenario)
            
            if scenario is None:
                error_list.append(docid)
                continue
                
            try:
                json_result = json.loads(scenario)
                
                m = json_result["document_analysis"]["main_topic"]
                k = json_result["document_analysis"]["key_points"]

                for explanation in json_result["explanation"]:
                    i = explanation["information_need"]
                    e = explanation["explanation"]
                    
                    result_data["main_topic"].append(m)
                    result_data["key_points"].append(k)
                    result_data["information_need"].append(i)
                    result_data["explanation"].append(e)
                    result_data["doc_id"].append(docid)
                    
            except Exception as e:
                error_list.append(docid)
        
        result_df = pd.DataFrame(result_data)
        pbar.update(1)
    
    # sava error list
    error_df = pd.DataFrame({
        "doc_id": error_list,
    })

    error_df.to_parquet(error_file_path, index=False)
    raw_df = pd.DataFrame({
        "doc_id": raw_docid_list,
        "scenario": raw_list
    })

    raw_df.to_parquet(raw_file_path, index=False)
    try:
        result_df.to_parquet(output_file_path, index=False)
    except:
        result_df = result_df.apply(lambda x: x.str.encode('utf-8', 'ignore').decode('utf-8') if x.dtype == 'object' else x)
        result_df.to_parquet(output_file_path, index=False)
        
    if error_list:
        tqdm.write(f"Errors occurred for {len(error_list)} documents")
    
    logging.warning(f"#Error: {len(error_list)}")
    logging.warning(f"Error list: {error_list}")
    
    
    
    
if __name__ == "__main__":
    config = parse_args()
    logging.basicConfig(level=logging.WARNING)
    main(config)