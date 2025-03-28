<p align="center">
  <img src="./asset/github.png" alt="SPIKE Framework Overview" width="100%">
</p>


<p align="center">
  <img src="./asset/main_figure.png" alt="SPIKE Framework Overview" width="80%">
</p>

<p align="center">
  <a href=""><b>Paper</b></a> | <a href="https://huggingface.co/yonsei-dli/Scenario-Generator-only_plain_doc"><b>Hugging Face</b></a>
</p>


This repository contains the implementation of **S**cenario-**P**rofiled **I**ndexing with **K**nowledge **E**xpansion (**SPIKE**), a dense retrieval framework that explicitly indexes potential implicit relevance within documents.

SPIKE reframes document representations into **hypothetical retrieval scenarios**, where each scenario encapsulates the reasoning process required to uncover implicit relevance between a hypothetical information need and the document content. This approach:

1. Enhances retrieval performance by explicitly modeling how a document addresses hypothetical information needs, capturing implicit relevance between query and document.
2. Effectively connects query-document pairs across different formats such as code snippets, enabling semantic alignment despite format differences.
3. Improves the retrieval experience for users by providing useful information while also serving as valuable context for LLMs in RAG settings.

## Implementation
SPIKE is implemented through the following process:

### Scenario Generator
Before generating scenarios, we train our scenario generator model to effectively identify implicit relevance:

1. **Scenario-augmented training data**: Using high-performing LLMs (like GPT-4o) to create high-quality scenarios
2. **Scenario Distillation**: Training a smaller model(Llama-3.2-3B-Instruct) to efficiently produce reasoning-driven scenarios

This process is implemented in `src/scenario_generator/`

### Scenario Generation
We generate hypothetical retrieval scenarios for each document using the trained scenario generator.
This process is implemented in `data/scenario_extract/generator_scenario/generate_scenario.py`.

For efficient scenario generation at scale, we utilize `vllm's OpenAI-compatible server`, which allows us to process multiple documents in parallel using asyncio, handle large batches efficiently, and maintain consistent generation quality while improving throughput. 

We also provide our trained scenario generator model for public use. The model is available at: [Scenario Generator HF Link](https://huggingface.co/yonsei-dli/Scenario-Generator-only_plain_doc)

This model can be used to generate hypothetical retrieval scenarios for your own documents, enabling you to implement the SPIKE framework in your retrieval applications.


### Embedding and Indexing
For efficiency and accuracy, we used different libraries to extract embeddings for different dense retrieval models. Specifically, the models used for each library are as follows:

- **Vllm**: E5-Mistral-7B, SFR, Qwen
- **Sentencetransformer**: SBERT(all-mpnet-base-v2), BGE-Large
- **GritLM**: GritLM

We also use FAISS for efficient similarity search.

The embedding and indexing process is implemented in `data/embedding/`, supporting various embedding models with different dimensions and context length limitations.

### Retrieval with Scenarios
During retrieval, SPIKE combines document-level and scenario-level relevance:

1. **Compute Document-level and Scenario-level scores**: Computing relevance scores between the query and both documents and scenarios
2. **Score Aggregation**: Combining document scores with the maximum scenario score using a weighted sum
3. **Result Ranking**: Producing the final ranked list based on the combined relevance scores

This process is implemented in `src/main_result`.

## Requirements
```
datasets==2.21.0
faiss-gpu-cu12==1.10.0
gritlm==1.0.2
lightning==2.5.1
openai==1.69.0
pyarrow==19.0.1
sentence-transformers==4.0.1
sentencepiece==0.2.0
tokenizers==0.21.1
torch==2.6.0
transformers==4.50.2
vllm==0.8.2
wandb==0.19.8
```