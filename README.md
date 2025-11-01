# 🧠 RAGVerse: The Complete Repository for Retrieval-Augmented Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Supported-green)
![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-blue)

---

## 📘 Overview

**RAGVerse** is a comprehensive, community-driven repository that explores and implements every concept of **Retrieval-Augmented Generation (RAG)** — from foundational components to advanced research-level systems.

The goal of this repository is to **simplify the complexity of RAG architectures** while offering **hands-on, production-ready implementations** that empower developers, data scientists, and AI researchers to master the entire RAG pipeline.

---

## 🎯 Aim & Objectives

This project aims to:

- Cover **all RAG concepts** — embeddings, retrievers, vector stores, augmentation, generation, and evaluation.
- Demonstrate **multiple RAG architectures** — Basic RAG, Hybrid RAG, Contextual RAG, and Multi-Agent RAG.
- Provide **hands-on notebooks** for each topic with clear explanations.
- Explore **advanced techniques** like query rewriting, reranking, hybrid retrieval, and context optimization.
- Include **evaluation metrics and benchmarks** for performance assessment.
- Share **summaries of key research papers** and novel methods in the RAG domain.
- Offer **deployment-ready examples** using FastAPI, Docker, and Streamlit.
- Encourage **open collaboration** to advance practical RAG implementations and learning.

---









🧠 What is RAG?

  RAG (Retrieval-Augmented Generation) is a framework that combines information retrieval and text generation to make Large Language Models (LLMs) more accurate, up-to-date, and explainable.

In simple terms:

Instead of expecting the LLM to “know everything,” we retrieve relevant external knowledge and augment the LLM’s input with it before generating the final answer.

⚙️ The Core Idea

Traditional LLMs (like GPT, Claude, etc.) are trained on fixed data — they can’t know anything after their last training cutoff.

So, if you ask:

“Summarize yesterday’s Google earnings call.”

→ The base model can’t answer, because that info wasn’t part of its training data.

RAG fixes this by doing two main things:

Retrieve: Fetch the most relevant documents/passages from a knowledge source (database, PDF, website, etc.)

Generate: Pass both the user’s question and the retrieved context into the LLM, so it can generate an informed, grounded answer.



🧰 Components of a RAG System:

| Step | Component          | Description                                     | Example Tools                           |
| ---- | ------------------ | ----------------------------------------------- | --------------------------------------- |
| 1️⃣  | **Data Ingestion** | Load raw data (PDFs, CSVs, web pages, etc.)     | LangChain loaders, LlamaIndex           |
| 2️⃣  | **Chunking**       | Split large documents into small passages       | LangChain text splitter                 |
| 3️⃣  | **Embeddings**     | Convert text chunks into vector representations | OpenAI Embeddings, SentenceTransformers |
| 4️⃣  | **Vector Store**   | Store embeddings for fast similarity search     | FAISS, Chroma, Weaviate, Pinecone       |
| 5️⃣  | **Retriever**      | Find top-k most relevant chunks                 | Semantic similarity search              |
| 6️⃣  | **Augmentation**   | Combine context with query                      | Prompt templates                        |
| 7️⃣  | **Generation**     | Generate grounded answer                        | GPT-4, Claude, Mistral                  |
| 8️⃣  | **Evaluation**     | Measure factual accuracy, faithfulness          | RAGAS, Trulens                          |






# Retrieval-Augmented Generation (RAG) System Types by Design Strategy
## Retrieval Strategy

Systems in this category differ by how they perform document retrieval, such as single‑hop vs multi‑hop search, query rewriting, or graph-based lookup
arxiv.org
aiengineering.academy
. They include variants that retrieve evidence in one go or iteratively refine their search. Examples of RAG types by retrieval strategy include:

Standard RAG (original single-pass retrieval model)
homayounsrp.medium.com

Naive RAG (basic retrieval without enhancements)
medium.com

Simple RAG (with memory) (RAG augmented by a memory cache)

Multi-Hop RAG
medium.com
 (retrieves and reasons over multiple evidence pieces)

Graph RAG (uses knowledge graphs or structured data for retrieval)
marktechpost.com

RAG-Sequence
wandb.ai
 (uses the same retrieved document for each token)

RAG-Token
wandb.ai
 (retrieves a new document for each token)

Query Transformation RAG (rewrites or augments the query before retrieval)

Self Query RAG (uses intermediate model outputs as new queries)

RAG Fusion
homayounsrp.medium.com
 (combines information from multiple retrieved documents)

HyDE RAG (uses Hypothetical Document Embeddings for retrieval)

Adaptive RAG (dynamically adjusts retrieval based on context)

Branched RAG (splits retrieval into multiple sub-queries in parallel)

Sentence Window RAG (retrieves with sliding window of context)
aiengineering.academy

Auto Merging RAG (dynamically merges retrieved segments)
aiengineering.academy

Hybrid RAG (combines sparse and dense retrieval methods)

RAPTOR (a method to refine retrieved passages)

CAG (Cache-Augmented Generation) (uses a cache to speed up retrieval)

## Retrieval Models

This category groups RAG systems by the underlying retrieval mechanism, e.g. lexical vs. semantic search
blog.gopenai.com
. Examples include RAG variants that use different retrievers or indexing strategies:

BM25 RAG
arxiv.org
 (uses BM25 sparse retrieval)

TF-IDF RAG
arxiv.org
 (uses TF–IDF vector matching)

ES-RAG (Elasticsearch-based retrieval)
arxiv.org

Dense RAG (uses dense vector retrievers like DPR)

DPR RAG (Dense Passage Retrieval)

ColBERT RAG
aiengineering.academy
 (uses the ColBERT dense retrieval model)

Multi-vector RAG (combines multiple vector indices)

Cross-Encoder RAG (uses a cross-attention retriever model)

Graph-Retriever RAG (queries over a knowledge graph)

Hybrid RAG (integrates both sparse and dense retrievers)

## Optimization Techniques

These RAG types focus on novel training or optimization methods for the retriever and/or generator. They include approaches that train retrieval end-to-end or refine output. Notable examples are:

RAG-DDR (Differentiable Data Rewards)
arxiv.org
 – jointly trains retriever and generator via differentiable rewards

D-RAG (Differentiable RAG)
openreview.net
 – an end-to-end differentiable RAG framework

Reinforced RAG (using reinforcement learning to tune retrieval)

Distilled RAG (RAG with knowledge distillation)

Fine-tuned RAG (RAG system with joint or sequential fine-tuning of components)

Retriever-Fine-Tuned RAG (specialized training of the retriever component)

Self-Training RAG (iteratively uses its own outputs to refine training)

## Reasoning Capabilities

This category covers RAG variants that incorporate enhanced reasoning or multi-step processing, such as self-reflection and chain-of-thought. For example, Self-Reflective RAG (Self-RAG) uses self-critique
arxiv.org
, and CoT-RAG embeds chain-of-thought planning
arxiv.org
. Key types include:

Self-RAG (Self-Reflective RAG)
arxiv.org
 – uses the model’s own critique for refinement

Speculative RAG
homayounsrp.medium.com
 – generates multiple answer candidates and selects the best

Corrective RAG
homayounsrp.medium.com
 – detects and corrects errors in its answers

Agentic RAG
homayounsrp.medium.com
 – embeds an autonomous agent that plans actions (e.g. tool use)

CoT-RAG (Chain-of-Thought RAG)
arxiv.org
 – explicitly reasons step-by-step with planning

Branched RAG – explores multiple reasoning branches in parallel

Dialogue RAG – designed for multi-turn conversation contexts

Recursive RAG – re-invokes retrieval on intermediate answers for deeper inference

## Multi-modality

Systems here can handle non-text data. Multimodal RAG variants retrieve and generate across text, images, audio, etc
ibm.com
. Examples include:

Multimodal RAG
ibm.com
 – a general framework for text+image+audio/video input

Vision RAG – RAG for image-grounded QA (e.g. image captioning with retrieval)

VideoRAG
arxiv.org
 – retrieves from and generates with video content

Path-RAG – focuses on retrieving relevant regions from pathology images

WavRAG
arxiv.org
 – integrates raw audio retrieval into RAG (spoken dialogue)

## Specialization

Domain-specific RAG systems tuned for particular fields fall here. Examples include RAG systems built for industries or topics:

LexRAG
arxiv.org
 – RAG benchmark for legal consultation (LexiT toolkit)

MedRAG
arxiv.org
 – RAG system for medical QA (MIRAGE benchmark)

FinanceRAG – RAG specialized on financial documents (e.g. Kaggle FinanceRAG)

BioRAG – a biomedical domain RAG (using scientific literature)

EduRAG – for educational content retrieval and tutoring

CustomerSupport RAG – tuned for helpdesk QA and knowledge bases

Each category above is defined by its retrieval or processing strategy
arxiv.org
ibm.com
. The listed RAG variants (names only) illustrate the diversity of approaches found in recent literature and practice. The examples span original methods (e.g. RAG-Token/Sequence
wandb.ai
) and newer proposals (e.g. WavRAG
arxiv.org
, VideoRAG
arxiv.org
, LexRAG
arxiv.org
) as of late 2025.
