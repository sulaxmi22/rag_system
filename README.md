# RAG System - Question-Answering Bot


<img width="1189" height="769" alt="image" src="https://github.com/user-attachments/assets/a572f2bf-aa2b-40f3-80dd-cdab86b21c72" />

<img width="1400" height="826" alt="image" src="https://github.com/user-attachments/assets/fc006073-c26d-409d-a798-0caf2de6b8ff" />

<img width="922" height="575" alt="rag zania" src="https://github.com/user-attachments/assets/7043d064-00af-4ced-a7c3-e7b6d2b8f962" />



## Overview

A production-ready Question-Answering bot that uses Large Language Models to answer questions based on document content. Built with LangChain framework and OpenAI's gpt-4o-mini model, supporting PDF and JSON file uploads for both documents and questions.

**Architecture:**
```
User Query → Gateway → Query Processing → Retrieval → Re-ranker → LLM Router → Validator → Cache → Response
```

**Ingestion Pipeline:**
```
Source Docs (PDF/JSON) → Parse → Chunk → Embed → Vector DB + Metadata
```

## Features

- PDF and JSON document upload support
- JSON question file upload
- Batch question processing with structured Q&A output
- LangChain framework integration
- OpenAI gpt-4o-mini model
- Vector database (Qdrant) for semantic search
- Hybrid search (vector + BM25)
- Response validation and caching

## Installation

### Prerequisites
- Python 3.9+
- OpenAI API key
- Qdrant (for vector storage)
- Redis (optional, for caching)

### Setup

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```


## Running the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## API Endpoints

### POST /upload/document
Upload a PDF or JSON document for ingestion.

**Request:**
- `file`: PDF or JSON file (multipart/form-data)
- `source`: Source identifier (optional, default: "upload")

**Response:**
```json
{
  "status": "success",
  "doc_id": "document_id",
  "message": "Document filename ingested successfully",
  "file_type": ".pdf"
}
```

### POST /upload/questions
Upload a JSON file containing a list of questions.

**Request:**
- `file`: JSON file with questions (multipart/form-data)

**JSON Format:**
```json
{
  "questions": [
    "Question 1?",
    "Question 2?"
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Uploaded 5 questions from questions.json",
  "question_count": 5
}
```

### POST /answer-questions
Answer all uploaded questions against the ingested document.

**Response:**
```json
{
  "question_answer_pairs": [
    {
      "question": "Question 1?",
      "answer": "Answer 1"
    }
  ],
  "total_questions": 5,
  "model_used": "gpt-4o-mini"
}
```

### POST /query
Query the RAG system with a single question.

### POST /ingest
Ingest text directly into the knowledge base.


## Usage Example

### Using the Workflow

1. **Upload a document:**
```bash
curl -X POST "http://localhost:8000/upload/document" \
  -F "file=@document.pdf" \
  -F "source=my_document"
```

2. **Upload questions:**
```bash
curl -X POST "http://localhost:8000/upload/questions" \
  -F "file=@questions.json"
```

3. **Get answers:**
```bash
curl -X POST "http://localhost:8000/answer-questions"
```

### Python Usage

```python
from ingestion import ingestion_pipeline

# Ingest a document
doc_id = ingestion_pipeline.ingest_text(
    text="Our refund policy allows returns within 30 days...",
    source="refund_policy"
)

# Query the system
from main import client
response = client.post("/query", json={
    "query": "What is the refund policy?",
    "user_id": "user123"
})
```

## Technology Stack

- **Framework**: LangChain
- **LLM**: OpenAI gpt-4o-mini
- **Vector Database**: Qdrant
- **Caching**: Redis (optional)
- **API Framework**: FastAPI
- **Document Processing**: PyPDF, LangChain document loaders
- **Embeddings**: OpenAI text-embedding-3-small

## System Components

1. **Gateway**: Authentication, rate limiting, PII scrubbing, injection detection
2. **Query Processing**: Intent classification, query rewriting, embedding generation
3. **Retrieval**: Hybrid search (vector + BM25) with RRF fusion
4. **Re-ranker**: Cross-encoder re-ranking for improved precision
5. **LLM Router**: Routes queries to appropriate models
6. **Validator**: Faithfulness, safety, PII, and format validation
7. **Cache**: Semantic caching with Redis
8. **Streaming**: Server-Sent Events for streaming responses
9. **Ingestion Pipeline**: Document parsing, chunking, embedding, and indexing

