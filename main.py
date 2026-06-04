"""
Main Application - FastAPI Server
This ties all components together into a working API
This is the entry point for the RAG system.
"""

import logging
import time
import json
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import all components
from config import settings
from gateway import gateway
from query_processor import query_processor
from retrieval import retrieval_system
from reranker import reranker
from llm_router import llm_router
from validator import validator
from cache import cache
from ingestion import ingestion_pipeline
from streaming import streaming_handler

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str
    user_id: str
    stream: bool = False
    conversation_history: Optional[list] = None


class IngestRequest(BaseModel):
    """Request model for ingestion endpoint"""
    text: str
    source: str
    doc_id: Optional[str] = None
    metadata: Optional[Dict] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    response: str
    model_used: str
    complexity: str
    confidence: float
    validation: Dict[str, Any]
    cached: bool
    latency_ms: float


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    components: Dict[str, str]


class StatsResponse(BaseModel):
    """Response model for statistics"""
    routing_stats: Dict[str, Any]
    cache_stats: Dict[str, Any]


class QuestionAnswerPair(BaseModel):
    """Structured question-answer pair"""
    question: str
    answer: str


class AnswerQuestionsResponse(BaseModel):
    """Response model for answer-questions endpoint"""
    question_answer_pairs: List[QuestionAnswerPair]
    total_questions: int
    model_used: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown
    Initialize resources on startup, cleanup on shutdown
    """
    logger.info("Starting RAG system...")
    # Initialize any resources here
    yield
    logger.info("Shutting down RAG system...")


# Create FastAPI app
app = FastAPI(
    title="Production RAG System",
    description="Retrieval-Augmented Generation system",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files for web interface
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """
    Redirect to web interface
    Simple web interface for real-time Q&A
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Simple health check for monitoring
    """
    return HealthResponse(
        status="healthy",
        components={
            "gateway": "ok",
            "query_processor": "ok",
            "retrieval": "ok",
            "reranker": "ok",
            "llm_router": "ok",
            "validator": "ok",
            "cache": "ok" if cache.enabled else "disabled"
        }
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get system statistics
    Useful for monitoring and demo
    """
    return StatsResponse(
        routing_stats=llm_router.get_routing_stats(),
        cache_stats=cache.get_stats()
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main query endpoint - complete RAG pipeline
    This demonstrates the full pipeline
    """
    start_time = time.time()
    
    try:
        # Step 1: Gateway check (Component #1)
        logger.info(f"Processing query for user {request.user_id}: {request.query[:50]}...")
        gateway_result = gateway.process_request(
            query=request.query,
            user_id=request.user_id
        )
        
        if not gateway_result.is_allowed:
            raise HTTPException(
                status_code=429 if gateway_result.retry_after else 403,
                detail=gateway_result.error_message
            )
        
        query = gateway_result.scrubbed_query
        
        # Step 2: Check JSON document cache for direct Q&A lookup
        json_cache = cache.get(query="json_documents", user_id="system")
        if json_cache:
            json_documents = json_cache['json_documents']
            # Search for matching question in all JSON documents
            for json_doc in json_documents:
                json_data = json_doc['json_data']
                if isinstance(json_data, list):
                    for item in json_data:
                        if isinstance(item, dict) and 'question' in item and 'answer' in item:
                            if item['question'].lower() in query.lower() or query.lower() in item['question'].lower():
                                logger.info(f"Found direct answer in JSON document: {json_doc['filename']}")
                                return QueryResponse(
                                    response=item['answer'],
                                    model_used="json_direct",
                                    complexity="simple",
                                    confidence=1.0,
                                    validation={},
                                    cached=True,
                                    latency_ms=(time.time() - start_time) * 1000
                                )
        
        # Step 3: Check cache (Component #7)
        cached_result = cache.get(query, user_id=request.user_id)
        if cached_result:
            logger.info("Cache hit")
            return QueryResponse(
                response=cached_result['response'],
                model_used=cached_result.get('model_used', 'cached'),
                complexity=cached_result.get('complexity', 'unknown'),
                confidence=cached_result.get('confidence', 1.0),
                validation=cached_result.get('validation', {}),
                cached=True,
                latency_ms=(time.time() - start_time) * 1000
            )
        
        # Step 3: Query processing (Component #2)
        processed = query_processor.process(
            query=query,
            conversation_history=request.conversation_history
        )
        
        if not processed['should_retrieve']:
            # Handle out-of-scope or escalation
            if processed['intent'] == 'out_of_scope':
                response = "I'm sorry, but that question is outside the scope of my knowledge base."
            elif processed['intent'] == 'escalate':
                response = "I'll connect you with a human agent who can better assist you."
            elif processed['intent'] == 'chitchat':
                response = "Hello! How can I help you today?"
            else:
                response = "I couldn't process your request."
            
            return QueryResponse(
                response=response,
                model_used="none",
                complexity="unknown",
                confidence=0.0,
                validation={},
                cached=False,
                latency_ms=(time.time() - start_time) * 1000
            )
        
        # Step 4: Retrieval (Component #3)
        documents = retrieval_system.retrieve(
            query_embedding=processed['embedding'],
            query_text=processed['final_query'],
            top_k=settings.top_k_retrieval,
            user_id=request.user_id
        )
        
        if not documents:
            response = "I couldn't find relevant information in my knowledge base to answer your question."
            return QueryResponse(
                response=response,
                model_used="none",
                complexity="unknown",
                confidence=0.0,
                validation={},
                cached=False,
                latency_ms=(time.time() - start_time) * 1000
            )
        
        # Step 5: Re-ranking (Component #4)
        top_docs = reranker.rerank(
            query=query,
            documents=documents,
            top_n=settings.top_k_final
        )
        
        # Step 6: LLM generation (Component #5)
        if request.stream:
            # Streaming response (Component #8)
            return streaming_handler.create_fastapi_streaming_response(
                prompt=query,
                model_id=settings.openai_model,  # Use OpenAI model for streaming
                max_tokens=1024,
                temperature=0.0
            )
        else:
            llm_result = llm_router.route(
                query=query,
                context_docs=top_docs,
                conversation_history=request.conversation_history,
                stream=False
            )
            
            response = llm_result['response']
            
            # Step 7: Validation (Component #6)
            validation_result = validator.validate(
                response=response,
                context_docs=top_docs,
                query=query
            )
            
            if not validation_result.is_valid:
                logger.warning(f"Validation failed: {validation_result.error_message}")
                # In production, you might retry here
                # For demo, we return the response anyway with validation info
                # response = "I couldn't find a reliable answer based on the available information."
            
            # Step 8: Cache response
            cache.set(
                query=query,
                response_data={
                    'response': response,
                    'model_used': llm_result['model_used'],
                    'complexity': llm_result['complexity'],
                    'confidence': llm_result['confidence'],
                    'validation': validation_result.__dict__
                },
                user_id=request.user_id
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return QueryResponse(
                response=response,
                model_used=llm_result['model_used'],
                complexity=llm_result['complexity'],
                confidence=llm_result['confidence'],
                validation=validation_result.__dict__,
                cached=False,
                latency_ms=latency_ms
            )
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest text into the knowledge base
    Demonstrates the ingestion pipeline (Component #9)
    """
    try:
        doc_id = ingestion_pipeline.ingest_text(
            text=request.text,
            source=request.source,
            doc_id=request.doc_id,
            metadata=request.metadata or {}
        )
        
        # Clear cache after ingestion
        background_tasks.add_task(cache.clear_all)
        
        return {
            "status": "success",
            "doc_id": doc_id,
            "message": "Document ingested successfully"
        }
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear")
async def clear_cache():
    """
    Clear the cache
    Admin endpoint for cache management
    """
    cache.clear_all()
    return {"status": "success", "message": "Cache cleared"}


@app.post("/stats/reset")
async def reset_stats():
    """
    Reset routing statistics
    Admin endpoint for statistics management
    """
    llm_router.reset_stats()
    return {"status": "success", "message": "Statistics reset"}


@app.post("/upload/document")
async def upload_document(file: UploadFile = File(...), source: str = "upload"):
    """
    Upload document (PDF or JSON) for ingestion
    Support PDF and JSON file uploads
    """
    try:
        # Save uploaded file temporarily
        import tempfile
        import os
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Determine file type and ingest
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension == '.pdf':
            doc_id = ingestion_pipeline.ingest_pdf(
                file_path=temp_file_path,
                source=source,
                metadata={"filename": file.filename}
            )
        elif file_extension == '.json':
            # Store JSON data for direct Q&A lookup
            import json
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Get existing JSON documents from cache
            existing_cache = cache.get(query="json_documents", user_id="system")
            if existing_cache:
                json_documents = existing_cache['json_documents']
            else:
                json_documents = []
            
            # Append new JSON document
            json_documents.append({
                'json_data': json_data,
                'filename': file.filename
            })
            
            # Cache all JSON documents
            cache.set(
                query="json_documents",
                response_data={
                    'json_documents': json_documents
                },
                user_id="system"
            )
            
            doc_id = ingestion_pipeline.ingest_json_file(
                file_path=temp_file_path,
                source=source,
                metadata={"filename": file.filename}
            )
        else:
            # Clean up temp file
            os.unlink(temp_file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Only PDF and JSON are supported."
            )
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        # Clear cache after ingestion
        cache.clear_all()
        
        return {
            "status": "success",
            "doc_id": doc_id,
            "message": f"Document {file.filename} ingested successfully",
            "file_type": file_extension
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/questions")
async def upload_questions(file: UploadFile = File(...)):
    """
    Upload JSON file containing list of questions or Q&A pairs
    Support JSON question file upload with or without answers
    """
    try:
        # Read and parse JSON file
        content = await file.read()
        questions_data = json.loads(content.decode('utf-8'))
        
        # Check if it's a list of Q&A objects (from Excel export)
        if isinstance(questions_data, list) and len(questions_data) > 0:
            if isinstance(questions_data[0], dict) and 'question' in questions_data[0]:
                # This is Q&A format from Excel
                qa_pairs = questions_data
                questions = [item['question'] for item in qa_pairs if 'question' in item]
                
                # Store Q&A pairs in cache for direct lookup
                cache.set(
                    query="uploaded_qa_pairs",
                    response_data={
                        'qa_pairs': qa_pairs,
                        'filename': file.filename
                    },
                    user_id="system"
                )
                
                # Also store just questions for backward compatibility
                cache.set(
                    query="uploaded_questions",
                    response_data={
                        'questions': questions,
                        'filename': file.filename
                    },
                    user_id="system"
                )
                
                return {
                    "status": "success",
                    "message": f"Uploaded {len(qa_pairs)} Q&A pairs from {file.filename}",
                    "question_count": len(questions),
                    "has_answers": True
                }
        
        # Validate format for simple questions list
        if isinstance(questions_data, dict):
            if 'questions' in questions_data:
                questions = questions_data['questions']
            else:
                raise HTTPException(
                    status_code=400,
                    detail="JSON must contain a 'questions' key with a list of questions"
                )
        elif isinstance(questions_data, list):
            questions = questions_data
        else:
            raise HTTPException(
                status_code=400,
                detail="JSON must be either a list of questions or an object with a 'questions' key"
            )
        
        # Store questions in cache for later use
        cache.set(
            query="uploaded_questions",
            response_data={
                'questions': questions,
                'filename': file.filename
            },
            user_id="system"
        )
        
        return {
            "status": "success",
            "message": f"Uploaded {len(questions)} questions from {file.filename}",
            "question_count": len(questions),
            "has_answers": False
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/answer-questions", response_model=AnswerQuestionsResponse)
async def answer_questions():
    """
    Answer all uploaded questions against the ingested document or Q&A pairs
    Process batch questions and return structured Q&A pairs
    Checks JSON Q&A pairs first, then falls back to RAG pipeline for PDF
    """
    try:
        # Get uploaded questions from cache
        cached_questions = cache.get(query="uploaded_questions", user_id="system")
        if not cached_questions:
            raise HTTPException(
                status_code=400,
                detail="No questions uploaded. Please upload questions first using /upload/questions"
            )
        
        questions = cached_questions['questions']
        
        # Check if we have Q&A pairs from Excel/JSON
        cached_qa_pairs = cache.get(query="uploaded_qa_pairs", user_id="system")
        qa_dict = {}
        if cached_qa_pairs:
            # Build a dictionary for quick lookup
            for qa in cached_qa_pairs['qa_pairs']:
                if 'question' in qa and 'answer' in qa:
                    qa_dict[qa['question']] = qa['answer']
        
        # Process each question
        question_answer_pairs = []
        
        for question in questions:
            try:
                # First check if answer exists in uploaded Q&A pairs
                if question in qa_dict:
                    answer = qa_dict[question]
                    question_answer_pairs.append(QuestionAnswerPair(question=question, answer=answer))
                    continue
                
                # Fall back to RAG pipeline
                # Process query through RAG pipeline
                processed = query_processor.process(question)
                
                if not processed['should_retrieve']:
                    answer = "I couldn't process this question."
                    question_answer_pairs.append(QuestionAnswerPair(question=question, answer=answer))
                    continue
                
                # Retrieval
                documents = retrieval_system.retrieve(
                    query_embedding=processed['embedding'],
                    query_text=processed['final_query'],
                    top_k=settings.top_k_retrieval,
                    user_id="system"
                )
                
                if not documents:
                    answer = "I couldn't find relevant information to answer this question."
                    question_answer_pairs.append(QuestionAnswerPair(question=question, answer=answer))
                    continue
                
                # Re-ranking
                top_docs = reranker.rerank(
                    query=question,
                    documents=documents,
                    top_n=settings.top_k_final
                )
                
                # LLM generation
                llm_result = llm_router.route(
                    query=question,
                    context_docs=top_docs,
                    stream=False
                )
                
                answer = llm_result['response']
                question_answer_pairs.append(QuestionAnswerPair(question=question, answer=answer))
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                question_answer_pairs.append(
                    QuestionAnswerPair(question=question, answer=f"Error processing question: {str(e)}")
                )
        
        return AnswerQuestionsResponse(
            question_answer_pairs=question_answer_pairs,
            total_questions=len(questions),
            model_used=settings.openai_model
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    """
    This is how you run the server
    In production, you'd use gunicorn or similar with multiple workers
    """
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,  # Enable for development
        log_level=settings.log_level.lower()
    )
