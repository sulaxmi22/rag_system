"""
Offline Document Ingestion Pipeline
Bottom of the template - Document Ingestion Pipeline (Parse · Chunk · Embed · Vector DB)
This module handles the offline pipeline for processing and indexing documents.
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import hashlib
import uuid
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_client import langchain_client
from retrieval import retrieval_system
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Document structure
    Standardized document format
    """
    content: str
    source: str
    doc_id: str
    metadata: Dict[str, Any]


class DocumentParser:
    """
    Parses documents from various formats
    Handles PDF, Word, and plain text
    Support multiple document formats
    """
    
    def __init__(self):
        logger.info("DocumentParser initialized")
    
    def parse_pdf(self, file_path: str) -> str:
        """
        Parse PDF document
        Extracts text from PDF
        """
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            logger.error("pypdf not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise
    
    def parse_word(self, file_path: str) -> str:
        """
        Parse Word document
        Extracts text from Word
        """
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            logger.error("python-docx not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to parse Word document: {e}")
            raise
    
    def parse_text(self, file_path: str) -> str:
        """
        Parse plain text document
        Simple text file reading
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to parse text file: {e}")
            raise
    
    def parse(self, file_path: str) -> str:
        """
        Parse document based on file extension
        Auto-detects format
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == '.pdf':
            return self.parse_pdf(file_path)
        elif extension in ['.docx', '.doc']:
            return self.parse_word(file_path)
        elif extension in ['.txt', '.md']:
            return self.parse_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")


class DocumentChunker:
    """
    Chunks documents into smaller pieces
    256-512 tokens with 20% overlap
    Smaller chunks are more precise for retrieval
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: float = 0.2):
        self.chunk_size = chunk_size
        self.chunk_overlap = int(chunk_size * chunk_overlap)
        
        # Use LangChain's text splitter
        # Tries to break at natural boundaries
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(f"DocumentChunker initialized (size: {chunk_size}, overlap: {self.chunk_overlap})")
    
    def chunk(self, document: Document) -> List[Dict]:
        """
        Chunk document into pieces
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunk dictionaries
        """
        try:
            chunks = self.splitter.split_text(document.content)
            
            chunked_docs = []
            for idx, chunk_text in enumerate(chunks):
                chunk_doc = {
                    'content': chunk_text,
                    'source': document.source,
                    'doc_id': str(uuid.uuid4()),
                    'metadata': {
                        **document.metadata,
                        'chunk_index': idx,
                        'parent_doc_id': document.doc_id
                    }
                }
                chunked_docs.append(chunk_doc)
            
            logger.info(f"Chunked document {document.doc_id} into {len(chunks)} chunks")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Failed to chunk document: {e}")
            raise


class EmbeddingGenerator:
    """
    Generates embeddings for document chunks
    Uses OpenAI embeddings via LangChain
    IMPORTANT: Must track model version for compatibility
    """
    
    def __init__(self):
        self.model_id = settings.openai_embedding_model
        self.model_hash = self._get_model_hash()
        logger.info(f"EmbeddingGenerator initialized (model: {self.model_id})")
    
    def _get_model_hash(self) -> str:
        """
        Generate hash for model version tracking
        Used to ensure query/index compatibility
        """
        return hashlib.md5(self.model_id.encode()).hexdigest()
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for chunks using LangChain
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Chunks with embeddings added
        """
        embedded_chunks = []
        
        # Get embeddings from LangChain
        texts = [chunk['content'] for chunk in chunks]
        try:
            embeddings = langchain_client.get_embeddings().embed_documents(texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding
                chunk['embed_model_hash'] = self.model_hash
                embedded_chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Fall back to individual embedding generation
            for chunk in chunks:
                try:
                    embedding = langchain_client.get_embeddings().embed_query(chunk['content'])
                    chunk['embedding'] = embedding
                    chunk['embed_model_hash'] = self.model_hash
                    embedded_chunks.append(chunk)
                except Exception as e:
                    logger.error(f"Failed to generate embedding for chunk {chunk.get('doc_id')}: {e}")
                    continue
        
        logger.info(f"Generated embeddings for {len(embedded_chunks)}/{len(chunks)} chunks")
        return embedded_chunks


class IngestionPipeline:
    """
    Main ingestion pipeline orchestrator
    Combines parsing, chunking, embedding, and indexing
    """
    
    def __init__(self):
        self.parser = DocumentParser()
        self.chunker = DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.embedding_generator = EmbeddingGenerator()
        
        logger.info("IngestionPipeline initialized")
    
    def ingest_document(
        self,
        file_path: str,
        source: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Ingest a single document
        
        Args:
            file_path: Path to document file
            source: Source identifier
            doc_id: Optional document ID (auto-generated if not provided)
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        # Generate doc_id if not provided
        if not doc_id:
            doc_id = hashlib.md5(file_path.encode()).hexdigest()
        
        if not metadata:
            metadata = {}
        
        # Step 1: Parse document
        logger.info(f"Parsing document: {file_path}")
        content = self.parser.parse(file_path)
        
        document = Document(
            content=content,
            source=source,
            doc_id=doc_id,
            metadata=metadata
        )
        
        # Step 2: Chunk document
        logger.info(f"Chunking document: {doc_id}")
        chunks = self.chunker.chunk(document)
        
        # Step 3: Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embedded_chunks = self.embedding_generator.generate_embeddings(chunks)
        
        # Step 4: Index in retrieval system
        logger.info(f"Indexing {len(embedded_chunks)} chunks")
        
        # Check if we should use simple retrieval system
        from retrieval import _use_simple_retrieval
        if _use_simple_retrieval:
            from retrieval_simple import simple_retrieval_system
            simple_retrieval_system.index_documents(embedded_chunks)
            logger.info("Indexed in simple in-memory retrieval system")
        else:
            try:
                retrieval_system.index_documents(embedded_chunks)
            except Exception as e:
                logger.warning(f"Vector indexing failed, using simple retrieval system: {e}")
                from retrieval_simple import simple_retrieval_system
                simple_retrieval_system.index_documents(embedded_chunks)
        
        logger.info(f"Successfully ingested document: {doc_id}")
        return doc_id
    
    def ingest_text(
        self,
        text: str,
        source: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Ingest text directly (without file)
        
        Args:
            text: Text content
            source: Source identifier
            doc_id: Optional document ID
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        if not doc_id:
            doc_id = hashlib.md5(text.encode()).hexdigest()
        
        if not metadata:
            metadata = {}
        
        document = Document(
            content=text,
            source=source,
            doc_id=doc_id,
            metadata=metadata
        )
        
        # Chunk
        chunks = self.chunker.chunk(document)
        
        # Embed
        embedded_chunks = self.embedding_generator.generate_embeddings(chunks)
        
        # Index
        retrieval_system.index_documents(embedded_chunks)
        
        logger.info(f"Successfully ingested text: {doc_id}")
        return doc_id
    
    def batch_ingest(self, file_paths: List[str], source: str) -> List[str]:
        """
        Batch ingest multiple documents
        
        Args:
            file_paths: List of file paths
            source: Source identifier
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        
        for file_path in file_paths:
            try:
                doc_id = self.ingest_document(file_path, source)
                doc_ids.append(doc_id)
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
                # In production, send to DLQ for retry
                continue
        
        logger.info(f"Batch ingestion complete: {len(doc_ids)}/{len(file_paths)} successful")
        return doc_ids
    
    def ingest_pdf(
        self,
        file_path: str,
        source: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Ingest PDF document using LangChain
        Support PDF file upload
        
        Args:
            file_path: Path to PDF file
            source: Source identifier
            doc_id: Optional document ID
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        if not doc_id:
            doc_id = hashlib.md5(file_path.encode()).hexdigest()
        
        if not metadata:
            metadata = {}
        
        # Load PDF using LangChain
        logger.info(f"Loading PDF: {file_path}")
        documents = langchain_client.load_pdf(file_path)
        
        # Combine all pages into single document
        content = "\n\n".join([doc.page_content for doc in documents])
        
        document = Document(
            content=content,
            source=source,
            doc_id=doc_id,
            metadata={**metadata, "file_type": "pdf"}
        )
        
        # Chunk
        chunks = self.chunker.chunk(document)
        
        # Embed
        embedded_chunks = self.embedding_generator.generate_embeddings(chunks)
        
        # Index
        retrieval_system.index_documents(embedded_chunks)
        
        logger.info(f"Successfully ingested PDF: {doc_id}")
        return doc_id
    
    def ingest_json_file(
        self,
        file_path: str,
        source: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Ingest JSON document using LangChain
        Support JSON file upload
        
        Args:
            file_path: Path to JSON file
            source: Source identifier
            doc_id: Optional document ID
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        if not doc_id:
            doc_id = hashlib.md5(file_path.encode()).hexdigest()
        
        if not metadata:
            metadata = {}
        
        # Load JSON using LangChain
        logger.info(f"Loading JSON: {file_path}")
        documents = langchain_client.load_json(file_path)
        
        # Combine all documents
        content = "\n\n".join([doc.page_content for doc in documents])
        
        document = Document(
            content=content,
            source=source,
            doc_id=doc_id,
            metadata={**metadata, "file_type": "json"}
        )
        
        # Chunk
        chunks = self.chunker.chunk(document)
        
        # Embed
        embedded_chunks = self.embedding_generator.generate_embeddings(chunks)
        
        # Index
        retrieval_system.index_documents(embedded_chunks)
        
        logger.info(f"Successfully ingested JSON: {doc_id}")
        return doc_id


# Global ingestion pipeline instance
ingestion_pipeline = IngestionPipeline()
