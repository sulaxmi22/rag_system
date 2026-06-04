"""
Simple Retrieval Module (No Docker Required)
In-memory for demo purposes
This replaces Qdrant with a simple in-memory vector store for demonstration.
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi
from config import settings
from bedrock_client import bedrock_client

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """Retrieved document with metadata"""
    content: str
    score: float
    source: str
    doc_id: str
    metadata: Dict[str, Any]
    retrieval_method: str


class InMemoryVectorStore:
    """
    Simple in-memory vector store
    For demo purposes when Docker is not available
    In production, use Qdrant, Pinecone, or Weaviate
    """
    
    def __init__(self):
        self.documents: List[Dict] = []
        self.embeddings: List[List[float]] = []
        logger.info("InMemoryVectorStore initialized")
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to the store"""
        for doc in documents:
            self.documents.append({
                'content': doc['content'],
                'source': doc.get('source', 'unknown'),
                'doc_id': doc.get('doc_id', f"doc_{len(self.documents)}"),
                'metadata': doc.get('metadata', {})
            })
            self.embeddings.append(doc['embedding'])
        
        logger.info(f"Added {len(documents)} documents to in-memory store")
    
    def search(self, query_embedding: List[float], top_k: int = 20) -> List[RetrievedDocument]:
        """Search using cosine similarity"""
        if not self.embeddings:
            return []
        
        try:
            # Convert to numpy arrays
            query_vec = np.array(query_embedding)
            doc_vecs = np.array(self.embeddings)
            
            # Calculate cosine similarity
            similarities = np.dot(doc_vecs, query_vec) / (
                np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
            )
            
            # Get top-k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            documents = []
            for idx in top_indices:
                doc = RetrievedDocument(
                    content=self.documents[idx]['content'],
                    score=float(similarities[idx]),
                    source=self.documents[idx]['source'],
                    doc_id=self.documents[idx]['doc_id'],
                    metadata=self.documents[idx]['metadata'],
                    retrieval_method='vector'
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []


class SimpleRetrievalSystem:
    """
    Simple retrieval system without Docker
    """
    
    def __init__(self):
        self.vector_store = InMemoryVectorStore()
        self.bm25_retriever = BM25Retriever()
        logger.info("SimpleRetrievalSystem initialized (no Docker required)")
    
    def index_documents(self, documents: List[Dict]):
        """Index documents"""
        self.vector_store.add_documents(documents)
        self.bm25_retriever.index_documents(documents)
    
    def retrieve(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 20,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """Retrieve documents using hybrid search"""
        # Vector search
        vector_results = self.vector_store.search(query_embedding, top_k)
        
        # BM25 search
        bm25_results = self.bm25_retriever.search(query_text, top_k)
        
        # Simple fusion (interleave results)
        fused = []
        for i in range(max(len(vector_results), len(bm25_results))):
            if i < len(vector_results):
                fused.append(vector_results[i])
            if i < len(bm25_results):
                if bm25_results[i].doc_id not in [d.doc_id for d in fused]:
                    fused.append(bm25_results[i])
        
        return fused[:top_k]


class BM25Retriever:
    """BM25 retriever (same as original)"""
    
    def __init__(self):
        self.documents: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.doc_metadata: List[Dict] = []
    
    def index_documents(self, documents: List[Dict]):
        self.documents = [doc['content'] for doc in documents]
        self.doc_metadata = [doc.get('metadata', {}) for doc in documents]
        self.tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info(f"Indexed {len(documents)} documents for BM25")
    
    def search(self, query: str, top_k: int = 20) -> List[RetrievedDocument]:
        if not self.bm25:
            return []
        
        try:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            documents = []
            for idx in top_indices:
                if scores[idx] > 0:
                    doc = RetrievedDocument(
                        content=self.documents[idx],
                        score=float(scores[idx]),
                        source=self.doc_metadata[idx].get('source', 'unknown'),
                        doc_id=self.doc_metadata[idx].get('doc_id', f"bm25_{idx}"),
                        metadata=self.doc_metadata[idx],
                        retrieval_method='bm25'
                    )
                    documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []


# Global instance
simple_retrieval_system = SimpleRetrievalSystem()
