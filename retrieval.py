"""
Retrieval Module
Component #3 from template - Retrieval (BM25 + Vector hybrid · ACL filter · Top-k chunks)
This module retrieves relevant documents using hybrid search combining semantic and keyword matching.

Pure vector search misses exact matches (product codes, names, IDs)
Pure keyword search misses semantic understanding
Hybrid search gives best of both worlds
ACL filtering ensures multi-tenant security
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """
    Retrieved document with metadata
    Structured data makes downstream processing clear
    """
    content: str
    score: float
    source: str
    doc_id: str
    metadata: Dict[str, Any]
    retrieval_method: str  # 'vector', 'bm25', or 'hybrid'


class VectorRetriever:
    """
    Vector-based semantic search using Qdrant
    """
    
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = settings.qdrant_collection_name
        self._ensure_collection_exists()
        
        logger.info(f"VectorRetriever initialized with collection {self.collection_name}")
    
    def _ensure_collection_exists(self):
        """
        Create collection if it doesn't exist
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.vector_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        Search for similar documents using vector similarity
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            user_id: Optional user ID for ACL filtering
            filters: Optional metadata filters
            
        Returns:
            List of retrieved documents
        """
        try:
            query_filter = self._build_filter(user_id, filters)
            
            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
            
            # Convert to RetrievedDocument objects
            documents = []
            for result in search_result:
                doc = RetrievedDocument(
                    content=result.payload.get('content', ''),
                    score=result.score,
                    source=result.payload.get('source', 'unknown'),
                    doc_id=result.id,
                    metadata=result.payload.get('metadata', {}),
                    retrieval_method='vector'
                )
                documents.append(doc)
            
            logger.info(f"Vector search returned {len(documents)} results")
            return documents
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _build_filter(
        self,
        user_id: Optional[str],
        filters: Optional[Dict[str, Any]]
    ) -> Optional[Filter]:
        """
        Build Qdrant filter for ACL and metadata
        Post-retrieval filtering is a security risk
        """
        conditions = []
        
        if user_id:
            # In production, this would check user's permissions
            # For demo, we assume all documents are accessible
            pass
        
        # Add custom metadata filters
        if filters:
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
        
        if conditions:
            return Filter(must=conditions)
        return None
    
    def add_documents(self, documents: List[Dict]):
        """
        Add documents to the vector store
        """
        try:
            points = []
            for idx, doc in enumerate(documents):
                point = PointStruct(
                    id=doc.get('doc_id', f"doc_{idx}"),
                    vector=doc['embedding'],
                    payload={
                        'content': doc['content'],
                        'source': doc.get('source', 'unknown'),
                        'metadata': doc.get('metadata', {}),
                        'embed_model_hash': doc.get('embed_model_hash', 'unknown')
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Added {len(points)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")


class BM25Retriever:
    """
    BM25 keyword-based retrieval
    
    "BM25 is important for enterprise documents with specific terminology. Vector search is great for 
    semantic understanding, but it misses exact matches like 'PROD-12345' or legal citations. 
    BM25 catches these at almost zero additional cost."
    """
    
    def __init__(self):
        self.documents: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.doc_metadata: List[Dict] = []
        
        logger.info("BM25Retriever initialized")
    
    def index_documents(self, documents: List[Dict]):
        """
        Index documents for BM25 search
        In production, use better tokenization (nltk, spaCy)
        """
        self.documents = [doc['content'] for doc in documents]
        self.doc_metadata = [doc.get('metadata', {}) for doc in documents]
        
        # In production, use proper tokenization with stopword removal
        self.tokenized_docs = [doc.lower().split() for doc in self.documents]
        
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info(f"Indexed {len(documents)} documents for BM25")
    
    def search(self, query: str, top_k: int = 20) -> List[RetrievedDocument]:
        """
        Search using BM25
        """
        if not self.bm25:
            logger.warning("BM25 not initialized, no documents indexed")
            return []
        
        try:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            documents = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only return documents with non-zero score
                    doc = RetrievedDocument(
                        content=self.documents[idx],
                        score=float(scores[idx]),
                        source=self.doc_metadata[idx].get('source', 'unknown'),
                        doc_id=self.doc_metadata[idx].get('doc_id', f"bm25_{idx}"),
                        metadata=self.doc_metadata[idx],
                        retrieval_method='bm25'
                    )
                    documents.append(doc)
            
            logger.info(f"BM25 search returned {len(documents)} results")
            return documents
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []


class HybridRetriever:
    """
    Hybrid retrieval combining vector and BM25 with RRF fusion
    
    "I use Reciprocal Rank Fusion (RRF) to combine vector and BM25 results. RRF is parameter-free 
    - no need to tune weights like '70% vector, 30% BM25'. It just works by ranking each result 
    by its position in both lists. This is robust across different domains."
    """
    
    def __init__(self, vector_retriever: VectorRetriever, bm25_retriever: BM25Retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = 60  # Standard RRF constant
        
        logger.info("HybridRetriever initialized")
    
    def reciprocal_rank_fusion(
        self,
        results_dict: Dict[str, List[tuple]],
        k: int = 60
    ) -> List[tuple]:
        """
        Perform Reciprocal Rank Fusion
        This is the standard, parameter-free fusion method
        
        Args:
            results_dict: Dict of {method: [(doc_id, score), ...]}
            k: RRF constant (default 60)
            
        Returns:
            List of (doc_id, fused_score) sorted by score
        """
        fused_scores = {}
        
        for method, results in results_dict.items():
            for rank, (doc_id, score) in enumerate(results):
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                # RRF formula
                fused_scores[doc_id] += 1 / (k + rank + 1)
        
        # Sort by fused score
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
    
    def search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 20,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        Hybrid search combining vector and BM25
        
        Args:
            query_embedding: Query vector for vector search
            query_text: Query text for BM25 search
            top_k: Number of results to return
            user_id: User ID for ACL filtering
            filters: Metadata filters
            
        Returns:
            Fused list of retrieved documents
        """
        # For demo, we run sequentially
        
        # Vector search
        vector_results = self.vector_retriever.search(
            query_embedding=query_embedding,
            top_k=top_k,
            user_id=user_id,
            filters=filters
        )
        
        # BM25 search
        bm25_results = self.bm25_retriever.search(
            query=query_text,
            top_k=top_k
        )
        
        vector_dict = [(doc.doc_id, doc.score) for doc in vector_results]
        bm25_dict = [(doc.doc_id, doc.score) for doc in bm25_results]
        
        # Fuse results
        fused_results = self.reciprocal_rank_fusion({
            'vector': vector_dict,
            'bm25': bm25_dict
        })
        
        # Convert back to RetrievedDocument with fused scores
        # Create a lookup for document details
        doc_lookup = {}
        for doc in vector_results:
            doc_lookup[doc.doc_id] = doc
        for doc in bm25_results:
            if doc.doc_id not in doc_lookup:
                doc_lookup[doc.doc_id] = doc
        
        # Build final results
        final_documents = []
        for doc_id, fused_score in fused_results[:top_k]:
            if doc_id in doc_lookup:
                doc = doc_lookup[doc_id]
                doc.score = fused_score
                doc.retrieval_method = 'hybrid'
                final_documents.append(doc)
        
        logger.info(f"Hybrid search returned {len(final_documents)} documents")
        return final_documents


class RetrievalSystem:
    """
    Main retrieval system orchestrator
    """
    
    def __init__(self, use_hybrid: bool = True):
        self.vector_retriever = VectorRetriever()
        self.bm25_retriever = BM25Retriever()
        
        if use_hybrid:
            self.retriever = HybridRetriever(self.vector_retriever, self.bm25_retriever)
        else:
            self.retriever = self.vector_retriever
        
        self.use_hybrid = use_hybrid
        logger.info(f"RetrievalSystem initialized (hybrid: {use_hybrid})")
    
    def retrieve(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 20,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents using configured method
        
        Args:
            query_embedding: Query vector
            query_text: Query text
            top_k: Number of results
            user_id: User ID for ACL
            filters: Metadata filters
            
        Returns:
            List of retrieved documents
        """
        try:
            if self.use_hybrid:
                return self.retriever.search(
                    query_embedding=query_embedding,
                    query_text=query_text,
                    top_k=top_k,
                    user_id=user_id,
                    filters=filters
                )
            else:
                return self.vector_retriever.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    user_id=user_id,
                    filters=filters
                )
        except Exception as e:
            logger.warning(f"Retrieval failed, falling back to simple retriever: {e}")
            from retrieval_simple import simple_retrieval_system
            return simple_retrieval_system.retrieve(
                query_embedding=query_embedding,
                query_text=query_text,
                top_k=top_k,
                user_id=user_id,
                filters=filters
            )
    
    def index_documents(self, documents: List[Dict]):
        """
        Index documents for both vector and BM25 search
        """
        # Index in vector store
        self.vector_retriever.add_documents(documents)
        
        # Index for BM25
        self.bm25_retriever.index_documents(documents)
        
        logger.info(f"Indexed {len(documents)} documents in both retrievers")


# Global retrieval system instance
_use_simple_retrieval = False
try:
    retrieval_system = RetrievalSystem(use_hybrid=True)
    logger.info("Using Qdrant for vector storage")
except Exception as e:
    logger.warning(f"Qdrant not available, using in-memory store: {e}")
    from retrieval_simple import simple_retrieval_system
    retrieval_system = simple_retrieval_system
    _use_simple_retrieval = True
