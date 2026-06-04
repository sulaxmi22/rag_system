"""
Re-ranker Module
Component #4 from template - Re-ranker (Cross-encoder score · Top-3 final context)
This module re-ranks retrieved documents to improve precision before sending to LLM.

Initial retrieval (top-20) is fast but less accurate
Re-ranking uses cross-encoder for higher precision
Only top-3 go to LLM, so accuracy matters
Trade-off: 100-200ms latency for better answers
"""

import logging
from typing import List, Optional
import cohere
from config import settings
from retrieval import RetrievedDocument

logger = logging.getLogger(__name__)


class CohereReranker:
    """
    Cohere API-based re-ranker
    State-of-the-art re-ranking, easy integration
    Best accuracy, no infrastructure to manage, simple API
    """
    
    def __init__(self):
        self.api_key = settings.cohere_api_key
        self.enabled = settings.enable_reranking
        
        if self.enabled and self.api_key:
            self.client = cohere.Client(self.api_key)
            logger.info("CohereReranker initialized")
        else:
            self.client = None
            if self.enabled:
                logger.warning("Cohere reranking enabled but no API key provided")
    
    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_n: int = 3
    ) -> List[RetrievedDocument]:
        """
        Re-rank documents using Cohere API
        
        Args:
            query: Original query
            documents: Retrieved documents to re-rank
            top_n: Number of top documents to return
            
        Returns:
            Re-ranked list of documents
        """
        if not self.enabled or not self.client:
            logger.info("Reranking disabled, returning original order")
            return documents[:top_n]
        
        if not documents:
            return []
        
        try:
            # Prepare documents for Cohere API
            docs_for_rerank = [
                {"text": doc.content, "id": doc.doc_id}
                for doc in documents
            ]
            
            # Call Cohere rerank API
            rerank_results = self.client.rerank(
                model="rerank-english-v2.0",
                query=query,
                documents=docs_for_rerank,
                top_n=top_n
            )
            
            # Re-rank based on Cohere scores
            reranked_docs = []
            doc_lookup = {doc.doc_id: doc for doc in documents}
            
            for result in rerank_results.results:
                doc_id = docs_for_rerank[result.index]["id"]
                if doc_id in doc_lookup:
                    doc = doc_lookup[doc_id]
                    doc.score = result.relevance_score
                    doc.retrieval_method = f"{doc.retrieval_method}_reranked"
                    reranked_docs.append(doc)
            
            logger.info(f"Re-ranked {len(documents)} documents to top-{len(reranked_docs)}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            # Better to return unranked results than to fail completely
            return documents[:top_n]


class LocalCrossEncoderReranker:
    """
    Local cross-encoder re-ranker (for large deployments)
    Requires GPU infrastructure, model management
    """
    
    def __init__(self):
        # from sentence_transformers import CrossEncoder
        # self.model = CrossEncoder('ms-marco-MiniLM-L-6-v2')
        self.model = None
        logger.info("LocalCrossEncoderReranker initialized (model not loaded in demo)")
    
    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_n: int = 3
    ) -> List[RetrievedDocument]:
        """
        Re-rank using local cross-encoder
        """
        if not self.model:
            logger.warning("Local model not loaded, returning original order")
            return documents[:top_n]
        
        try:
            # Prepare query-document pairs
            pairs = [[query, doc.content] for doc in documents]
            
            # Score pairs
            scores = self.model.predict(pairs)
            
            # Sort by scores
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-n
            reranked_docs = []
            for doc, score in scored_docs[:top_n]:
                doc.score = float(score)
                doc.retrieval_method = f"{doc.retrieval_method}_reranked"
                reranked_docs.append(doc)
            
            logger.info(f"Local cross-encoder re-ranked to top-{len(reranked_docs)}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Local reranking failed: {e}")
            return documents[:top_n]


class Reranker:
    """
    Main re-ranker orchestrator
    """
    
    def __init__(self, use_local: bool = False):
        if use_local:
            self.reranker = LocalCrossEncoderReranker()
        else:
            self.reranker = CohereReranker()
        
        logger.info(f"Reranker initialized (local: {use_local})")
    
    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_n: int = 3
    ) -> List[RetrievedDocument]:
        """
        Re-rank documents
        
        Args:
            query: Original query
            documents: Retrieved documents
            top_n: Number of top documents to keep
            
        Returns:
            Re-ranked documents
        """
        # No point re-ranking if we already have few results
        if len(documents) <= top_n:
            logger.info(f"Only {len(documents)} documents, skipping re-rank")
            return documents
        
        return self.reranker.rerank(query, documents, top_n)


# Global re-ranker instance
reranker = Reranker(use_local=False)
