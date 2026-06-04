"""
Query Processing Module
Component #2 from template - Query Processing (Intent classify · HyDE · Query rewrite · Embed)
This module prepares the query for retrieval by understanding intent and generating embeddings.
"""

import logging
from typing import Dict, Optional, Tuple
from enum import Enum
from langchain_client import langchain_client
from config import settings

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """
    Query intent classification
    Helps route queries to appropriate handlers
    """
    IN_SCOPE = "in_scope"  # Query is relevant to our knowledge base
    OUT_OF_SCOPE = "out_of_scope"  # Query is not relevant
    ESCALATE_TO_HUMAN = "escalate"  # Needs human intervention
    CHITCHAT = "chitchat"  # Casual conversation, not information seeking


class IntentClassifier:
    """
    Classifies query intent before expensive retrieval
    Lightweight classifier to prevent wasted retrieval costs
    60-70% of queries might be out-of-scope in some domains
    """
    """
    
    def __init__(self):
        # In production, use a fine-tuned small model or LLM classifier
        self.out_of_scope_keywords = [
            'weather', 'sports', 'politics', 'celebrity', 'gossip',
            'recipe', 'joke', 'horoscope', 'stock price', 'news'
        ]
        
        self.escalation_keywords = [
            'complaint', 'angry', 'sue',
            'legal action', 'manager', 'supervisor', 'escalate',
            'speak to manager', 'talk to supervisor', 'unhappy with service'
        ]
        
        self.chitchat_keywords = [
            'hello', 'hi', 'how are you', 'thanks', 'bye',
            'good morning', 'good evening', 'nice to meet you'
        ]
    
    def classify(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Classify query intent
        
        Args:
            query: User query text
            
        Returns:
            (intent, confidence_score)
        """
        query_lower = query.lower()
        
        for keyword in self.escalation_keywords:
            if keyword in query_lower:
                return QueryIntent.ESCALATE_TO_HUMAN, 0.9
        
        # Check for chitchat
        for keyword in self.chitchat_keywords:
            if keyword in query_lower:
                return QueryIntent.CHITCHAT, 0.8
        
        # Check for out-of-scope
        for keyword in self.out_of_scope_keywords:
            if keyword in query_lower:
                return QueryIntent.OUT_OF_SCOPE, 0.7
        
        # Default to in-scope
        return QueryIntent.IN_SCOPE, 0.6
    
    def classify_with_llm(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Classify intent using LLM (more accurate but slower)
        
        Args:
            query: User query text
            
        Returns:
            (intent, confidence_score)
        """
        prompt = f"""Classify the following query into one of these categories:
1. in_scope - Query is relevant to our knowledge base
2. out_of_scope - Query is not relevant to our domain
3. escalate - Query requires human intervention
4. chitchat - Casual conversation

Query: {query}

Respond with just the category name."""
        
        try:
            llm = langchain_client.get_llm()
            response = llm.invoke(prompt, temperature=0.0)
            response_text = response.content.lower().strip()
            
            if 'escalate' in response_text:
                return QueryIntent.ESCALATE_TO_HUMAN, 0.95
            elif 'chitchat' in response_text:
                return QueryIntent.CHITCHAT, 0.9
            elif 'out_of_scope' in response_text or 'out of scope' in response_text:
                return QueryIntent.OUT_OF_SCOPE, 0.9
            else:
                return QueryIntent.IN_SCOPE, 0.9
                
        except Exception as e:
            logger.error(f"LLM classification failed, defaulting to in_scope: {e}")
            # Fail open - if classifier fails, assume in-scope
            # Better to waste retrieval cost than to block valid queries
            return QueryIntent.IN_SCOPE, 0.5


class QueryRewriter:
    """
    Rewrites queries for better retrieval
    
    "I add query rewriting for ambiguous queries. For example, 'how do I reset it?' becomes 
    'how do I reset the password?' based on context. This improves retrieval recall by 15-20%."
    """
    
    def __init__(self):
        pass
    
    def rewrite(self, query: str, conversation_history: Optional[list] = None) -> str:
        """
        Rewrite query for better retrieval
        
        Args:
            query: Original user query
            conversation_history: Optional conversation context
            
        Returns:
            Rewritten query
        """
        # For demo, simple rewrite logic
        # In production, use LLM to rewrite based on conversation history
        
        # Simple pronoun resolution (very basic)
        query_lower = query.lower()
        
        # If query has pronouns but no context, return as-is
        if any(pronoun in query_lower for pronoun in ['it', 'they', 'this', 'that']):
            if not conversation_history:
                logger.warning("Query has pronouns but no conversation history")
                return query
        
        # 1. Use conversation history to resolve references
        # 2. Expand abbreviations (e.g., "API" -> "Application Programming Interface")
        # 3. Add domain-specific context
        # 4. Handle multi-turn conversations
        
        return query
    
    def rewrite_with_llm(self, query: str, conversation_history: Optional[list] = None) -> str:
        """
        Rewrite query using LLM (more sophisticated)
        Using OpenAI gpt-4o-mini via LangChain
        Use this for production systems with conversation history
        """
        history_context = ""
        if conversation_history:
            history_context = "\n".join([
                f"Q: {turn['query']}\nA: {turn['answer']}"
                for turn in conversation_history[-3:]  # Last 3 turns
            ])
        
        prompt = f"""Rewrite the following query to be more specific and clear for document retrieval.
{f'Conversation history:\n{history_context}\n\n' if history_context else ''}
Original query: {query}

Rewritten query:"""
        
        try:
            llm = langchain_client.get_llm()
            response = llm.invoke(prompt, temperature=0.0)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Query rewrite failed, using original: {e}")
            return query


class HyDEGenerator:
    """
    Hypothetical Document Embeddings (HyDE) generator
    Generates a hypothetical answer, embeds that instead of query
    Bridges query-document embedding gap for short queries
    When to use: When retrieval recall is below 0.75 without it
    Trade-off: Adds one LLM call per query (latency + cost)
    """
    
    def __init__(self):
        self.enabled = False  # Disabled by default, enable based on metrics
    
    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate a hypothetical answer to the query
        
        Args:
            query: User query
            
        Returns:
            Hypothetical answer text
        """
        prompt = f"""Write a detailed answer to the following question. 
Make it comprehensive and informative, as if it were from a knowledge base article.

Question: {query}

Answer:"""
        
        try:
            llm = langchain_client.get_llm()
            response = llm.invoke(prompt, temperature=0.0)
            return response.content
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return query  # Fall back to original query


class QueryProcessor:
    """
    Main query processing orchestrator
    Combines all query processing steps
    """
    
    def __init__(self, use_llm_classifier: bool = False, use_hyde: bool = False):
        self.intent_classifier = IntentClassifier()
        self.query_rewriter = QueryRewriter()
        self.hyde_generator = HyDEGenerator()
        self.use_llm_classifier = use_llm_classifier
        self.use_hyde = use_hyde
        
        logger.info(f"QueryProcessor initialized (LLM classifier: {use_llm_classifier}, HyDE: {use_hyde})")
    
    def process(
        self,
        query: str,
        conversation_history: Optional[list] = None
    ) -> Dict:
        """
        Process query through all steps
        
        Args:
            query: Original user query
            conversation_history: Optional conversation context
            
        Returns:
            Dictionary with processed query and metadata
        """
        result = {
            'original_query': query,
            'intent': None,
            'intent_confidence': 0.0,
            'rewritten_query': None,
            'final_query': None,
            'embedding': None,
            'should_retrieve': True
        }
        
        # Step 1 - Intent classification
        if self.use_llm_classifier:
            intent, confidence = self.intent_classifier.classify_with_llm(query)
        else:
            intent, confidence = self.intent_classifier.classify(query)
        
        result['intent'] = intent
        result['intent_confidence'] = confidence
        
        # Handle different intents
        if intent == QueryIntent.OUT_OF_SCOPE:
            result['should_retrieve'] = False
            logger.info(f"Query out of scope: {query}")
            return result
        
        if intent == QueryIntent.ESCALATE_TO_HUMAN:
            result['should_retrieve'] = False
            logger.info(f"Query requires escalation: {query}")
            return result
        
        if intent == QueryIntent.CHITCHAT:
            result['should_retrieve'] = False
            logger.info(f"Query is chitchat: {query}")
            return result
        
        # Step 2 - Query rewriting
        rewritten_query = self.query_rewriter.rewrite(query, conversation_history)
        result['rewritten_query'] = rewritten_query
        
        # Step 3 - HyDE (if enabled)
        if self.use_hyde:
            hypothetical_answer = self.hyde_generator.generate_hypothetical_answer(rewritten_query)
            final_query = hypothetical_answer
            result['used_hyde'] = True
        else:
            final_query = rewritten_query
            result['used_hyde'] = False
        
        result['final_query'] = final_query
        
        # Step 4 - Generate embedding
        # IMPORTANT: Use same model as document indexing
        try:
            embedding = langchain_client.get_embeddings().embed_query(final_query)
            result['embedding'] = embedding
            logger.debug(f"Generated embedding for query: {final_query[:50]}...")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            result['should_retrieve'] = False
        
        return result


# Global query processor instance
query_processor = QueryProcessor()
