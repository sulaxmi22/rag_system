"""
LLM Router Module
Component #5 from template - LLM (Router: cheap vs large · Prompt assembly · Gen)
This module routes queries to appropriate LLM models based on complexity.
"""

import logging
from typing import Dict, Optional, Tuple
from enum import Enum
from langchain_client import langchain_client
from config import settings

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """
    Model tier classification
    Different tiers for different complexity levels
    """
    CHEAP = "cheap"  # Fast, cheap, good for simple queries
    LARGE = "large"  # Slower, expensive, better for complex queries


class QueryComplexity(Enum):
    """
    Query complexity classification
    Determines which model to use
    """
    SIMPLE = "simple"  # FAQ-type, straightforward
    COMPLEX = "complex"  # Requires reasoning, synthesis, or nuance


class ComplexityClassifier:
    """
    Classifies query complexity to route to appropriate model
    Lightweight classifier to save costs
    60-70% of queries are simple and don't need expensive models
    """
    
    def __init__(self):
        # Simple heuristics for demo
        # In production, use a fine-tuned small model or LLM classifier
        
        # Simple query indicators
        self.simple_indicators = [
            'what is', 'how do i', 'where is', 'who is', 'when is',
            'define', 'explain', 'describe', 'list', 'show me',
            'faq', 'help', 'support', 'troubleshoot'
        ]
        
        # Complex query indicators
        self.complex_indicators = [
            'compare', 'analyze', 'evaluate', 'synthesize',
            'how might', 'what if', 'consider',
            'implications', 'trade-off', 'pros and cons',
            'relationship between', 'difference between'
        ]
        
        # Domain-specific complexity (customize for your domain)
        self.complex_domains = [
            'legal', 'compliance', 'security', 'architecture',
            'strategy', 'financial', 'medical'
        ]
    
    def classify(self, query: str) -> Tuple[QueryComplexity, float]:
        """
        Classify query complexity
        
        Args:
            query: User query
            
        Returns:
            (complexity, confidence_score)
        """
        query_lower = query.lower()
        
        # Check for complex indicators
        for indicator in self.complex_indicators:
            if indicator in query_lower:
                return QueryComplexity.COMPLEX, 0.8
        
        # Check for complex domains
        for domain in self.complex_domains:
            if domain in query_lower:
                return QueryComplexity.COMPLEX, 0.75
        
        # Check for simple indicators
        for indicator in self.simple_indicators:
            if indicator in query_lower:
                return QueryComplexity.SIMPLE, 0.7
        
        # Longer queries tend to be more complex
        if len(query.split()) > 15:
            return QueryComplexity.COMPLEX, 0.6
        else:
            return QueryComplexity.SIMPLE, 0.6
    
    def classify_with_llm(self, query: str) -> Tuple[QueryComplexity, float]:
        """
        Classify complexity using LLM (more accurate but slower)
        """
        prompt = f"""Classify the following query as either 'simple' or 'complex'.

Simple queries: FAQ-type, straightforward, factual, single-step reasoning
Complex queries: Requires analysis, synthesis, comparison, multi-step reasoning

Query: {query}

Respond with just 'simple' or 'complex'."""
        
        try:
            llm = langchain_client.get_llm()
            response = llm.invoke(prompt, temperature=0.0)
            response_text = response.content.lower().strip()
            
            if 'complex' in response_text:
                return QueryComplexity.COMPLEX, 0.9
            else:
                return QueryComplexity.SIMPLE, 0.9
                
        except Exception as e:
            logger.error(f"LLM classification failed, using heuristic: {e}")
            return self.classify(query)


class PromptBuilder:
    """
    Builds prompts for LLM generation
    Important for consistent, high-quality outputs
    Well-structured prompts produce better answers
    """
    
    def build_rag_prompt(
        self,
        query: str,
        context_docs: list,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Build RAG prompt with context
        
        Args:
            query: User query
            context_docs: List of retrieved documents
            system_prompt: Optional system instructions
            
        Returns:
            (system_prompt, user_prompt)
        """
        # Default system prompt if none provided
        if not system_prompt:
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
- Use only the information from the context to answer questions
- If the context doesn't contain the answer, say "I don't have enough information to answer this question"
- Cite your sources using the document IDs provided
- Be concise but thorough
- If you're uncertain, acknowledge it"""
        
        # Build context section
        context_parts = []
        for idx, doc in enumerate(context_docs, 1):
            context_parts.append(f"""
Document {idx} (ID: {doc.doc_id}, Source: {doc.source}):
{doc.content}
""")
        
        context_text = "\n".join(context_parts)
        
        # Build user prompt
        user_prompt = f"""Context:
{context_text}

Question: {query}

Answer:"""
        
        return system_prompt, user_prompt
    
    def build_chat_prompt(
        self,
        query: str,
        conversation_history: list,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Build prompt for conversational queries
        Includes conversation history for context
        """
        if not system_prompt:
            system_prompt = """You are a helpful AI assistant. Be conversational and friendly."""
        
        # Build conversation history
        history_text = ""
        if conversation_history:
            history_parts = []
            for turn in conversation_history[-5:]:  # Last 5 turns
                history_parts.append(f"User: {turn['query']}")
                history_parts.append(f"Assistant: {turn['answer']}")
            history_text = "\n".join(history_parts) + "\n\n"
        
        user_prompt = f"""{history_text}User: {query}
Assistant:"""
        
        return system_prompt, user_prompt


class LLMRouter:
    """
    Routes queries to appropriate LLM models
    Main router that orchestrates complexity classification and model selection
    """
    
    def __init__(self, use_llm_classifier: bool = False):
        self.complexity_classifier = ComplexityClassifier()
        self.prompt_builder = PromptBuilder()
        self.use_llm_classifier = use_llm_classifier
        
        # Using single OpenAI gpt-4o-mini model as required
        self.model = settings.openai_model
        
        # Track routing decisions for monitoring
        self.routing_stats = {
            'total': 0,
            'simple': 0,
            'complex': 0
        }
        
        logger.info(f"LLMRouter initialized (LLM classifier: {use_llm_classifier}, model: {self.model})")
    
    def route(
        self,
        query: str,
        context_docs: Optional[list] = None,
        conversation_history: Optional[list] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> Dict:
        """
        Route query to appropriate model and generate response
        
        Args:
            query: User query
            context_docs: Retrieved context documents (for RAG)
            conversation_history: Conversation history (for chat)
            system_prompt: System instructions
            stream: Whether to stream response
            
        Returns:
            Dictionary with response and metadata
        """
        self.routing_stats['total'] += 1
        
        # Classify complexity (for tracking only)
        if self.use_llm_classifier:
            complexity, confidence = self.complexity_classifier.classify_with_llm(query)
        else:
            complexity, confidence = self.complexity_classifier.classify(query)
        
        # Track complexity but always use gpt-4o-mini
        if complexity == QueryComplexity.SIMPLE:
            self.routing_stats['simple'] += 1
        else:
            self.routing_stats['complex'] += 1
        logger.info(f"Query complexity: {complexity.value} (confidence: {confidence})")
        
        # Build prompt
        if context_docs:
            sys_prompt, user_prompt = self.prompt_builder.build_rag_prompt(
                query=query,
                context_docs=context_docs,
                system_prompt=system_prompt
            )
        else:
            sys_prompt, user_prompt = self.prompt_builder.build_chat_prompt(
                query=query,
                conversation_history=conversation_history or [],
                system_prompt=system_prompt
            )
        
        # Generate response using LangChain
        try:
            llm = langchain_client.get_llm()
            
            if stream:
                # Streaming not implemented for LangChain in this version
                # Fall back to non-streaming
                logger.warning("Streaming not implemented, using non-streaming")
            
            # Combine system and user prompts for LangChain
            full_prompt = f"{sys_prompt}\n\n{user_prompt}"
            
            response = llm.invoke(full_prompt, temperature=0.0)
            
            return {
                'response': response.content,
                'model_used': self.model,
                'complexity': complexity.value,
                'confidence': confidence,
                'is_streaming': False
            }
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def get_routing_stats(self) -> Dict:
        """
        Get routing statistics
        Useful for monitoring and optimization
        """
        stats = self.routing_stats.copy()
        if stats['total'] > 0:
            stats['simple_percentage'] = (stats['simple'] / stats['total']) * 100
            stats['complex_percentage'] = (stats['complex'] / stats['total']) * 100
        return stats
    
    def reset_stats(self):
        """Reset routing statistics"""
        self.routing_stats = {'total': 0, 'simple': 0, 'complex': 0}


# Global LLM router instance
llm_router = LLMRouter(use_llm_classifier=False)
