"""
Streaming Response Module
Component #8 from template - Stream (SSE to client · TTFT < 1s target)
This module handles streaming responses to clients using Server-Sent Events (SSE).

Reduces Time To First Token (TTFT) - users see progress immediately
Better user experience for long responses
Perceived latency is lower even if total time is same
Important for conversational interfaces

TTFT = Time To First Token
- Target: < 1 second
- Measured from request receipt to first token arrival
- Important for user satisfaction
"""

import logging
import json
from typing import AsyncGenerator, Optional
from fastapi.responses import StreamingResponse
from langchain_client import langchain_client

logger = logging.getLogger(__name__)


class StreamingResponseHandler:
    """
    Handles streaming responses using SSE
    Using OpenAI via LangChain
    Wraps OpenAI streaming generator in SSE format
    SSE is simpler than WebSockets for one-way streaming
    """
    
    def __init__(self):
        logger.info("StreamingResponseHandler initialized")
    
    async def stream_response(
        self,
        prompt: str,
        model_id: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from LLM using SSE format
        
        Args:
            prompt: User prompt
            model_id: OpenAI model ID
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Yields:
            SSE-formatted chunks
        """
        try:
            # Get LLM from langchain_client
            llm = langchain_client.get_llm()
            
            # Build full prompt with system prompt if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Stream response from OpenAI
            stream = llm.stream(full_prompt, temperature=temperature, max_tokens=max_tokens)
            
            # Stream chunks
            for chunk in stream:
                # Format as SSE
                sse_data = json.dumps({
                    "type": "chunk",
                    "content": chunk.content
                })
                yield f"data: {sse_data}\n\n"
            
            # Send completion signal
            completion_data = json.dumps({
                "type": "done",
                "content": ""
            })
            yield f"data: {completion_data}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            # Send error signal
            error_data = json.dumps({
                "type": "error",
                "content": str(e)
            })
            yield f"data: {error_data}\n\n"
    
    def create_fastapi_streaming_response(
        self,
        prompt: str,
        model_id: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> StreamingResponse:
        """
        Create FastAPI StreamingResponse
        
        Args:
            prompt: User prompt
            model_id: OpenAI model ID
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            
        Returns:
            FastAPI StreamingResponse object
        """
        return StreamingResponse(
            self.stream_response(
                prompt=prompt,
                model_id=model_id,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )


class NonStreamingResponseHandler:
    """
    Handles non-streaming responses
    """
    
    def __init__(self):
        logger.info("NonStreamingResponseHandler initialized")
    
    def get_response(
        self,
        prompt: str,
        model_id: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> str:
        """
        Get non-streaming response
        
        Args:
            prompt: User prompt
            model_id: OpenAI model ID
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            
        Returns:
            Complete response text
        """
        try:
            # Get LLM from langchain_client
            llm = langchain_client.get_llm()
            
            # Build full prompt with system prompt if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Get response from OpenAI
            response = llm.invoke(full_prompt, temperature=temperature, max_tokens=max_tokens)
            return response.content
        except Exception as e:
            logger.error(f"Non-streaming response error: {e}")
            raise


# Global streaming handler instance
streaming_handler = StreamingResponseHandler()
non_streaming_handler = NonStreamingResponseHandler()
