"""
AWS Bedrock Client Module
Wrapper around AWS Bedrock API for LLM and embedding calls
"""

import boto3
import json
import logging
from typing import Dict, List, Optional, Any
from config import settings

logger = logging.getLogger(__name__)


class BedrockClient:
    """
    AWS Bedrock client for LLM and embedding operations
    Singleton pattern to reuse connections
    Connection pooling is handled by boto3 automatically
    """
    
    def __init__(self):
        """Initialize Bedrock client with AWS credentials"""
        self.region = settings.aws_region
        
        # Initialize boto3 clients
        # Separate clients for runtime and embeddings
        # Different endpoints, different API contracts
        self.runtime_client = self._create_runtime_client()
        self.client = self._create_client()
        
        logger.info(f"Bedrock client initialized for region {self.region}")
    
    def _create_runtime_client(self) -> boto3.client:
        """
        Create Bedrock Runtime client for inference
        Runtime client is for actual model invocations
        """
        try:
            if settings.aws_access_key_id and settings.aws_secret_access_key:
                return boto3.client(
                    'bedrock-runtime',
                    region_name=self.region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key
                )
            else:
                # Use default credential chain (env, IAM role, etc.)
                # This is the AWS best practice - don't hardcode credentials
                return boto3.client('bedrock-runtime', region_name=self.region)
        except Exception as e:
            logger.error(f"Failed to create Bedrock runtime client: {e}")
            raise
    
    def _create_client(self) -> boto3.client:
        """
        Create Bedrock client for management operations
        Management client for listing models, etc.
        """
        try:
            if settings.aws_access_key_id and settings.aws_secret_access_key:
                return boto3.client(
                    'bedrock',
                    region_name=self.region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key
                )
            else:
                return boto3.client('bedrock', region_name=self.region)
        except Exception as e:
            logger.error(f"Failed to create Bedrock client: {e}")
            raise
    
    def invoke_claude(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Invoke Claude model via Bedrock
        Using Claude 3 via Converse API (newer, more consistent)
        
        Args:
            prompt: User query
            model_id: Bedrock model ID (e.g., claude-3-haiku, claude-3-sonnet)
            max_tokens: Maximum tokens in response
            temperature: 0.0 = deterministic, higher = more creative
            system_prompt: System instructions for the model
            
        Returns:
            Model response text
        """
        try:
            # Build message format for Converse API
            messages = [{"role": "user", "content": [{"text": prompt}]}]
            
            # System prompt guides model behavior
            # Important for consistent outputs in production
            inference_config = {
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": 0.9
            }
            
            kwargs = {
                "modelId": model_id,
                "messages": messages,
                "inferenceConfig": inference_config
            }
            
            if system_prompt:
                kwargs["system"] = [{"text": system_prompt}]
            
            # Call Bedrock
            response = self.runtime_client.converse(**kwargs)
            
            # Extract response text
            output_text = response['output']['message']['content'][0]['text']
            
            # Log token usage for cost tracking
            # Important for production cost management
            input_tokens = response['usage']['inputTokens']
            output_tokens = response['usage']['outputTokens']
            logger.info(f"Model {model_id} - Input: {input_tokens}, Output: {output_tokens}")
            
            return output_text
            
        except Exception as e:
            logger.error(f"Error invoking Claude model {model_id}: {e}")
            raise
    
    def invoke_claude_streaming(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None
    ):
        """
        Invoke Claude model with streaming response
        Streaming reduces Time To First Token (TTFT)
        Important for user experience - shows progress immediately
        
        Args:
            prompt: User query
            model_id: Bedrock model ID
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            system_prompt: System instructions
            
        Yields:
            Chunks of response text as they arrive
        """
        try:
            messages = [{"role": "user", "content": [{"text": prompt}]}]
            
            inference_config = {
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": 0.9
            }
            
            kwargs = {
                "modelId": model_id,
                "messages": messages,
                "inferenceConfig": inference_config
            }
            
            if system_prompt:
                kwargs["system"] = [{"text": system_prompt}]
            
            # Streaming call
            response = self.runtime_client.converse_stream(**kwargs)
            
            # Yield chunks as they arrive
            stream = response.get('stream')
            if stream:
                for event in stream:
                    if 'contentBlockDelta' in event:
                        chunk = event['contentBlockDelta']['delta']['text']
                        yield chunk
                        
        except Exception as e:
            logger.error(f"Error in streaming invoke: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get text embedding using Bedrock Titan model
        Using Amazon Titan for embeddings
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # Titan Embeddings v1 - 1536 dimensions
            # Same model must be used for indexing and querying
            response = self.runtime_client.invoke_model(
                modelId=settings.bedrock_embedding_model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "inputText": text
                })
            )
            
            response_body = json.loads(response.get('body').read())
            embedding = response_body.get('embedding')
            
            logger.debug(f"Generated embedding with dimension {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def list_available_models(self) -> List[str]:
        """
        List available foundation models in Bedrock
        Useful for discovery and debugging
        """
        try:
            response = self.client.list_foundation_models()
            models = [model['modelId'] for model in response['modelSummaries']]
            return models
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []


# Global singleton instance
# Reuse client across requests for connection pooling
bedrock_client = BedrockClient()
