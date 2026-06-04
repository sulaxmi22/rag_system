"""
Configuration Module
Centralized configuration using Pydantic Settings
Type-safe, environment variable support, validation, easy to test
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings with validation and type safety
    Using Pydantic for automatic validation and type conversion
    This prevents runtime errors from misconfigured environment variables
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    
    # OpenAI Models
    # Pin specific model versions for reproducibility
    # Important for production - model updates can break your system
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Vector Database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "rag_documents"
    vector_dimension: int = 1536
    
    # Cache Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl_seconds: int = 3600
    
    # Rate Limiting
    # Token bucket algorithm parameters
    # Burst allows short bursts, rate limits sustained traffic
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst_size: int = 10
    
    # Retrieval Configuration
    top_k_retrieval: int = 20
    top_k_final: int = 3
    chunk_size: int = 512
    chunk_overlap: float = 0.2
    
    # Re-ranking
    cohere_api_key: Optional[str] = None
    enable_reranking: bool = True
    
    # Application
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    
    # Validation
    enable_validation: bool = True
    max_retries: int = 1
    faithfulness_threshold: float = 0.7
    
    def validate_config(self) -> bool:
        """
        Validate important configuration
        Fail fast on misconfiguration
        Better to fail at startup than in production
        """
        if self.chunk_overlap < 0 or self.chunk_overlap > 0.5:
            raise ValueError("Chunk overlap must be between 0 and 0.5")
        
        if self.top_k_final > self.top_k_retrieval:
            raise ValueError("top_k_final must be <= top_k_retrieval")
        
        if self.faithfulness_threshold < 0 or self.faithfulness_threshold > 1:
            raise ValueError("Faithfulness threshold must be between 0 and 1")
        
        return True


# Global settings instance
# Singleton pattern for configuration
# Ensures consistent configuration across the application
settings = Settings()
settings.validate_config()
