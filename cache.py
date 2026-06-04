"""
Semantic Cache Module
Component #7 from template - Semantic Cache (Store response · TTL by data type)
This module caches responses to avoid redundant LLM calls.
"""

import logging
from typing import Optional, Dict, Any
import hashlib
import json
from redis import Redis
from config import settings

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic cache using Redis
    Caches responses based on semantic similarity
    Catches similar queries, not just exact matches
    """
    
    def __init__(self):
        self.redis_client = None
        self.enabled = False
        
        try:
            self.redis_client = Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logger.info("SemanticCache initialized with Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed, caching disabled: {e}")
            self.enabled = False
    
    def _generate_cache_key(self, query: str, user_id: Optional[str] = None) -> str:
        """
        Generate cache key from query
        Hash the query for consistent keys
        In production, use semantic embedding for similarity matching
        """
        # For demo, use simple hash
        # In production, use embedding similarity
        key_data = f"{query}:{user_id or 'default'}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached response for query
        
        Args:
            query: User query
            user_id: Optional user ID for per-user caching
            
        Returns:
            Cached response data or None
        """
        if not self.enabled:
            return None
        
        try:
            cache_key = self._generate_cache_key(query, user_id)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return json.loads(cached_data)
            
            logger.info(f"Cache miss for query: {query[:50]}...")
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None
    
    def set(
        self,
        query: str,
        response_data: Dict[str, Any],
        ttl: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Cache response for query
        
        Args:
            query: User query
            response_data: Response data to cache
            ttl: Time to live in seconds (default from config)
            user_id: Optional user ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            cache_key = self._generate_cache_key(query, user_id)
            ttl = ttl or settings.cache_ttl_seconds
            
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(response_data)
            )
            
            logger.info(f"Cached response for query: {query[:50]}... (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False
    
    def invalidate(self, query: str, user_id: Optional[str] = None) -> bool:
        """
        Invalidate cached response for query
        
        Args:
            query: User query
            user_id: Optional user ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            cache_key = self._generate_cache_key(query, user_id)
            self.redis_client.delete(cache_key)
            logger.info(f"Invalidated cache for query: {query[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
            return False
    
    def clear_all(self) -> bool:
        """
        Clear all cached responses
        Use with caution - admin function
        """
        if not self.enabled:
            return False
        
        try:
            self.redis_client.flushdb()
            logger.info("Cleared all cache entries")
            return True
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        Useful for monitoring cache effectiveness
        """
        if not self.enabled:
            return {'enabled': False}
        
        try:
            info = self.redis_client.info()
            return {
                'enabled': True,
                'total_keys': self.redis_client.dbsize(),
                'memory_used': info.get('used_memory_human', 'unknown'),
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'enabled': True, 'error': str(e)}


class InMemoryCache:
    """
    Simple in-memory cache for small deployments
    No persistence, no distributed support
    Simpler setup for small deployments
    """
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.enabled = True
        logger.info("InMemoryCache initialized")
    
    def _generate_cache_key(self, query: str, user_id: Optional[str] = None) -> str:
        """Generate cache key"""
        key_data = f"{query}:{user_id or 'default'}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        if not self.enabled:
            return None
        
        cache_key = self._generate_cache_key(query, user_id)
        return self.cache.get(cache_key)
    
    def set(
        self,
        query: str,
        response_data: Dict[str, Any],
        ttl: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """Cache response"""
        if not self.enabled:
            return False
        
        cache_key = self._generate_cache_key(query, user_id)
        self.cache[cache_key] = response_data
        return True
    
    def invalidate(self, query: str, user_id: Optional[str] = None) -> bool:
        """Invalidate cached response"""
        if not self.enabled:
            return False
        
        cache_key = self._generate_cache_key(query, user_id)
        if cache_key in self.cache:
            del self.cache[cache_key]
        return True
    
    def clear_all(self) -> bool:
        """Clear all cache"""
        self.cache.clear()
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'enabled': True,
            'total_keys': len(self.cache)
        }


# Global cache instance
# Try Redis first, fall back to in-memory
try:
    cache = SemanticCache()
except Exception:
    cache = InMemoryCache()
