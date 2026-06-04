"""
Gateway Module
First line of defense - every request passes through here
COMPONENT #1 from template: Gateway (Auth · Rate Limit · PII scrub · Injection detect · Token limit)
"""

import time
import re
import logging
from typing import Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class GatewayResult:
    """
    Result of gateway processing
    Structured result makes downstream processing clear
    """
    is_allowed: bool
    error_message: Optional[str] = None
    scrubbed_query: Optional[str] = None
    user_id: Optional[str] = None
    retry_after: Optional[int] = None


class RateLimiter:
    """
    Token bucket rate limiter
    Token bucket algorithm allows bursts while limiting sustained rate
    Better than fixed window (allows bursts), better than sliding window (simpler)
    """
    
    def __init__(self):
        # In production, use Redis for distributed rate limiting
        # Local dict is fine for single-instance demo
        self.buckets: Dict[str, Dict] = defaultdict(lambda: {
            'tokens': settings.rate_limit_burst_size,
            'last_update': time.time()
        })
    
    def is_allowed(self, user_id: str) -> Tuple[bool, Optional[int]]:
        """
        Check if request is allowed under rate limit
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            (is_allowed, retry_after_seconds)
        """
        now = time.time()
        bucket = self.buckets[user_id]
        
        # Refill tokens based on time elapsed
        # Rate = requests per minute / 60 seconds
        refill_rate = settings.rate_limit_requests_per_minute / 60.0
        time_elapsed = now - bucket['last_update']
        tokens_to_add = time_elapsed * refill_rate
        
        bucket['tokens'] = min(
            settings.rate_limit_burst_size,
            bucket['tokens'] + tokens_to_add
        )
        bucket['last_update'] = now
        
        # Check if we have tokens
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True, None
        else:
            # Calculate retry after
            tokens_needed = 1 - bucket['tokens']
            retry_after = int(tokens_needed / refill_rate)
            return False, retry_after
    
    def reset_user(self, user_id: str):
        """Reset rate limit for a user (admin function)"""
        if user_id in self.buckets:
            del self.buckets[user_id]


class PIIScrubber:
    """
    PII detection and anonymization
    Important for GDPR/HIPAA compliance
    Prevents PII from being sent to third-party LLM APIs
    """
    
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # Define what PII we care about
        # Customize based on your compliance requirements
        self.pii_entities = [
            'PERSON',           # Names
            'EMAIL_ADDRESS',    # Emails
            'PHONE_NUMBER',     # Phone numbers
            'US_SSN',          # Social Security Numbers
            'CREDIT_CARD',     # Credit card numbers
            'IBAN_CODE',       # Bank account numbers
            'IP_ADDRESS',      # IP addresses
        ]
    
    def scrub_text(self, text: str) -> Tuple[str, Dict]:
        """
        Detect and remove PII from text
        
        Args:
            text: Input text that may contain PII
            
        Returns:
            (scrubbed_text, detected_pii_info)
        """
        try:
            # Detect PII
            results = self.analyzer.analyze(
                text=text,
                entities=self.pii_entities,
                language='en'
            )
            
            if not results:
                return text, {}
            
            # Log what we detected (for audit trail)
            # But don't log the actual PII values
            detected_info = {
                'count': len(results),
                'types': list(set([r.entity_type for r in results]))
            }
            logger.info(f"Detected PII: {detected_info}")
            
            # Anonymize PII
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results
            )
            
            return anonymized.text, detected_info
            
        except Exception as e:
            logger.error(f"Error scrubbing PII: {e}")
            # Fail open - if PII detection fails, still process
            # But log the error for investigation
            return text, {'error': str(e)}


class InjectionDetector:
    """
    Prompt injection detection
    Prevents adversarial prompts from manipulating the system
    Security - prevents jailbreaks, data exfiltration, system prompt leaks
    """
    
    # Common injection patterns
    # In production, maintain this list and update based on attack trends
    INJECTION_PATTERNS = [
        r'ignore\s+(all\s+)?(previous|above)?\s*instructions',
        r'system\s*prompt',
        r'forget\s+(everything|all\s+instructions)',
        r'override\s+(your\s+)?programming',
        r'disregard\s+(previous|above)',
        r'new\s+instructions:',
        r'act\s+as\s+a\s+different',
        r'pretend\s+(you\s+are|to\s+be)',
        r'jailbreak',
        r'dan\s+mode',
        r'developer\s+mode',
        r'print\s+your\s+(system\s+)?prompt',
        r'show\s+your\s+(system\s+)?instructions',
    ]
    
    def __init__(self):
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.INJECTION_PATTERNS]
    
    def detect_injection(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if text contains prompt injection
        
        Args:
            text: Input text to check
            
        Returns:
            (is_injection, matched_pattern)
        """
        for pattern in self.patterns:
            if pattern.search(text):
                matched = pattern.pattern
                logger.warning(f"Injection detected: {matched}")
                return True, matched
        
        return False, None


class TokenValidator:
    """
    Token limit validation
    Prevents oversized queries that would be expensive
    Cost control - prevents users from sending massive queries
    """
    
    def __init__(self):
        # Using tiktoken for accurate token counting
        # For demo, we'll use a simple heuristic (4 chars ≈ 1 token)
        self.max_tokens = 4096  # Conservative limit
    
    def validate_token_count(self, text: str) -> Tuple[bool, int]:
        """
        Validate text doesn't exceed token limit
        
        Args:
            text: Input text
            
        Returns:
            (is_valid, estimated_tokens)
        """
        # Simple heuristic for demo
        # In production, use tiktoken for accurate counting
        estimated_tokens = len(text) // 4
        
        if estimated_tokens > self.max_tokens:
            return False, estimated_tokens
        
        return True, estimated_tokens


class Gateway:
    """
    Main gateway class that orchestrates all checks
    This is the facade pattern - single entry point for all gateway logic
    """
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.pii_scrubber = PIIScrubber()
        self.injection_detector = InjectionDetector()
        self.token_validator = TokenValidator()
        
        logger.info("Gateway initialized with all protection layers")
    
    def process_request(
        self,
        query: str,
        user_id: str,
        api_key: Optional[str] = None
    ) -> GatewayResult:
        """
        Process a request through all gateway checks
        
        Order matters - fail fast on cheap checks first
        1. Auth (fastest)
        2. Rate limit (fast)
        3. Token limit (fast)
        4. Injection detection (fast)
        5. PII scrubbing (slower, but important)
        
        Args:
            query: User query text
            user_id: User identifier
            api_key: Optional API key for auth
            
        Returns:
            GatewayResult with processing outcome
        """
        
        # Step 1 - Authentication
        # In production, validate JWT tokens or API keys
        # For demo, we accept any user_id
        if not self._authenticate(user_id, api_key):
            return GatewayResult(
                is_allowed=False,
                error_message="Authentication failed"
            )
        
        # Step 2 - Rate limiting
        is_allowed, retry_after = self.rate_limiter.is_allowed(user_id)
        if not is_allowed:
            return GatewayResult(
                is_allowed=False,
                error_message="Rate limit exceeded",
                retry_after=retry_after
            )
        
        # Step 3 - Token limit validation
        is_valid, token_count = self.token_validator.validate_token_count(query)
        if not is_valid:
            return GatewayResult(
                is_allowed=False,
                error_message=f"Query too long ({token_count} tokens, max {self.token_validator.max_tokens})"
            )
        
        # Step 4 - Injection detection
        is_injection, pattern = self.injection_detector.detect_injection(query)
        if is_injection:
            return GatewayResult(
                is_allowed=False,
                error_message="Potential prompt injection detected"
            )
        
        # Step 5 - PII scrubbing
        # This is the most expensive check, so we do it last
        scrubbed_query, pii_info = self.pii_scrubber.scrub_text(query)
        
        logger.info(f"Gateway check passed for user {user_id}")
        
        return GatewayResult(
            is_allowed=True,
            scrubbed_query=scrubbed_query,
            user_id=user_id
        )
    
    def _authenticate(self, user_id: str, api_key: Optional[str]) -> bool:
        """
        Authenticate user
        In production, validate against auth service
        For demo, we accept any non-empty user_id
        """
        if not user_id:
            return False
        
        # In production, you would:
        # - Validate JWT signature
        # - Check API key against database
        # - Verify user is active
        # - Check permissions
        
        return True


# Global gateway instance
# Singleton for consistency across requests
gateway = Gateway()
