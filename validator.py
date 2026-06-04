"""
Validator Module
Component #6 from template - Validator (Faithfulness · Safety · PII leak · Format check)
This module validates LLM responses before returning them to users.

Prevents hallucinations from reaching users
Ensures responses don't contain PII
Checks for safety violations
Validates response format
Important for trust and compliance
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from langchain_client import langchain_client
from gateway import PIIScrubber
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of validation
    Structured result makes handling clear
    """
    is_valid: bool
    faithfulness_score: float
    safety_score: float
    pii_detected: bool
    format_valid: bool
    error_message: Optional[str] = None


class FaithfulnessValidator:
    """
    Validates that response is faithful to retrieved context
    Using OpenAI gpt-4o-mini via LangChain
    Checks for hallucinations
    Prevents model from making things up not in context
    Skip validation (risk of hallucinations), NLI model (more accurate)
    
    
    "I use an LLM-as-judge to check faithfulness — does the response contain claims not supported 
by the context? This catches hallucinations before they reach the user. The judge model is a 
cheap Haiku-tier model, so the cost is minimal compared to the risk of bad answers."
    """
    
    def __init__(self):
        self.threshold = settings.faithfulness_threshold
        logger.info(f"FaithfulnessValidator initialized (threshold: {self.threshold})")
    
    def validate(
        self,
        response: str,
        context_docs: list,
        query: str
    ) -> Tuple[bool, float]:
        """
        Validate response faithfulness
        
        Args:
            response: LLM-generated response
            context_docs: Retrieved context documents
            query: Original query
            
        Returns:
            (is_faithful, faithfulness_score)
        """
        try:
            # Build context text
            context_text = "\n\n".join([doc.content for doc in context_docs])
            
            # Use LLM as judge
            prompt = f"""You are a faithfulness judge. Evaluate if the response is faithful to the context.

Context:
{context_text}

Query: {query}

Response: {response}

Evaluate on a range of 0 to 1:
- 1.0: Response is fully supported by context
- 0.5: Response is partially supported, some claims not in context
- 0.0: Response contains significant hallucinations not in context

Respond with just the numeric score (0.0, 0.5, or 1.0)."""
            
            llm = langchain_client.get_llm()
            response_text = llm.invoke(prompt, temperature=0.0)
            score_str = response_text.content
            
            # Parse score
            try:
                score = float(score_str.strip())
            except ValueError:
                logger.warning(f"Could not parse faithfulness score: {score_str}")
                score = 0.5  # Default to uncertain
            
            is_faithful = score >= self.threshold
            
            logger.info(f"Faithfulness check: score={score}, valid={is_faithful}")
            return is_faithful, score
            
        except Exception as e:
            logger.error(f"Faithfulness validation failed: {e}")
            # Fail open - if validation fails, assume faithful
            # Better to return potentially unfaithful answer than to fail completely
            return True, 0.5


class SafetyValidator:
    """
    Validates response for safety violations
    Checks for harmful content
    Prevents harmful, illegal, or inappropriate content
    Llama Guard (specialized), Content moderation APIs (AWS, etc.)
    
    
    "I use a keyword-based safety filter for common violations (hate speech, violence, etc.). 
For high-security deployments, I'd use Llama Guard or AWS Content Moderation API for more 
comprehensive coverage."
    """
    
    def __init__(self):
        # Simple keyword filter for demo
        # In production, use dedicated moderation service
        self.violation_keywords = [
            'kill', 'murder', 'suicide', 'bomb', 'terrorist',
            'hate', 'racist', 'nazi', 'hitler',
            'illegal', 'drugs', 'weapon'
        ]
        
        logger.info("SafetyValidator initialized")
    
    def validate(self, response: str) -> Tuple[bool, float]:
        """
        Validate response safety
        
        Args:
            response: LLM-generated response
            
        Returns:
            (is_safe, safety_score)
        """
        response_lower = response.lower()
        
        violations_found = []
        for keyword in self.violation_keywords:
            if keyword in response_lower:
                violations_found.append(keyword)
        
        if violations_found:
            logger.warning(f"Safety violations found: {violations_found}")
            return False, 0.0
        
        logger.info("Safety check passed")
        return True, 1.0


class PIIValidator:
    """
    Validates response for PII leakage
    Checks if response contains PII that shouldn't be there
    Model might accidentally include PII from training data
    Skip (risk), Presidio (what we use)
    
    
    "Even though we scrub PII from the input, the model might still generate PII in its response 
from its training data. I run a PII check on the output to catch this. If PII is detected, 
I either redact it or reject the response entirely."
    """
    
    def __init__(self):
        self.pii_scrubber = PIIScrubber()
        logger.info("PIIValidator initialized")
    
    def validate(self, response: str) -> Tuple[bool, bool]:
        """
        Validate response for PII
        
        Args:
            response: LLM-generated response
            
        Returns:
            (is_valid, has_pii)
        """
        # Check for PII
        _, pii_info = self.pii_scrubber.scrub_text(response)
        
        has_pii = pii_info.get('count', 0) > 0
        
        if has_pii:
            logger.warning(f"PII detected in response: {pii_info}")
            return False, True
        
        logger.info("PII check passed")
        return True, False


class FormatValidator:
    """
    Validates response format
    Checks basic format requirements
    Ensures response is well-formed and usable
    Skip (risk of malformed responses)
    """
    
    def __init__(self):
        self.min_length = 10
        self.max_length = 10000
        logger.info("FormatValidator initialized")
    
    def validate(self, response: str) -> Tuple[bool, str]:
        """
        Validate response format
        
        Args:
            response: LLM-generated response
            
        Returns:
            (is_valid, error_message)
        """
        # Check length
        if len(response) < self.min_length:
            return False, f"Response too short ({len(response)} < {self.min_length})"
        
        if len(response) > self.max_length:
            return False, f"Response too long ({len(response)} > {self.max_length})"
        
        # Check for empty or whitespace-only
        if not response.strip():
            return False, "Response is empty or whitespace"
        
        logger.info("Format check passed")
        return True, ""


class Validator:
    """
    Main validator orchestrator
    Runs all validation checks
    """
    
    def __init__(self):
        self.faithfulness_validator = FaithfulnessValidator()
        self.safety_validator = SafetyValidator()
        self.pii_validator = PIIValidator()
        self.format_validator = FormatValidator()
        
        self.enabled = settings.enable_validation
        self.max_retries = settings.max_retries
        
        logger.info(f"Validator initialized (enabled: {self.enabled}, max_retries: {self.max_retries})")
    
    def validate(
        self,
        response: str,
        context_docs: list,
        query: str
    ) -> ValidationResult:
        """
        Run all validation checks
        
        Args:
            response: LLM-generated response
            context_docs: Retrieved context
            query: Original query
            
        Returns:
            ValidationResult with all check results
        """
        if not self.enabled:
            logger.info("Validation disabled, returning valid")
            return ValidationResult(
                is_valid=True,
                faithfulness_score=1.0,
                safety_score=1.0,
                pii_detected=False,
                format_valid=True
            )
        
        # Run all checks
        # In production, run in parallel where possible
        
        # Faithfulness check
        is_faithful, faithfulness_score = self.faithfulness_validator.validate(
            response, context_docs, query
        )
        
        # Safety check
        is_safe, safety_score = self.safety_validator.validate(response)
        
        # PII check
        pii_valid, has_pii = self.pii_validator.validate(response)
        
        # Format check
        format_valid, format_error = self.format_validator.validate(response)
        
        # Overall validity
        is_valid = is_faithful and is_safe and pii_valid and format_valid
        
        error_message = None
        if not is_valid:
            errors = []
            if not is_faithful:
                errors.append("Response not faithful to context")
            if not is_safe:
                errors.append("Safety violation detected")
            if not pii_valid:
                errors.append("PII detected in response")
            if not format_valid:
                errors.append(format_error)
            error_message = "; ".join(errors)
        
        return ValidationResult(
            is_valid=is_valid,
            faithfulness_score=faithfulness_score,
            safety_score=safety_score,
            pii_detected=has_pii,
            format_valid=format_valid,
            error_message=error_message
        )
    
    def validate_with_retry(
        self,
        response: str,
        context_docs: list,
        query: str,
        retry_callback: callable
    ) -> Tuple[str, ValidationResult]:
        """
        Validate with retry logic
        Retry once with stricter prompt if validation fails
        Sometimes stricter prompt fixes the issue
        
        Args:
            response: Initial response
            context_docs: Retrieved context
            query: Original query
            retry_callback: Function to call for retry (should return new response)
            
        Returns:
            (final_response, final_validation_result)
        """
        # First validation
        validation_result = self.validate(response, context_docs, query)
        
        if validation_result.is_valid:
            return response, validation_result
        
        # Retry with stricter prompt
        logger.warning(f"Validation failed: {validation_result.error_message}")
        
        for attempt in range(self.max_retries):
            logger.info(f"Retry attempt {attempt + 1}/{self.max_retries}")
            
            # Get new response with stricter prompt
            new_response = retry_callback(attempt + 1)
            
            # Validate new response
            new_validation = self.validate(new_response, context_docs, query)
            
            if new_validation.is_valid:
                logger.info("Retry successful")
                return new_response, new_validation
        
        # All retries failed
        # Return fallback message instead of bad response
        logger.error("All validation retries failed")
        fallback_message = "I couldn't find a reliable answer based on the available information."
        
        fallback_validation = ValidationResult(
            is_valid=True,  # Fallback is always valid
            faithfulness_score=1.0,
            safety_score=1.0,
            pii_detected=False,
            format_valid=True,
            error_message="Used fallback message"
        )
        
        return fallback_message, fallback_validation


# Global validator instance
validator = Validator()
