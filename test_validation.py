"""
Simple validation test script
Tests each validator component directly without running full RAG pipeline
"""

from validator import (
    FaithfulnessValidator,
    SafetyValidator,
    PIIValidator,
    FormatValidator
)

def test_faithfulness():
    """Test faithfulness validator"""
    print("\n" + "="*60)
    print("TEST: Faithfulness Validator")
    print("="*60)
    
    validator = FaithfulnessValidator()
    
    # Test case 1: Faithful answer
    response = "The refund policy allows returns within 30 days."
    context = [{"content": "Our refund policy allows returns within 30 days with original receipt."}]
    query = "What is the refund policy?"
    
    is_valid, score = validator.validate(response, context, query)
    print(f"\nTest 1 - Faithful answer:")
    print(f"  Response: {response}")
    print(f"  Score: {score}")
    print(f"  Valid: {is_valid}")
    
    # Test case 2: Hallucinated answer
    response = "The refund policy allows returns within 90 days."
    is_valid, score = validator.validate(response, context, query)
    print(f"\nTest 2 - Hallucinated answer:")
    print(f"  Response: {response}")
    print(f"  Score: {score}")
    print(f"  Valid: {is_valid}")


def test_safety():
    """Test safety validator"""
    print("\n" + "="*60)
    print("TEST: Safety Validator")
    print("="*60)
    
    validator = SafetyValidator()
    
    # Test case 1: Safe response
    response = "You can return items within 30 days."
    is_valid, score = validator.validate(response)
    print(f"\nTest 1 - Safe response:")
    print(f"  Response: {response}")
    print(f"  Score: {score}")
    print(f"  Valid: {is_valid}")
    
    # Test case 2: Unsafe response
    response = "I will help you kill someone."
    is_valid, score = validator.validate(response)
    print(f"\nTest 2 - Unsafe response:")
    print(f"  Response: {response}")
    print(f"  Score: {score}")
    print(f"  Valid: {is_valid}")


def test_pii():
    """Test PII validator"""
    print("\n" + "="*60)
    print("TEST: PII Validator")
    print("="*60)
    
    validator = PIIValidator()
    
    # Test case 1: No PII
    response = "The refund policy is 30 days."
    is_valid, has_pii = validator.validate(response)
    print(f"\nTest 1 - No PII:")
    print(f"  Response: {response}")
    print(f"  Has PII: {has_pii}")
    print(f"  Valid: {is_valid}")
    
    # Test case 2: With PII
    response = "Contact John Smith at john@example.com or 555-123-4567."
    is_valid, has_pii = validator.validate(response)
    print(f"\nTest 2 - With PII:")
    print(f"  Response: {response}")
    print(f"  Has PII: {has_pii}")
    print(f"  Valid: {is_valid}")


def test_format():
    """Test format validator"""
    print("\n" + "="*60)
    print("TEST: Format Validator")
    print("="*60)
    
    validator = FormatValidator()
    
    # Test case 1: Valid format
    response = "This is a valid response with adequate length."
    is_valid, error = validator.validate(response)
    print(f"\nTest 1 - Valid format:")
    print(f"  Response: {response}")
    print(f"  Valid: {is_valid}")
    print(f"  Error: {error}")
    
    # Test case 2: Too short
    response = "Hi"
    is_valid, error = validator.validate(response)
    print(f"\nTest 2 - Too short:")
    print(f"  Response: {response}")
    print(f"  Valid: {is_valid}")
    print(f"  Error: {error}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SIMPLE VALIDATION TESTS")
    print("="*60)
    print("\nTesting individual validation components...")
    
    try:
        test_faithfulness()
        test_safety()
        test_pii()
        test_format()
        
        print("\n" + "="*60)
        print("✓ All validation tests completed")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Faithfulness validator requires OpenAI API key")
