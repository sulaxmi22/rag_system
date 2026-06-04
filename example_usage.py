"""
Example Usage Script
This demonstrates the complete RAG pipeline end-to-end
Use this to test the system
"""

import logging
from ingestion import ingestion_pipeline
from gateway import gateway
from query_processor import query_processor
from retrieval import retrieval_system
from reranker import reranker
from llm_router import llm_router
from validator import validator
from cache import cache
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    """
    Complete end-to-end demonstration of the RAG system
    Walk through this step by step
    """
    
    print_section("RAG SYSTEM - COMPLETE DEMONSTRATION")
    
    # ========================================================================
    # STEP 1: INGEST DOCUMENTS (Offline Pipeline)
    # ========================================================================
    print_section("STEP 1: INGESTING DOCUMENTS")
    
    sample_documents = [
        {
            "text": """Our company has a comprehensive refund policy. Customers can return products 
within 30 days of purchase for a full refund. The product must be in its original condition 
with all tags attached. To initiate a return, customers should contact customer service at 
support@example.com or call 1-800-555-0123. Refunds are processed within 5-7 business days 
after we receive the returned item.""",
            "source": "refund_policy",
            "metadata": {"category": "policy", "department": "customer_service"}
        },
        {
            "text": """Employee benefits include health insurance, dental coverage, and a 401(k) 
retirement plan with company matching up to 4%. Full-time employees receive 20 days of paid 
time off annually, plus 10 paid holidays. The company also offers flexible work arrangements 
and remote work options for eligible positions.""",
            "source": "employee_handbook",
            "metadata": {"category": "hr", "department": "human_resources"}
        },
        {
            "text": """Our product pricing is tiered based on usage. The Basic plan starts at $29/month 
for up to 1,000 API calls. The Pro plan is $99/month for up to 10,000 API calls. Enterprise 
plans are custom-priced based on specific requirements. All plans include 24/7 email support, 
with phone support available for Pro and Enterprise tiers.""",
            "source": "pricing",
            "metadata": {"category": "sales", "department": "sales"}
        }
    ]
    
    doc_ids = []
    for doc in sample_documents:
        doc_id = ingestion_pipeline.ingest_text(
            text=doc["text"],
            source=doc["source"],
            metadata=doc["metadata"]
        )
        doc_ids.append(doc_id)
        print(f"✓ Ingested document: {doc['source']} (ID: {doc_id})")
    
    print(f"\nTotal documents ingested: {len(doc_ids)}")
    
    # ========================================================================
    # STEP 2: PROCESS A QUERY (Online Pipeline)
    # ========================================================================
    print_section("STEP 2: PROCESSING A QUERY")
    
    query = "What is the refund policy?"
    user_id = "demo_user"
    
    print(f"User Query: {query}")
    print(f"User ID: {user_id}\n")
    
    # ========================================================================
    # COMPONENT 1: GATEWAY
    # ========================================================================
    print("--- COMPONENT 1: GATEWAY ---")
    print("Checking authentication, rate limits, PII scrubbing...")
    
    gateway_result = gateway.process_request(
        query=query,
        user_id=user_id
    )
    
    if not gateway_result.is_allowed:
        print(f"✗ Request blocked: {gateway_result.error_message}")
        return
    
    print(f"✓ Gateway check passed")
    print(f"  Scrubbed query: {gateway_result.scrubbed_query}")
    
    query = gateway_result.scrubbed_query
    
    # ========================================================================
    # COMPONENT 7: CACHE CHECK
    # ========================================================================
    print("\n--- COMPONENT 7: CACHE ---")
    print("Checking for cached response...")
    
    cached = cache.get(query, user_id=user_id)
    if cached:
        print(f"✓ Cache hit! Returning cached response")
        print(f"  Response: {cached['response']}")
        return
    
    print("✓ Cache miss - proceeding with full pipeline")
    
    # ========================================================================
    # COMPONENT 2: QUERY PROCESSING
    # ========================================================================
    print("\n--- COMPONENT 2: QUERY PROCESSING ---")
    print("Classifying intent and generating embedding...")
    
    processed = query_processor.process(query)
    
    print(f"✓ Query processed")
    print(f"  Intent: {processed['intent']}")
    print(f"  Confidence: {processed['intent_confidence']}")
    print(f"  Should retrieve: {processed['should_retrieve']}")
    
    if not processed['should_retrieve']:
        print("✗ Query is out of scope or requires escalation")
        return
    
    # ========================================================================
    # COMPONENT 3: RETRIEVAL
    # ========================================================================
    print("\n--- COMPONENT 3: RETRIEVAL ---")
    print(f"Retrieving top-{settings.top_k_retrieval} documents using hybrid search...")
    
    documents = retrieval_system.retrieve(
        query_embedding=processed['embedding'],
        query_text=processed['final_query'],
        top_k=settings.top_k_retrieval,
        user_id=user_id
    )
    
    print(f"✓ Retrieved {len(documents)} documents")
    for i, doc in enumerate(documents[:3], 1):
        print(f"  {i}. Score: {doc.score:.4f} | Source: {doc.source}")
        print(f"     Content: {doc.content[:100]}...")
    
    if not documents:
        print("✗ No documents retrieved")
        return
    
    # ========================================================================
    # COMPONENT 4: RE-RANKING
    # ========================================================================
    print("\n--- COMPONENT 4: RE-RANKING ---")
    print(f"Re-ranking to top-{settings.top_k_final} documents...")
    
    top_docs = reranker.rerank(
        query=query,
        documents=documents,
        top_n=settings.top_k_final
    )
    
    print(f"✓ Re-ranked to {len(top_docs)} documents")
    for i, doc in enumerate(top_docs, 1):
        print(f"  {i}. Score: {doc.score:.4f} | Source: {doc.source}")
    
    # ========================================================================
    # COMPONENT 5: LLM ROUTER & GENERATION
    # ========================================================================
    print("\n--- COMPONENT 5: LLM ROUTER ---")
    print("Routing to appropriate model and generating response...")
    
    llm_result = llm_router.route(
        query=query,
        context_docs=top_docs,
        stream=False
    )
    
    print(f"✓ Response generated")
    print(f"  Model used: {llm_result['model_used']}")
    print(f"  Complexity: {llm_result['complexity']}")
    print(f"  Confidence: {llm_result['confidence']}")
    print(f"\n  Response:")
    print(f"  {llm_result['response']}")
    
    # ========================================================================
    # COMPONENT 6: VALIDATION
    # ========================================================================
    print("\n--- COMPONENT 6: VALIDATION ---")
    print("Checking faithfulness, safety, and PII...")
    
    validation_result = validator.validate(
        response=llm_result['response'],
        context_docs=top_docs,
        query=query
    )
    
    print(f"✓ Validation complete")
    print(f"  Is valid: {validation_result.is_valid}")
    print(f"  Faithfulness score: {validation_result.faithfulness_score}")
    print(f"  Safety score: {validation_result.safety_score}")
    print(f"  PII detected: {validation_result.pii_detected}")
    print(f"  Format valid: {validation_result.format_valid}")
    
    if not validation_result.is_valid:
        print(f"  Error: {validation_result.error_message}")
    
    # ========================================================================
    # COMPONENT 7: CACHE RESPONSE
    # ========================================================================
    print("\n--- COMPONENT 7: CACHE RESPONSE ---")
    print("Caching response for future queries...")
    
    cache.set(
        query=query,
        response_data={
            'response': llm_result['response'],
            'model_used': llm_result['model_used'],
            'complexity': llm_result['complexity'],
            'confidence': llm_result['confidence'],
            'validation': validation_result.__dict__
        },
        user_id=user_id
    )
    
    print("✓ Response cached")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section("SUMMARY")
    
    print(f"✓ Query processed successfully")
    print(f"✓ Response: {llm_result['response']}")
    print(f"\nSystem Statistics:")
    print(f"  Routing stats: {llm_router.get_routing_stats()}")
    print(f"  Cache stats: {cache.get_stats()}")
    
    # ========================================================================
    # DEMONSTRATE CACHE HIT
    # ========================================================================
    print_section("DEMONSTRATING CACHE HIT")
    
    print("Asking the same question again...")
    cached = cache.get(query, user_id=user_id)
    
    if cached:
        print(f"✓ Cache hit! Response retrieved from cache")
        print(f"  Response: {cached['response']}")
        print(f"  This avoided an expensive LLM call!")
    
    # ========================================================================
    # DEMONSTRATE DIFFERENT QUERY
    # ========================================================================
    print_section("PROCESSING A DIFFERENT QUERY")
    
    query2 = "What employee benefits are available?"
    print(f"User Query: {query2}\n")
    
    # Gateway
    gateway_result2 = gateway.process_request(query=query2, user_id=user_id)
    query2 = gateway_result2.scrubbed_query
    
    # Query processing
    processed2 = query_processor.process(query2)
    
    # Retrieval
    docs2 = retrieval_system.retrieve(
        query_embedding=processed2['embedding'],
        query_text=processed2['final_query'],
        top_k=settings.top_k_retrieval,
        user_id=user_id
    )
    
    # Re-rank
    top_docs2 = reranker.rerank(query=query2, documents=docs2, top_n=3)
    
    # LLM
    llm_result2 = llm_router.route(query=query2, context_docs=top_docs2, stream=False)
    
    # Validation
    validation2 = validator.validate(
        response=llm_result2['response'],
        context_docs=top_docs2,
        query=query2
    )
    
    print(f"✓ Response: {llm_result2['response']}")
    print(f"  Model used: {llm_result2['model_used']}")
    print(f"  Faithfulness: {validation2.faithfulness_score}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print_section("FINAL SYSTEM STATISTICS")
    
    print(f"Total queries processed: {llm_router.get_routing_stats()['total']}")
    print(f"Queries routed to cheap model: {llm_router.get_routing_stats()['cheap']}")
    print(f"Queries routed to large model: {llm_router.get_routing_stats()['large']}")
    print(f"\nCache statistics: {cache.get_stats()}")
    
    print_section("DEMONSTRATION COMPLETE")
    print("The RAG system is working end-to-end!")
    print("All components have been demonstrated successfully.\n")


if __name__ == "__main__":
    main()
