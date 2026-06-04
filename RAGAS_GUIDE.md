# RAGAS Evaluation Guide

## Overview

RAGAS (Retrieval Augmented Generation Assessment) is a framework for evaluating RAG systems using LLM-based metrics. This guide explains how to use RAGAS with your existing RAG system.

## What is RAGAS?

RAGAS provides automated evaluation of RAG systems using an "LLM-as-judge" approach. Instead of manual human evaluation, RAGAS uses LLMs to assess the quality of your RAG outputs across multiple dimensions.

## Key Metrics

### Faithfulness
- **What it measures**: Whether the generated answer is grounded in the retrieved context
- **Why it matters**: Detects hallucinations - answers that contain information not present in the context
- **Score range**: 0-1, higher is better
- **Target**: > 0.7 for production systems

### Context Precision
- **What it measures**: How relevant the retrieved context is to the question
- **Why it matters**: Indicates retrieval quality - are we getting the right documents?
- **Score range**: 0-1, higher is better
- **Target**: > 0.7 for production systems

### Context Recall
- **What it measures**: How much of the relevant information was retrieved from ground truth
- **Why it matters**: Indicates completeness - are we missing important information?
- **Score range**: 0-1, higher is better
- **Target**: > 0.7 for production systems
- **Note**: Requires ground truth answers

### Answer Relevancy
- **What it measures**: How relevant the generated answer is to the question
- **Why it matters**: Indicates whether the answer actually addresses the user's question
- **Score range**: 0-1, higher is better
- **Target**: > 0.7 for production systems

## Installation

RAGAS is already included in `requirements.txt`. To install:

```bash
pip install -r requirements.txt
```

## Setup

### Option 1: Using OpenAI (Default)

RAGAS uses OpenAI by default for the LLM-as-judge evaluation:

```bash
export OPENAI_API_KEY='your-openai-api-key'
```

Get an API key from: https://platform.openai.com/api-keys

### Option 2: Using Other LLM Providers

RAGAS 0.1.9 primarily supports OpenAI. To use other providers (Bedrock, Anthropic, etc.), you would need to:

1. **Use a newer version of RAGAS** with broader provider support
2. **Implement custom evaluators** using your preferred LLM
3. **Configure RAGAS** to use a custom LLM instance

See `ragas_evaluator.py` for configuration details.

## Usage

### Basic Example

```python
from ragas_evaluator import RAGASEvaluator, print_evaluation_report

# Initialize evaluator
evaluator = RAGASEvaluator()

# Prepare your data
questions = ["What is the refund policy?"]
answers = ["Our refund policy allows returns within 30 days."]
contexts = [["Our refund policy allows returns within 30 days with original receipt."]]
ground_truths = ["Returns are accepted within 30 days with original receipt."]

# Run evaluation
result = evaluator.evaluate(
    questions=questions,
    answers=answers,
    contexts=contexts,
    ground_truths=ground_truths
)

# Print results
print_evaluation_report(result)
```

### Evaluating Your RAG Pipeline

```python
from ragas_evaluator import RAGASEvaluator
from gateway import gateway
from query_processor import query_processor
from retrieval import retrieval_system
from reranker import reranker
from llm_router import llm_router

# Initialize evaluator
evaluator = RAGASEvaluator()

# Run queries through your RAG system
rag_results = []
for query in test_queries:
    # ... run through your RAG pipeline ...
    rag_results.append({
        "question": query,
        "answer": generated_answer,
        "contexts": retrieved_contexts,
        "ground_truth": ground_truth  # optional
    })

# Evaluate
result = evaluator.evaluate_from_rag_pipeline(rag_results)
print_evaluation_report(result)
```

### Running the Examples

```bash
# Without API key (shows mock evaluation)
python ragas_example.py

# With OpenAI API key (runs real evaluation)
export OPENAI_API_KEY='your-key'
python ragas_example.py
```

## Evaluation Strategies

### 1. Offline Evaluation (Recommended)

**When to use**: Regular quality checks, A/B testing, comparing configurations

**How**:
1. Create a test dataset with questions, answers, and ground truths
2. Run evaluation on the dataset
3. Track scores over time

**Benefits**:
- Consistent, reproducible results
- Can compare different configurations
- Lower cost (run offline, not on every query)

### 2. Online Evaluation

**When to use**: Monitoring production performance, detecting regressions

**How**:
1. Sample a percentage of production queries
2. Evaluate in real-time or near real-time
3. Alert on score drops

**Benefits**:
- Detect issues quickly
- Monitor real-world performance
- Higher cost (evaluates production traffic)

### 3. Single Query Evaluation

**When to use**: Development, debugging, quick checks

**How**:
```python
result = evaluator.evaluate_single(
    question="What is the refund policy?",
    answer="Our refund policy allows returns within 30 days.",
    contexts=["Our refund policy allows returns within 30 days."]
)
```

**Benefits**:
- Fast feedback during development
- Debug specific issues
- No dataset required

## Interpreting Results

### Overall Score

The overall score is the average of all computed metrics:

- **> 0.8**: Excellent - Your RAG system is performing very well
- **0.6 - 0.8**: Good - Your RAG system is performing adequately
- **0.4 - 0.6**: Fair - Your RAG system needs improvement
- **< 0.4**: Poor - Your RAG system needs significant improvement

### Metric-Specific Recommendations

**Low Faithfulness (< 0.7)**:
- Add better validation in your validator module
- Improve prompt engineering to reduce hallucinations
- Consider adding a faithfulness check before returning answers

**Low Context Precision (< 0.7)**:
- Improve retrieval quality (better embeddings, hybrid search)
- Add or improve re-ranking
- Tune top_k parameters
- Improve chunking strategy

**Low Context Recall (< 0.7)**:
- Increase top_k retrieval
- Improve chunking to capture more relevant information
- Add query expansion or HyDE
- Improve document indexing

**Low Answer Relevancy (< 0.7)**:
- Improve query understanding
- Better prompt engineering
- Add query rewriting
- Improve context selection

## A/B Testing

Use RAGAS to compare different configurations:

```python
# Configuration A (baseline)
result_a = evaluator.evaluate(
    questions=questions,
    answers=answers_a,
    contexts=contexts_a
)

# Configuration B (improved)
result_b = evaluator.evaluate(
    questions=questions,
    answers=answers_b,
    contexts=contexts_b
)

# Compare
improvement = result_b.overall_score - result_a.overall_score
print(f"Improvement: {improvement:+.4f}")
```

## Best Practices

### 1. Create a Good Test Dataset

- **Diverse questions**: Cover different topics, difficulty levels
- **Ground truths**: Ideal answers for context_recall metric
- **Representative**: Reflect real user queries
- **Size**: 50-100 questions for reliable results

### 2. Track Scores Over Time

- Store evaluation results in a database
- Plot trends to see improvements
- Set alerts for score drops
- Correlate with system changes

### 3. Use Multiple Metrics

Don't rely on a single metric. Use all available metrics to get a complete picture:
- Faithfulness + Context Precision = Good retrieval and generation
- Context Recall = Completeness (requires ground truth)
- Answer Relevancy = User satisfaction

### 4. Cost Management

RAGAS uses LLM-as-judge, which costs money:
- Run offline on samples, not all traffic
- Use cheaper models for evaluation (e.g., GPT-3.5 instead of GPT-4)
- Cache evaluation results
- Run evaluations periodically, not on every query

### 5. Combine with Human Evaluation

RAGAS is great for automated evaluation, but:
- Still do periodic human evaluation
- Use human evaluation to validate RAGAS scores
- Human evaluation catches edge cases RAGAS might miss

## Integration with Existing RAG System

The RAGAS evaluator integrates seamlessly with your existing RAG components:

- **Gateway**: Evaluate after gateway processing
- **Query Processor**: Evaluate query understanding
- **Retrieval**: Evaluate retrieval quality (context_precision, context_recall)
- **Re-ranker**: Evaluate re-ranking effectiveness
- **LLM Router**: Evaluate generation quality (faithfulness, answer_relevancy)
- **Validator**: Compare validator results with RAGAS faithfulness

## Troubleshooting

### "OPENAI_API_KEY not set"

RAGAS requires an LLM for evaluation. Set the environment variable:
```bash
export OPENAI_API_KEY='your-key'
```

### "Timeout during evaluation"

Increase the timeout in the evaluator:
```python
result = evaluate(
    dataset=dataset,
    metrics=metrics,
    run_config=RunConfig(timeout=120)  # 120 seconds
)
```

### "Low scores across all metrics"

Check:
- Are your documents properly indexed?
- Is your embedding model working correctly?
- Are you retrieving relevant context?
- Is your LLM generating quality answers?

### "Context recall not computed"

Context recall requires ground truth answers. Provide ground_truths parameter:
```python
result = evaluator.evaluate(
    questions=questions,
    answers=answers,
    contexts=contexts,
    ground_truths=ground_truths  # Required for context_recall
)
```

## Files

- `ragas_evaluator.py`: Core RAGAS evaluation module
- `ragas_example.py`: Example usage scripts
- `RAGAS_GUIDE.md`: This documentation file

## Next Steps

1. **Create a test dataset** with questions and ground truths
2. **Run baseline evaluation** to establish current performance
3. **Implement improvements** (re-ranking, better chunking, etc.)
4. **Re-evaluate** to measure improvements
5. **Track scores over time** to monitor performance
6. **Set up alerts** for score drops in production

## Resources

- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS Metrics Explained](https://docs.ragas.io/en/stable/concepts/metrics/index.html)