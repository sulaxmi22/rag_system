"""
RAGAS Evaluator Module
Automated evaluation of RAG systems using LLM-based metrics
Measures faithfulness, context precision, context recall, answer relevance

This module integrates RAGAS with the existing RAG system to evaluate:
- Faithfulness: Is the answer grounded in retrieved context?
- Context Precision: Is the retrieved context relevant to the question?
- Context Recall: Did we retrieve all relevant information?
- Answer Relevance: Is the answer relevant to the question?
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

# Import existing RAG components
from langchain_client import langchain_client
from config import settings


@dataclass
class EvaluationResult:
    """Container for RAGAS evaluation results"""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: Optional[float] = None
    overall_score: float = 0.0
    
    def __post_init__(self):
        """Calculate overall score as average of available metrics"""
        metrics = [self.faithfulness, self.answer_relevancy, self.context_precision]
        if self.context_recall is not None:
            metrics.append(self.context_recall)
        self.overall_score = sum(metrics) / len(metrics)


class RAGASEvaluator:
    """
    RAGAS Evaluator for RAG system assessment
    """
    
    def __init__(self):
        """
        Initialize RAGAS evaluator with OpenAI
        """
        self._configure_ragas()
    
    def _configure_ragas(self):
        """
        Configure RAGAS to use OpenAI for evaluation
        We configure it to use the same OpenAI model as our RAG system
        """
        import os
        
        # Configure RAGAS to use OpenAI
        try:
            # Set OpenAI API key for RAGAS
            if settings.openai_api_key:
                os.environ["OPENAI_API_KEY"] = settings.openai_api_key
                print(f"✓ RAGAS configured to use OpenAI: {settings.openai_model}")
            else:
                raise ValueError("OPENAI_API_KEY not found in settings")
                
        except Exception as e:
            print(f"⚠ Warning: Could not configure OpenAI for RAGAS: {e}")
            print("Please set OPENAI_API_KEY in your .env file")
            raise
    
    def prepare_evaluation_data(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dataset:
        """
        Prepare data in RAGAS format
        
        Args:
            questions: List of user questions
            answers: List of generated answers from RAG system
            contexts: List of retrieved contexts for each question
            ground_truths: Optional list of ground truth answers (for context_recall)
        
        Returns:
            Dataset in RAGAS format
        """
        data_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truths:
            data_dict["ground_truth"] = ground_truths
        
        return Dataset.from_dict(data_dict)
    
    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> EvaluationResult:
        """
        Run RAGAS evaluation on RAG system outputs
        
        Args:
            questions: List of user questions
            answers: List of generated answers from RAG system
            contexts: List of retrieved contexts for each question
            ground_truths: Optional list of ground truth answers
            metrics: Optional list of metrics to compute
                    If None, computes all available metrics
        
        Returns:
            EvaluationResult with metric scores
        """
        # Prepare dataset
        dataset = self.prepare_evaluation_data(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths
        )
        
        # Select metrics
        if metrics is None:
            # Default metrics: faithfulness, answer_relevancy, context_precision
            selected_metrics = [faithfulness, answer_relevancy, context_precision]
            
            # Add context_recall if ground_truths are provided
            if ground_truths:
                selected_metrics.append(context_recall)
        else:
            # Map metric names to RAGAS metrics
            metric_map = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
            }
            selected_metrics = [metric_map[m] for m in metrics if m in metric_map]
        
        # Run evaluation
        # Consider running this offline on a sample of queries
        
        # Configure RAGAS with OpenAI
        run_config = RunConfig(timeout=60)  # 60 second timeout per evaluation
        
        result = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            run_config=run_config
        )
        
        # Convert to DataFrame for easier handling
        df = result.to_pandas()
        
        # Calculate average scores
        avg_faithfulness = df["faithfulness"].mean()
        avg_answer_relevancy = df["answer_relevancy"].mean()
        avg_context_precision = df["context_precision"].mean()
        
        avg_context_recall = None
        if "context_recall" in df.columns:
            avg_context_recall = df["context_recall"].mean()
        
        return EvaluationResult(
            faithfulness=avg_faithfulness,
            answer_relevancy=avg_answer_relevancy,
            context_precision=avg_context_precision,
            context_recall=avg_context_recall
        )
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single Q&A pair
        
        Args:
            question: User question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer
        
        Returns:
            Dictionary of metric scores
        """
        return self.evaluate(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth] if ground_truth else None
        ).__dict__
    
    def evaluate_from_rag_pipeline(
        self,
        rag_pipeline_results: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        Evaluate results from the RAG pipeline
        
        Args:
            rag_pipeline_results: List of dicts with keys:
                - question: str
                - answer: str
                - contexts: List[str]
                - ground_truth: Optional[str]
        
        Returns:
            EvaluationResult with metric scores
        """
        questions = [r["question"] for r in rag_pipeline_results]
        answers = [r["answer"] for r in rag_pipeline_results]
        contexts = [r["contexts"] for r in rag_pipeline_results]
        ground_truths = [r.get("ground_truth") for r in rag_pipeline_results]
        
        # Filter out None ground_truths
        has_ground_truth = [gt is not None for gt in ground_truths]
        
        if any(has_ground_truth):
            # Use only samples with ground truth for context_recall
            return self.evaluate(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truths
            )
        else:
            # Evaluate without context_recall
            return self.evaluate(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=None
            )


def print_evaluation_report(result: EvaluationResult):
    """
    Print a formatted evaluation report
    
    Args:
        result: EvaluationResult from RAGAS evaluation
    """
    print("\n" + "="*60)
    print("RAGAS EVALUATION REPORT")
    print("="*60)
    print(f"Faithfulness:       {result.faithfulness:.4f}")
    print(f"Answer Relevancy:   {result.answer_relevancy:.4f}")
    print(f"Context Precision:   {result.context_precision:.4f}")
    if result.context_recall is not None:
        print(f"Context Recall:      {result.context_recall:.4f}")
    print("-"*60)
    print(f"Overall Score:       {result.overall_score:.4f}")
    print("="*60)
    
    # Interpretation
    print("\nINTERPRETATION:")
    if result.overall_score >= 0.8:
        print("✓ Excellent: Your RAG system is performing very well")
    elif result.overall_score >= 0.6:
        print("✓ Good: Your RAG system is performing adequately")
    elif result.overall_score >= 0.4:
        print("⚠ Fair: Your RAG system needs improvement")
    else:
        print("✗ Poor: Your RAG system needs significant improvement")
    
    print("\nRECOMMENDATIONS:")
    if result.faithfulness < 0.7:
        print("- Improve faithfulness: Add better validation, reduce hallucinations")
    if result.context_precision < 0.7:
        print("- Improve context precision: Improve retrieval quality, add re-ranking")
    if result.answer_relevancy < 0.7:
        print("- Improve answer relevancy: Better prompt engineering, query understanding")
    if result.context_recall is not None and result.context_recall < 0.7:
        print("- Improve context recall: Increase top_k, improve chunking strategy")


# Example usage
if __name__ == "__main__":
    # This is a simple example - see ragas_example.py for full integration
    evaluator = RAGASEvaluator()
    
    # Sample data
    questions = [
        "What is the refund policy?",
        "How do I contact support?",
    ]
    
    answers = [
        "The refund policy allows returns within 30 days of purchase.",
        "You can contact support via email at support@example.com.",
    ]
    
    contexts = [
        ["Our refund policy allows returns within 30 days of purchase with original receipt."],
        ["Contact our support team at support@example.com or call 1-800-SUPPORT."],
    ]
    
    ground_truths = [
        "Returns are accepted within 30 days with original receipt.",
        "Support is available at support@example.com or 1-800-SUPPORT.",
    ]
    
    result = evaluator.evaluate(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths
    )
    
    print_evaluation_report(result)
