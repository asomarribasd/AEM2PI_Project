#!/usr/bin/env python3
"""
Evaluator Agent for RAG-based FAQ Support Chatbot

This script evaluates the quality of chatbot responses by:
1. Analyzing chunk relevance to user questions
2. Assessing answer accuracy and completeness
3. Checking for hallucinations or contradictions
4. Providing structured feedback with 0-10 scoring

Evaluation Criteria:
- Chunk Relevance (30%): How well retrieved chunks match the question
- Answer Accuracy (40%): Factual correctness based on provided context
- Answer Completeness (20%): Coverage of all relevant aspects
- Response Quality (10%): Clarity, professionalism, helpfulness
"""

import os
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

class ResponseEvaluator:
    """Evaluates chatbot responses using LLM-based assessment"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize evaluator with OpenAI model
        
        Args:
            model: OpenAI model for evaluation
        """
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def evaluate_response(self, user_question: str, system_answer: str, 
                         chunks_related: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a complete chatbot response
        
        Args:
            user_question: Original user question
            system_answer: Chatbot's generated answer
            chunks_related: Retrieved chunks used for answer generation
            
        Returns:
            Dictionary with evaluation results and scoring
        """
        # Prepare context from chunks
        context_text = self._format_chunks_for_evaluation(chunks_related)
        
        # Create evaluation prompt
        evaluation_prompt = self._create_evaluation_prompt(
            user_question, system_answer, context_text, chunks_related
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=1500
            )
            
            evaluation_text = response.choices[0].message.content.strip()
            return self._parse_evaluation_response(evaluation_text)
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {
                "overall_score": 0,
                "evaluation_breakdown": {},
                "reasoning": f"Evaluation failed due to error: {e}",
                "recommendations": ["Fix evaluation system error"],
                "evaluation_success": False
            }
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for evaluation"""
        return """You are an expert evaluator for HR FAQ chatbot responses. Your job is to assess the quality of responses based on multiple criteria.

Evaluation Criteria (with weights):
1. Chunk Relevance (30%): How well do the retrieved chunks match the user's question?
2. Answer Accuracy (40%): Is the answer factually correct based on the provided context?
3. Answer Completeness (20%): Does the answer fully address the question?
4. Response Quality (10%): Is the answer clear, professional, and helpful?

Scoring Scale:
- 9-10: Excellent - Exceptional quality, addresses all aspects perfectly
- 7-8: Good - High quality with minor issues
- 5-6: Adequate - Acceptable but has notable gaps or issues
- 3-4: Poor - Significant problems, partially useful
- 1-2: Very Poor - Major issues, minimally helpful
- 0: Unacceptable - Completely wrong or unhelpful

Your response MUST follow this exact JSON format:
{
  "chunk_relevance_score": X,
  "chunk_relevance_reasoning": "explanation",
  "accuracy_score": X,
  "accuracy_reasoning": "explanation", 
  "completeness_score": X,
  "completeness_reasoning": "explanation",
  "quality_score": X,
  "quality_reasoning": "explanation",
  "overall_score": X.X,
  "reasoning": "overall explanation",
  "recommendations": ["improvement 1", "improvement 2"],
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"]
}"""
    
    def _create_evaluation_prompt(self, user_question: str, system_answer: str, 
                                 context_text: str, chunks_related: List[Dict[str, Any]]) -> str:
        """Create evaluation prompt with all necessary information"""
        
        chunk_summary = f"Retrieved {len(chunks_related)} chunks"
        if chunks_related:
            sections = set(chunk.get('section_title', 'Unknown') for chunk in chunks_related)
            chunk_summary += f" from sections: {', '.join(sections)}"
            
            avg_similarity = sum(chunk.get('similarity_score', 0) for chunk in chunks_related) / len(chunks_related)
            chunk_summary += f" (average similarity: {avg_similarity:.2f})"
        
        return f"""Please evaluate this HR chatbot response:

USER QUESTION:
{user_question}

SYSTEM ANSWER:
{system_answer}

CONTEXT USED ({chunk_summary}):
{context_text}

Please provide a thorough evaluation following the JSON format specified in the system prompt. Focus on:

1. Chunk Relevance: Do the retrieved chunks contain information relevant to answering the user's question?
2. Answer Accuracy: Is the answer factually correct based on the context provided? Are there any hallucinations?
3. Answer Completeness: Does the answer fully address all aspects of the question? What might be missing?
4. Response Quality: Is the answer well-structured, professional, and helpful for an HR context?

Provide specific reasoning for each score and actionable recommendations for improvement."""
    
    def _format_chunks_for_evaluation(self, chunks_related: List[Dict[str, Any]]) -> str:
        """Format chunks for evaluation prompt"""
        if not chunks_related:
            return "No chunks were retrieved."
        
        formatted_chunks = []
        for i, chunk in enumerate(chunks_related, 1):
            section = chunk.get('section_title', 'Unknown Section')
            similarity = chunk.get('similarity_score', 0)
            text = chunk.get('text_preview', chunk.get('text', ''))
            
            formatted_chunks.append(
                f"Chunk {i} (Section: {section}, Similarity: {similarity:.2f}):\n{text}"
            )
        
        return "\n\n".join(formatted_chunks)
    
    def _parse_evaluation_response(self, evaluation_text: str) -> Dict[str, Any]:
        """Parse LLM evaluation response into structured format"""
        try:
            # Try to extract JSON from the response
            start_idx = evaluation_text.find('{')
            end_idx = evaluation_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = evaluation_text[start_idx:end_idx]
            evaluation_data = json.loads(json_str)
            
            # Validate required fields
            required_fields = [
                'chunk_relevance_score', 'accuracy_score', 
                'completeness_score', 'quality_score', 'overall_score'
            ]
            
            for field in required_fields:
                if field not in evaluation_data:
                    evaluation_data[field] = 0
            
            # Calculate overall score if not provided or incorrect
            weights = {
                'chunk_relevance_score': 0.3,
                'accuracy_score': 0.4,
                'completeness_score': 0.2,
                'quality_score': 0.1
            }
            
            calculated_overall = sum(
                evaluation_data.get(field, 0) * weight 
                for field, weight in weights.items()
            )
            
            evaluation_data['calculated_overall_score'] = round(calculated_overall, 1)
            evaluation_data['evaluation_success'] = True
            
            # Ensure lists exist
            for list_field in ['recommendations', 'strengths', 'weaknesses']:
                if list_field not in evaluation_data:
                    evaluation_data[list_field] = []
            
            return evaluation_data
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback parsing if JSON extraction fails
            return self._fallback_parse_evaluation(evaluation_text, str(e))
    
    def _fallback_parse_evaluation(self, evaluation_text: str, error: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails"""
        # Try to extract scores using regex or simple text parsing
        import re
        
        scores = {}
        
        # Look for score patterns
        score_patterns = {
            'overall_score': r'overall[_\s]*score[:\s]*(\d+(?:\.\d+)?)',
            'accuracy_score': r'accuracy[_\s]*score[:\s]*(\d+)',
            'chunk_relevance_score': r'chunk[_\s]*relevance[_\s]*score[:\s]*(\d+)',
            'completeness_score': r'completeness[_\s]*score[:\s]*(\d+)',
            'quality_score': r'quality[_\s]*score[:\s]*(\d+)'
        }
        
        for score_name, pattern in score_patterns.items():
            match = re.search(pattern, evaluation_text, re.IGNORECASE)
            if match:
                scores[score_name] = float(match.group(1))
        
        return {
            "overall_score": scores.get('overall_score', 0),
            "chunk_relevance_score": scores.get('chunk_relevance_score', 0),
            "accuracy_score": scores.get('accuracy_score', 0),
            "completeness_score": scores.get('completeness_score', 0),
            "quality_score": scores.get('quality_score', 0),
            "reasoning": f"Evaluation parsing failed: {error}. Extracted from text: {evaluation_text[:500]}...",
            "recommendations": ["Improve evaluation response parsing"],
            "strengths": [],
            "weaknesses": ["Evaluation system error"],
            "evaluation_success": False,
            "parsing_error": error
        }

def evaluate_sample_response():
    """Demonstrate evaluation with a sample response"""
    sample_data = {
        "user_question": "What is the remote work policy?",
        "system_answer": "Our hybrid work model allows employees to work remotely up to 3 days per week, with mandatory in-office presence on Tuesdays and Thursdays for team collaboration. Remote work requires manager approval and completion of our remote work agreement. All remote workers must maintain a professional home office setup with reliable internet (minimum 25 Mbps upload/download), ergonomic workspace, and appropriate lighting for video calls.",
        "chunks_related": [
            {
                "chunk_id": 8,
                "section_title": "Employee Onboarding & Policies",
                "similarity_score": 0.89,
                "text_preview": "Our hybrid work model allows employees to work remotely up to 3 days per week, with mandatory in-office presence on Tuesdays and Thursdays for team collaboration. Remote work requires manager approval..."
            }
        ]
    }
    
    evaluator = ResponseEvaluator()
    evaluation = evaluator.evaluate_response(
        sample_data["user_question"],
        sample_data["system_answer"],
        sample_data["chunks_related"]
    )
    
    print("Sample Evaluation Results:")
    print("=" * 50)
    print(json.dumps(evaluation, indent=2))

def main():
    """Command line interface for evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate FAQ chatbot responses")
    parser.add_argument("--input", "-i", type=str, help="JSON file with response to evaluate")
    parser.add_argument("--output", "-o", type=str, help="Output file for evaluation results")
    parser.add_argument("--sample", action="store_true", help="Run evaluation on sample data")
    
    args = parser.parse_args()
    
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    if args.sample:
        evaluate_sample_response()
        return
    
    if args.input:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                response_data = json.load(f)
            
            evaluator = ResponseEvaluator()
            evaluation = evaluator.evaluate_response(
                response_data["user_question"],
                response_data["system_answer"],
                response_data["chunks_related"]
            )
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(evaluation, f, indent=2, ensure_ascii=False)
                print(f"Evaluation saved to {args.output}")
            else:
                print(json.dumps(evaluation, indent=2, ensure_ascii=False))
                
        except FileNotFoundError:
            print(f"Error: Input file {args.input} not found")
        except KeyError as e:
            print(f"Error: Missing required field in input: {e}")
        except Exception as e:
            print(f"Error evaluating response: {e}")
    
    else:
        print("Please provide --input file or use --sample for demonstration")
        parser.print_help()

if __name__ == "__main__":
    main()