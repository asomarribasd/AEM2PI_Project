#!/usr/bin/env python3
"""
Query Pipeline for RAG-based FAQ Support Chatbot

This script handles user queries by:
1. Converting user questions to embeddings
2. Performing vector search using k-NN/cosine similarity
3. Retrieving relevant chunks
4. Generating answers using OpenAI LLM
5. Returning structured JSON response

Technical Choices:
- Vector Search: Cosine similarity with k-NN for relevant chunk retrieval
- LLM Model: GPT-4o-mini for cost-effective, high-quality response generation
- Search Strategy: Top-K retrieval with relevance threshold filtering
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# Load environment variables
load_dotenv()

class VectorSearchEngine:
    """Vector search engine using cosine similarity"""
    
    def __init__(self, vector_store_path: str):
        """
        Initialize search engine with vector store
        
        Args:
            vector_store_path: Path to vector store JSON file
        """
        self.vector_store_path = vector_store_path
        self.chunks = []
        self.embeddings = []
        self.load_vector_store()
    
    def load_vector_store(self):
        """Load chunks and embeddings from vector store"""
        try:
            with open(self.vector_store_path, 'r', encoding='utf-8') as f:
                store_data = json.load(f)
            
            self.chunks = store_data.get('chunks', [])
            self.embeddings = np.array([chunk['embedding'] for chunk in self.chunks])
            
            print(f"Loaded vector store with {len(self.chunks)} chunks")
            
        except FileNotFoundError:
            print(f"Error: Vector store not found at {self.vector_store_path}")
            print("Please run build_index.py first to create the vector store")
            raise
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5, 
               similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for most relevant chunks using cosine similarity
        
        Args:
            query_embedding: Embedding vector for user query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score to include
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if not query_embedding:
            return []
        
        query_vector = np.array(query_embedding).reshape(1, -1)
        
        # Calculate cosine similarity with all chunks
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        
        # Get indices of top-k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by similarity threshold and create results
        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            
            if similarity_score >= similarity_threshold:
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(similarity_score)
                results.append(chunk)
            else:
                break  # Since we sorted by similarity, we can break early
        
        return results

class AnswerGenerator:
    """Generate answers using OpenAI LLM"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize answer generator
        
        Args:
            model: OpenAI model name for answer generation
        """
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def generate_answer(self, user_question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate answer using relevant chunks and LLM
        
        Args:
            user_question: User's original question
            relevant_chunks: List of relevant chunks from vector search
            
        Returns:
            Generated answer string
        """
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question. Please try rephrasing or contact HR support for assistance."
        
        # Prepare context from relevant chunks
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            section_info = f" (from {chunk.get('section_title', 'FAQ')})" if chunk.get('section_title') else ""
            context_parts.append(f"Context {i}{section_info}:\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Create system prompt for HR FAQ assistant
        system_prompt = """You are an HR support assistant for a SaaS company. Your role is to provide accurate, helpful answers to employee questions based on the company's FAQ documentation.

Instructions:
1. Answer questions using ONLY the provided context information
2. Be concise but comprehensive in your responses
3. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation
4. Maintain a professional, helpful tone
5. Reference specific policies or procedures when relevant
6. If multiple sections are relevant, synthesize the information cohesively

Do not:
- Make up information not found in the context
- Provide advice that contradicts company policies
- Include personal opinions or external information"""
        
        user_prompt = f"""Based on the following context from our HR FAQ documentation, please answer this employee question:

Question: {user_question}

Context:
{context}

Please provide a clear, accurate answer based on the information provided."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for factual accuracy
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I encountered an error while generating an answer. Please try again or contact HR support."

class EmbeddingGenerator:
    """Generate embeddings for user queries (reused from build_index.py)"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for user query"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []

class FAQChatbot:
    """Main chatbot class that orchestrates the RAG pipeline"""
    
    def __init__(self, vector_store_path: str = None):
        """Initialize chatbot with vector store"""
        if vector_store_path is None:
            vector_store_path = os.getenv('VECTOR_DB_PATH', './data/vector_store.json')
        
        self.search_engine = VectorSearchEngine(vector_store_path)
        self.embedding_generator = EmbeddingGenerator()
        self.answer_generator = AnswerGenerator()
        self.top_k = int(os.getenv('TOP_K_CHUNKS', 5))
    
    def answer_question(self, user_question: str, 
                       top_k: Optional[int] = None,
                       similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Process user question and return structured JSON response
        
        Args:
            user_question: User's question
            top_k: Number of chunks to retrieve (optional)
            similarity_threshold: Minimum similarity for chunk inclusion
            
        Returns:
            Dictionary with user_question, system_answer, and chunks_related
        """
        if top_k is None:
            top_k = self.top_k
        
        print(f"Processing question: {user_question}")
        
        # Step 1: Generate query embedding
        print("1. Generating query embedding...")
        query_embedding = self.embedding_generator.generate_embedding(user_question)
        
        if not query_embedding:
            return {
                "user_question": user_question,
                "system_answer": "Error processing your question. Please try again.",
                "chunks_related": []
            }
        
        # Step 2: Search for relevant chunks
        print("2. Searching for relevant chunks...")
        relevant_chunks = self.search_engine.search(
            query_embedding, 
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        print(f"   Found {len(relevant_chunks)} relevant chunks")
        
        # Step 3: Generate answer
        print("3. Generating answer...")
        answer = self.answer_generator.generate_answer(user_question, relevant_chunks)
        
        # Step 4: Format response
        # Clean chunks for response (remove embeddings to reduce size)
        chunks_for_response = []
        for chunk in relevant_chunks:
            chunk_info = {
                'chunk_id': chunk['chunk_id'],
                'section_title': chunk.get('section_title', ''),
                'similarity_score': chunk['similarity_score'],
                'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            }
            chunks_for_response.append(chunk_info)
        
        response = {
            "user_question": user_question,
            "system_answer": answer,
            "chunks_related": chunks_for_response
        }
        
        return response

def main():
    """Command line interface for the chatbot"""
    parser = argparse.ArgumentParser(description="FAQ Support Chatbot")
    parser.add_argument("--question", "-q", type=str, help="Question to ask the chatbot")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--output", "-o", type=str, help="Save response to JSON file")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in .env file")
        return
    
    # Initialize chatbot
    try:
        chatbot = FAQChatbot()
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")
        return
    
    if args.interactive:
        # Interactive mode
        print("FAQ Support Chatbot - Interactive Mode")
        print("Type 'quit' or 'exit' to end the session")
        print("=" * 50)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    print("Please enter a question.")
                    continue
                
                response = chatbot.answer_question(question, top_k=args.top_k)
                
                print(f"\nAnswer: {response['system_answer']}")
                print(f"\nBased on {len(response['chunks_related'])} relevant sections:")
                for chunk in response['chunks_related']:
                    print(f"  - {chunk['section_title']} (similarity: {chunk['similarity_score']:.2f})")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.question:
        # Single question mode
        response = chatbot.answer_question(args.question, top_k=args.top_k)
        
        # Print response
        print(json.dumps(response, indent=2, ensure_ascii=False))
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            print(f"\nResponse saved to {args.output}")
    
    else:
        print("Please provide a question using --question or run in --interactive mode")
        parser.print_help()

if __name__ == "__main__":
    main()