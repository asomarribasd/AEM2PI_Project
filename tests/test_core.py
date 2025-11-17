#!/usr/bin/env python3
"""
Test Suite for RAG-based FAQ Support Chatbot

Tests core functionality including:
- Document chunking and text processing
- Embedding generation and storage
- Vector search and retrieval
- Response generation and formatting
- Evaluation system
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from build_index import DocumentChunker, EmbeddingGenerator, VectorStore
from query import VectorSearchEngine, AnswerGenerator, FAQChatbot
from evaluator import ResponseEvaluator

class TestDocumentChunker(unittest.TestCase):
    """Test document chunking functionality"""
    
    def setUp(self):
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        
        # Sample document for testing
        self.sample_doc = """# HR Policies

## Remote Work Policy
Our company supports remote work. Employees can work from home up to 3 days per week.
Manager approval is required for remote work arrangements.

## Benefits Overview
We offer comprehensive health insurance and retirement benefits.
All employees are eligible after 90 days of employment.

### Health Insurance
Multiple plan options are available including PPO and HMO plans.
Company pays 80% of premium costs for most plans.

### Retirement Plans
401k with company match available.
Immediate vesting for company contributions."""
    
    def test_token_counting(self):
        """Test token counting functionality"""
        text = "Hello world, this is a test."
        token_count = self.chunker.count_tokens(text)
        self.assertIsInstance(token_count, int)
        self.assertGreater(token_count, 0)
    
    def test_section_splitting(self):
        """Test document splitting by sections"""
        sections = self.chunker.split_by_sections(self.sample_doc)
        self.assertGreater(len(sections), 1)
        # First section should contain HR Policies
        self.assertIn("HR Policies", sections[0])
        # Should have sections for Remote Work and Benefits
        section_text = "\n".join(sections)
        self.assertIn("Remote Work Policy", section_text)
        self.assertIn("Benefits Overview", section_text)
    
    def test_chunk_document(self):
        """Test complete document chunking"""
        chunks = self.chunker.chunk_document(self.sample_doc)
        
        # Should create multiple chunks
        self.assertGreater(len(chunks), 3)
        
        # Each chunk should have required metadata
        for chunk in chunks:
            self.assertIn('chunk_id', chunk)
            self.assertIn('text', chunk)
            self.assertIn('token_count', chunk)
            self.assertIn('section_title', chunk)
            
            # Check token count is reasonable
            self.assertGreater(chunk['token_count'], 0)
            self.assertLessEqual(chunk['token_count'], self.chunker.chunk_size * 1.2)  # Allow some overflow
    
    def test_chunk_ids_unique(self):
        """Test that chunk IDs are unique and sequential"""
        chunks = self.chunker.chunk_document(self.sample_doc)
        chunk_ids = [chunk['chunk_id'] for chunk in chunks]
        
        # Should be unique
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        
        # Should be sequential starting from 1
        self.assertEqual(min(chunk_ids), 1)
        self.assertEqual(max(chunk_ids), len(chunks))

class TestVectorStore(unittest.TestCase):
    """Test vector store functionality"""
    
    def setUp(self):
        # Create temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.store_path = self.temp_file.name
        self.vector_store = VectorStore(self.store_path)
        
        # Sample chunks and embeddings
        self.sample_chunks = [
            {
                'chunk_id': 1,
                'text': 'Sample text 1',
                'token_count': 10,
                'section_title': 'Section 1'
            },
            {
                'chunk_id': 2,
                'text': 'Sample text 2',
                'token_count': 15,
                'section_title': 'Section 2'
            }
        ]
        
        self.sample_embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
    
    def tearDown(self):
        # Clean up temporary file
        if os.path.exists(self.store_path):
            os.unlink(self.store_path)
    
    def test_add_chunks(self):
        """Test adding chunks to vector store"""
        self.vector_store.add_chunks(self.sample_chunks, self.sample_embeddings)
        
        self.assertEqual(len(self.vector_store.chunks), 2)
        self.assertEqual(len(self.vector_store.embeddings), 2)
        
        # Check embeddings are added to chunks
        for chunk in self.vector_store.chunks:
            self.assertIn('embedding', chunk)
    
    def test_save_and_load(self):
        """Test saving and loading vector store"""
        self.vector_store.add_chunks(self.sample_chunks, self.sample_embeddings)
        
        # Save
        self.vector_store.save()
        self.assertTrue(os.path.exists(self.store_path))
        
        # Load into new instance
        new_store = VectorStore(self.store_path)
        success = new_store.load()
        
        self.assertTrue(success)
        self.assertEqual(len(new_store.chunks), 2)
        self.assertEqual(len(new_store.embeddings), 2)
    
    def test_mismatched_chunks_embeddings(self):
        """Test error handling for mismatched chunks and embeddings"""
        with self.assertRaises(ValueError):
            self.vector_store.add_chunks(self.sample_chunks, [[0.1, 0.2]])  # Only one embedding

class TestVectorSearchEngine(unittest.TestCase):
    """Test vector search functionality"""
    
    def setUp(self):
        # Create temporary vector store
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.store_path = self.temp_file.name
        
        # Create sample vector store data
        store_data = {
            'metadata': {'total_chunks': 3},
            'chunks': [
                {
                    'chunk_id': 1,
                    'text': 'Remote work policy information',
                    'section_title': 'Policies',
                    'embedding': [0.8, 0.1, 0.1]
                },
                {
                    'chunk_id': 2,
                    'text': 'Benefits and compensation details',
                    'section_title': 'Benefits', 
                    'embedding': [0.1, 0.8, 0.1]
                },
                {
                    'chunk_id': 3,
                    'text': 'IT support and technical help',
                    'section_title': 'IT Support',
                    'embedding': [0.1, 0.1, 0.8]
                }
            ]
        }
        
        with open(self.store_path, 'w') as f:
            json.dump(store_data, f)
        
        self.search_engine = VectorSearchEngine(self.store_path)
    
    def tearDown(self):
        if os.path.exists(self.store_path):
            os.unlink(self.store_path)
    
    def test_load_vector_store(self):
        """Test loading vector store"""
        self.assertEqual(len(self.search_engine.chunks), 3)
        self.assertEqual(self.search_engine.embeddings.shape, (3, 3))
    
    def test_search_functionality(self):
        """Test search with query embedding"""
        # Query similar to first chunk (remote work)
        query_embedding = [0.9, 0.05, 0.05]
        
        results = self.search_engine.search(query_embedding, top_k=2)
        
        # Should return results
        self.assertGreater(len(results), 0)
        
        # First result should be most similar (chunk 1)
        self.assertEqual(results[0]['chunk_id'], 1)
        
        # Should have similarity scores
        for result in results:
            self.assertIn('similarity_score', result)
            self.assertGreater(result['similarity_score'], 0)
    
    def test_similarity_threshold(self):
        """Test similarity threshold filtering"""
        # Query with very low similarity
        query_embedding = [0.0, 0.0, 0.0]
        
        results = self.search_engine.search(
            query_embedding, 
            top_k=5, 
            similarity_threshold=0.9
        )
        
        # Should return no results due to high threshold
        self.assertEqual(len(results), 0)
    
    def test_empty_query_embedding(self):
        """Test handling of empty query embedding"""
        results = self.search_engine.search([])
        self.assertEqual(len(results), 0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        # Mock OpenAI API calls
        self.mock_openai = patch('openai.OpenAI')
        self.mock_client = self.mock_openai.start()
        
        # Mock embedding response
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        self.mock_client.return_value.embeddings.create.return_value = mock_embedding_response
        
        # Mock chat response
        mock_chat_response = Mock()
        mock_chat_response.choices = [Mock(message=Mock(content="Test answer"))]
        self.mock_client.return_value.chat.completions.create.return_value = mock_chat_response
    
    def tearDown(self):
        self.mock_openai.stop()
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_embedding_generation(self):
        """Test embedding generation"""
        generator = EmbeddingGenerator()
        embedding = generator.generate_embedding("test text")
        
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 3)  # Based on mock
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_answer_generation(self):
        """Test answer generation"""
        generator = AnswerGenerator()
        
        chunks = [
            {
                'chunk_id': 1,
                'text': 'Remote work is allowed up to 3 days per week',
                'section_title': 'Policies'
            }
        ]
        
        answer = generator.generate_answer("What is the remote work policy?", chunks)
        
        self.assertIsInstance(answer, str)
        self.assertEqual(answer, "Test answer")  # Based on mock
    
    def test_empty_chunks_handling(self):
        """Test handling of empty chunks"""
        generator = AnswerGenerator()
        answer = generator.generate_answer("Test question", [])
        
        self.assertIn("couldn't find relevant information", answer.lower())

class TestResponseEvaluator(unittest.TestCase):
    """Test response evaluation functionality"""
    
    def setUp(self):
        # Mock OpenAI API
        self.mock_openai = patch('openai.OpenAI')
        self.mock_client = self.mock_openai.start()
        
        # Mock evaluation response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="""{
            "chunk_relevance_score": 8,
            "accuracy_score": 9,
            "completeness_score": 7,
            "quality_score": 8,
            "overall_score": 8.0,
            "reasoning": "Good response with minor issues",
            "recommendations": ["Improve completeness"],
            "strengths": ["Accurate information"],
            "weaknesses": ["Could be more complete"]
        }"""))]
        
        self.mock_client.return_value.chat.completions.create.return_value = mock_response
        
        self.evaluator = ResponseEvaluator()
    
    def tearDown(self):
        self.mock_openai.stop()
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_response_evaluation(self):
        """Test complete response evaluation"""
        evaluation = self.evaluator.evaluate_response(
            "What is the remote work policy?",
            "Remote work is allowed up to 3 days per week",
            [{'chunk_id': 1, 'text': 'Remote work policy...', 'section_title': 'Policies'}]
        )
        
        # Check required fields
        required_fields = [
            'chunk_relevance_score', 'accuracy_score', 
            'completeness_score', 'quality_score', 'overall_score'
        ]
        
        for field in required_fields:
            self.assertIn(field, evaluation)
            self.assertIsInstance(evaluation[field], (int, float))
        
        # Check calculated overall score
        self.assertIn('calculated_overall_score', evaluation)
        
        # Check list fields
        for field in ['recommendations', 'strengths', 'weaknesses']:
            self.assertIn(field, evaluation)
            self.assertIsInstance(evaluation[field], list)

def run_performance_tests():
    """Run performance tests for chunking large documents"""
    print("\nRunning Performance Tests...")
    print("=" * 50)
    
    # Generate large document
    large_doc = "# Section {}\n\n" + "This is a test paragraph with multiple sentences. " * 20
    large_doc = "\n\n".join([large_doc.format(i) for i in range(50)])
    
    import time
    
    start_time = time.time()
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_document(large_doc)
    end_time = time.time()
    
    print(f"Document size: {len(large_doc)} characters")
    print(f"Chunks created: {len(chunks)}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Average chunk size: {sum(chunk['token_count'] for chunk in chunks) / len(chunks):.1f} tokens")

if __name__ == '__main__':
    # Run unit tests
    print("Running FAQ Chatbot Test Suite")
    print("=" * 50)
    
    # Configure test verbosity
    unittest.main(verbosity=2, exit=False, argv=[''])
    
    # Run performance tests
    run_performance_tests()
    
    print("\nTest Suite Completed!")