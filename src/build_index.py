#!/usr/bin/env python3
"""
Data Pipeline for RAG-based FAQ Support Chatbot

This script builds the knowledge base by:
1. Loading the FAQ document
2. Chunking text intelligently 
3. Generating embeddings for each chunk
4. Storing embeddings and chunks for retrieval

Technical Choices:
- Chunking Strategy: Recursive character-based splitting with overlap to preserve context
- Embedding Model: OpenAI text-embedding-3-small for high-quality semantic representations
- Storage: JSON format for simplicity and transparency in development
"""

import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
import openai
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

class DocumentChunker:
    """Intelligent text chunking with overlap for context preservation"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize chunker with configurable parameters
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Tokens to overlap between chunks for context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def split_by_sections(self, text: str) -> List[str]:
        """
        Split document by sections (headers) first for logical boundaries
        """
        # Split by markdown headers and section breaks
        sections = re.split(r'\n(?=#{1,3}\s)', text)
        return [section.strip() for section in sections if section.strip()]
    
    def recursive_chunk(self, text: str) -> List[str]:
        """
        Recursively chunk text by different delimiters if needed
        
        Strategy:
        1. Try to keep sections together if they fit
        2. Split by paragraphs if section too large
        3. Split by sentences if paragraph too large
        4. Character split as last resort
        """
        if self.count_tokens(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        
        # First try splitting by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Check if adding this paragraph would exceed chunk size
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if self.count_tokens(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If paragraph itself is too large, split by sentences
                if self.count_tokens(paragraph) > self.chunk_size:
                    sentence_chunks = self._split_by_sentences(paragraph)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split large paragraph by sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self.count_tokens(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If single sentence is too large, character split
                if self.count_tokens(sentence) > self.chunk_size:
                    char_chunks = self._character_split(sentence)
                    chunks.extend(char_chunks)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _character_split(self, text: str) -> List[str]:
        """Last resort: split by characters with word boundaries"""
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word
            
            if self.count_tokens(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def chunk_document(self, document: str) -> List[Dict[str, Any]]:
        """
        Main chunking method that returns structured chunks with metadata
        """
        # First split by logical sections
        sections = self.split_by_sections(document)
        
        all_chunks = []
        chunk_id = 1
        
        for section_idx, section in enumerate(sections):
            # Extract section title if exists
            section_title = ""
            lines = section.split('\n')
            if lines and lines[0].startswith('#'):
                section_title = lines[0].strip('#').strip()
            
            # Chunk each section
            section_chunks = self.recursive_chunk(section)
            
            for chunk_text in section_chunks:
                chunk_metadata = {
                    'chunk_id': chunk_id,
                    'text': chunk_text.strip(),
                    'section_title': section_title,
                    'section_index': section_idx,
                    'token_count': self.count_tokens(chunk_text),
                    'character_count': len(chunk_text)
                }
                all_chunks.append(chunk_metadata)
                chunk_id += 1
        
        return all_chunks

class EmbeddingGenerator:
    """Generate embeddings using OpenAI's API"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize embedding generator
        
        Args:
            model: OpenAI embedding model name
        """
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def generate_batch_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                print(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {e}")
                # Add empty embeddings for failed batch
                all_embeddings.extend([[] for _ in batch_texts])
        
        return all_embeddings

class VectorStore:
    """Simple vector store using JSON for development"""
    
    def __init__(self, storage_path: str):
        """
        Initialize vector store
        
        Args:
            storage_path: Path to save vector store JSON file
        """
        self.storage_path = storage_path
        self.chunks = []
        self.embeddings = []
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add chunks and their embeddings to the store"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:  # Only add if embedding was successfully generated
                chunk['embedding'] = embedding
                self.chunks.append(chunk)
                self.embeddings.append(embedding)
    
    def save(self):
        """Save vector store to JSON file"""
        store_data = {
            'metadata': {
                'total_chunks': len(self.chunks),
                'embedding_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
                'chunk_size': os.getenv('CHUNK_SIZE', 500),
                'chunk_overlap': os.getenv('CHUNK_OVERLAP', 50)
            },
            'chunks': self.chunks
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(store_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.chunks)} chunks to {self.storage_path}")
    
    def load(self) -> bool:
        """Load vector store from JSON file"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                store_data = json.load(f)
            
            self.chunks = store_data.get('chunks', [])
            self.embeddings = [chunk['embedding'] for chunk in self.chunks]
            
            print(f"Loaded {len(self.chunks)} chunks from {self.storage_path}")
            return True
        
        except FileNotFoundError:
            print(f"Vector store not found at {self.storage_path}")
            return False
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False

def build_knowledge_base():
    """Main function to build the knowledge base"""
    print("Building RAG Knowledge Base...")
    print("=" * 50)
    
    # Load configuration
    chunk_size = int(os.getenv('CHUNK_SIZE', 500))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))
    doc_path = './data/faq_document.txt'
    vector_store_path = os.getenv('VECTOR_DB_PATH', './data/vector_store.json')
    
    # 1. Load document
    print("1. Loading FAQ document...")
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
        print(f"   Document loaded: {len(document_text)} characters")
    except FileNotFoundError:
        print(f"   Error: Document not found at {doc_path}")
        return
    
    # 2. Chunk document
    print("2. Chunking document...")
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_document(document_text)
    
    print(f"   Created {len(chunks)} chunks")
    print(f"   Average chunk size: {sum(chunk['token_count'] for chunk in chunks) / len(chunks):.1f} tokens")
    
    # Display chunk statistics
    token_counts = [chunk['token_count'] for chunk in chunks]
    print(f"   Token count range: {min(token_counts)} - {max(token_counts)}")
    
    # 3. Generate embeddings
    print("3. Generating embeddings...")
    if not os.getenv('OPENAI_API_KEY'):
        print("   Error: OPENAI_API_KEY not found in environment variables")
        print("   Please set your OpenAI API key in .env file")
        return
    
    embedding_generator = EmbeddingGenerator()
    chunk_texts = [chunk['text'] for chunk in chunks]
    embeddings = embedding_generator.generate_batch_embeddings(chunk_texts)
    
    # 4. Store chunks and embeddings
    print("4. Saving to vector store...")
    vector_store = VectorStore(vector_store_path)
    vector_store.add_chunks(chunks, embeddings)
    vector_store.save()
    
    # 5. Summary
    print("\nKnowledge Base Built Successfully!")
    print("=" * 50)
    print(f"Total chunks: {len(vector_store.chunks)}")
    print(f"Storage location: {vector_store_path}")
    print(f"Embedding model: {os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')}")
    
    # Display sample chunks
    print("\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {chunk['chunk_id']}:")
        print(f"  Section: {chunk['section_title']}")
        print(f"  Tokens: {chunk['token_count']}")
        print(f"  Preview: {chunk['text'][:100]}...")

if __name__ == "__main__":
    build_knowledge_base()