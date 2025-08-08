# rag_pipeline.py
# This file contains the core logic for the Retrieval-Augmented Generation pipeline.

import os
import requests
import faiss
import networkx as nx
import numpy as np
import google.generativeai as genai
import torch # Added for GPU detection
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, genai_key: str, embed_model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initializes the RAG pipeline, loading models and setting up configurations.
        """
        logger.info("Initializing RAG Pipeline...")
        
        # --- Configuration ---
        self.CHUNK_SIZE = 300
        self.OVERLAP = 50
        self.SIMILARITY_THRESHOLD = 0.75 
        self.NEIGHBORS_TO_FETCH = 3
        self.TOP_K_FAISS = 5

        # --- Model and API Setup ---
        if not genai_key:
            raise ValueError("GENAI_KEY is required for the RAG pipeline.")
            
        genai.configure(api_key=genai_key)
        # Updated Gemini model name
        self.gemini = genai.GenerativeModel("gemini-2.0-flash-lite")
        
        # Determine the device to use (GPU if available, otherwise CPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading embedding model: {embed_model_name} onto device: {device}...")
        self.embedder = SentenceTransformer(embed_model_name, device=device)
        logger.info("Embedding model loaded.")

        # --- State variables to hold document-specific data ---
        self.chunks = None
        self.embeddings = None
        self.faiss_index = None
        self.graph = None
        self.is_ready = False
        logger.info("RAG Pipeline initialized.")

    def _download_and_extract_text(self, url: str) -> str:
        """Downloads a PDF from a URL and extracts its text content."""
        logger.info(f"Downloading document from {url}...")
        try:
            local_path = "downloaded_doc.pdf"
            response = requests.get(url, timeout=30)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(local_path, "wb") as f:
                f.write(response.content)
            
            reader = PdfReader(local_path)
            full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            logger.info("Document text extracted successfully.")
            return full_text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

    def _chunk_text(self, text: str) -> list[str]:
        """Splits the text into overlapping chunks."""
        logger.info("Chunking text...")
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i : i + self.CHUNK_SIZE]
            chunks.append(" ".join(chunk))
            i += self.CHUNK_SIZE - self.OVERLAP
        logger.info(f"Created {len(chunks)} chunks.")
        return chunks

    def _get_embeddings(self, chunks: list[str]) -> np.ndarray:
        """Generates and normalizes embeddings for text chunks."""
        logger.info("Generating embeddings...")
        # Prepending "passage: " as recommended by the BGE model documentation for retrieval tasks
        embeddings = self.embedder.encode([f"passage: {c}" for c in chunks], convert_to_numpy=True)
        logger.info("Embeddings generated.")
        return normalize(embeddings)

    def _build_faiss(self, embeddings: np.ndarray):
        """Builds a FAISS index for efficient similarity search."""
        logger.info("Building FAISS index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Using Inner Product (cosine similarity on normalized vectors)
        index.add(embeddings)
        logger.info("FAISS index built.")
        return index

    def _build_graph(self, chunks: list[str], embeddings: np.ndarray):
        """Builds a graph connecting semantically similar chunks."""
        logger.info("Building similarity graph...")
        G = nx.Graph()
        for i, chunk_text in enumerate(chunks):
            G.add_node(i, text=chunk_text)
        
        sim_matrix = cosine_similarity(embeddings)
        edges_added = 0
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                if sim_matrix[i][j] > self.SIMILARITY_THRESHOLD:
                    G.add_edge(i, j, weight=sim_matrix[i][j])
                    edges_added += 1
        logger.info(f"Similarity graph built with {edges_added} edges.")
        return G

    def process_document(self, doc_url: str):
        """
        Main method to process a document: download, chunk, embed, and index.
        This prepares the pipeline for answering questions.
        """
        logger.info("--- Starting new document processing ---")
        raw_text = self._download_and_extract_text(doc_url)
        self.chunks = self._chunk_text(raw_text)
        self.embeddings = self._get_embeddings(self.chunks)
        self.faiss_index = self._build_faiss(self.embeddings)
        self.graph = self._build_graph(self.chunks, self.embeddings)
        self.is_ready = True
        logger.info("--- Document processing complete. Pipeline is ready. ---")

    def _retrieve_chunks(self, query: str) -> list[str]:
        """Retrieves relevant chunks using both FAISS and the graph."""
        if not self.is_ready:
            raise RuntimeError("Pipeline is not ready. Please process a document first.")
        
        logger.info(f"Retrieving chunks for query: '{query[:50]}...'")
        # Prepending "query: " as recommended by the BGE model documentation
        q_embed = self.embedder.encode([f"query: {query}"], convert_to_numpy=True)
        q_embed = normalize(q_embed)
        
        _, indices = self.faiss_index.search(q_embed, self.TOP_K_FAISS)
        
        # Combine FAISS results with graph neighbors for richer context
        base_indices = set(indices[0])
        for i in indices[0]:
            if self.graph.has_node(i):
                neighbors = list(self.graph.neighbors(i))[:self.NEIGHBORS_TO_FETCH]
                base_indices.update(neighbors)
        
        retrieved = [self.chunks[i] for i in sorted(list(base_indices))]
        logger.info(f"Retrieved {len(retrieved)} chunks.")
        return retrieved

    def _generate_answer(self, original_query: str, retrieved_chunks: list[str]) -> str:
        """Generates a final answer using the Gemini model with the retrieved context."""
        logger.info("Generating final answer with Gemini...")
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""You are answering a question using the following insurance policy context make it precise and to the point.

            Context:
            {context}

            Question:
            {original_query}

            Answer: """
        
        try:
            response = self.gemini.generate_content(prompt)
            logger.info("Answer generated successfully.")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error during Gemini content generation: {e}")
            return "Error: Could not generate an answer."

    def answer_questions(self, questions: list[str]) -> list[str]:
        """
        Answers a list of questions based on the processed document.
        """
        if not self.is_ready:
            raise RuntimeError("Pipeline is not ready. Please process a document first.")
            
        final_answers = []
        for question in questions:
            retrieved = self._retrieve_chunks(question)
            answer = self._generate_answer(question, retrieved)
            final_answers.append(answer)
        return final_answers
