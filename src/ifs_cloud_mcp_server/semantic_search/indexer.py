"""
FAISS Indexing for IFS Semantic Search
======================================

This module manages vector similarity search using Facebook's FAISS library.
It provides efficient similarity search for production inference, supporting
both exact and approximate nearest neighbor search.
"""

import numpy as np
import pickle
import json
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
import time

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logging.warning("FAISS not available. Using fallback similarity search.")

from .data_structures import CodeChunk, SearchResult


@dataclass
class IndexConfig:
    """
    Configuration for FAISS index creation and management.

    INDEX CONFIGURATION PHILOSOPHY:
    ------------------------------
    FAISS offers many index types with different tradeoffs:

    1. EXACT SEARCH: IndexFlatIP (inner product) for perfect accuracy
    2. APPROXIMATE: IndexIVFFlat for speed vs accuracy tradeoff
    3. COMPRESSED: IndexIVFPQ for memory efficiency
    4. GPU ACCELERATION: GPU variants for ultra-fast search

    We choose based on:
    - Dataset size (number of vectors)
    - Embedding dimension
    - Accuracy requirements
    - Memory constraints
    - Query speed requirements
    """

    # Index type selection
    use_gpu: bool = False
    approximate_search: bool = True  # Use approximate for large datasets

    # Index parameters for IVF (Inverted File) indexes
    n_clusters: int = 100  # Number of clusters for IVF
    n_probe: int = 10  # Number of clusters to search

    # PQ (Product Quantization) parameters for compression
    use_pq: bool = False  # Enable product quantization
    pq_m: int = 8  # Number of PQ codes
    pq_bits: int = 8  # Bits per PQ code

    # Training parameters
    train_size: int = 10000  # Number of vectors to use for training

    # Performance parameters
    batch_size: int = 1000  # Batch size for adding vectors

    def __post_init__(self):
        """Validate and adjust configuration."""
        if not HAS_FAISS:
            logging.warning("FAISS not available, configuration will be ignored")
            return

        # Adjust parameters based on availability
        if self.use_gpu and not hasattr(faiss, "StandardGpuResources"):
            logging.warning("GPU FAISS not available, using CPU")
            self.use_gpu = False

        # For small datasets, use exact search
        if self.train_size < 1000:
            self.approximate_search = False
            logging.info("Small dataset detected, using exact search")


class FAISSIndexManager:
    """
    Manages FAISS indexes for semantic search.

    INDEX MANAGEMENT PHILOSOPHY:
    ---------------------------
    We manage the full lifecycle of similarity indexes:

    1. CREATION: Build indexes from embeddings with optimal configuration
    2. TRAINING: Train index structures (for IVF, PQ) on representative data
    3. POPULATION: Add all vectors with metadata tracking
    4. SEARCH: Fast similarity queries with result ranking
    5. PERSISTENCE: Save/load indexes for production deployment
    6. MAINTENANCE: Update indexes as new code is added

    Production considerations:
    - Memory-mapped indexes for large datasets
    - Batch processing for efficiency
    - Thread-safe search operations
    - Graceful fallback when FAISS unavailable
    """

    def __init__(
        self, embedding_dim: int, config: IndexConfig = None, index_dir: Path = None
    ):
        """
        Initialize the FAISS index manager.

        Args:
            embedding_dim: Dimension of embeddings to index
            config: Index configuration
            index_dir: Directory for saving/loading indexes
        """
        self.embedding_dim = embedding_dim
        self.config = config or IndexConfig()
        self.index_dir = index_dir or Path("indexes/semantic_search")

        # Create index directory
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Initialize FAISS components
        if HAS_FAISS:
            self.index = None
            self.gpu_resources = None
            if self.config.use_gpu:
                try:
                    self.gpu_resources = faiss.StandardGpuResources()
                    logging.info("GPU resources initialized for FAISS")
                except Exception as e:
                    logging.warning(f"GPU initialization failed: {e}, using CPU")
                    self.config.use_gpu = False
        else:
            self.index = None

        # Metadata tracking
        self.chunk_ids = []  # Parallel array of chunk IDs
        self.chunk_metadata = {}  # Full metadata by chunk ID
        self.is_trained = False
        self.total_vectors = 0

        # Statistics
        self.search_stats = {
            "total_queries": 0,
            "total_search_time": 0.0,
            "average_search_time": 0.0,
        }

    def create_index(self, embedding_dim: int = None) -> bool:
        """
        Create a new FAISS index based on configuration.

        INDEX CREATION STRATEGY:
        -----------------------
        We select the optimal index type based on:
        1. Dataset size and search accuracy requirements
        2. Memory constraints
        3. Query speed requirements
        4. Available hardware (CPU vs GPU)

        Returns True if index created successfully, False otherwise.
        """
        if not HAS_FAISS:
            logging.error("Cannot create FAISS index - FAISS not available")
            return False

        if embedding_dim:
            self.embedding_dim = embedding_dim

        try:
            if self.config.approximate_search:
                # Use IVF (Inverted File) index for large datasets
                if self.config.use_pq:
                    # IVF with Product Quantization for memory efficiency
                    quantizer = faiss.IndexFlatIP(self.embedding_dim)
                    self.index = faiss.IndexIVFPQ(
                        quantizer,
                        self.embedding_dim,
                        self.config.n_clusters,
                        self.config.pq_m,
                        self.config.pq_bits,
                    )
                    index_type = "IVFPQ"
                else:
                    # Standard IVF index
                    quantizer = faiss.IndexFlatIP(self.embedding_dim)
                    self.index = faiss.IndexIVFFlat(
                        quantizer, self.embedding_dim, self.config.n_clusters
                    )
                    index_type = "IVFFlat"

                # Set search parameters
                self.index.nprobe = self.config.n_probe

            else:
                # Exact search for small datasets or high accuracy needs
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                index_type = "FlatIP"

            # Move to GPU if requested and available
            if self.config.use_gpu and self.gpu_resources:
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                index_type += "_GPU"

            logging.info(
                f"Created {index_type} index for {self.embedding_dim}D embeddings"
            )
            return True

        except Exception as e:
            logging.error(f"Failed to create FAISS index: {e}")
            return False

    def train_index(self, training_embeddings: np.ndarray) -> bool:
        """
        Train the index on a representative sample of embeddings.

        INDEX TRAINING RATIONALE:
        ------------------------
        Some FAISS indexes (IVF, PQ) need training to learn:
        1. Cluster centroids for partitioning data
        2. Quantization parameters for compression
        3. Optimal parameters for the specific data distribution

        We use a representative sample for efficiency while maintaining quality.
        """
        if not HAS_FAISS or self.index is None:
            logging.error("Cannot train index - FAISS index not available")
            return False

        if not hasattr(self.index, "is_trained"):
            # Index doesn't require training (e.g., FlatIP)
            self.is_trained = True
            return True

        if self.index.is_trained:
            logging.info("Index already trained")
            self.is_trained = True
            return True

        try:
            # Sample training data if too large
            if len(training_embeddings) > self.config.train_size:
                indices = np.random.choice(
                    len(training_embeddings), size=self.config.train_size, replace=False
                )
                train_sample = training_embeddings[indices]
                logging.info(f"Using {len(train_sample)} samples for index training")
            else:
                train_sample = training_embeddings
                logging.info(f"Using all {len(train_sample)} embeddings for training")

            # Ensure embeddings are in correct format
            if train_sample.dtype != np.float32:
                train_sample = train_sample.astype(np.float32)

            # Normalize embeddings for cosine similarity (inner product with normalized vectors)
            faiss.normalize_L2(train_sample)

            # Train the index
            start_time = time.time()
            self.index.train(train_sample)
            training_time = time.time() - start_time

            self.is_trained = True
            logging.info(f"Index training completed in {training_time:.2f} seconds")
            return True

        except Exception as e:
            logging.error(f"Index training failed: {e}")
            return False

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[CodeChunk]) -> bool:
        """
        Add embeddings and corresponding chunks to the index.

        EMBEDDING ADDITION STRATEGY:
        ---------------------------
        We add embeddings in batches for memory efficiency and track
        all metadata needed for search result reconstruction:

        1. Normalize embeddings for cosine similarity
        2. Add to FAISS index in batches
        3. Store parallel metadata arrays
        4. Update statistics and health checks
        """
        if not HAS_FAISS or self.index is None:
            logging.error("Cannot add embeddings - FAISS index not available")
            return False

        if not self.is_trained and hasattr(self.index, "is_trained"):
            if not self.train_index(embeddings):
                logging.error("Failed to train index before adding embeddings")
                return False

        if len(embeddings) != len(chunks):
            logging.error(
                f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks"
            )
            return False

        try:
            # Prepare embeddings
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)

            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)

            # Add embeddings in batches
            batch_size = self.config.batch_size
            total_added = 0

            for i in range(0, len(embeddings), batch_size):
                end_idx = min(i + batch_size, len(embeddings))
                batch_embeddings = embeddings[i:end_idx]
                batch_chunks = chunks[i:end_idx]

                # Add to FAISS index
                self.index.add(batch_embeddings)

                # Update metadata tracking
                for chunk in batch_chunks:
                    self.chunk_ids.append(chunk.chunk_id)
                    self.chunk_metadata[chunk.chunk_id] = chunk

                total_added += len(batch_embeddings)

                if total_added % 1000 == 0:
                    logging.info(
                        f"Added {total_added}/{len(embeddings)} embeddings to index"
                    )

            self.total_vectors += len(embeddings)

            logging.info(f"Successfully added {len(embeddings)} embeddings to index")
            logging.info(f"Total vectors in index: {self.total_vectors}")
            return True

        except Exception as e:
            logging.error(f"Failed to add embeddings to index: {e}")
            return False

    def search(
        self, query_embeddings: np.ndarray, k: int = 10, score_threshold: float = 0.0
    ) -> List[List[SearchResult]]:
        """
        Search for similar embeddings and return ranked results.

        SEARCH STRATEGY:
        ---------------
        1. Normalize query embeddings for cosine similarity
        2. Perform FAISS similarity search
        3. Filter results by score threshold
        4. Reconstruct SearchResult objects with rich metadata
        5. Apply additional ranking/filtering if needed

        Args:
            query_embeddings: Query vectors to search for
            k: Number of results to return per query
            score_threshold: Minimum similarity score to include

        Returns:
            List of search result lists (one per query)
        """
        if not HAS_FAISS or self.index is None:
            logging.error("Cannot search - FAISS index not available")
            return self._fallback_search(query_embeddings, k)

        if self.total_vectors == 0:
            logging.warning("Index is empty - no results available")
            return [[] for _ in range(len(query_embeddings))]

        try:
            start_time = time.time()

            # Prepare query embeddings
            if query_embeddings.dtype != np.float32:
                query_embeddings = query_embeddings.astype(np.float32)

            # Normalize for cosine similarity
            faiss.normalize_L2(query_embeddings)

            # Perform search
            scores, indices = self.index.search(query_embeddings, k)

            # Update statistics
            search_time = time.time() - start_time
            self.search_stats["total_queries"] += len(query_embeddings)
            self.search_stats["total_search_time"] += search_time
            self.search_stats["average_search_time"] = (
                self.search_stats["total_search_time"]
                / self.search_stats["total_queries"]
            )

            # Convert to SearchResult objects
            all_results = []
            for query_idx in range(len(query_embeddings)):
                query_results = []

                for rank, (score, chunk_idx) in enumerate(
                    zip(scores[query_idx], indices[query_idx])
                ):
                    # Skip invalid indices
                    if chunk_idx == -1 or score < score_threshold:
                        continue

                    # Get chunk metadata
                    if chunk_idx < len(self.chunk_ids):
                        chunk_id = self.chunk_ids[chunk_idx]
                        chunk = self.chunk_metadata.get(chunk_id)

                        if chunk:
                            # Create search result
                            result = SearchResult(
                                chunk=chunk,
                                similarity_score=float(score),
                                rank=rank + 1,
                                relevance_explanation=self._generate_relevance_explanation(
                                    chunk, score, rank
                                ),
                            )
                            query_results.append(result)

                all_results.append(query_results)

            logging.debug(
                f"Search completed in {search_time:.4f}s for {len(query_embeddings)} queries"
            )
            return all_results

        except Exception as e:
            logging.error(f"Search failed: {e}")
            return self._fallback_search(query_embeddings, k)

    def _fallback_search(
        self, query_embeddings: np.ndarray, k: int
    ) -> List[List[SearchResult]]:
        """
        Fallback similarity search when FAISS is not available.

        FALLBACK IMPLEMENTATION:
        -----------------------
        Uses simple numpy operations for similarity search:
        1. Compute cosine similarity manually
        2. Sort and select top-k results
        3. Return formatted results

        This is much slower than FAISS but provides functionality.
        """
        logging.info("Using fallback similarity search")

        if not self.chunk_metadata:
            return [[] for _ in range(len(query_embeddings))]

        # Get all stored embeddings (this is inefficient but works for fallback)
        stored_embeddings = []
        stored_chunks = []

        for chunk in self.chunk_metadata.values():
            if chunk.embedding is not None:
                stored_embeddings.append(chunk.embedding)
                stored_chunks.append(chunk)

        if not stored_embeddings:
            logging.warning("No embeddings available for fallback search")
            return [[] for _ in range(len(query_embeddings))]

        stored_embeddings = np.array(stored_embeddings)

        # Normalize embeddings
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_embeddings = query_embeddings / (query_norms + 1e-8)

        stored_norms = np.linalg.norm(stored_embeddings, axis=1, keepdims=True)
        stored_embeddings = stored_embeddings / (stored_norms + 1e-8)

        all_results = []
        for query_embedding in query_embeddings:
            # Compute cosine similarities
            similarities = np.dot(stored_embeddings, query_embedding)

            # Get top-k
            top_k_indices = np.argsort(similarities)[-k:][::-1]

            query_results = []
            for rank, idx in enumerate(top_k_indices):
                chunk = stored_chunks[idx]
                score = similarities[idx]

                result = SearchResult(
                    chunk=chunk,
                    similarity_score=float(score),
                    rank=rank + 1,
                    relevance_explanation=f"Cosine similarity: {score:.3f}",
                )
                query_results.append(result)

            all_results.append(query_results)

        return all_results

    def _generate_relevance_explanation(
        self, chunk: CodeChunk, score: float, rank: int
    ) -> str:
        """Generate explanation for why a result is relevant."""
        explanations = []

        # Score-based explanation
        if score > 0.8:
            explanations.append("Very high semantic similarity")
        elif score > 0.6:
            explanations.append("High semantic similarity")
        elif score > 0.4:
            explanations.append("Moderate semantic similarity")
        else:
            explanations.append("Lower semantic similarity")

        # Content-based explanation
        if chunk.function_name:
            explanations.append(f"matches function '{chunk.function_name}'")

        if chunk.business_terms:
            terms = ", ".join(chunk.business_terms[:3])
            explanations.append(f"contains business concepts: {terms}")

        if chunk.module:
            explanations.append(f"from {chunk.module} module")

        return "; ".join(explanations)

    def save_index(self, name: str = "semantic_search") -> bool:
        """
        Save the index to disk for persistence.

        PERSISTENCE STRATEGY:
        --------------------
        We save:
        1. FAISS index file (binary format)
        2. Metadata mappings (chunk IDs, metadata)
        3. Configuration and statistics
        4. Version information for compatibility
        """
        if not HAS_FAISS or self.index is None:
            logging.error("Cannot save index - FAISS index not available")
            return False

        try:
            # Save FAISS index
            index_path = self.index_dir / f"{name}.faiss"
            faiss.write_index(self.index, str(index_path))

            # Save metadata
            metadata = {
                "chunk_ids": self.chunk_ids,
                "chunk_metadata": {
                    chunk_id: {
                        "chunk_id": chunk.chunk_id,
                        "file_path": chunk.file_path,
                        "function_name": chunk.function_name,
                        "module": chunk.module,
                        "layer": chunk.layer,
                        "business_terms": chunk.business_terms,
                        "api_calls": chunk.api_calls,
                        "has_error_handling": chunk.has_error_handling,
                        "complexity_score": chunk.complexity_score,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "raw_content": chunk.raw_content,
                    }
                    for chunk_id, chunk in self.chunk_metadata.items()
                },
                "total_vectors": self.total_vectors,
                "embedding_dim": self.embedding_dim,
                "config": {
                    "use_gpu": self.config.use_gpu,
                    "approximate_search": self.config.approximate_search,
                    "n_clusters": self.config.n_clusters,
                    "n_probe": self.config.n_probe,
                },
                "stats": self.search_stats,
                "version": "1.0",
            }

            metadata_path = self.index_dir / f"{name}_metadata.pkl"
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)

            # Save config as JSON for human readability
            config_path = self.index_dir / f"{name}_config.json"
            with open(config_path, "w") as f:
                json.dump(metadata["config"], f, indent=2)

            logging.info(f"Index saved: {index_path}")
            logging.info(f"Metadata saved: {metadata_path}")
            return True

        except Exception as e:
            logging.error(f"Failed to save index: {e}")
            return False

    def load_index(self, name: str = "semantic_search") -> bool:
        """
        Load index from disk.

        LOADING STRATEGY:
        ----------------
        1. Load FAISS index file
        2. Restore metadata mappings
        3. Validate compatibility
        4. Initialize GPU resources if needed
        """
        if not HAS_FAISS:
            logging.error("Cannot load index - FAISS not available")
            return False

        try:
            index_path = self.index_dir / f"{name}.faiss"
            metadata_path = self.index_dir / f"{name}_metadata.pkl"

            if not index_path.exists() or not metadata_path.exists():
                logging.error(f"Index files not found: {index_path}")
                return False

            # Load FAISS index
            self.index = faiss.read_index(str(index_path))

            # Move to GPU if requested
            if self.config.use_gpu and self.gpu_resources:
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                logging.info("Moved index to GPU")

            # Load metadata
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            # Restore state
            self.chunk_ids = metadata["chunk_ids"]
            self.total_vectors = metadata["total_vectors"]
            self.embedding_dim = metadata["embedding_dim"]
            self.search_stats = metadata.get("stats", {})

            # Reconstruct chunk metadata
            self.chunk_metadata = {}
            for chunk_id, chunk_data in metadata["chunk_metadata"].items():
                # Create CodeChunk from saved data
                chunk = CodeChunk(
                    chunk_id=chunk_data["chunk_id"],
                    file_path=chunk_data["file_path"],
                    start_line=chunk_data["start_line"],
                    end_line=chunk_data["end_line"],
                    raw_content=chunk_data["raw_content"],
                    processed_content=chunk_data.get("processed_content", ""),
                    chunk_type=chunk_data.get("chunk_type", "unknown"),
                    function_name=chunk_data.get("function_name"),
                    module=chunk_data.get("module"),
                    layer=chunk_data.get("layer"),
                    business_terms=chunk_data.get("business_terms", []),
                    api_calls=chunk_data.get("api_calls", []),
                    has_error_handling=chunk_data.get("has_error_handling", False),
                    complexity_score=chunk_data.get("complexity_score", 0.0),
                )
                self.chunk_metadata[chunk_id] = chunk

            self.is_trained = True

            logging.info(
                f"Index loaded: {len(self.chunk_metadata)} chunks, {self.total_vectors} vectors"
            )
            return True

        except Exception as e:
            logging.error(f"Failed to load index: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics and health information."""
        stats = {
            "total_vectors": self.total_vectors,
            "embedding_dimension": self.embedding_dim,
            "is_trained": self.is_trained,
            "search_stats": self.search_stats.copy(),
            "has_faiss": HAS_FAISS,
            "config": {
                "approximate_search": self.config.approximate_search,
                "use_gpu": self.config.use_gpu,
                "n_clusters": self.config.n_clusters,
                "n_probe": self.config.n_probe,
            },
        }

        if HAS_FAISS and self.index:
            stats["index_type"] = type(self.index).__name__
            stats["ntotal"] = self.index.ntotal

            if hasattr(self.index, "nprobe"):
                stats["nprobe"] = self.index.nprobe

        return stats

    def clear(self):
        """Clear the index and reset all data."""
        self.index = None
        self.chunk_ids = []
        self.chunk_metadata = {}
        self.is_trained = False
        self.total_vectors = 0
        self.search_stats = {
            "total_queries": 0,
            "total_search_time": 0.0,
            "average_search_time": 0.0,
        }

        logging.info("Index cleared")


# Utility functions
def optimize_index_config(
    dataset_size: int, embedding_dim: int, memory_limit_gb: float = 4.0
) -> IndexConfig:
    """
    Automatically optimize index configuration based on dataset characteristics.

    OPTIMIZATION STRATEGY:
    ---------------------
    Based on dataset size and available memory:
    - Small (<10K): Exact search
    - Medium (10K-1M): IVF with moderate clusters
    - Large (>1M): IVF with PQ compression
    """
    config = IndexConfig()

    if dataset_size < 10000:
        # Small dataset - use exact search
        config.approximate_search = False
        logging.info("Small dataset: using exact search")

    elif dataset_size < 1000000:
        # Medium dataset - use IVF
        config.approximate_search = True
        config.n_clusters = min(dataset_size // 100, 1000)
        config.n_probe = min(config.n_clusters // 10, 50)
        config.use_pq = False
        logging.info(f"Medium dataset: using IVF with {config.n_clusters} clusters")

    else:
        # Large dataset - use IVF with PQ
        config.approximate_search = True
        config.n_clusters = min(dataset_size // 1000, 4000)
        config.n_probe = min(config.n_clusters // 20, 100)

        # Estimate memory usage and enable PQ if needed
        estimated_memory = dataset_size * embedding_dim * 4 / (1024**3)  # GB
        if estimated_memory > memory_limit_gb:
            config.use_pq = True
            config.pq_m = min(embedding_dim // 8, 64)
            logging.info(f"Large dataset with memory constraints: using IVFPQ")
        else:
            logging.info(f"Large dataset: using IVF with {config.n_clusters} clusters")

    return config


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the indexer
    print("Testing FAISS Index Manager...")

    # Create test data
    embedding_dim = 128
    num_vectors = 1000

    # Generate random embeddings
    embeddings = np.random.randn(num_vectors, embedding_dim).astype(np.float32)

    # Create dummy chunks
    from .data_structures import CodeChunk

    chunks = []
    for i in range(num_vectors):
        chunk = CodeChunk(
            chunk_id=f"test_chunk_{i}",
            file_path=f"/test/file_{i}.plsql",
            start_line=1,
            end_line=10,
            raw_content=f"test content {i}",
            processed_content=f"processed {i}",
            chunk_type="test",
            function_name=f"test_func_{i}",
            module="TEST",
            layer="business",
        )
        chunk.embedding = embeddings[i]
        chunks.append(chunk)

    # Create and test index manager
    config = optimize_index_config(num_vectors, embedding_dim)
    manager = FAISSIndexManager(embedding_dim, config)

    # Create index
    if manager.create_index():
        print("✓ Index created successfully")
    else:
        print("✗ Index creation failed")
        exit(1)

    # Add embeddings
    if manager.add_embeddings(embeddings, chunks):
        print("✓ Embeddings added successfully")
    else:
        print("✗ Adding embeddings failed")

    # Test search
    query_embeddings = np.random.randn(5, embedding_dim).astype(np.float32)
    results = manager.search(query_embeddings, k=10)

    print(f"✓ Search completed, found {len(results)} result sets")
    for i, result_set in enumerate(results):
        print(f"  Query {i}: {len(result_set)} results")
        for j, result in enumerate(result_set[:3]):  # Show top 3
            print(
                f"    {j+1}. {result.chunk.function_name} (score: {result.similarity_score:.3f})"
            )

    # Test save/load
    if manager.save_index("test_index"):
        print("✓ Index saved successfully")

        # Create new manager and load
        manager2 = FAISSIndexManager(embedding_dim, config)
        if manager2.load_index("test_index"):
            print("✓ Index loaded successfully")

            # Test search on loaded index
            results2 = manager2.search(query_embeddings[:1], k=5)
            print(f"✓ Search on loaded index: {len(results2[0])} results")
        else:
            print("✗ Index loading failed")
    else:
        print("✗ Index saving failed")

    # Show statistics
    stats = manager.get_statistics()
    print("\nIndex Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("All tests completed!")
