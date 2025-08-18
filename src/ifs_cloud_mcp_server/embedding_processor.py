"""Production-Ready Embedding Framework for IFS Cloud.

A clean, sequential embedding framework that incorporates all learnings:
- Pre-processing with PageRank importance rankings
- Sequential processing for reliability
- Progress tracking and ETA estimates
- Checkpoint/cache system
- AI summarization on full files with metadata
- Phi-4 Mini with 64K context optimization
- Ollama CLI integration
- BM25S indexing for hybrid search
- ColBERT-style fusion preparation (see COLBERT_HYBRID_SEARCH_ARCHITECTURE.md)

The framework prepares data for ColBERT hybrid search combining:
- Dense retrieval (FAISS) for semantic understanding
- Sparse retrieval (BM25S) for lexical matching
- Late interaction fusion for optimal relevance
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
import re
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle

# Search engine imports
try:
    import faiss
    import numpy as np
    import bm25s
    import torch
    from transformers import AutoTokenizer, AutoModel

    SEARCH_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Search dependencies not available: {e}")
    SEARCH_DEPENDENCIES_AVAILABLE = False

logger = logging.getLogger(__name__)


class BuiltInPageRankAnalyzer:
    """Built-in PageRank analyzer for PL/SQL files."""

    def __init__(self, work_dir: Path, max_context_tokens: int = 65536):
        self.work_dir = Path(work_dir)
        self.max_context_tokens = max_context_tokens

        # Regex patterns for PL/SQL analysis
        self.api_call_patterns = [
            re.compile(r"([A-Z][a-z]+(?:[A-Z][a-z]*)*_API)\.", re.IGNORECASE),
            re.compile(r"PROCEDURE\s+([A-Z][a-z_]+)", re.IGNORECASE),
            re.compile(r"FUNCTION\s+([A-Z][a-z_]+)", re.IGNORECASE),
        ]

        self.error_message_pattern = re.compile(
            r'Error_SYS\.(?:Record_General|Appl_General)\s*\(\s*[\'"]([^\'"]+)[\'"]',
            re.IGNORECASE | re.MULTILINE,
        )

        self.comment_pattern = re.compile(r"--.*$", re.MULTILINE)

    def extract_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata and references from a PL/SQL file."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Extract API calls
            api_calls = set()
            for pattern in self.api_call_patterns:
                matches = pattern.findall(content)
                api_calls.update(
                    match.upper() if isinstance(match, str) else match[0].upper()
                    for match in matches
                )

            # Count procedures and functions
            procedure_count = len(re.findall(r"\bPROCEDURE\s+", content, re.IGNORECASE))
            function_count = len(re.findall(r"\bFUNCTION\s+", content, re.IGNORECASE))

            # Extract error messages
            error_messages = self.error_message_pattern.findall(content)

            # Count comments
            comments = self.comment_pattern.findall(content)

            # Determine API name from file path
            file_name = file_path.name
            api_name = file_name.replace(".plsql", "") + "_API"
            if not api_name.endswith("_API"):
                api_name = api_name.replace("_API", "") + "_API"

            return {
                "file_path": str(file_path),
                "relative_path": str(file_path.relative_to(self.work_dir)),
                "file_name": file_name,
                "api_name": api_name,
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                "procedure_count": procedure_count,
                "function_count": function_count,
                "error_message_count": len(error_messages),
                "comment_count": len(comments),
                "api_calls": list(api_calls),
            }

        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            return None

    def build_reference_graph(
        self, files_info: List[Dict[str, Any]]
    ) -> Dict[str, Set[str]]:
        """Build reference graph between files."""
        # Create API name to file mapping
        api_to_file = {}
        for info in files_info:
            if info:
                api_to_file[info["api_name"]] = info["file_path"]

        # Build reference graph
        reference_graph = {}
        for info in files_info:
            if not info:
                continue

            file_path = info["file_path"]
            reference_graph[file_path] = set()

            # Find references to other APIs
            for api_call in info["api_calls"]:
                if api_call in api_to_file and api_to_file[api_call] != file_path:
                    reference_graph[file_path].add(api_to_file[api_call])

        return reference_graph

    def calculate_pagerank(
        self,
        reference_graph: Dict[str, Set[str]],
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> Dict[str, float]:
        """Calculate PageRank scores for files."""
        files = list(reference_graph.keys())
        n = len(files)

        if n == 0:
            return {}

        # Initialize PageRank scores
        pagerank = {file: 1.0 / n for file in files}

        for iteration in range(max_iterations):
            new_pagerank = {}

            for file in files:
                # Base score from damping
                new_score = (1 - damping) / n

                # Add contributions from referencing files
                for other_file in files:
                    if file in reference_graph[other_file]:
                        out_links = len(reference_graph[other_file])
                        if out_links > 0:
                            new_score += damping * pagerank[other_file] / out_links

                new_pagerank[file] = new_score

            # Check convergence
            diff = sum(abs(new_pagerank[file] - pagerank[file]) for file in files)
            if diff < tolerance:
                logger.info(f"PageRank converged after {iteration + 1} iterations")
                break

            pagerank = new_pagerank

        return pagerank

    def analyze_files(self) -> Dict[str, Any]:
        """Perform complete PageRank analysis of PL/SQL files."""
        logger.info("🔍 Starting built-in PageRank analysis...")

        # Find all PL/SQL files
        plsql_files = list(self.work_dir.rglob("*.plsql"))
        logger.info(f"Found {len(plsql_files)} PL/SQL files")

        if not plsql_files:
            raise ValueError(f"No PL/SQL files found in {self.work_dir}")

        # Extract file information
        logger.info("📊 Extracting file metadata...")
        files_info = []
        for i, file_path in enumerate(plsql_files):
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(plsql_files)} files")

            info = self.extract_file_info(file_path)
            if info:
                files_info.append(info)

        logger.info(f"Successfully processed {len(files_info)} files")

        # Build reference graph
        logger.info("🔗 Building reference graph...")
        reference_graph = self.build_reference_graph(files_info)

        # Calculate PageRank
        logger.info("⚡ Calculating PageRank scores...")
        pagerank_scores = self.calculate_pagerank(reference_graph)

        # Create reverse reference count (who calls this file)
        reference_counts = {}
        for file_path in reference_graph:
            reference_counts[file_path] = 0

        for file_path, references in reference_graph.items():
            for referenced_file in references:
                reference_counts[referenced_file] += 1

        # Combine all data and calculate importance scores
        file_rankings = []
        for info in files_info:
            if not info:
                continue

            file_path = info["file_path"]
            pagerank_score = pagerank_scores.get(file_path, 0)
            reference_count = reference_counts.get(file_path, 0)

            # Combined importance score
            importance_score = pagerank_score * 10000 + reference_count * 0.8

            file_rankings.append(
                {
                    **info,
                    "pagerank_score": pagerank_score,
                    "reference_count": reference_count,
                    "combined_importance_score": importance_score,
                    "calls_count": len(info["api_calls"]),
                    "called_by_count": reference_count,
                }
            )

        # Sort by importance score
        file_rankings.sort(key=lambda x: x["combined_importance_score"], reverse=True)

        # Add rank numbers
        for i, ranking in enumerate(file_rankings):
            ranking["rank"] = i + 1

        # Prepare final analysis
        analysis_metadata = {
            "work_directory": str(self.work_dir),
            "max_context_tokens": self.max_context_tokens,
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "processing_stats": {
                "total_files_found": len(plsql_files),
                "total_files_processed": len(files_info),
                "total_procedures_found": sum(
                    info["procedure_count"] for info in files_info
                ),
                "total_functions_found": sum(
                    info["function_count"] for info in files_info
                ),
                "total_error_messages_found": sum(
                    info["error_message_count"] for info in files_info
                ),
                "total_comments_found": sum(
                    info["comment_count"] for info in files_info
                ),
            },
        }

        return {"analysis_metadata": analysis_metadata, "file_rankings": file_rankings}


@dataclass
class FileMetadata:
    """Metadata for a single file to be processed."""

    rank: int
    file_path: str
    relative_path: str
    file_name: str
    api_name: str
    combined_importance_score: float
    pagerank_score: float
    reference_count: int
    file_size_mb: float
    procedure_count: int
    function_count: int
    error_message_count: int
    comment_count: int
    called_by_count: int
    calls_count: int


@dataclass
class ProcessingResult:
    """Result of processing a single file."""

    file_metadata: FileMetadata
    success: bool
    processing_time: float
    content_hash: str
    summary: Optional[str] = None
    embedding: Optional[List[float]] = None
    bm25_text: Optional[str] = None  # Text used for BM25S indexing
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None


@dataclass
class ProcessingProgress:
    """Current processing progress and statistics."""

    current_file: int
    total_files: int
    files_processed: int
    files_successful: int
    files_failed: int
    start_time: datetime
    estimated_completion: Optional[datetime]
    current_rate: float  # files per second
    total_processing_time: float
    checkpoint_saved: bool


class UniXCoderEmbeddingGenerator:
    """Generates code embeddings using UniXCoder for semantic search."""

    def __init__(self, model_name: str = "microsoft/unixcoder-base"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None

    def initialize_model(self) -> bool:
        """Initialize the UniXCoder model for embedding generation."""
        if not SEARCH_DEPENDENCIES_AVAILABLE:
            logger.warning("Embedding dependencies not available")
            return False

        try:
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            # Load UniXCoder tokenizer and model
            logger.info(f"Loading UniXCoder model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            logger.info("✅ UniXCoder model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize UniXCoder model: {e}")
            return False

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for the given text using UniXCoder."""
        if not self.model or not self.tokenizer:
            logger.warning("UniXCoder model not initialized")
            return None

        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,  # UniXCoder's max length
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()

            # Convert to CPU numpy array then to list
            embedding_list = embedding.cpu().numpy().tolist()

            return embedding_list

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if not self.model:
            return 768  # Default UniXCoder dimension
        return self.model.config.hidden_size


class FAISSIndexManager:
    """Manages FAISS index creation and persistence for embeddings."""

    def __init__(self, index_dir: Path, embedding_dim: int = 768):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_dim = embedding_dim
        self.faiss_index_file = self.index_dir / "faiss_index.bin"
        self.embeddings_file = self.index_dir / "embeddings.npy"
        self.metadata_file = self.index_dir / "faiss_metadata.json"

        # In-memory storage
        self.embeddings = []
        self.embedding_metadata = []
        self.faiss_index = None

    def add_embedding(self, embedding: List[float], metadata: dict) -> None:
        """Add an embedding and its metadata to the index."""
        self.embeddings.append(embedding)
        self.embedding_metadata.append(metadata)

    def build_faiss_index(self) -> bool:
        """Build FAISS index from collected embeddings."""
        if not SEARCH_DEPENDENCIES_AVAILABLE:
            logger.warning("FAISS not available, skipping index build")
            return False

        if not self.embeddings:
            logger.warning("No embeddings available for FAISS indexing")
            return False

        try:
            logger.info(
                f"Building FAISS index with {len(self.embeddings)} embeddings..."
            )

            # Convert embeddings to numpy array
            embeddings_np = np.array(self.embeddings, dtype=np.float32)

            # Create FAISS index (using IndexFlatIP for cosine similarity)
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_np)

            # Add embeddings to index
            self.faiss_index.add(embeddings_np)

            # Save to disk
            faiss.write_index(self.faiss_index, str(self.faiss_index_file))
            np.save(self.embeddings_file, embeddings_np)

            # Save metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "embedding_count": len(self.embeddings),
                "embedding_dimension": self.embedding_dim,
                "index_type": "FAISS_IndexFlatIP",
                "model": "microsoft/unixcoder-base",
            }
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"✅ FAISS index built successfully: {len(self.embeddings)} embeddings"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            return False

    def load_existing_index(self) -> bool:
        """Load existing FAISS index if available."""
        if not SEARCH_DEPENDENCIES_AVAILABLE:
            return False

        try:
            if (
                self.faiss_index_file.exists()
                and self.embeddings_file.exists()
                and self.metadata_file.exists()
            ):

                self.faiss_index = faiss.read_index(str(self.faiss_index_file))
                embeddings_np = np.load(self.embeddings_file)
                self.embeddings = embeddings_np.tolist()

                logger.info(
                    f"Loaded existing FAISS index with {len(self.embeddings)} embeddings"
                )
                return True

        except Exception as e:
            logger.warning(f"Failed to load existing FAISS index: {e}")

        return False


class BM25SIndexer:
    """Handles BM25S indexing for lexical search capabilities."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # BM25S index files
        self.bm25_index_file = self.index_dir / "bm25s_index.pkl"
        self.bm25_corpus_file = self.index_dir / "bm25s_corpus.pkl"
        self.bm25_metadata_file = self.index_dir / "bm25s_metadata.json"

        # In-memory storage for batch processing
        self.corpus_texts = []
        self.corpus_metadata = []
        self.bm25_index = None

    def prepare_text_for_bm25(self, summary: str, metadata: FileMetadata) -> str:
        """Prepare text for BM25S indexing optimized for ColBERT hybrid fusion.

        Creates structured text that supports both lexical matching and
        late interaction with dense embeddings in ColBERT-style fusion.
        """
        # Structure text for optimal ColBERT fusion
        searchable_parts = [
            # Core identifiers (high weight in sparse retrieval)
            f"file:{metadata.file_name}",
            f"api:{metadata.api_name}",
            # Ranking context (for query-document alignment)
            f"rank:{metadata.rank}",
            f"importance:{metadata.combined_importance_score:.1f}",
            f"references:{metadata.reference_count}",
            # Code structure (for technical queries)
            f"procedures:{metadata.procedure_count}",
            f"functions:{metadata.function_count}",
            f"errors:{metadata.error_message_count}",
            # Rich content (for semantic-lexical bridge)
            summary or "",
        ]

        # Join with spaces for proper tokenization in ColBERT late interaction
        return " ".join(filter(None, searchable_parts))

    def add_document(self, result: ProcessingResult) -> None:
        """Add a processed document to the BM25S corpus."""
        if not result.success or not result.summary:
            return

        bm25_text = self.prepare_text_for_bm25(result.summary, result.file_metadata)

        self.corpus_texts.append(bm25_text)
        self.corpus_metadata.append(
            {
                "file_path": result.file_metadata.file_path,
                "file_name": result.file_metadata.file_name,
                "api_name": result.file_metadata.api_name,
                "rank": result.file_metadata.rank,
                "importance_score": result.file_metadata.combined_importance_score,
                "reference_count": result.file_metadata.reference_count,
                "content_hash": result.content_hash,
            }
        )

        # Update result with BM25 text for checkpointing
        result.bm25_text = bm25_text

    def build_index(self) -> bool:
        """Build the BM25S index from accumulated corpus."""
        if not SEARCH_DEPENDENCIES_AVAILABLE:
            logger.warning("BM25S dependencies not available, skipping index build")
            return False

        if not self.corpus_texts:
            logger.warning("No corpus texts available for BM25S indexing")
            return False

        try:
            logger.info(
                f"Building BM25S index with {len(self.corpus_texts)} documents..."
            )

            # Tokenize corpus (BM25S will handle tokenization)
            corpus_tokens = bm25s.tokenize(self.corpus_texts, show_progress=True)

            # Build BM25S index
            self.bm25_index = bm25s.BM25()
            self.bm25_index.index(corpus_tokens, show_progress=True)

            # Save index to disk
            with open(self.bm25_index_file, "wb") as f:
                pickle.dump(self.bm25_index, f)

            with open(self.bm25_corpus_file, "wb") as f:
                pickle.dump(self.corpus_texts, f)

            # Save metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "document_count": len(self.corpus_texts),
                "index_type": "BM25S",
                "version": "0.2.6+",
            }
            with open(self.bm25_metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"✅ BM25S index built successfully: {len(self.corpus_texts)} documents"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to build BM25S index: {e}")
            return False

    def load_existing_index(self) -> bool:
        """Load existing BM25S index if available."""
        if not SEARCH_DEPENDENCIES_AVAILABLE:
            return False

        try:
            if (
                self.bm25_index_file.exists()
                and self.bm25_corpus_file.exists()
                and self.bm25_metadata_file.exists()
            ):

                with open(self.bm25_index_file, "rb") as f:
                    self.bm25_index = pickle.load(f)

                with open(self.bm25_corpus_file, "rb") as f:
                    self.corpus_texts = pickle.load(f)

                logger.info(
                    f"Loaded existing BM25S index with {len(self.corpus_texts)} documents"
                )
                return True

        except Exception as e:
            logger.warning(f"Failed to load existing BM25S index: {e}")

        return False


class EmbeddingCheckpointManager:
    """Manages checkpoints and caching for embedding processing."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.progress_file = checkpoint_dir / "progress.json"
        self.results_file = checkpoint_dir / "results.jsonl"
        self.metadata_file = checkpoint_dir / "metadata.json"

    def save_progress(self, progress: ProcessingProgress) -> None:
        """Save current progress to checkpoint."""
        progress_data = asdict(progress)
        progress_data["start_time"] = progress.start_time.isoformat()
        if progress.estimated_completion:
            progress_data["estimated_completion"] = (
                progress.estimated_completion.isoformat()
            )

        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, indent=2)

    def load_progress(self) -> Optional[ProcessingProgress]:
        """Load progress from checkpoint if exists."""
        if not self.progress_file.exists():
            return None

        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            data["start_time"] = datetime.fromisoformat(data["start_time"])
            if data.get("estimated_completion"):
                data["estimated_completion"] = datetime.fromisoformat(
                    data["estimated_completion"]
                )

            return ProcessingProgress(**data)
        except Exception as e:
            logger.warning(f"Failed to load progress checkpoint: {e}")
            return None

    def save_result(self, result: ProcessingResult) -> None:
        """Append processing result to results file."""
        result_data = asdict(result)
        result_data["timestamp"] = datetime.now().isoformat()

        with open(self.results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_data) + "\n")

    def load_processed_files(self) -> Set[str]:
        """Load set of already processed files."""
        processed = set()

        if self.results_file.exists():
            try:
                with open(self.results_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line)
                            if result.get("success"):
                                processed.add(result["file_metadata"]["file_path"])
            except Exception as e:
                logger.warning(f"Failed to load processed files: {e}")

        return processed

    def load_all_results(self) -> List[ProcessingResult]:
        """Load all processing results from checkpoint files."""
        results = []

        if self.results_file.exists():
            try:
                with open(self.results_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            result_data = json.loads(line)
                            # Convert back to ProcessingResult
                            result = ProcessingResult(
                                file_path=result_data["file_path"],
                                success=result_data["success"],
                                content_hash=result_data["content_hash"],
                                summary=result_data.get("summary", ""),
                                error_message=result_data.get("error_message", ""),
                                processing_time=result_data.get("processing_time", 0.0),
                                token_count=result_data.get("token_count", 0),
                                embedding=result_data.get("embedding"),
                            )
                            results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load all results: {e}")

        return results

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save processing metadata."""
        metadata["saved_at"] = datetime.now().isoformat()
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


class OllamaProcessor:
    """Handles AI processing using Ollama CLI for optimal performance."""

    def __init__(self, model: str = "phi4-mini:3.8b-q4_K_M", max_tokens: int = 65536):
        self.model = model
        self.max_tokens = max_tokens
        self.context_limit = 64000  # Leave some buffer for response

    def check_ollama_availability(self) -> bool:
        """Check if Ollama is available and model is pulled."""
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0 and self.model in result.stdout
        except Exception as e:
            logger.error(f"Ollama availability check failed: {e}")
            return False

    def ensure_model(self) -> bool:
        """Ensure the model is available, pull if necessary."""
        if self.check_ollama_availability():
            return True

        logger.info(f"Pulling model {self.model}...")
        try:
            result = subprocess.run(
                ["ollama", "pull", self.model],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes for model pull
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False

    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count (chars / 4)."""
        return len(text) // 4

    def extract_valuable_content(
        self, content: str, metadata: FileMetadata
    ) -> Dict[str, List[str]]:
        """Extract valuable signals from PL/SQL content for balanced analysis."""
        lines = content.split("\n")

        signals = {
            "error_messages": [],
            "procedures": [],
            "functions": [],
            "comments": [],
            "api_calls": [],
            "database_operations": [],
        }

        current_block = []
        current_type = None
        in_block_comment = False

        for line in lines:
            line_stripped = line.strip()
            line_upper = line_stripped.upper()

            # Handle block comments
            if "/*" in line:
                in_block_comment = True
            if "*/" in line:
                in_block_comment = False
                if line_stripped.startswith("--") or in_block_comment:
                    signals["comments"].append(line)
                continue

            # Extract error messages (highest priority signal)
            if (
                "Error_SYS.Record_General" in line
                or "RAISE_APPLICATION_ERROR" in line_upper
                or "Error_SYS.Appl_General" in line
                or "RAISE" in line_upper
                and ("EXCEPTION" in line_upper or "-20" in line)
            ):
                signals["error_messages"].append(line.strip())
                continue

            # Extract procedure/function signatures (key identification signal)
            if (
                line_upper.startswith("PROCEDURE ")
                or "PROCEDURE " in line_upper
                and ("AS" in line_upper or "BEGIN" in line_upper)
            ):
                if current_block and current_type:
                    signals[current_type].extend(current_block)
                current_block = [line]
                current_type = "procedures"
                continue

            elif (
                line_upper.startswith("FUNCTION ")
                or "FUNCTION " in line_upper
                and ("RETURN" in line_upper or "AS" in line_upper)
            ):
                if current_block and current_type:
                    signals[current_type].extend(current_block)
                current_block = [line]
                current_type = "functions"
                continue

            # Extract comments (valuable context signal)
            if line_stripped.startswith("--") or in_block_comment:
                signals["comments"].append(line)
                continue

            # Extract API calls (integration points signal)
            if "_API." in line and not line_upper.startswith("PACKAGE"):
                signals["api_calls"].append(line.strip())
                continue

            # Extract database operations (data access signal)
            if any(
                keyword in line_upper
                for keyword in [
                    "SELECT",
                    "INSERT",
                    "UPDATE",
                    "DELETE",
                    "FROM ",
                    "INTO ",
                ]
            ):
                signals["database_operations"].append(line.strip())
                continue

            # Add to current block if we're in a procedure/function
            if current_type and current_block:
                current_block.append(line)
                # Limit block size to prevent any single procedure from dominating
                if len(current_block) > 50:  # Max 50 lines per procedure/function
                    signals[current_type].extend(current_block[:50])
                    current_block = []
                    current_type = None

        # Add any remaining block
        if current_block and current_type:
            signals[current_type].extend(current_block)

        return signals

    def balance_signals(
        self,
        signals: Dict[str, List[str]],
        metadata: FileMetadata,
        available_tokens: int,
    ) -> str:
        """Create balanced content from extracted signals within token limits."""

        # Define signal priorities and allocation percentages
        signal_priorities = {
            "error_messages": 0.25,  # 25% - Critical for understanding failures
            "procedures": 0.30,  # 30% - Main functionality
            "functions": 0.25,  # 25% - Key operations
            "comments": 0.15,  # 15% - Context and documentation
            "api_calls": 0.03,  # 3% - Integration points
            "database_operations": 0.02,  # 2% - Data access patterns
        }

        # Adjust priorities based on file metadata
        if metadata.error_message_count > 10:
            signal_priorities["error_messages"] = 0.35  # Increase if many errors
            signal_priorities["procedures"] = 0.25

        if metadata.procedure_count > metadata.function_count * 2:
            signal_priorities["procedures"] = 0.35  # File is procedure-heavy
            signal_priorities["functions"] = 0.20
        elif metadata.function_count > metadata.procedure_count * 2:
            signal_priorities["functions"] = 0.35  # File is function-heavy
            signal_priorities["procedures"] = 0.20

        # Calculate token allocation for each signal
        balanced_content = []

        for signal_type, priority in signal_priorities.items():
            signal_data = signals.get(signal_type, [])
            if not signal_data:
                continue

            allocated_tokens = int(available_tokens * priority)
            allocated_chars = allocated_tokens * 4  # Rough char estimation

            # Join and truncate signal data
            signal_text = "\n".join(signal_data)
            if len(signal_text) > allocated_chars:
                signal_text = signal_text[:allocated_chars]
                # Try to end at line boundary
                last_newline = signal_text.rfind("\n")
                if (
                    last_newline > allocated_chars * 0.8
                ):  # If we're close to a line boundary
                    signal_text = signal_text[:last_newline]

            if signal_text.strip():
                balanced_content.append(
                    f"\n--- {signal_type.upper().replace('_', ' ')} ---"
                )
                balanced_content.append(signal_text)

        return "\n".join(balanced_content)

    def truncate_content(self, content: str, metadata: FileMetadata) -> Tuple[str, int]:
        """Intelligently extract and balance valuable signals within context limits."""
        estimated_tokens = self.estimate_tokens(content)

        if estimated_tokens <= self.context_limit:
            return content, estimated_tokens

        # Extract valuable signals
        signals = self.extract_valuable_content(content, metadata)

        # Calculate available tokens (reserve space for prompt and response)
        available_tokens = self.context_limit - 2000

        # Create balanced content
        balanced_content = self.balance_signals(signals, metadata, available_tokens)
        final_tokens = self.estimate_tokens(balanced_content)

        return balanced_content, final_tokens

    def create_summary_prompt(self, content: str, metadata: FileMetadata) -> str:
        """Create optimized prompt for file summarization using balanced signals."""
        return f"""Analyze this IFS Cloud PL/SQL file using the provided signals and create a comprehensive summary for semantic search purposes.

FILE METADATA & CONTEXT:
- File: {metadata.file_name}
- API: {metadata.api_name}  
- PageRank Importance: {metadata.combined_importance_score:.2f} (Rank #{metadata.rank} of 9,750)
- System Integration: Referenced by {metadata.reference_count} other files
- Code Structure: {metadata.procedure_count} procedures, {metadata.function_count} functions
- Error Handling: {metadata.error_message_count} error scenarios
- Calls Other APIs: {metadata.calls_count} external dependencies

EXTRACTED SIGNALS (BALANCED FOR SEMANTIC ANALYSIS):
{content}

ANALYSIS INSTRUCTIONS:
Create a detailed summary optimized for semantic search and system understanding:

1. **PRIMARY BUSINESS PURPOSE**: What core business function does this API serve in IFS Cloud?

2. **SYSTEM IMPORTANCE**: Why is this file ranked #{metadata.rank}? What makes it critical to {metadata.reference_count} other components?

3. **KEY FUNCTIONALITY**: Based on the procedures/functions, what are the 3-5 most important operations this API provides?

4. **ERROR SCENARIOS**: From the error messages, what are the main failure cases and business rules this API enforces?

5. **INTEGRATION PATTERNS**: What other systems/APIs does this interact with? What role does it play in larger business processes?

6. **BUSINESS DOMAIN**: Which IFS Cloud functional area (Finance, Manufacturing, HR, etc.) does this primarily support?

7. **DATA OPERATIONS**: What main business entities/tables does this API manage or access?

8. **TECHNICAL ARCHITECTURE**: Any notable patterns, complexity, or architectural considerations?

Focus on business value, system relationships, and searchable concepts rather than implementation details. This summary will be used for semantic similarity matching to help users find relevant code components.
"""

    def process_file(self, content: str, metadata: FileMetadata) -> ProcessingResult:
        """Process a single file with AI summarization."""
        start_time = time.time()
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

        try:
            # Truncate content if necessary
            processed_content, tokens_used = self.truncate_content(content, metadata)

            # Create prompt
            prompt = self.create_summary_prompt(processed_content, metadata)

            # Call Ollama CLI
            logger.debug(f"Processing {metadata.file_name} with {tokens_used} tokens")

            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=300,  # 5 minutes per file
            )

            processing_time = time.time() - start_time

            if result.returncode == 0 and result.stdout.strip():
                return ProcessingResult(
                    file_metadata=metadata,
                    success=True,
                    processing_time=processing_time,
                    content_hash=content_hash,
                    summary=result.stdout.strip(),
                    tokens_used=tokens_used,
                )
            else:
                error_msg = result.stderr or "No output from Ollama"
                return ProcessingResult(
                    file_metadata=metadata,
                    success=False,
                    processing_time=processing_time,
                    content_hash=content_hash,
                    error_message=error_msg,
                    tokens_used=tokens_used,
                )

        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                file_metadata=metadata,
                success=False,
                processing_time=processing_time,
                content_hash=content_hash,
                error_message=str(e),
            )


class ProductionEmbeddingFramework:
    """Production-ready embedding framework with all learnings incorporated."""

    def __init__(
        self,
        work_dir: Path,
        analysis_file: Path,
        checkpoint_dir: Path,
        model: str = "phi4-mini:3.8b-q4_K_M",
        max_files: Optional[int] = None,
    ):
        self.work_dir = Path(work_dir)
        self.analysis_file = Path(analysis_file)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_files = max_files

        # Initialize components
        self.checkpoint_manager = EmbeddingCheckpointManager(self.checkpoint_dir)
        self.ollama_processor = OllamaProcessor(model)
        self.bm25_indexer = BM25SIndexer(self.checkpoint_dir / "search_indexes")
        self.embedding_generator = UniXCoderEmbeddingGenerator()
        self.faiss_manager = FAISSIndexManager(
            self.checkpoint_dir / "search_indexes",
            embedding_dim=768,  # UniXCoder default dimension
        )

        # Initialize built-in analyzer
        self.analyzer = BuiltInPageRankAnalyzer(self.work_dir)

        # Load or generate analysis data
        self.file_rankings = self._load_or_generate_analysis_data()

        logger.info(
            f"Initialized embedding framework with {len(self.file_rankings)} files"
        )

        # Try to load existing indexes
        self.bm25_indexer.load_existing_index()
        self.faiss_manager.load_existing_index()

    def _load_analysis_data(self) -> List[FileMetadata]:
        """Load and parse the comprehensive analysis data."""
        try:
            with open(self.analysis_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            rankings = []
            for item in data.get("file_rankings", []):
                metadata = FileMetadata(**item)
                rankings.append(metadata)

                # Apply max_files limit if specified
                if self.max_files and len(rankings) >= self.max_files:
                    break

            logger.info(f"Loaded {len(rankings)} file rankings from analysis")
            return rankings

        except Exception as e:
            logger.error(f"Failed to load analysis data: {e}")
            raise

    def _load_or_generate_analysis_data(self) -> List[FileMetadata]:
        """Load existing analysis or generate new analysis if missing."""
        # Try to load existing analysis first
        if self.analysis_file.exists():
            logger.info(f"📊 Loading existing analysis from {self.analysis_file}")
            return self._load_analysis_data()

        # Generate new analysis
        logger.info(f"📊 Analysis file not found, generating new PageRank analysis...")
        logger.info(f"   This may take a few minutes for {self.work_dir}...")

        # Run built-in analyzer
        analysis_data = self.analyzer.analyze_files()

        # Save analysis to file
        with open(self.analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Analysis saved to {self.analysis_file}")

        # Convert to FileMetadata format
        rankings = []
        for item in analysis_data.get("file_rankings", []):
            metadata = FileMetadata(**item)
            rankings.append(metadata)

            # Apply max_files limit if specified
            if self.max_files and len(rankings) >= self.max_files:
                break

        logger.info(f"Generated analysis for {len(rankings)} files")
        return rankings

    def _read_file_content(self, file_path: str) -> Optional[str]:
        """Read file content safely."""
        try:
            full_path = self.work_dir / file_path.replace("_work\\", "").replace(
                "_work/", ""
            )
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return None

    def _calculate_progress_stats(
        self,
        current_idx: int,
        start_time: datetime,
        processed_count: int,
        successful_count: int,
        failed_count: int,
    ) -> ProcessingProgress:
        """Calculate current progress and ETA."""
        now = datetime.now()
        elapsed = (now - start_time).total_seconds()

        # Calculate processing rate
        rate = processed_count / elapsed if elapsed > 0 else 0.0

        # Estimate completion time
        remaining_files = len(self.file_rankings) - processed_count
        estimated_completion = None
        if rate > 0:
            remaining_seconds = remaining_files / rate
            estimated_completion = now + timedelta(seconds=remaining_seconds)

        return ProcessingProgress(
            current_file=current_idx + 1,
            total_files=len(self.file_rankings),
            files_processed=processed_count,
            files_successful=successful_count,
            files_failed=failed_count,
            start_time=start_time,
            estimated_completion=estimated_completion,
            current_rate=rate,
            total_processing_time=elapsed,
            checkpoint_saved=False,
        )

    def _print_progress(self, progress: ProcessingProgress) -> None:
        """Print formatted progress information."""
        percent = (progress.files_processed / progress.total_files) * 100

        print(f"\n{'='*60}")
        print(f"🚀 EMBEDDING PROGRESS: {percent:.1f}% Complete")
        print(f"{'='*60}")
        print(f"📊 Files: {progress.files_processed:,}/{progress.total_files:,}")
        print(f"✅ Successful: {progress.files_successful:,}")
        print(f"❌ Failed: {progress.files_failed:,}")
        print(f"⚡ Rate: {progress.current_rate:.2f} files/sec")

        if progress.estimated_completion:
            eta_str = progress.estimated_completion.strftime("%H:%M:%S")
            remaining = progress.estimated_completion - datetime.now()
            remaining_str = str(remaining).split(".")[0]  # Remove microseconds
            print(f"⏰ ETA: {eta_str} (in {remaining_str})")

        elapsed = timedelta(seconds=progress.total_processing_time)
        elapsed_str = str(elapsed).split(".")[0]  # Remove microseconds
        print(f"⏱️  Elapsed: {elapsed_str}")

        if progress.current_file <= len(self.file_rankings):
            current_file = self.file_rankings[progress.current_file - 1]
            print(f"📄 Current: {current_file.file_name}")
            print(f"🎯 Importance: {current_file.combined_importance_score:.1f}")

        print(f"{'='*60}")

    async def run_embedding_pipeline(self, resume: bool = True) -> Dict[str, Any]:
        """Run the complete embedding pipeline with separated AI summarization."""
        logger.info("Starting production embedding pipeline (2-phase approach)")

        # Phase 1: Basic processing and embeddings
        phase1_stats = await self._run_phase1_basic_processing(resume)

        # Phase 2: AI summarization pass
        phase2_stats = await self._run_phase2_ai_summarization(resume)

        # Combine stats
        final_stats = {**phase1_stats, **phase2_stats, "pipeline_completed": True}

        return final_stats

    async def _run_phase1_basic_processing(self, resume: bool = True) -> Dict[str, Any]:
        """Phase 1: Process files for basic embeddings and BM25S indexing without AI summaries."""
        logger.info("🚀 Phase 1: Basic processing and embeddings")

        # Initialize UniXCoder model
        logger.info("Initializing UniXCoder embedding model...")
        if not self.embedding_generator.initialize_model():
            logger.warning(
                "UniXCoder model initialization failed - embeddings will be skipped"
            )
            embedding_available = False
        else:
            embedding_available = True

        # Load checkpoint if resuming
        processed_files = set()
        start_idx = 0
        start_time = datetime.now()
        successful_count = 0
        failed_count = 0

        if resume:
            processed_files = self.checkpoint_manager.load_processed_files()
            progress = self.checkpoint_manager.load_progress()
            if progress:
                start_idx = progress.files_processed
                start_time = progress.start_time
                successful_count = progress.files_successful
                failed_count = progress.files_failed
                logger.info(
                    f"Resuming Phase 1 from file {start_idx + 1}/{len(self.file_rankings)}"
                )

        # Save initial metadata
        metadata = {
            "framework_version": "1.0.0",
            "phase": "basic_processing",
            "total_files": len(self.file_rankings),
            "started_at": start_time.isoformat(),
            "work_directory": str(self.work_dir),
            "analysis_file": str(self.analysis_file),
        }
        self.checkpoint_manager.save_metadata(metadata)

        # Process files sequentially - Phase 1 (basic processing)
        for idx, file_metadata in enumerate(
            self.file_rankings[start_idx:], start=start_idx
        ):

            # Skip if already processed
            if file_metadata.file_path in processed_files:
                continue

            # Read file content
            content = self._read_file_content(file_metadata.file_path)
            if content is None:
                failed_count += 1
                continue

            # Phase 1: Create basic result without AI summarization
            logger.info(
                f"Phase 1: Processing {file_metadata.file_name} (rank {file_metadata.rank})"
            )

            # Create basic result with content hash and metadata
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Create a basic result for Phase 1 (no AI summary yet)
            basic_result = ProcessingResult(
                file_path=file_metadata.file_path,
                success=True,
                content_hash=content_hash,
                summary="",  # Will be filled in Phase 2
                error_message="",
                processing_time=0.0,
                token_count=0,
                embedding=None,
            )

            # Generate UniXCoder embedding directly from raw content (not summary)
            if embedding_available:
                logger.debug(
                    f"Generating UniXCoder embedding for {file_metadata.file_name}"
                )
                # Use truncated content for embedding to avoid memory issues
                truncated_content = content[:8000] if len(content) > 8000 else content
                embedding = self.embedding_generator.generate_embedding(
                    truncated_content
                )
                if embedding:
                    basic_result.embedding = embedding
                    # Add to FAISS manager
                    self.faiss_manager.add_embedding(
                        embedding,
                        {
                            "file_path": file_metadata.file_path,
                            "file_name": file_metadata.file_name,
                            "api_name": file_metadata.api_name,
                            "rank": file_metadata.rank,
                            "importance_score": file_metadata.combined_importance_score,
                            "reference_count": file_metadata.reference_count,
                            "content_hash": content_hash,
                        },
                    )

            # Save basic result (no AI summary yet) and add placeholder to BM25S
            self.checkpoint_manager.save_result(basic_result)
            # For Phase 1, we'll create a basic BM25S document using content excerpts
            basic_result.summary = self._create_content_excerpt(content, file_metadata)
            self.bm25_indexer.add_document(basic_result)

            successful_count += 1
            logger.info(
                f"✅ Phase 1: {file_metadata.file_name} - embedding: {'✓' if basic_result.embedding is not None else '✗'}"
            )

            processed_count = (idx - start_idx) + 1

            # Update progress every 10 files or at the end
            if processed_count % 10 == 0 or processed_count == len(self.file_rankings):
                progress = self._calculate_progress_stats(
                    idx, start_time, processed_count, successful_count, failed_count
                )
                progress.checkpoint_saved = True
                self.checkpoint_manager.save_progress(progress)
                self._print_progress(progress)

        # Phase 1 Final statistics
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Build search indexes from Phase 1 data
        logger.info("🔍 Building Phase 1 search indexes...")
        bm25_success = self.bm25_indexer.build_index()

        faiss_success = False
        if embedding_available:
            logger.info("🧠 Building FAISS embedding index...")
            faiss_success = self.faiss_manager.build_faiss_index()
        else:
            logger.info("⚠️ Skipping FAISS index (UniXCoder not available)")

        phase1_stats = {
            "phase1_completed_at": end_time.isoformat(),
            "phase1_processing_time": total_time,
            "phase1_files_processed": successful_count + failed_count,
            "phase1_files_successful": successful_count,
            "phase1_files_failed": failed_count,
            "phase1_success_rate": (
                successful_count / (successful_count + failed_count)
                if (successful_count + failed_count) > 0
                else 0
            ),
            "phase1_average_rate": (
                (successful_count + failed_count) / total_time if total_time > 0 else 0
            ),
            "bm25_index_built": bm25_success,
            "bm25_documents": (
                len(self.bm25_indexer.corpus_texts) if bm25_success else 0
            ),
            "faiss_index_built": faiss_success,
            "faiss_embeddings": (
                len(self.faiss_manager.embeddings) if faiss_success else 0
            ),
        }

        logger.info("✅ Phase 1 completed: Basic processing and embeddings")
        logger.info(
            f"📊 Phase 1 stats: {successful_count}/{len(self.file_rankings)} files processed"
        )

        return phase1_stats

    def _create_content_excerpt(self, content: str, file_metadata: FileMetadata) -> str:
        """Create a content excerpt for basic BM25S indexing (Phase 1)."""
        # Extract key elements for search indexing
        lines = content.split("\n")

        # Get first few lines of comments/documentation
        comments = []
        procedures = []
        functions = []

        for line in lines[:100]:  # Look at first 100 lines
            stripped = line.strip()
            if stripped.startswith("--") and len(stripped) > 5:
                comments.append(stripped[2:].strip())
            elif "PROCEDURE" in line.upper():
                procedures.append(line.strip())
            elif "FUNCTION" in line.upper():
                functions.append(line.strip())

        # Create structured excerpt
        excerpt_parts = []
        if comments:
            excerpt_parts.append("Comments: " + " | ".join(comments[:3]))
        if procedures:
            excerpt_parts.append("Procedures: " + " | ".join(procedures[:3]))
        if functions:
            excerpt_parts.append("Functions: " + " | ".join(functions[:3]))

        excerpt = " | ".join(excerpt_parts)

        # Fallback to content start if no structured content found
        if not excerpt:
            excerpt = content[:1000]

        return excerpt[:2000]  # Limit excerpt size

    async def _run_phase2_ai_summarization(self, resume: bool = True) -> Dict[str, Any]:
        """Phase 2: AI summarization pass using Ollama."""
        logger.info("🤖 Phase 2: AI Summarization Pass")

        # Check Ollama availability
        if not self.ollama_processor.ensure_model():
            logger.warning("Ollama model not available - skipping AI summarization")
            return {
                "phase2_completed": False,
                "phase2_error": "Ollama model not available",
            }

        start_time = datetime.now()
        summarized_count = 0
        failed_count = 0

        # Get all results that need AI summarization
        all_results = self.checkpoint_manager.load_all_results()

        logger.info(
            f"Phase 2: Processing {len(all_results)} files for AI summarization"
        )

        for idx, result in enumerate(all_results):
            # Skip if already has AI summary
            if (
                result.summary and len(result.summary) > 100
            ):  # Assume AI summaries are longer
                continue

            # Find the file metadata
            file_metadata = None
            for metadata in self.file_rankings:
                if metadata.file_path == result.file_path:
                    file_metadata = metadata
                    break

            if not file_metadata:
                failed_count += 1
                continue

            # Read file content again
            content = self._read_file_content(file_metadata.file_path)
            if content is None:
                failed_count += 1
                continue

            logger.info(
                f"Phase 2: AI summarizing {file_metadata.file_name} ({idx+1}/{len(all_results)})"
            )

            # Process with Ollama
            ai_result = self.ollama_processor.process_file(content, file_metadata)

            if ai_result.success and ai_result.summary:
                # Update the result with AI summary
                result.summary = ai_result.summary
                result.token_count = ai_result.token_count
                result.processing_time = ai_result.processing_time

                # Re-save with AI summary
                self.checkpoint_manager.save_result(result)

                summarized_count += 1
                logger.info(
                    f"✅ Phase 2: {file_metadata.file_name}: {len(ai_result.summary)} chars"
                )
            else:
                failed_count += 1
                logger.warning(
                    f"❌ Phase 2: {file_metadata.file_name}: {ai_result.error_message}"
                )

            # Progress update every 10 files
            if (idx + 1) % 10 == 0:
                logger.info(
                    f"Phase 2 Progress: {idx+1}/{len(all_results)} files processed"
                )

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        phase2_stats = {
            "phase2_completed_at": end_time.isoformat(),
            "phase2_processing_time": total_time,
            "phase2_files_processed": summarized_count + failed_count,
            "phase2_files_successful": summarized_count,
            "phase2_files_failed": failed_count,
            "phase2_success_rate": (
                summarized_count / (summarized_count + failed_count)
                if (summarized_count + failed_count) > 0
                else 0
            ),
            "phase2_average_rate": (
                (summarized_count + failed_count) / total_time if total_time > 0 else 0
            ),
        }

        logger.info("✅ Phase 2 completed: AI Summarization")
        logger.info(f"📊 Phase 2 stats: {summarized_count} files summarized")

        return phase2_stats


# CLI Interface Functions
def setup_embedding_directories(base_dir: Path) -> Tuple[Path, Path, Path]:
    """Set up required directories for embedding processing."""
    work_dir = base_dir / "_work"
    checkpoint_dir = base_dir / "embedding_checkpoints"
    analysis_file = base_dir / "comprehensive_plsql_analysis.json"

    if not work_dir.exists():
        raise FileNotFoundError(f"Work directory not found: {work_dir}")

    # Analysis file will be generated automatically if it doesn't exist
    # (Removed the requirement check since we have built-in analyzer)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return work_dir, checkpoint_dir, analysis_file


async def run_embedding_command(args) -> int:
    """Handle the embedding command."""
    try:
        base_dir = Path.cwd()
        work_dir, checkpoint_dir, analysis_file = setup_embedding_directories(base_dir)

        # Initialize framework
        framework = ProductionEmbeddingFramework(
            work_dir=work_dir,
            analysis_file=analysis_file,
            checkpoint_dir=checkpoint_dir,
            model=args.model,
            max_files=args.max_files,
        )

        # Run the pipeline
        stats = await framework.run_embedding_pipeline(resume=not args.no_resume)

        # Print final summary
        print(f"\n🎉 Embedding pipeline completed!")
        print(f"✅ Successfully processed: {stats['files_successful']:,} files")
        print(f"❌ Failed: {stats['files_failed']:,} files")
        print(f"📊 Success rate: {stats['success_rate']:.1%}")
        print(f"⚡ Average rate: {stats['average_rate']:.2f} files/sec")
        print(f"⏱️  Total time: {timedelta(seconds=stats['total_processing_time'])}")

        if stats.get("bm25_index_built"):
            print(f"🔍 BM25S index built: {stats['bm25_documents']:,} documents")
        else:
            print(f"⚠️  BM25S index build failed (check logs)")

        if stats.get("faiss_index_built"):
            print(f"🧠 FAISS index built: {stats['faiss_embeddings']:,} embeddings")
            print(f"📁 Search indexes saved to: {framework.bm25_indexer.index_dir}")
        elif stats.get("faiss_embeddings", 0) > 0:
            print(
                f"⚠️  FAISS index build failed but {stats['faiss_embeddings']} embeddings generated"
            )
        else:
            print(f"⚠️  UniXCoder embeddings skipped (dependencies not available)")

        return 0

    except Exception as e:
        logger.error(f"Embedding pipeline failed: {e}")
        return 1
