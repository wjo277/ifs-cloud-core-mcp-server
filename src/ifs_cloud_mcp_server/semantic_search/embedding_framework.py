"""
Comprehensive Embedding Creation Framework

This framework combines all the techniques we've developed:
- AST-based PL/SQL parsing with ConservativePLSQLAnalyzer
- AI-powered code summarization using Qwen3-8B
- UniXcoder embeddings for semantic search
- Robust checkpointing and recovery system
- Batch processing with progress tracking
- Error handling and fallback mechanisms

Author: AI Assistant
Date: August 17, 2025
"""

import json
import logging
import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pickle

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.asyncio import tqdm

from ..plsql_analyzer import ConservativePLSQLAnalyzer
from .ai_summarizer import AISummarizer
from .data_loader import IFSCodeDataset, ChunkingConfig
from .data_structures import CodeChunk


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding creation framework"""

    # Model settings
    model_name: str = "microsoft/unixcoder-base"
    max_tokens: int = 510  # Leave room for special tokens
    batch_size: int = 8

    # AI enhancement settings
    enable_ai_summaries: bool = True
    ai_batch_size: int = 5
    ai_model: str = "qwen3:8b"

    # Processing settings
    max_workers: int = 4
    chunk_overlap: int = 50

    # Checkpointing settings
    checkpoint_frequency: int = 100  # Save every N chunks
    backup_frequency: int = 500  # Create backup every N chunks

    # Output settings
    output_dir: Path = Path("embeddings")
    cache_dir: Path = Path("cache")

    # Quality settings
    validate_embeddings: bool = True
    compute_similarity_metrics: bool = True


@dataclass
class ProcessingStats:
    """Statistics for the embedding creation process"""

    total_files: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    ai_enhanced_chunks: int = 0
    start_time: datetime = None
    last_checkpoint: datetime = None
    estimated_completion: datetime = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            **asdict(self),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_checkpoint": (
                self.last_checkpoint.isoformat() if self.last_checkpoint else None
            ),
            "estimated_completion": (
                self.estimated_completion.isoformat()
                if self.estimated_completion
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingStats":
        """Create from dictionary"""
        for field in ["start_time", "last_checkpoint", "estimated_completion"]:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)


class EmbeddingCheckpoint:
    """Handles checkpointing and recovery for embedding creation"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.checkpoint_dir = config.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def save_checkpoint(
        self,
        chunks: List[CodeChunk],
        embeddings: np.ndarray,
        processed_files: Set[str],
        stats: ProcessingStats,
        checkpoint_id: str = None,
    ) -> str:
        """Save a checkpoint of the current processing state"""
        if checkpoint_id is None:
            checkpoint_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"

        checkpoint_data = {
            "config": asdict(self.config),
            "chunks": chunks,
            "embeddings": embeddings,
            "processed_files": list(processed_files),
            "stats": stats.to_dict(),
            "checkpoint_id": checkpoint_id,
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
        }

        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Also save metadata as JSON for easy inspection
            metadata_path = (
                self.checkpoint_dir / f"checkpoint_{checkpoint_id}_metadata.json"
            )
            metadata = {
                "checkpoint_id": checkpoint_id,
                "stats": stats.to_dict(),
                "num_chunks": len(chunks),
                "embedding_shape": embeddings.shape if embeddings is not None else None,
                "num_processed_files": len(processed_files),
                "created_at": checkpoint_data["created_at"],
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
            return checkpoint_id

        except Exception as e:
            self.logger.error(f"❌ Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, checkpoint_id: str = None) -> Optional[Dict[str, Any]]:
        """Load the most recent or specified checkpoint"""
        try:
            if checkpoint_id is None:
                # Find the most recent checkpoint
                checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
                if not checkpoint_files:
                    return None
                checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            else:
                checkpoint_path = (
                    self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"
                )
                if not checkpoint_path.exists():
                    return None

            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)

            self.logger.info(f"📂 Loaded checkpoint: {checkpoint_path}")
            return checkpoint_data

        except Exception as e:
            self.logger.error(f"❌ Failed to load checkpoint: {e}")
            return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata"""
        checkpoints = []

        for metadata_file in self.checkpoint_dir.glob("checkpoint_*_metadata.json"):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                checkpoints.append(metadata)
            except Exception as e:
                self.logger.warning(
                    f"⚠️ Could not read checkpoint metadata: {metadata_file} - {e}"
                )

        return sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)

    def cleanup_old_checkpoints(self, keep_count: int = 5):
        """Remove old checkpoints, keeping only the most recent ones"""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= keep_count:
            return

        old_checkpoints = checkpoints[keep_count:]

        for checkpoint in old_checkpoints:
            checkpoint_id = checkpoint["checkpoint_id"]
            try:
                # Remove pickle file
                pickle_file = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"
                if pickle_file.exists():
                    pickle_file.unlink()

                # Remove metadata file
                metadata_file = (
                    self.checkpoint_dir / f"checkpoint_{checkpoint_id}_metadata.json"
                )
                if metadata_file.exists():
                    metadata_file.unlink()

                self.logger.info(f"🗑️ Cleaned up old checkpoint: {checkpoint_id}")

            except Exception as e:
                self.logger.warning(
                    f"⚠️ Could not clean up checkpoint {checkpoint_id}: {e}"
                )


class ComprehensiveEmbeddingFramework:
    """
    Comprehensive embedding creation framework with all advanced features:
    - AST-based PL/SQL parsing
    - AI-powered summarization
    - UniXcoder embeddings
    - Robust checkpointing
    - Progress tracking and recovery
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.checkpoint_manager = EmbeddingCheckpoint(config)
        self.ai_summarizer = AISummarizer() if config.enable_ai_summaries else None
        self.plsql_analyzer = ConservativePLSQLAnalyzer()

        # Model components (loaded lazily)
        self.tokenizer = None
        self.model = None
        self.device = None

        # Processing state
        self.stats = ProcessingStats()
        self.processed_files: Set[str] = set()
        self.all_chunks: List[CodeChunk] = []
        self.embeddings: Optional[np.ndarray] = None

        # Create output directories
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        """Load the UniXcoder model and tokenizer"""
        if self.tokenizer is not None:
            return

        self.logger.info(f"🤖 Loading model: {self.config.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name)

            # Set up device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"✅ Model loaded on device: {self.device}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute hash of file for change detection"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(file_path).encode()).hexdigest()

    async def _create_chunks_for_file(self, file_path: Path) -> List[CodeChunk]:
        """Create chunks for a single file using AST-based parsing"""
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse with AST analyzer
            result = self.plsql_analyzer.analyze_file(content)

            chunks = []
            for func_info in result.functions:
                chunk = CodeChunk(
                    content=func_info.full_text,
                    file_path=str(file_path),
                    start_line=func_info.start_line,
                    end_line=func_info.end_line,
                    function_name=func_info.name,
                    chunk_type="function",
                    language="plsql",
                    module_name=self._extract_module_name(file_path),
                    metadata={
                        "file_hash": self._compute_file_hash(file_path),
                        "parsing_method": "ast",
                        "has_body": func_info.has_body,
                        "signature": func_info.signature,
                    },
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            self.logger.error(f"❌ Failed to create chunks for {file_path}: {e}")
            return []

    def _extract_module_name(self, file_path: Path) -> str:
        """Extract module name from file path"""
        # Extract from path like _work/module_name/...
        parts = file_path.parts
        if len(parts) >= 2 and parts[-2] != "_work":
            return parts[-2]
        return "unknown"

    async def _enhance_chunks_with_ai(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Enhance chunks with AI-generated summaries"""
        if not self.ai_summarizer or not self.config.enable_ai_summaries:
            return chunks

        self.logger.info(f"🤖 Enhancing {len(chunks)} chunks with AI summaries...")

        enhanced_chunks = []

        # Process in batches
        for i in range(0, len(chunks), self.config.ai_batch_size):
            batch = chunks[i : i + self.config.ai_batch_size]

            try:
                enhanced_batch = await self.ai_summarizer.enhance_chunks_batch(batch)
                enhanced_chunks.extend(enhanced_batch)
                self.stats.ai_enhanced_chunks += len(enhanced_batch)

            except Exception as e:
                self.logger.warning(
                    f"⚠️ AI enhancement failed for batch {i//self.config.ai_batch_size + 1}: {e}"
                )
                # Fallback to original chunks
                enhanced_chunks.extend(batch)

        return enhanced_chunks

    def _create_embeddings_batch(self, chunks: List[CodeChunk]) -> np.ndarray:
        """Create embeddings for a batch of chunks"""
        if not chunks:
            return np.array([])

        # Prepare texts for embedding
        texts = []
        for chunk in chunks:
            # Use enhanced content if available, otherwise original
            if hasattr(chunk, "ai_summary") and chunk.ai_summary:
                text = f"{chunk.ai_summary}\n\n{chunk.content}"
            else:
                text = chunk.content
            texts.append(text)

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_tokens,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

    async def _process_files_batch(
        self, files: List[Path]
    ) -> Tuple[List[CodeChunk], np.ndarray]:
        """Process a batch of files and create embeddings"""
        all_batch_chunks = []

        # Create chunks for all files
        for file_path in files:
            if str(file_path) in self.processed_files:
                continue

            chunks = await self._create_chunks_for_file(file_path)
            if chunks:
                all_batch_chunks.extend(chunks)
                self.processed_files.add(str(file_path))

        if not all_batch_chunks:
            return [], np.array([])

        # Enhance with AI
        enhanced_chunks = await self._enhance_chunks_with_ai(all_batch_chunks)

        # Create embeddings
        embeddings = self._create_embeddings_batch(enhanced_chunks)

        return enhanced_chunks, embeddings

    def _update_stats(self):
        """Update processing statistics"""
        now = datetime.now()

        if self.stats.processed_chunks > 0:
            elapsed = (now - self.stats.start_time).total_seconds()
            rate = self.stats.processed_chunks / elapsed
            remaining = self.stats.total_chunks - self.stats.processed_chunks
            if rate > 0:
                eta_seconds = remaining / rate
                self.stats.estimated_completion = now + datetime.timedelta(
                    seconds=eta_seconds
                )

    def _should_checkpoint(self) -> bool:
        """Check if we should create a checkpoint"""
        return (
            self.stats.processed_chunks > 0
            and self.stats.processed_chunks % self.config.checkpoint_frequency == 0
        )

    async def create_embeddings(
        self, source_directories: List[Path], resume_from_checkpoint: str = None
    ) -> Dict[str, Any]:
        """
        Create embeddings for all PL/SQL files in source directories

        Args:
            source_directories: List of directories to process
            resume_from_checkpoint: Checkpoint ID to resume from

        Returns:
            Dictionary with processing results and metadata
        """
        self.logger.info("🚀 Starting comprehensive embedding creation...")

        # Try to resume from checkpoint
        if resume_from_checkpoint:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(
                resume_from_checkpoint
            )
            if checkpoint_data:
                self.logger.info("📂 Resuming from checkpoint...")
                self._restore_from_checkpoint(checkpoint_data)

        # Load model
        self._load_model()

        # Initialize statistics
        if not self.stats.start_time:
            self.stats.start_time = datetime.now()

        # Collect all PL/SQL files
        all_files = []
        for source_dir in source_directories:
            if source_dir.exists():
                plsql_files = list(source_dir.rglob("*.pls")) + list(
                    source_dir.rglob("*.sql")
                )
                all_files.extend(plsql_files)

        self.stats.total_files = len(all_files)

        # Estimate total chunks (rough estimate)
        if self.stats.total_chunks == 0:
            self.stats.total_chunks = len(all_files) * 5  # Rough estimate

        self.logger.info(f"📁 Found {len(all_files)} PL/SQL files to process")

        # Process files in batches
        progress_bar = tqdm(total=len(all_files), desc="Processing files", unit="files")

        try:
            batch_size = self.config.batch_size
            for i in range(0, len(all_files), batch_size):
                batch_files = all_files[i : i + batch_size]

                # Process batch
                batch_chunks, batch_embeddings = await self._process_files_batch(
                    batch_files
                )

                if batch_chunks:
                    # Add to collections
                    self.all_chunks.extend(batch_chunks)

                    if self.embeddings is None:
                        self.embeddings = batch_embeddings
                    else:
                        self.embeddings = np.vstack([self.embeddings, batch_embeddings])

                    # Update statistics
                    self.stats.processed_chunks += len(batch_chunks)
                    self._update_stats()

                    # Check for checkpoint
                    if self._should_checkpoint():
                        checkpoint_id = self.checkpoint_manager.save_checkpoint(
                            self.all_chunks,
                            self.embeddings,
                            self.processed_files,
                            self.stats,
                        )
                        self.stats.last_checkpoint = datetime.now()

                progress_bar.update(len(batch_files))
                progress_bar.set_postfix(
                    {
                        "chunks": len(self.all_chunks),
                        "ai_enhanced": self.stats.ai_enhanced_chunks,
                        "processed_files": len(self.processed_files),
                    }
                )

        finally:
            progress_bar.close()

        # Final statistics update
        self.stats.total_chunks = len(self.all_chunks)
        self._update_stats()

        # Save final results
        results = await self._save_final_results()

        # Save final checkpoint
        final_checkpoint_id = self.checkpoint_manager.save_checkpoint(
            self.all_chunks, self.embeddings, self.processed_files, self.stats, "final"
        )

        # Cleanup old checkpoints
        self.checkpoint_manager.cleanup_old_checkpoints()

        self.logger.info("✅ Embedding creation completed successfully!")

        return {
            "total_chunks": len(self.all_chunks),
            "total_files": len(self.processed_files),
            "ai_enhanced_chunks": self.stats.ai_enhanced_chunks,
            "embedding_shape": (
                self.embeddings.shape if self.embeddings is not None else None
            ),
            "final_checkpoint": final_checkpoint_id,
            "stats": self.stats.to_dict(),
            **results,
        }

    def _restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Restore state from checkpoint data"""
        self.all_chunks = checkpoint_data["chunks"]
        self.embeddings = checkpoint_data["embeddings"]
        self.processed_files = set(checkpoint_data["processed_files"])
        self.stats = ProcessingStats.from_dict(checkpoint_data["stats"])

        self.logger.info(f"📂 Restored {len(self.all_chunks)} chunks from checkpoint")

    async def _save_final_results(self) -> Dict[str, Any]:
        """Save final embedding results to disk"""
        output_files = {}

        try:
            # Save embeddings
            embeddings_path = self.config.output_dir / "embeddings.npy"
            if self.embeddings is not None:
                np.save(embeddings_path, self.embeddings)
                output_files["embeddings"] = str(embeddings_path)

            # Save chunks metadata
            chunks_path = self.config.output_dir / "chunks.json"
            chunks_data = [
                {
                    "content": (
                        chunk.content[:500] + "..."
                        if len(chunk.content) > 500
                        else chunk.content
                    ),
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "function_name": chunk.function_name,
                    "chunk_type": chunk.chunk_type,
                    "language": chunk.language,
                    "module_name": chunk.module_name,
                    "metadata": chunk.metadata,
                    "ai_summary": getattr(chunk, "ai_summary", None),
                    "ai_keywords": getattr(chunk, "ai_keywords", None),
                }
                for chunk in self.all_chunks
            ]

            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            output_files["chunks"] = str(chunks_path)

            # Save processing statistics
            stats_path = self.config.output_dir / "processing_stats.json"
            with open(stats_path, "w") as f:
                json.dump(self.stats.to_dict(), f, indent=2)
            output_files["stats"] = str(stats_path)

            # Save configuration
            config_path = self.config.output_dir / "config.json"
            config_dict = asdict(self.config)
            # Convert Path objects to strings
            for key, value in config_dict.items():
                if isinstance(value, Path):
                    config_dict[key] = str(value)

            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            output_files["config"] = str(config_path)

            self.logger.info(f"💾 Results saved to: {self.config.output_dir}")
            return output_files

        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {e}")
            raise

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        return self.checkpoint_manager.list_checkpoints()

    async def validate_embeddings(self) -> Dict[str, Any]:
        """Validate the quality of created embeddings"""
        if self.embeddings is None or len(self.all_chunks) == 0:
            return {"error": "No embeddings to validate"}

        validation_results = {
            "total_embeddings": len(self.embeddings),
            "embedding_dimension": self.embeddings.shape[1],
            "chunks_with_ai_summary": sum(
                1
                for chunk in self.all_chunks
                if hasattr(chunk, "ai_summary") and chunk.ai_summary
            ),
            "average_chunk_length": np.mean(
                [len(chunk.content) for chunk in self.all_chunks]
            ),
            "modules_covered": len(set(chunk.module_name for chunk in self.all_chunks)),
            "embedding_stats": {
                "mean": float(np.mean(self.embeddings)),
                "std": float(np.std(self.embeddings)),
                "min": float(np.min(self.embeddings)),
                "max": float(np.max(self.embeddings)),
            },
        }

        # Test similarity computation
        if len(self.embeddings) > 1:
            from sklearn.metrics.pairwise import cosine_similarity

            sample_similarities = cosine_similarity(
                self.embeddings[: min(10, len(self.embeddings))]
            )
            validation_results["sample_similarity_range"] = {
                "min": float(np.min(sample_similarities)),
                "max": float(np.max(sample_similarities)),
                "mean": float(np.mean(sample_similarities)),
            }

        return validation_results


# Convenience function for easy usage
async def create_comprehensive_embeddings(
    source_directories: List[Path],
    config: EmbeddingConfig = None,
    resume_from: str = None,
) -> Dict[str, Any]:
    """
    Convenience function to create embeddings with all advanced features

    Args:
        source_directories: Directories containing PL/SQL files
        config: Configuration (uses defaults if None)
        resume_from: Checkpoint ID to resume from

    Returns:
        Results dictionary with all processing information
    """
    if config is None:
        config = EmbeddingConfig()

    framework = ComprehensiveEmbeddingFramework(config)
    return await framework.create_embeddings(source_directories, resume_from)


if __name__ == "__main__":
    # Example usage
    async def main():
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Create configuration
        config = EmbeddingConfig(
            enable_ai_summaries=True,
            batch_size=4,
            ai_batch_size=3,
            checkpoint_frequency=50,
            output_dir=Path("comprehensive_embeddings"),
        )

        # Source directories
        source_dirs = [Path("_work")]

        # Create embeddings
        results = await create_comprehensive_embeddings(
            source_directories=source_dirs, config=config
        )

        print("\n🎉 Embedding Creation Complete!")
        print(f"📊 Total chunks: {results['total_chunks']}")
        print(f"📁 Total files: {results['total_files']}")
        print(f"🤖 AI-enhanced chunks: {results['ai_enhanced_chunks']}")
        print(f"📐 Embedding shape: {results['embedding_shape']}")

    # Run the example
    # asyncio.run(main())
