"""
Simple Sequential Production Embedding Framework
===============================================

Simple, clean sequential processing with better progress reporting:
- Processes files one by one in batches of 50
- Shows current module and progress clearly
- Estimates time remaining
- Uses existing cache to avoid wasting work
- Suppresses HTTP request logs for cleaner output

Author: AI Assistant
Date: August 17, 2025
"""

import json
import logging
import asyncio
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.asyncio import tqdm

from src.ifs_cloud_mcp_server.plsql_analyzer import ConservativePLSQLAnalyzer
from src.ifs_cloud_mcp_server.semantic_search.ai_summarizer import AISummarizer
from src.ifs_cloud_mcp_server.semantic_search.production_data_structures import (
    ProductionCodeChunk,
    CodeContentLoader,
    convert_to_production_chunk,
)
from src.ifs_cloud_mcp_server.semantic_search.production_embedding_framework import (
    ProductionEmbeddingConfig,
    ProductionProcessingStats,
    ProductionEmbeddingFramework,
)


class SimpleSequentialFramework(ProductionEmbeddingFramework):
    """
    Simple sequential processing framework with clear progress reporting
    """

    def __init__(self, config: ProductionEmbeddingConfig):
        super().__init__(config)
        self.batch_size_files = 50  # Process 50 files at a time
        self.start_time = None
        self.processed_file_count = 0

        # Suppress HTTP request logs from ollama
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    def _estimate_remaining_time(self, processed_files: int, total_files: int) -> str:
        """Estimate remaining time based on current progress"""
        if not self.start_time or processed_files == 0:
            return "calculating..."

        elapsed = time.time() - self.start_time
        rate = processed_files / elapsed  # files per second

        if rate == 0:
            return "calculating..."

        remaining_files = total_files - processed_files
        remaining_seconds = remaining_files / rate

        # Convert to human readable format
        if remaining_seconds < 60:
            return f"{int(remaining_seconds)}s"
        elif remaining_seconds < 3600:
            minutes = int(remaining_seconds / 60)
            seconds = int(remaining_seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(remaining_seconds / 3600)
            minutes = int((remaining_seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def _get_module_name(self, file_path: Path) -> str:
        """Extract module name from file path"""
        parts = file_path.parts
        if len(parts) >= 2:
            for i, part in enumerate(parts):
                if part == "_work" and i + 1 < len(parts):
                    return parts[i + 1]
        return "unknown"

    async def _process_file_batch(
        self, file_batch: List[Path], batch_num: int
    ) -> Tuple[List[ProductionCodeChunk], Dict[str, str]]:
        """Process a batch of files"""
        self.logger.info(f"📁 Processing batch {batch_num} ({len(file_batch)} files)")

        batch_chunks = []
        batch_temp_content = {}

        # Show sample file
        if file_batch:
            self.logger.info(f"📄 Sample file: {file_batch[0].name}")

        # Process each file in the batch
        for file_path in file_batch:
            if str(file_path) in self.processed_files:
                continue

            try:
                # Create production chunks for this file
                file_chunks = await self._create_production_chunks_for_file(file_path)

                if file_chunks:
                    # Read file content for AI processing
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        file_content = f.read()

                    lines = file_content.split("\n")
                    for chunk in file_chunks:
                        if chunk.end_line <= len(lines):
                            chunk_content = "\n".join(
                                lines[chunk.start_line - 1 : chunk.end_line]
                            )
                            batch_temp_content[chunk.chunk_id] = chunk_content

                    batch_chunks.extend(file_chunks)
                    self.processed_files.add(str(file_path))

            except Exception as e:
                self.logger.error(f"❌ Failed to process {file_path}: {e}")
                continue

        # Show sample chunk
        if batch_chunks:
            sample_chunk = batch_chunks[0]
            self.logger.info(
                f"📦 Sample chunk: {sample_chunk.function_name or 'unnamed'} from {sample_chunk.module_name or 'unknown'}"
            )

        return batch_chunks, batch_temp_content

    async def _enhance_batch_with_ai(
        self, chunks: List[ProductionCodeChunk], temp_content: Dict[str, str]
    ) -> List[ProductionCodeChunk]:
        """Generate AI summaries for a batch of chunks"""
        if not self.ai_summarizer or not self.config.enable_ai_summaries or not chunks:
            return chunks

        self.logger.info(f"🤖 Generating AI summaries for {len(chunks)} chunks...")

        # Create temporary chunks for AI processing
        temp_chunks = []
        for chunk in chunks:
            temp_chunk_content = temp_content.get(chunk.chunk_id, "")
            if temp_chunk_content:
                temp_chunk = type(
                    "TempChunk",
                    (),
                    {
                        "chunk_id": chunk.chunk_id,
                        "processed_content": temp_chunk_content,
                        "raw_content": temp_chunk_content,
                        "function_name": chunk.function_name,
                        "file_path": chunk.relative_file_path,
                        "chunk_type": chunk.chunk_type,
                        "module": chunk.module_name,
                        "business_terms": [],
                        "database_tables": [],
                        "api_calls": chunk.api_calls or [],
                        "layer": None,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    },
                )()
                temp_chunks.append(temp_chunk)

        if not temp_chunks:
            return chunks

        try:
            # Generate AI summaries
            summary_results = await self.ai_summarizer.summarize_chunks(
                temp_chunks, batch_size=self.config.ai_batch_size
            )

            # Apply summaries to production chunks
            for chunk in chunks:
                if chunk.chunk_id in summary_results:
                    summary_data = summary_results[chunk.chunk_id]
                    chunk.ai_summary = summary_data
                    chunk.ai_purpose = summary_data.get("purpose", "")
                    chunk.ai_keywords = summary_data.get("keywords", None)

                    if (
                        chunk.ai_summary
                        and len(chunk.ai_summary) >= self.config.min_summary_length
                    ):
                        self.stats.ai_enhanced_chunks += 1

                        # Show first few AI summaries as samples
                        if self.stats.ai_enhanced_chunks <= 3:
                            self.logger.info(
                                f"🤖 Sample AI Summary #{self.stats.ai_enhanced_chunks}:"
                            )
                            self.logger.info(
                                f"   Function: {chunk.function_name or 'unnamed'}"
                            )
                            self.logger.info(
                                f"   Module: {chunk.module_name or 'unknown'}"
                            )
                            self.logger.info(f"   File: {chunk.relative_file_path}")
                            self.logger.info(f"   Summary: {chunk.ai_summary}")
                            if chunk.ai_purpose:
                                self.logger.info(f"   Purpose: {chunk.ai_purpose}")
                            if chunk.ai_keywords:
                                self.logger.info(
                                    f"   Keywords: {', '.join(chunk.ai_keywords[:8])}"
                                )
                            self.logger.info("   " + "-" * 50)

            return chunks

        except Exception as e:
            self.logger.warning(f"⚠️ AI enhancement failed: {e}")
            return chunks

    async def create_simple_sequential_embeddings(
        self, source_directories: List[Path], workspace_root: Path = None
    ) -> Dict[str, Any]:
        """Create embeddings using simple sequential processing with clear progress reporting"""

        self.logger.info("🚀 Creating embeddings with simple sequential processing...")
        self.start_time = time.time()

        # Set workspace root
        if workspace_root:
            self.config.workspace_root = workspace_root
        elif source_directories:
            self.config.workspace_root = source_directories[0].parent

        # Load model
        self._load_model()

        # Check AI summarizer status
        if self.ai_summarizer:
            self.logger.info("🤖 AI Summarizer Status:")
            self.logger.info(
                f"   Ollama available: {self.ai_summarizer.ollama_available}"
            )
            self.logger.info(f"   Model: {self.ai_summarizer.model_name}")
            if hasattr(self.ai_summarizer.cache, "summaries"):
                cached_count = len(self.ai_summarizer.cache.summaries)
                self.logger.info(
                    f"   🗄️  Using existing cache with {cached_count} summaries"
                )

        # Initialize statistics
        self.stats.start_time = datetime.now()

        # Collect all files
        all_files = []
        for source_dir in source_directories:
            if source_dir.exists():
                plsql_files = (
                    list(source_dir.rglob("*.plsql"))
                    + list(source_dir.rglob("*.plsvc"))
                    + list(source_dir.rglob("*.pls"))
                    + list(source_dir.rglob("*.sql"))
                )
                all_files.extend(plsql_files)

        self.stats.total_files = len(all_files)
        total_files = len(all_files)
        self.logger.info(f"📁 Found {total_files} files to process")

        # Process files in batches of 50
        total_batches = (
            total_files + self.batch_size_files - 1
        ) // self.batch_size_files
        self.logger.info(
            f"📦 Will process in {total_batches} batches of up to {self.batch_size_files} files each"
        )

        current_module = ""

        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size_files
            end_idx = min(start_idx + self.batch_size_files, total_files)
            file_batch = all_files[start_idx:end_idx]

            # Track module for progress reporting
            if file_batch:
                batch_module = self._get_module_name(file_batch[0])
                if batch_module != current_module:
                    current_module = batch_module
                    self.logger.info(f"📂 Processing module: {current_module}")

            # Show progress with ETA
            files_processed = start_idx
            eta = self._estimate_remaining_time(files_processed, total_files)
            self.logger.info(
                f"📊 Progress: {files_processed}/{total_files} files ({files_processed/total_files*100:.1f}%) | ETA: {eta}"
            )

            # Process this batch of files
            batch_chunks, batch_temp_content = await self._process_file_batch(
                file_batch, batch_num + 1
            )

            if batch_chunks:
                # Generate AI summaries for this batch immediately
                enhanced_chunks = await self._enhance_batch_with_ai(
                    batch_chunks, batch_temp_content
                )

                # Add to main collection
                self.production_chunks.extend(enhanced_chunks)

            # Clear temp content to save memory
            batch_temp_content.clear()

            files_completed = end_idx
            self.processed_file_count = files_completed

            self.logger.info(f"✅ Batch {batch_num + 1}/{total_batches} complete")
            self.logger.info(
                f"   Files: {len(file_batch)} processed ({files_completed}/{total_files} total)"
            )
            self.logger.info(
                f"   Chunks: {len(batch_chunks)} created ({len(self.production_chunks)} total)"
            )
            self.logger.info(f"   AI enhanced: {self.stats.ai_enhanced_chunks} chunks")

        # Final progress update
        self.logger.info(
            f"📊 Final Progress: {total_files}/{total_files} files (100%) | Processing complete!"
        )

        self.logger.info(
            f"✅ File processing complete: {len(self.production_chunks)} chunks from {len(self.processed_files)} files"
        )

        # Create embeddings from AI summaries and metadata
        self.logger.info("📐 Creating embeddings from AI summaries and metadata...")
        self.embeddings = self._create_embeddings_from_summaries(self.production_chunks)

        # Show sample searchable text
        if len(self.production_chunks) > 0:
            sample_chunk = self.production_chunks[0]
            sample_searchable_text = sample_chunk.get_searchable_text()
            self.logger.info(f"📐 Sample searchable text (first chunk):")
            self.logger.info(f"   Function: {sample_chunk.function_name or 'unnamed'}")
            for line in sample_searchable_text.split("\n"):
                if line.strip():
                    self.logger.info(f"     {line}")
            self.logger.info("   " + "-" * 50)

        if self.embeddings is not None and len(self.embeddings) > 0:
            self.logger.info(f"📐 Created embeddings: shape={self.embeddings.shape}")

        # Update statistics
        self.stats.total_chunks = len(self.production_chunks)
        self.stats.processed_chunks = len(self.production_chunks)
        self.stats.chunks_without_content = len(
            [c for c in self.production_chunks if not c.ai_summary]
        )

        # Save results
        results = await self._save_production_results()

        # Final statistics
        elapsed_time = time.time() - self.start_time
        self.logger.info("📊 Sequential Processing Complete!")
        self.logger.info(f"   Total files processed: {len(self.processed_files)}")
        self.logger.info(f"   Total chunks created: {len(self.production_chunks)}")
        self.logger.info(f"   AI-enhanced chunks: {self.stats.ai_enhanced_chunks}")
        self.logger.info(
            f"   Chunks without AI summary: {self.stats.chunks_without_content}"
        )
        self.logger.info(f"   Total processing time: {elapsed_time/60:.1f} minutes")

        return {
            "total_chunks": len(self.production_chunks),
            "total_files": len(self.processed_files),
            "ai_enhanced_chunks": self.stats.ai_enhanced_chunks,
            "chunks_without_content": self.stats.chunks_without_content,
            "embedding_shape": (
                self.embeddings.shape if self.embeddings is not None else None
            ),
            "copyright_compliant": True,
            "processing_method": "simple_sequential",
            "batch_size_files": self.batch_size_files,
            "processing_time_minutes": elapsed_time / 60,
            "stats": self.stats.to_dict(),
            **results,
        }

        if self.embeddings is not None and len(self.embeddings) > 0:
            self.logger.info(f"📐 Created embeddings: shape={self.embeddings.shape}")

        # Update statistics
        self.stats.total_chunks = len(self.production_chunks)
        self.stats.processed_chunks = len(self.production_chunks)
        self.stats.chunks_without_content = len(
            [c for c in self.production_chunks if not c.ai_summary]
        )

        # Save results
        results = await self._save_production_results()

        # Final statistics
        self.logger.info("📊 Sequential Processing Complete!")
        self.logger.info(f"   Total files processed: {len(self.processed_files)}")
        self.logger.info(f"   Total chunks created: {len(self.production_chunks)}")
        self.logger.info(f"   AI-enhanced chunks: {self.stats.ai_enhanced_chunks}")
        self.logger.info(
            f"   Chunks without AI summary: {self.stats.chunks_without_content}"
        )

        return {
            "total_chunks": len(self.production_chunks),
            "total_files": len(self.processed_files),
            "ai_enhanced_chunks": self.stats.ai_enhanced_chunks,
            "chunks_without_content": self.stats.chunks_without_content,
            "embedding_shape": (
                self.embeddings.shape if self.embeddings is not None else None
            ),
            "copyright_compliant": True,
            "processing_method": "simple_sequential",
            "batch_size_files": self.batch_size_files,
            "stats": self.stats.to_dict(),
            **results,
        }


# Convenience function
async def create_simple_sequential_embeddings(
    source_directories: List[Path],
    workspace_root: Path,
    config: ProductionEmbeddingConfig = None,
) -> Dict[str, Any]:
    """Create embeddings using simple sequential processing"""
    if config is None:
        config = ProductionEmbeddingConfig()
        config.batch_size = 12
        config.ai_batch_size = 8
        config.enable_ai_summaries = True

    config.workspace_root = workspace_root

    framework = SimpleSequentialFramework(config)
    return await framework.create_simple_sequential_embeddings(
        source_directories, workspace_root
    )


if __name__ == "__main__":

    async def main():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        config = ProductionEmbeddingConfig(
            enable_ai_summaries=True,
            batch_size=12,
            ai_batch_size=8,
            output_dir=Path("simple_production_embeddings"),
            cache_dir=Path("cache"),  # Use existing cache with 495 summaries
            copyright_owner="IFS AB",
        )

        workspace_root = Path(".")
        source_dirs = [Path("_work")]

        print("🚀 Starting Simple Sequential Embedding Creation")
        print("=" * 60)

        results = await create_simple_sequential_embeddings(
            source_directories=source_dirs, workspace_root=workspace_root, config=config
        )

        print("\n" + "=" * 60)
        print("🎉 Simple Sequential Embedding Creation Complete!")
        print("=" * 60)
        print(f"📁 Total files: {results['total_files']}")
        print(f"📊 Total chunks: {results['total_chunks']}")
        print(f"🤖 AI enhanced: {results['ai_enhanced_chunks']}")
        print(f"📦 Processing method: {results['processing_method']}")
        print(f"⚖️ Copyright compliant: {results['copyright_compliant']}")
        print(f"📐 Embedding shape: {results['embedding_shape']}")
        print("=" * 60)

    asyncio.run(main())
