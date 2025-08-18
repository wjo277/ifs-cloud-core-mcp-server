#!/usr/bin/env python3
"""
AI-Powered Code Summarization for Enhanced Semantic Search
==============================================                logger.info(f"💡 To enable AI summaries, install Ollama and run: ollama pull qwen3:8b")===========

This module uses Qwen3-8B to generate intelligent, natural language summaries
of code functions and procedures, dramatically improving semantic search quality.

WHY AI SUMMARIES IMPROVE SEARCH:
--------------------------------
1. NATURAL LANGUAGE: Converts technical code to human-readable descriptions
2. BUSINESS CONTEXT: Explains what the code does in domain-specific terms
3. INTENT CAPTURE: Describes the purpose and use cases, not just implementation
4. BETTER EMBEDDINGS: UniXcoder creates richer vector representations from natural text
5. QUERY MATCHING: Users search with business questions, not technical syntax

ARCHITECTURE:
-------------
- Uses Ollama for local Qwen3-8B inference (privacy-preserving)
- Intelligent prompting for IFS Cloud business context
- Caching to avoid re-summarizing unchanged code
- Batch processing for efficiency
- Fallback to lightweight summaries if AI is unavailable
"""

import logging
import hashlib
import json
import asyncio
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import aiofiles

# Optional dependency for development-time summarization
try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

# Optional tiktoken for accurate token counting
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

from .data_structures import CodeChunk

logger = logging.getLogger(__name__)


@dataclass
class OllamaClientPool:
    """
    Connection pool for Ollama AsyncClients to enable true parallel processing
    """

    pool_size: int = 4
    clients: List = field(default_factory=list)
    current_index: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        # Initialize client pool
        for i in range(self.pool_size):
            if OLLAMA_AVAILABLE and ollama:
                client = ollama.AsyncClient()
                self.clients.append(client)
            else:
                self.clients.append(None)

    async def get_client(self):
        """Get next available client from the pool (round-robin)"""
        if not self.clients or not self.clients[0]:  # No valid clients
            return None

        async with self.lock:
            client = self.clients[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.clients)
            return client


@dataclass
class ParallelComplexityAnalyzer:
    """
    Parallel complexity analyzer using ProcessPoolExecutor for CPU-intensive tasks
    """

    executor: ProcessPoolExecutor = field(
        default_factory=lambda: ProcessPoolExecutor(
            max_workers=max(1, multiprocessing.cpu_count() // 2)
        )
    )

    def __del__(self):
        if hasattr(self, "executor") and self.executor:
            self.executor.shutdown(wait=False)


@dataclass
class SummaryResult:
    """Result from AI summarization process"""

    chunk_id: str
    title: str
    summary: str
    key_concepts: List[str]
    reasoning: str


@dataclass
class SummaryCache:
    """
    Improved cache using multiple NDJSON files for better performance
    - Splits cache into multiple files capped at 32MB each
    - Uses NDJSON for append-only operations
    - Provides compatibility layer for old JSON format
    """

    cache_dir: Path
    max_file_size_mb: int = 32
    summaries: Dict[str, Dict] = field(default_factory=dict)
    _current_file_index: int = 0
    _current_file_size: int = 0
    _written_hashes: set = field(
        default_factory=set
    )  # Track what's been written to disk

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.load_cache()

    def _get_cache_file_path(self, index: int = None) -> Path:
        """Get path for cache file by index"""
        if index is None:
            index = self._current_file_index
        if index == 0:
            return self.cache_dir / "summaries_000.ndjson"
        return self.cache_dir / f"summaries_{index:03d}.ndjson"

    def _get_legacy_cache_path(self) -> Path:
        """Get path for legacy JSON cache file"""
        return self.cache_dir / "summaries.json"

    def _estimate_ndjson_line_size(self, content_hash: str, summary_data: Dict) -> int:
        """Estimate the size of an NDJSON line"""
        line_data = {"hash": content_hash, "summary": summary_data}
        return (
            len(json.dumps(line_data, ensure_ascii=False).encode("utf-8")) + 1
        )  # +1 for newline

    def _load_legacy_cache(self) -> int:
        """Load and migrate legacy JSON cache format"""
        legacy_path = self._get_legacy_cache_path()
        if not legacy_path.exists():
            return 0

        try:
            with open(legacy_path, "r", encoding="utf-8") as f:
                legacy_summaries = json.load(f)

            logger.info(
                f"� Migrating {len(legacy_summaries)} summaries from legacy format..."
            )

            # Add all legacy summaries to current cache
            for content_hash, summary_data in legacy_summaries.items():
                self.summaries[content_hash] = summary_data

            # Save in new format
            self._save_all_summaries()

            # Backup and remove legacy file
            backup_path = legacy_path.with_suffix(".json.backup")
            legacy_path.rename(backup_path)
            logger.info(f"✅ Legacy cache migrated and backed up to {backup_path}")

            return len(legacy_summaries)

        except Exception as e:
            logger.warning(f"⚠️ Failed to migrate legacy cache: {e}")
            return 0

    def _save_all_summaries(self):
        """Save all summaries to NDJSON files, splitting as needed - with safe backup"""
        try:
            # Step 1: Create backup directory and move existing files there
            backup_dir = self.cache_dir / "_to_be_deleted"
            existing_files = list(self.cache_dir.glob("summaries_*.ndjson"))

            if existing_files:
                backup_dir.mkdir(exist_ok=True)
                logger.info(
                    f"📦 Backing up {len(existing_files)} existing cache files..."
                )

                for file_path in existing_files:
                    backup_path = backup_dir / file_path.name
                    # If backup already exists, add timestamp
                    if backup_path.exists():
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = (
                            backup_dir / f"{file_path.stem}_{timestamp}.ndjson"
                        )
                    file_path.rename(backup_path)

            # Step 2: Write new cache files
            current_file_index = 0
            current_file_size = 0
            max_size_bytes = self.max_file_size_mb * 1024 * 1024
            current_file = None
            files_written = []

            try:
                for content_hash, summary_data in self.summaries.items():
                    line_data = {"hash": content_hash, "summary": summary_data}
                    line = json.dumps(line_data, ensure_ascii=False) + "\n"
                    line_size = len(line.encode("utf-8"))

                    # Check if we need a new file
                    if current_file is None or (
                        current_file_size + line_size > max_size_bytes
                    ):
                        if current_file:
                            current_file.close()

                        file_path = self._get_cache_file_path(current_file_index)
                        files_written.append(file_path)
                        current_file = open(file_path, "w", encoding="utf-8")
                        current_file_size = 0
                        current_file_index += 1

                    current_file.write(line)
                    current_file_size += line_size

                if current_file:
                    current_file.close()

                # Step 3: Verify all files were written successfully
                total_written_lines = 0
                for file_path in files_written:
                    if not file_path.exists():
                        raise Exception(f"Failed to create cache file: {file_path}")

                    # Count lines to verify integrity
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = sum(1 for line in f if line.strip())
                        total_written_lines += lines

                if total_written_lines != len(self.summaries):
                    raise Exception(
                        f"Data integrity check failed: wrote {total_written_lines} lines but expected {len(self.summaries)}"
                    )

                # Step 4: Update tracking variables
                self._current_file_index = (
                    current_file_index - 1 if current_file_index > 0 else 0
                )
                self._current_file_size = current_file_size

                # Mark all summaries as written to disk
                self._written_hashes = set(self.summaries.keys())

                logger.info(
                    f"✅ Successfully wrote {len(self.summaries)} summaries to {len(files_written)} NDJSON files"
                )

                # Step 5: Only now delete the backup files (they're safe to remove)
                if backup_dir.exists():
                    import shutil

                    shutil.rmtree(backup_dir)
                    logger.info(f"🗑️ Cleaned up backup files")

            except Exception as write_error:
                # Step 6: If writing failed, restore from backup
                logger.error(f"❌ Failed to write new cache files: {write_error}")

                # Close any open files
                if current_file:
                    current_file.close()

                # Remove any partially written files
                for file_path in files_written:
                    if file_path.exists():
                        file_path.unlink()
                        logger.info(f"🗑️ Removed partial file: {file_path.name}")

                # Restore from backup
                if backup_dir.exists():
                    for backup_file in backup_dir.glob("summaries_*.ndjson"):
                        restore_path = (
                            self.cache_dir / backup_file.name.split("_")[0]
                            + "_"
                            + backup_file.name.split("_")[1]
                            + ".ndjson"
                        )
                        # Handle timestamped backups
                        if len(backup_file.name.split("_")) > 2:
                            restore_path = (
                                self.cache_dir
                                / f"summaries_{backup_file.name.split('_')[1]}.ndjson"
                            )
                        backup_file.rename(restore_path)
                        logger.info(f"♻️ Restored: {restore_path.name}")

                    backup_dir.rmdir()

                raise write_error

        except Exception as e:
            logger.error(f"❌ Failed to save all summaries: {e}")
            raise

    def load_cache(self):
        """Load existing summaries from NDJSON cache files with legacy compatibility"""
        try:
            # First check for legacy format and migrate if needed
            migrated_count = self._load_legacy_cache()
            if migrated_count > 0:
                return  # Already loaded during migration

            # Load from NDJSON files
            total_loaded = 0
            cache_files = sorted(self.cache_dir.glob("summaries_*.ndjson"))

            for file_path in cache_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue

                            try:
                                data = json.loads(line)
                                content_hash = data["hash"]
                                summary_data = data["summary"]
                                self.summaries[content_hash] = summary_data
                                self._written_hashes.add(
                                    content_hash
                                )  # Mark as written to disk
                                total_loaded += 1

                            except (json.JSONDecodeError, KeyError) as e:
                                logger.warning(
                                    f"⚠️ Skipping invalid line {line_num} in {file_path.name}: {e}"
                                )
                                continue

                except Exception as e:
                    logger.warning(f"⚠️ Failed to load cache file {file_path}: {e}")
                    continue

            # Track the highest file index and size
            if cache_files:
                last_file_path = cache_files[-1]
                self._current_file_index = int(last_file_path.stem.split("_")[-1])
                self._current_file_size = last_file_path.stat().st_size
            else:
                self._current_file_index = 0
                self._current_file_size = 0

            if total_loaded > 0:
                logger.info(
                    f"📂 Loaded {total_loaded} cached summaries from {len(cache_files)} NDJSON files"
                )
            else:
                logger.info("📂 No summary cache found, starting fresh")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load summary cache: {e}")
            self.summaries = {}

    def save_cache(self, force_full_rebuild: bool = False):
        """
        Save summaries to NDJSON cache files
        - During batch operations: only append new summaries (incremental)
        - On shutdown or force: rebuild all files (full)
        """
        try:
            if force_full_rebuild:
                # Full rebuild - used on shutdown or migration
                self._save_all_summaries()
                logger.info(
                    f"💾 Full rebuild: {len(self.summaries)} summaries to NDJSON cache"
                )
            else:
                # Incremental save - only append new summaries
                unwritten_count = self._save_new_summaries()
                if unwritten_count > 0:
                    logger.debug(
                        f"💾 Appended {unwritten_count} new summaries to cache"
                    )
        except Exception as e:
            logger.error(f"❌ Failed to save summary cache: {e}")

    def _save_new_summaries(self) -> int:
        """Append only new summaries that haven't been written to disk yet"""
        unwritten_count = 0
        max_size_bytes = self.max_file_size_mb * 1024 * 1024

        try:
            for content_hash, summary_data in self.summaries.items():
                if content_hash in self._written_hashes:
                    continue  # Already written to disk

                line_data = {"hash": content_hash, "summary": summary_data}
                line = json.dumps(line_data, ensure_ascii=False) + "\n"
                line_size = len(line.encode("utf-8"))

                # Check if we need a new file
                if self._current_file_size + line_size > max_size_bytes:
                    self._current_file_index += 1
                    self._current_file_size = 0

                file_path = self._get_cache_file_path()

                # Append to file
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(line)

                self._current_file_size += line_size
                self._written_hashes.add(content_hash)
                unwritten_count += 1

        except Exception as e:
            logger.warning(f"⚠️ Failed to append new summaries: {e}")

        return unwritten_count

    def get_summary(self, content_hash: str) -> Optional[Dict]:
        """Get cached summary by content hash"""
        return self.summaries.get(content_hash)

    def set_summary(self, content_hash: str, summary_data: Dict):
        """Cache a summary - will be written to disk on next save_cache() call"""
        summary_data["cached_at"] = datetime.now().isoformat()
        self.summaries[content_hash] = summary_data
        # Note: summary will be appended to NDJSON file on next save_cache() call


class AISummarizer:
    """
    AI-powered code summarization using Phi4-14B (16K context) for enhanced semantic search
    """

    # FIXED OLLAMA PARAMETERS - NEVER CHANGE THESE TO AVOID MODEL RELOADING
    # These are optimized for phi4-mini with 64K context window to ensure model stays loaded
    OLLAMA_OPTIONS = {
        "temperature": 0.2,  # Consistent for focused summaries
        "top_p": 0.7,  # Consistent quality
        "num_ctx": 65536,  # phi4-mini context window (64K tokens) - NEVER CHANGE
        "num_predict": 4000,  # Large response buffer (up to 4K tokens) - NEVER CHANGE
        "repeat_penalty": 1.1,  # Prevent repetition
    }

    def __init__(self, cache_dir: Path = None, model_name: str = "phi4-mini:latest"):
        self.model_name = model_name
        self.cache_dir = cache_dir or Path("cache/ai_summaries")
        self.cache = SummaryCache(self.cache_dir)  # Pass directory, not file
        self.complexity_threshold = 5  # Minimum complexity score for AI processing
        self.ollama_available = False
        self.async_client = None
        self.client_pool = None
        self.complexity_analyzer = None

        # Initialize Ollama connection and parallel components
        self._check_ollama()

        if self.ollama_available:
            # DISABLED CLIENT POOL: Use single client only to prevent GPU overload
            # self.client_pool = OllamaClientPool(pool_size=2)
            logger.info(
                "🔄 Using single Ollama client for sequential processing (optimal for GPU)"
            )
            # Disable parallel complexity analyzer for now to avoid multiprocessing issues
            # self.complexity_analyzer = ParallelComplexityAnalyzer()

    def _check_ollama(self):
        """Check if Ollama is available and the model is installed"""
        if not OLLAMA_AVAILABLE:
            logger.info("💡 Ollama not installed. AI summaries disabled.")
            logger.info("   To enable AI summaries in development:")
            logger.info("   1. Install dev dependencies: uv sync --dev")
            logger.info("   2. Install Ollama: https://ollama.ai")
            logger.info("   3. Pull model: ollama pull phi4-mini")
            self.ollama_available = False
            return

        try:
            # Check if Ollama is running
            models = ollama.list()
            available_models = [
                m.get("model", m.get("name", "")) for m in models["models"]
            ]

            if self.model_name in available_models:
                self.ollama_available = True
                self.async_client = ollama.AsyncClient()  # Initialize async client
                logger.info(
                    f"✅ Ollama available with {self.model_name} (async client)"
                )
            else:
                logger.warning(
                    f"⚠️ Model {self.model_name} not found. Available: {available_models}"
                )
                # Try to pull the model
                logger.info(f"🔄 Pulling {self.model_name}...")
                ollama.pull(self.model_name)
                self.ollama_available = True
                self.async_client = ollama.AsyncClient()  # Initialize async client
                logger.info(f"✅ Successfully pulled {self.model_name} (async client)")

        except Exception as e:
            logger.warning(f"⚠️ Ollama not available: {e}")
            logger.info(
                "💡 To enable AI summaries, install Ollama and run: ollama pull phi4-mini"
            )
            self.ollama_available = False

    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash of the code content for caching"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _create_ifs_prompt(self, chunk: CodeChunk) -> str:
        """
        Create an intelligent prompt for summarizing IFS Cloud code
        """
        # Determine the context based on chunk metadata
        context_parts = []

        if chunk.module:
            context_parts.append(f"IFS {chunk.module} module")

        if chunk.layer:
            context_parts.append(f"{chunk.layer} layer")

        if chunk.function_name:
            function_context = f"{chunk.chunk_type.replace('plsql_', '')} named '{chunk.function_name}'"
        else:
            function_context = f"{chunk.chunk_type.replace('plsql_', '')}"

        # Build business context
        business_context = ""
        if chunk.business_terms:
            business_context = (
                f" This code deals with: {', '.join(chunk.business_terms[:5])}."
            )

        if chunk.database_tables:
            business_context += f" It works with database tables: {', '.join(chunk.database_tables[:3])}."

        if chunk.api_calls:
            business_context += f" It calls APIs: {', '.join(chunk.api_calls[:3])}."

        context = " - ".join(context_parts) if context_parts else "IFS Cloud"

        # Create the prompt optimized for semantic search and similarity matching
        prompt = f"""You are an expert in IFS Cloud ERP system. Analyze this {function_context} from the {context}.

CODE TO ANALYZE:
```sql
{chunk.processed_content[:2000]}  
```

CONTEXT: {business_context}

Create a summary optimized for semantic search and similarity matching. Use varied vocabulary and synonyms so users can find this code with different search terms.

Provide response in this exact format:

SUMMARY: [Direct business action using action verbs - describe what it does, how it works, what business problem it solves]
PURPOSE: [When/why developers would use this - include common scenarios, use cases, business situations]  
KEYWORDS: [8-12 diverse keywords covering: business terms, technical terms, action verbs, domain concepts, synonyms]

Optimization guidelines:
- Use rich, varied vocabulary and synonyms (e.g., "calculate/compute/determine", "validate/verify/check", "process/handle/manage")
- Include both business domain terms AND technical implementation terms
- Add context about business scenarios and use cases
- Include related concepts someone might search for
- Use natural language that matches how developers think and search
- Avoid repetitive corporate language - be specific and descriptive

Example good keywords: "account validation, financial verification, code part validation, business rules, data integrity, accounting standards, compliance checking, error handling\""""

        return prompt

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        Simple approximation: ~4 chars per token for English text.
        """
        if TIKTOKEN_AVAILABLE and tiktoken:
            try:
                encoding = tiktoken.get_encoding(
                    "cl100k_base"
                )  # GPT-4 encoding as approximation
                return len(encoding.encode(text))
            except Exception:
                pass

        # Fallback estimation: ~4 characters per token
        return len(text) // 4

    def _group_chunks_by_file(
        self, chunks: List[CodeChunk]
    ) -> Dict[str, List[CodeChunk]]:
        """Group chunks by source file for better context in batches"""
        file_groups = defaultdict(list)

        for chunk in chunks:
            # Use the base filename without path for grouping
            filename = Path(chunk.file_path).name if chunk.file_path else "unknown"
            file_groups[filename].append(chunk)

        return dict(file_groups)

    def _group_chunks_by_file_with_complexity(
        self, chunk_complexity_pairs: List[Tuple[CodeChunk, int]]
    ) -> Dict[str, List[Tuple[CodeChunk, int]]]:
        """
        Group chunks by their source file with complexity scores for parallel processing
        """
        file_groups = {}
        for chunk, complexity_score in chunk_complexity_pairs:
            filename = Path(chunk.file_path).name if chunk.file_path else "unknown"
            if filename not in file_groups:
                file_groups[filename] = []
            file_groups[filename].append((chunk, complexity_score))
        return file_groups

    def _calculate_optimal_batch_size(
        self, chunks: List[CodeChunk], target_utilization: float = 0.8
    ) -> int:
        """
        Dynamically calculate optimal batch size to achieve target context utilization

        Args:
            chunks: List of chunks to analyze for size estimation
            target_utilization: Target utilization ratio (0.8 = 80%)

        Returns:
            Optimal batch size (minimum 1, maximum len(chunks))
        """
        if not chunks:
            return 1

        # Use fixed context window from OLLAMA_OPTIONS - NEVER CHANGE THIS
        max_context_tokens = self.OLLAMA_OPTIONS["num_ctx"]

        # Base prompt overhead (estimated)
        base_prompt = """You are an expert in IFS Cloud ERP system. Analyze these PL/SQL functions and provide summaries. 

For each function, respond with exactly this format (using === as delimiters):

===FUNCTION_START===
FUNCTION_ID: [number]
SUMMARY: [what this function does - business action and purpose]
PURPOSE: [when developers use this - scenarios and use cases]  
KEYWORDS: [8-10 keywords separated by commas]
===FUNCTION_END===

Functions to analyze:

"""
        base_tokens = self._estimate_tokens(base_prompt)

        # Sample a few chunks to estimate average size
        sample_size = min(5, len(chunks))
        sample_chunks = chunks[:sample_size]

        total_chunk_tokens = 0
        for chunk in sample_chunks:
            # Estimate function text size (header + content + formatting)
            function_text = (
                f"\n--- FUNCTION X: {chunk.function_name or 'Unnamed'} ---\n"
            )
            function_text += f"File: {chunk.file_path}\n"
            function_text += chunk.raw_content[:3000]  # Truncated like in actual prompt
            function_text += "\n"

            chunk_tokens = self._estimate_tokens(function_text)
            total_chunk_tokens += chunk_tokens

        # Average tokens per chunk
        avg_tokens_per_chunk = (
            total_chunk_tokens / sample_size if sample_size > 0 else 800
        )

        # Response tokens (400 per function based on our testing)
        tokens_per_response = 400

        # Calculate target tokens for content (leaving headroom)
        target_content_tokens = (
            int(max_context_tokens * target_utilization) - base_tokens
        )

        # Calculate how many chunks fit in target utilization
        tokens_per_complete_chunk = avg_tokens_per_chunk + tokens_per_response
        optimal_batch_size = max(
            1, int(target_content_tokens / tokens_per_complete_chunk)
        )

        # Cap at available chunks
        optimal_batch_size = min(optimal_batch_size, len(chunks))

        logger.debug(
            f"📊 Batch size calculation: avg_chunk_tokens={avg_tokens_per_chunk:.0f}, "
            f"base_tokens={base_tokens}, target_utilization={target_utilization:.0%}, "
            f"max_context={max_context_tokens}, optimal_batch_size={optimal_batch_size}"
        )

        return optimal_batch_size

    def _validate_batch_size(
        self, chunks: List[CodeChunk], batch_size: int
    ) -> Tuple[int, float]:
        """
        Validate that a batch size will fit within context limits

        Returns:
            Tuple of (validated_batch_size, estimated_utilization)
        """
        # Use fixed context window from OLLAMA_OPTIONS - NEVER CHANGE THIS
        max_context_tokens = self.OLLAMA_OPTIONS["num_ctx"]

        # Test with the proposed batch size
        test_batch = chunks[:batch_size]
        test_prompt, fits_in_context = self._create_batch_prompt(test_batch)

        if fits_in_context:
            estimated_tokens = self._estimate_tokens(test_prompt) + (
                len(test_batch) * 400
            )  # Add response tokens
            utilization = estimated_tokens / max_context_tokens
            return batch_size, utilization
        else:
            # Binary search to find maximum batch size that fits
            left, right = 1, batch_size
            best_size = 1
            best_utilization = 0.0

            while left <= right:
                mid = (left + right) // 2
                test_batch = chunks[:mid]
                test_prompt, fits = self._create_batch_prompt(test_batch)

                if fits:
                    estimated_tokens = self._estimate_tokens(test_prompt) + (
                        len(test_batch) * 400
                    )
                    utilization = estimated_tokens / max_context_tokens
                    best_size = mid
                    best_utilization = utilization
                    left = mid + 1
                else:
                    right = mid - 1

            return best_size, best_utilization

    async def _process_file_group_batched(
        self, file_path: str, file_chunks: List[Tuple[CodeChunk, int]]
    ) -> List[SummaryResult]:
        """
        Process a group of chunks from the same file with dynamic batch sizing
        """
        results = []
        chunks_only = [chunk for chunk, _ in file_chunks]

        # Dynamic batch sizing for optimal context utilization with 64K window
        # Use higher utilization since we have more context space
        max_context_tokens = self.OLLAMA_OPTIONS["num_ctx"]
        optimal_batch_size = self._calculate_optimal_batch_size(
            chunks_only, target_utilization=0.85  # Increased from 0.8 for 64K context
        )

        # Validate and adjust batch size
        validated_batch_size, actual_utilization = self._validate_batch_size(
            chunks_only, optimal_batch_size
        )

        logger.info(
            f"📊 Dynamic batch sizing (64K context): optimal={optimal_batch_size}, validated={validated_batch_size} chunks "
            f"(estimated {actual_utilization:.1%} utilization of {max_context_tokens} tokens)"
        )

        for i in range(0, len(chunks_only), validated_batch_size):
            batch = chunks_only[i : i + validated_batch_size]

            try:
                # Use single async client for sequential processing
                batch_results = await self._process_batch_with_retry(
                    batch, self.async_client
                )

                # Convert to SummaryResult format
                for chunk_id, summary_data in batch_results.items():
                    result = SummaryResult(
                        chunk_id=chunk_id,
                        title=summary_data.get("title", ""),
                        summary=summary_data.get("summary", ""),
                        key_concepts=summary_data.get("keywords", []),
                        reasoning=summary_data.get(
                            "reasoning", "AI processing successful"
                        ),
                    )
                    results.append(result)

            except Exception as e:
                logger.error(f"❌ Failed to process batch in {file_path}: {e}")

                # If batch fails, try individual processing
                logger.info(
                    f"🔄 Attempting individual processing for {len(batch)} chunks..."
                )
                for chunk in batch:
                    try:
                        # Use single async client for sequential processing
                        individual_results = await self._process_batch_with_retry(
                            [chunk], self.async_client
                        )

                        for chunk_id, summary_data in individual_results.items():
                            result = SummaryResult(
                                chunk_id=chunk_id,
                                title=summary_data.get("title", ""),
                                summary=summary_data.get("summary", ""),
                                key_concepts=summary_data.get("keywords", []),
                                reasoning=summary_data.get(
                                    "reasoning", "Individual processing successful"
                                ),
                            )
                            results.append(result)

                    except Exception as individual_e:
                        logger.error(
                            f"❌ Individual processing failed for chunk {chunk.chunk_id}: {individual_e}"
                        )
                        # Add error result for this chunk
                        result = SummaryResult(
                            chunk_id=chunk.chunk_id,
                            title="Processing Failed",
                            summary="",
                            key_concepts=[],
                            reasoning=f"Both batch and individual processing failed: {str(individual_e)}",
                        )
                        results.append(result)

        return results

    def _clean_plsql_for_summarization(
        self, raw_content: str, function_name: str = ""
    ) -> str:
        """
        Clean PL/SQL code for AI summarization by removing unnecessary sections.
        Keeps: function signature, main business logic, key comments
        Removes: verbose variable declarations, excessive whitespace, non-printable chars
        """
        import re
        import string

        # First pass: Remove non-printable characters and normalize whitespace
        # Keep printable ASCII + common Unicode chars, but remove control chars
        printable_chars = set(string.printable)
        printable_chars.update(
            ["\u00a0", "\u2013", "\u2014", "\u2018", "\u2019", "\u201c", "\u201d"]
        )  # Common Unicode

        # Clean non-printable characters
        cleaned_content = "".join(
            c if c in printable_chars else " " for c in raw_content
        )

        # Normalize different types of whitespace and quotes
        cleaned_content = re.sub(
            r"[\u00a0\u2000-\u200b\u202f\u205f]", " ", cleaned_content
        )  # Non-breaking spaces
        cleaned_content = re.sub(
            r"[\u2018\u2019]", "'", cleaned_content
        )  # Smart quotes to regular
        cleaned_content = re.sub(
            r"[\u201c\u201d]", '"', cleaned_content
        )  # Smart double quotes
        cleaned_content = re.sub(
            r"[\u2013\u2014]", "-", cleaned_content
        )  # En/em dashes to hyphens

        # Normalize multiple spaces to single space (except at line start for indentation)
        lines = cleaned_content.split("\n")
        normalized_lines = []
        for line in lines:
            # Preserve leading whitespace for indentation, normalize the rest
            leading_space = len(line) - len(line.lstrip())
            if leading_space > 0:
                # Keep reasonable indentation (max 8 spaces)
                indent = min(leading_space, 8)
                rest_of_line = re.sub(r"\s+", " ", line.lstrip()).strip()
                if rest_of_line:
                    normalized_lines.append(" " * indent + rest_of_line)
                else:
                    normalized_lines.append("")
            else:
                normalized = re.sub(r"\s+", " ", line).strip()
                if normalized:
                    normalized_lines.append(normalized)
                else:
                    normalized_lines.append("")

        lines = normalized_lines
        cleaned_lines = []

        # State tracking for PL/SQL structure
        in_declaration_section = False
        in_begin_section = False
        declaration_section_lines = []

        for i, line in enumerate(lines):
            if not line.strip():  # Skip completely empty lines during processing
                continue

            stripped = line.strip().upper()

            # Always keep function/procedure signature
            if (
                stripped.startswith(("PROCEDURE", "FUNCTION"))
                and function_name.upper() in stripped
            ):
                cleaned_lines.append(line)
                continue

            # Track if we're in declaration section (between signature and BEGIN)
            if stripped == "IS" or stripped.endswith(" IS"):
                in_declaration_section = True
                cleaned_lines.append(line)
                continue

            if stripped == "BEGIN":
                in_begin_section = True
                in_declaration_section = False

                # Add minimal variable declarations (only the important ones)
                important_declarations = []
                for decl_line in declaration_section_lines:
                    decl_upper = decl_line.strip().upper()
                    # Keep important declarations: exceptions, constants, cursors
                    if any(
                        keyword in decl_upper
                        for keyword in [
                            "EXCEPTION",
                            "CONSTANT",
                            "CURSOR",
                            "TYPE",
                            "%ROWTYPE",
                            "%TYPE",
                            "PRAGMA",
                            "SUBTYPE",
                        ]
                    ):
                        important_declarations.append(decl_line)

                # Add important declarations with a summary comment if we removed some
                if important_declarations:
                    if len(declaration_section_lines) > len(important_declarations):
                        cleaned_lines.append(
                            "   -- [Key declarations only - other variables omitted]"
                        )
                    cleaned_lines.extend(important_declarations)
                elif len(declaration_section_lines) > 0:
                    cleaned_lines.append("   -- [Variable declarations omitted]")

                cleaned_lines.append(line)  # Add the BEGIN line
                continue

            # If in declaration section, collect but don't add yet
            if in_declaration_section:
                declaration_section_lines.append(line)
                continue

            # In BEGIN section or after - keep business logic
            if in_begin_section or not in_declaration_section:
                cleaned_lines.append(line)

        # Join lines and do final cleanup
        cleaned_content = "\n".join(cleaned_lines)

        # Remove excessive empty lines
        cleaned_content = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned_content)

        # Remove trailing whitespace from lines
        lines = cleaned_content.split("\n")
        cleaned_content = "\n".join(line.rstrip() for line in lines)

        # Remove leading/trailing empty lines
        cleaned_content = cleaned_content.strip()

        # Log the reduction if significant
        original_length = len(raw_content)
        cleaned_length = len(cleaned_content)
        if original_length > cleaned_length:
            reduction_pct = (original_length - cleaned_length) / original_length * 100
            logger.debug(
                f"📝 Cleaned {function_name}: {original_length} → {cleaned_length} chars "
                f"({reduction_pct:.1f}% reduction)"
            )

        return cleaned_content

    def _calculate_max_function_length(self, chunks: List[CodeChunk]) -> int:
        """
        Calculate maximum safe length for individual functions based on context window
        and number of chunks to ensure we never exceed context limits.
        """
        max_context_tokens = self.OLLAMA_OPTIONS["num_ctx"]

        # Base prompt overhead (estimated tokens)
        base_overhead = 300  # Base prompt structure

        # Response tokens (estimate 400 tokens per function)
        response_tokens = len(chunks) * 400

        # Safety margin (20% of context window)
        safety_margin = int(max_context_tokens * 0.2)

        # Available tokens for function content
        available_tokens = (
            max_context_tokens - base_overhead - response_tokens - safety_margin
        )

        # Distribute available tokens among chunks
        tokens_per_chunk = max(
            200, available_tokens // len(chunks)
        )  # Minimum 200 tokens per chunk

        # Convert tokens to characters (rough approximation: 4 chars per token)
        max_chars_per_function = tokens_per_chunk * 4

        # Cap at reasonable limits
        max_chars_per_function = min(
            max_chars_per_function, 2000
        )  # Never more than 2000 chars
        max_chars_per_function = max(
            max_chars_per_function, 500
        )  # Never less than 500 chars

        logger.debug(
            f"📏 Function length calculation: {tokens_per_chunk} tokens = {max_chars_per_function} chars max per function"
        )

        return max_chars_per_function

    def _create_batch_prompt(self, chunks: List[CodeChunk]) -> Tuple[str, bool]:
        """
        Create a batch prompt for multiple chunks with context window management.
        Uses FIXED context window to prevent model reloading.
        Dynamically calculates safe truncation length for functions.
        Returns (prompt, fits_in_context)
        """
        # Use fixed context window from OLLAMA_OPTIONS - NEVER CHANGE THIS
        max_context_tokens = self.OLLAMA_OPTIONS["num_ctx"]

        # Calculate maximum safe length for each function
        max_function_length = self._calculate_max_function_length(chunks)

        base_prompt = """Analyze these PL/SQL functions from IFS Cloud ERP. Code has been cleaned for analysis.

CRITICAL RULES:
- Be CONCISE - avoid repetitive "This function" language
- If functions are identical/very similar to previous ones, mark EXCLUDE: YES
- Only include functions with unique business logic
- Start summaries with action verbs (Generate, Calculate, Process, etc.)

Format exactly (use === delimiters):

===FUNCTION_START===
FUNCTION_ID: [number]
SUMMARY: [concise action - what business logic this performs]
PURPOSE: [when/why developers use this specific function]
KEYWORDS: [6-8 relevant technical/business keywords]
EXCLUDE: [YES if duplicate/similar, NO if unique]
===FUNCTION_END===

Functions:

"""

        # Estimate base prompt tokens
        current_tokens = self._estimate_tokens(base_prompt)

        # Reserve tokens for response (estimate ~400 tokens per function summary)
        reserved_response_tokens = len(chunks) * 400
        available_tokens = (
            max_context_tokens - current_tokens - reserved_response_tokens
        )

        if available_tokens < 1000:  # Need minimum space for at least one function
            logger.warning(f"⚠️  Insufficient tokens available: {available_tokens}")
            return base_prompt, False

        # Add functions while staying within token limit
        included_chunks = []

        for i, chunk in enumerate(chunks, 1):
            function_text = (
                f"\n--- FUNCTION {i}: {chunk.function_name or 'Unnamed'} ---\n"
            )
            function_text += f"File: {chunk.file_path}\n"

            # First clean the function for summarization
            cleaned_content = self._clean_plsql_for_summarization(
                chunk.raw_content, chunk.function_name
            )

            # Use dynamic truncation based on available space
            if len(cleaned_content) > max_function_length:
                truncated_content = cleaned_content[:max_function_length]
                truncated_content += "\n... [TRUNCATED - function continues] ..."
                logger.debug(
                    f"📏 After cleaning, truncated {chunk.function_name} from {len(cleaned_content)} to {max_function_length} chars"
                )
            else:
                truncated_content = cleaned_content
                if len(chunk.raw_content) != len(cleaned_content):
                    logger.debug(
                        f"📝 Cleaned {chunk.function_name} from {len(chunk.raw_content)} to {len(cleaned_content)} chars"
                    )

            function_text += truncated_content
            function_text += "\n"

            function_tokens = self._estimate_tokens(function_text)

            if current_tokens + function_tokens > available_tokens:
                logger.warning(
                    f"⚠️  Batch prompt would exceed context limit. Including {i-1}/{len(chunks)} chunks."
                )
                break

            base_prompt += function_text
            current_tokens += function_tokens
            included_chunks.append(chunk)

        fits_in_context = len(included_chunks) == len(chunks)

        # Final safety check
        final_estimated_tokens = (
            self._estimate_tokens(base_prompt) + reserved_response_tokens
        )
        if final_estimated_tokens > max_context_tokens * 0.9:  # 90% threshold
            logger.warning(
                f"⚠️  Final prompt uses {final_estimated_tokens} tokens ({final_estimated_tokens/max_context_tokens:.1%} of context)"
            )

        logger.debug(
            f"📊 Batch prompt: {current_tokens} input + {reserved_response_tokens} response = {final_estimated_tokens} total tokens"
        )

        return base_prompt, fits_in_context

    def _parse_batch_response(
        self, response_text: str, chunks: List[CodeChunk]
    ) -> Dict[str, Dict]:
        """
        Parse batch AI response to extract individual summaries using delimiter-based format
        """
        results = {}

        # Split response into individual function blocks
        function_blocks = re.split(r"===FUNCTION_START===", response_text)

        for i, chunk in enumerate(chunks):
            try:
                # Find the corresponding function block (skip first empty split)
                if i + 1 < len(function_blocks):
                    block = function_blocks[i + 1]

                    # Extract function ID to verify correct mapping
                    function_id_match = re.search(
                        r"FUNCTION_ID:\s*(\d+)", block, re.IGNORECASE
                    )
                    expected_id = str(i + 1)

                    if function_id_match and function_id_match.group(1) == expected_id:
                        # Extract summary
                        summary_match = re.search(
                            r"SUMMARY:\s*(.+?)(?=PURPOSE:|KEYWORDS:|===FUNCTION_END===|$)",
                            block,
                            re.DOTALL | re.IGNORECASE,
                        )
                        summary = (
                            summary_match.group(1).strip() if summary_match else ""
                        )

                        # Extract purpose
                        purpose_match = re.search(
                            r"PURPOSE:\s*(.+?)(?=KEYWORDS:|===FUNCTION_END===|$)",
                            block,
                            re.DOTALL | re.IGNORECASE,
                        )
                        purpose = (
                            purpose_match.group(1).strip() if purpose_match else ""
                        )

                        # Extract keywords
                        keywords_match = re.search(
                            r"KEYWORDS:\s*(.+?)(?====FUNCTION_END===|$)",
                            block,
                            re.DOTALL | re.IGNORECASE,
                        )
                        keywords_text = (
                            keywords_match.group(1).strip() if keywords_match else ""
                        )

                        # Parse keywords list
                        keywords = []
                        if keywords_text:
                            keywords = [
                                k.strip() for k in keywords_text.split(",") if k.strip()
                            ]
                            keywords = keywords[:12]  # Limit to 12 keywords

                        # Only include if we got meaningful data
                        if summary and len(summary) > 10:
                            results[chunk.chunk_id] = {
                                "summary": summary,
                                "purpose": purpose,
                                "keywords": keywords,
                                "model": f"{self.model_name}-batch",
                                "timestamp": datetime.utcnow().isoformat(),
                                "batch_processed": True,
                            }
                        else:
                            logger.warning(
                                f"⚠️  Failed to parse summary for function {i+1}: {chunk.function_name} - insufficient content"
                            )
                    else:
                        logger.warning(
                            f"⚠️  Function ID mismatch for function {i+1}: expected {expected_id}, got {function_id_match.group(1) if function_id_match else 'none'}"
                        )
                else:
                    logger.warning(
                        f"⚠️  No function block found for function {i+1}: {chunk.function_name}"
                    )

            except Exception as e:
                logger.warning(
                    f"⚠️  Failed to parse summary for function {i+1}: {chunk.function_name} - {e}"
                )

        return results

    async def _process_batch_with_retry(
        self, chunks: List[CodeChunk], client=None
    ) -> Dict[str, Dict]:
        """
        Process a batch of chunks with retry logic for failed parsing
        Uses fixed context window to prevent model reloading
        """
        results = {}
        actual_client = client if client else self.client

        # First attempt: full batch
        batch_prompt, fits_in_context = self._create_batch_prompt(chunks)

        if not fits_in_context:
            logger.warning(
                f"⚠️  Batch of {len(chunks)} chunks exceeds context window. Will split into sub-batches."
            )
            return await self._process_oversized_batch(chunks)

        try:
            # Send batch request
            logger.debug(
                f"🤖 Processing batch of {len(chunks)} chunks (estimated {self._estimate_tokens(batch_prompt)} tokens)"
            )

            response = await actual_client.generate(
                model=self.model_name,
                prompt=batch_prompt,
                options=self.OLLAMA_OPTIONS,  # Use fixed options to prevent model reloading
            )

            batch_response = response["response"].strip()

            # Parse responses
            batch_results = self._parse_batch_response(batch_response, chunks)
            results.update(batch_results)

            # Check for failed parsing and retry
            failed_chunks = [
                chunk for chunk in chunks if chunk.chunk_id not in batch_results
            ]

            if failed_chunks:
                logger.info(
                    f"🔄 Retrying {len(failed_chunks)} chunks with failed parsing..."
                )
                retry_results = await self._retry_failed_chunks(
                    failed_chunks, actual_client
                )
                results.update(retry_results)

        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            # Fall back to individual processing
            logger.info(
                f"🔄 Falling back to individual processing for {len(chunks)} chunks..."
            )
            for chunk in chunks:
                try:
                    individual_result = await self.summarize_chunk(chunk)
                    if individual_result:
                        summary_data, was_cached = individual_result
                        results[chunk.chunk_id] = summary_data
                except Exception as individual_error:
                    logger.error(
                        f"❌ Individual processing failed for {chunk.function_name}: {individual_error}"
                    )

        return results

    async def _process_oversized_batch(
        self, chunks: List[CodeChunk]
    ) -> Dict[str, Dict]:
        """
        Split an oversized batch into smaller sub-batches that fit in context window
        Uses fixed context window to prevent model reloading
        """
        results = {}

        # Use fixed context window from OLLAMA_OPTIONS
        max_context_tokens = self.OLLAMA_OPTIONS["num_ctx"]

        # Start with batch size that should fit
        estimated_tokens_per_chunk = 800  # Conservative estimate
        max_chunks_per_batch = max(
            1, (max_context_tokens - 5000) // estimated_tokens_per_chunk
        )  # Reserve 5K for prompt overhead

        logger.info(
            f"🔄 Splitting {len(chunks)} chunks into sub-batches of ~{max_chunks_per_batch} chunks each"
        )

        for i in range(0, len(chunks), max_chunks_per_batch):
            sub_batch = chunks[i : i + max_chunks_per_batch]
            logger.debug(
                f"📦 Processing sub-batch {i//max_chunks_per_batch + 1}: {len(sub_batch)} chunks"
            )

            sub_results = await self._process_batch_with_retry(sub_batch)
            results.update(sub_results)

            # Small delay between sub-batches to be gentle on the API
            await asyncio.sleep(0.5)

        return results

    async def _retry_failed_chunks(
        self, failed_chunks: List[CodeChunk], client=None
    ) -> Dict[str, Dict]:
        """
        Retry chunks that failed to parse in the batch response
        """
        if not failed_chunks:
            return {}

        actual_client = client if client else self.client

        if len(failed_chunks) == 1:
            # Single chunk - use individual processing
            chunk = failed_chunks[0]
            try:
                result = await self.summarize_chunk(chunk)
                if result:
                    summary_data, was_cached = result
                    return {chunk.chunk_id: summary_data}
            except Exception as e:
                logger.error(f"❌ Retry failed for {chunk.function_name}: {e}")
                return {}

        # Multiple chunks - try smaller batch
        logger.debug(
            f"🔄 Retrying batch of {len(failed_chunks)} chunks with simpler prompt"
        )

        try:
            # Simpler retry prompt
            retry_prompt = """Analyze these PL/SQL functions and provide summaries. For each function, provide:

FUNCTION_X_SUMMARY: [Brief business description]
FUNCTION_X_PURPOSE: [When to use this]
FUNCTION_X_KEYWORDS: [Key terms for search]

Functions:
"""

            for i, chunk in enumerate(failed_chunks, 1):
                retry_prompt += (
                    f"\n--- FUNCTION {i}: {chunk.function_name or 'Unnamed'} ---\n"
                )
                retry_prompt += chunk.processed_content[
                    :1500
                ]  # Shorter content for retry
                retry_prompt += "\n"

            response = await actual_client.generate(
                model=self.model_name,
                prompt=retry_prompt,
                options=self.OLLAMA_OPTIONS,  # Use fixed options to prevent model reloading
            )

            retry_response = response["response"].strip()
            return self._parse_batch_response(retry_response, failed_chunks)

        except Exception as e:
            logger.error(f"❌ Retry batch processing failed: {e}")
            return {}

    async def _get_ai_summary(self, chunk: CodeChunk) -> Optional[Dict]:
        """Generate AI summary using Qwen3-8B with async client"""
        if not self.ollama_available or not OLLAMA_AVAILABLE or not self.async_client:
            return None

        try:
            prompt = self._create_ifs_prompt(chunk)

            # Generate summary using Ollama AsyncClient for true concurrency
            logger.debug(
                f"🤖 Generating AI summary for {chunk.function_name or 'unnamed chunk'}"
            )

            response = await self.async_client.generate(
                model=self.model_name,
                prompt=prompt,
                options=self.OLLAMA_OPTIONS,  # Use fixed options to prevent model reloading
            )

            summary_text = response["response"].strip()

            # Parse the structured response
            summary_data = self._parse_ai_response(summary_text)
            summary_data["model"] = self.model_name
            summary_data["generated_at"] = datetime.now().isoformat()

            return summary_data

        except Exception as e:
            logger.error(f"❌ AI summary generation failed: {e}")
            return None

    def _parse_ai_response(self, response: str) -> Dict:
        """Parse the structured AI response"""
        result = {
            "summary": "",
            "purpose": "",
            "keywords": [],
            "full_response": response,
        }

        # Clean up response by removing thinking tokens
        cleaned_response = response
        thinking_patterns = [
            "<think>",
            "</think>",
            "<thought>",
            "</thought>",
            "<reasoning>",
            "</reasoning>",
        ]
        for pattern in thinking_patterns:
            cleaned_response = cleaned_response.replace(pattern, "")

        # Remove content inside thinking tags
        import re

        cleaned_response = re.sub(
            r"<think>.*?</think>", "", cleaned_response, flags=re.DOTALL
        )
        cleaned_response = re.sub(
            r"<thought>.*?</thought>", "", cleaned_response, flags=re.DOTALL
        )
        cleaned_response = re.sub(
            r"<reasoning>.*?</reasoning>", "", cleaned_response, flags=re.DOTALL
        )

        lines = cleaned_response.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("SUMMARY:"):
                result["summary"] = line.replace("SUMMARY:", "").strip()
            elif line.startswith("PURPOSE:"):
                result["purpose"] = line.replace("PURPOSE:", "").strip()
            elif line.startswith("KEYWORDS:"):
                keywords_text = line.replace("KEYWORDS:", "").strip()
                result["keywords"] = [
                    k.strip() for k in keywords_text.split(",") if k.strip()
                ]

        # Fallback parsing if structured format isn't followed
        if not result["summary"] and cleaned_response:
            # Take first meaningful sentence as summary
            sentences = cleaned_response.replace("\n", " ").split(".")
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and not sentence.startswith(
                    ("Okay", "Let", "I", "The user")
                ):
                    result["summary"] = sentence.strip()
                    break

        return result

    def _create_fallback_summary(self, chunk: CodeChunk) -> Dict:
        """Create a basic summary when AI is not available"""

        # Extract key info from function name and content
        function_name = chunk.function_name or "unnamed"

        # Simple keyword extraction from content
        content_lower = chunk.processed_content.lower()

        # Business action keywords
        action_keywords = []
        actions = [
            "create",
            "update",
            "delete",
            "insert",
            "modify",
            "validate",
            "check",
            "calculate",
            "process",
            "handle",
            "execute",
            "retrieve",
            "fetch",
            "get",
            "set",
        ]
        for action in actions:
            if action in content_lower or action in function_name.lower():
                action_keywords.append(action)

        # Domain keywords from metadata
        domain_keywords = []
        if chunk.business_terms:
            domain_keywords.extend(chunk.business_terms[:3])
        if chunk.module:
            domain_keywords.append(chunk.module.lower())

        # Create simple summary
        chunk_type = chunk.chunk_type.replace("plsql_", "")
        summary = f"IFS Cloud {chunk_type}"

        if chunk.module:
            summary += f" in {chunk.module} module"

        if action_keywords:
            summary += f" that handles {', '.join(action_keywords[:2])} operations"

        if chunk.database_tables:
            summary += f" for {chunk.database_tables[0]} data"

        return {
            "summary": summary,
            "purpose": f"Used for {chunk_type} operations in IFS Cloud system",
            "keywords": action_keywords + domain_keywords,
            "generated_at": datetime.now().isoformat(),
            "method": "fallback",
        }

    async def summarize_chunk(self, chunk: CodeChunk) -> Tuple[Dict, bool]:
        """
        Generate or retrieve cached summary for a code chunk
        Returns: (summary_data, was_cached)
        """
        # Generate content hash for caching
        content_hash = self._generate_content_hash(chunk.processed_content)

        # Check cache first
        cached_summary = self.cache.get_summary(content_hash)
        if cached_summary:
            logger.debug(f"📂 Using cached summary for {chunk.function_name}")
            return cached_summary, True

        # Generate new summary
        logger.debug(f"🤖 Generating new summary for {chunk.function_name}")

        summary_data = await self._get_ai_summary(chunk)

        if not summary_data:
            # Fallback to rule-based summary
            summary_data = self._create_fallback_summary(chunk)
            logger.debug(f"📝 Generated fallback summary for {chunk.function_name}")

        # Cache the result
        self.cache.set_summary(content_hash, summary_data)

        return summary_data, False

    def _should_skip_chunk(self, chunk: CodeChunk) -> bool:
        """
        Selective filtering for moderate to high complexity procedures

        Rules:
        1. Skip low complexity procedures (complexity score < 5)
        2. Skip built-in IFS standard procedures (developers already know these)
        3. Process moderate to high complexity business logic procedures (5+ complexity)
        """
        content_lower = chunk.processed_content.lower()
        function_name = getattr(chunk, "function_name", "").lower()
        lines = chunk.processed_content.split("\n")

        # Skip chunks that are mostly comments
        comment_lines = sum(
            1
            for line in lines
            if line.strip().startswith("--")
            or line.strip().startswith("/*")
            or line.strip().startswith("*")
        )
        if comment_lines > len(lines) * 0.6:
            return True

        # Rule 2: Skip built-in IFS standard procedures (developers know these)
        standard_ifs_patterns = [
            # Standard CRUD operations
            "new___",
            "modify___",
            "remove___",
            "delete___",
            "insert___",
            "update___",
            "check_insert___",
            "check_update___",
            "check_delete___",
            "check_common___",
            # Standard getters/setters
            "get_",
            "set_",
            "get_obj_state___",
            "set_obj_state___",
            # Standard validation/existence checks (keeping validate___ for complex business rules)
            "check_exist___",
            "exist_control___",
            "check_ref___",
            # Standard object operations
            "unpack___",
            "pack___",
            "finite_state___",
            "do_",
            "prepare_insert___",
            "prepare_update___",
            "prepare_delete___",
            # Standard security/access functions
            "check_security___",
            "check_access___",
            "user_allowed_site_api",
            # Standard utility functions
            "lock___",
            "unlock___",
            "exist___",
            "decode",
            "encode",
        ]

        # Check if function name matches standard IFS patterns
        if any(function_name.startswith(pattern) for pattern in standard_ifs_patterns):
            return True

        # Rule 1: Analyze complexity - only process moderate to high complexity procedures
        complexity_score = self._calculate_complexity(content_lower, lines)

        # Only process moderate to high complexity procedures (complexity score >= 5)
        # This ensures we focus on meaningful business logic while being more inclusive
        if complexity_score < 5:
            return True

        return False

    def _calculate_complexity(self, content_lower: str, lines: List[str]) -> int:
        """
        Calculate complexity score based on control structures, SQL operations, and business logic

        Returns:
        - 0-5: Low complexity (simple getters, basic validation)
        - 6-14: Moderate complexity (some business logic, but not critical enough)
        - 15+: VERY HIGH complexity (complex business logic, definitely worth AI summarizing)
        """
        return self._calculate_complexity_static(content_lower, lines)

    @staticmethod
    def _calculate_complexity_static(content_lower: str, lines: List[str]) -> int:
        """
        Static version for multiprocessing compatibility
        """
        complexity_score = 0

        # Control flow structures (high value indicators)
        control_structures = {
            "loop": 2,
            "while": 2,
            "for": 2,
            "if": 1,
            "case": 2,
            "when": 1,
            "cursor": 2,
            "bulk collect": 2,
            "forall": 2,
        }

        for structure, score in control_structures.items():
            count = content_lower.count(structure)
            complexity_score += count * score

        # Exception handling (indicates business logic)
        exception_handling = ["exception", "raise", "pragma exception_init"]
        complexity_score += (
            sum(1 for eh in exception_handling if eh in content_lower) * 2
        )

        # Database operations (moderate complexity)
        db_operations = ["select", "insert", "update", "delete", "merge"]
        db_count = sum(1 for op in db_operations if op in content_lower)
        if db_count > 2:  # Multiple DB operations indicate complexity
            complexity_score += 2
        elif db_count > 0:
            complexity_score += 1

        # Transaction control (indicates important business logic)
        transaction_control = ["commit", "rollback", "savepoint"]
        complexity_score += (
            sum(1 for tc in transaction_control if tc in content_lower) * 2
        )

        # Business logic indicators
        business_logic = [
            "calculate",
            "process",
            "validate",
            "transform",
            "convert",
            "generate",
            "create",
            "build",
        ]
        complexity_score += sum(
            1
            for bl in business_logic
            if bl in content_lower and len(content_lower) > 300
        )

        # Function calls (excluding simple API calls)
        function_call_count = content_lower.count("(") - content_lower.count(
            "--"
        )  # Rough estimate
        if function_call_count > 10:  # Many function calls indicate complexity
            complexity_score += 2
        elif function_call_count > 5:
            complexity_score += 1

        # Line count factor (longer procedures are often more complex)
        non_comment_lines = [
            line for line in lines if line.strip() and not line.strip().startswith("--")
        ]
        if len(non_comment_lines) > 100:
            complexity_score += 2
        elif len(non_comment_lines) > 50:
            complexity_score += 1

        # Variable declarations (more variables = more complex logic)
        variable_declarations = sum(
            1
            for line in lines
            if ":=" in line or "varchar2" in line.lower() or "number" in line.lower()
        )
        if variable_declarations > 10:
            complexity_score += 1

        return complexity_score

    def _filter_chunks_needing_summaries(
        self, chunks: List[CodeChunk]
    ) -> List[CodeChunk]:
        """
        Filter chunks that don't already have cached summaries
        """
        uncached_chunks = []
        for chunk in chunks:
            content_hash = self._generate_content_hash(chunk.processed_content)
            cached_summary = self.cache.get_summary(content_hash)
            if not cached_summary:
                uncached_chunks.append(chunk)
        return uncached_chunks

    async def _analyze_chunks_parallel(
        self, chunks: List[CodeChunk]
    ) -> List[Tuple[CodeChunk, int]]:
        """
        Calculate complexity scores in parallel using multiple CPU cores
        """
        if not self.complexity_analyzer:
            # Fallback to sequential processing
            results = []
            for chunk in chunks:
                score = self._calculate_complexity(
                    chunk.processed_content.lower(), chunk.processed_content.split("\n")
                )
                results.append((chunk, score))
            return results

        loop = asyncio.get_event_loop()

        # Run CPU-intensive complexity calculation in process pool
        futures = [
            loop.run_in_executor(
                self.complexity_analyzer.executor,
                self._calculate_complexity_static,
                chunk.processed_content.lower(),
                chunk.processed_content.split("\n"),
            )
            for chunk in chunks
        ]

        try:
            scores = await asyncio.gather(*futures)
            return list(zip(chunks, scores))
        except Exception as e:
            logger.warning(
                f"⚠️ Parallel complexity analysis failed: {e}, falling back to sequential"
            )
            # Fallback to sequential
            results = []
            for chunk in chunks:
                score = self._calculate_complexity(
                    chunk.processed_content.lower(), chunk.processed_content.split("\n")
                )
                results.append((chunk, score))
            return results

    async def summarize_chunks(
        self, chunks: List[CodeChunk], batch_size: int = 10
    ) -> Dict[str, Dict]:
        """
        Summarize multiple chunks efficiently using intelligent batch processing
        with context window management, retry logic, and source file grouping.
        """
        results = {}
        total_new_summaries = 0
        total_cached_summaries = 0

        logger.info(
            f"🤖 Starting AI summarization of {len(chunks)} chunks with advanced batch processing..."
        )

        # Filter out low-value chunks to reduce noise with detailed complexity analysis
        valuable_chunks = []
        filter_stats = {
            "too_short": 0,
            "mostly_comments": 0,
            "standard_ifs_procedures": 0,
            "low_complexity": 0,
            "excessive_boilerplate": 0,
        }
        complexity_distribution = []

        for chunk in chunks:
            if self._should_skip_chunk(chunk):
                # Determine skip reason for statistics
                content_lower = chunk.processed_content.lower()
                function_name = getattr(chunk, "function_name", "").lower()
                lines = chunk.processed_content.split("\n")

                if len(chunk.processed_content.strip()) < 100:
                    filter_stats["too_short"] += 1
                elif (
                    sum(1 for line in lines if line.strip().startswith("--"))
                    > len(lines) * 0.6
                ):
                    filter_stats["mostly_comments"] += 1
                elif any(
                    pattern in function_name
                    for pattern in [
                        "new___",
                        "modify___",
                        "get_",
                        "set_",
                        "check_insert___",
                        "check_exist___",
                        "check_common___",
                        "insert___",
                        "update___",
                        "delete___",
                    ]
                ):
                    filter_stats["standard_ifs_procedures"] += 1
                elif self._calculate_complexity(content_lower, lines) < 5:
                    filter_stats["low_complexity"] += 1
                else:
                    filter_stats["excessive_boilerplate"] += 1
            else:
                valuable_chunks.append(chunk)
                complexity_score = self._calculate_complexity(
                    chunk.processed_content.lower(), chunk.processed_content.split("\n")
                )
                complexity_distribution.append(complexity_score)

        skipped_count = len(chunks) - len(valuable_chunks)

        if skipped_count > 0:
            filter_percentage = (skipped_count / len(chunks)) * 100
            avg_complexity = (
                sum(complexity_distribution) / len(complexity_distribution)
                if complexity_distribution
                else 0
            )
            logger.info(
                f"⏭️ Filtered out {skipped_count} low-value chunks ({filter_percentage:.1f}% reduction)"
            )
            logger.info(
                f"   📊 Filter breakdown: Short={filter_stats['too_short']}, Comments={filter_stats['mostly_comments']}, Standard={filter_stats['standard_ifs_procedures']}, LowComplex={filter_stats['low_complexity']}, Boilerplate={filter_stats['excessive_boilerplate']}"
            )
            logger.info(
                f"   🎯 Kept chunks avg complexity: {avg_complexity:.1f} (5+ = moderate complexity threshold)"
            )

        if len(valuable_chunks) == 0:
            logger.warning(
                "⚠️ All chunks were filtered out - consider loosening filter criteria"
            )
            return {}

        # Check cache first for all chunks
        cached_chunks = []
        uncached_chunks = []

        for chunk in valuable_chunks:
            content_hash = self._generate_content_hash(chunk.processed_content)
            cached_summary = self.cache.get_summary(content_hash)

            if cached_summary:
                results[chunk.chunk_id] = cached_summary
                total_cached_summaries += 1
                cached_chunks.append(chunk)
            else:
                uncached_chunks.append(chunk)

        logger.info(
            f"📋 Cache analysis: {total_cached_summaries} cached, {len(uncached_chunks)} need processing"
        )

        if not uncached_chunks:
            logger.info("✅ All summaries found in cache!")
            return results

        # Group uncached chunks by source file for better context
        file_groups = self._group_chunks_by_file(uncached_chunks)
        logger.info(
            f"📁 Grouped {len(uncached_chunks)} chunks into {len(file_groups)} files"
        )

        # Model context window limits (40K tokens for qwen3:8b)
        max_context_tokens = 40000

        # Process each file group with intelligent batching
        batch_number = 0
        total_batches = sum(
            len(chunks) for chunks in file_groups.values()
        ) // batch_size + len(file_groups)

        for filename, file_chunks in file_groups.items():
            logger.info(f"📄 Processing {len(file_chunks)} chunks from {filename}")

            # Process file chunks in optimal batches
            for i in range(0, len(file_chunks), batch_size):
                batch = file_chunks[i : i + batch_size]
                batch_number += 1

                logger.info(
                    f"🔄 Processing batch {batch_number}/{total_batches}: {len(batch)} chunks from {filename}"
                )

                # Use advanced batch processing with context management and retry logic
                batch_results = await self._process_batch_with_retry(
                    batch, max_context_tokens
                )

                # Store results and update cache
                for chunk_id, summary_data in batch_results.items():
                    results[chunk_id] = summary_data
                    total_new_summaries += 1

                    # Find the chunk and store in cache
                    chunk = next((c for c in batch if c.chunk_id == chunk_id), None)
                    if chunk:
                        content_hash = self._generate_content_hash(
                            chunk.processed_content
                        )
                        self.cache.set_summary(content_hash, summary_data)

                # Progress update
                progress = f"✅ Batch {batch_number}/{total_batches} complete (new: {total_new_summaries}, cached: {total_cached_summaries})"
                print(f"\r{progress}", end="", flush=True)

                # Small delay between batches to be API-friendly
                await asyncio.sleep(0.2)

        # Final cache save - incremental
        self.cache.save_cache(force_full_rebuild=False)

        # Print newline to finish the progress line, then final summary
        print()  # Move to next line after progress updates

        success_rate = (
            (total_new_summaries + total_cached_summaries) / len(valuable_chunks) * 100
        )
        logger.info(
            f"🎉 Advanced batch summarization complete: {len(results)} summaries total "
            f"({total_new_summaries} new, {total_cached_summaries} cached) - {success_rate:.1f}% success rate"
        )

        # Report any failed chunks
        failed_chunks = len(valuable_chunks) - len(results)
        if failed_chunks > 0:
            logger.warning(
                f"⚠️  {failed_chunks} chunks failed to process and were skipped"
            )

        return results

    async def summarize_chunks_parallel(
        self,
        chunks: List[CodeChunk],
        use_ai: bool = True,
        fallback_to_existing: bool = True,
    ) -> List[SummaryResult]:
        """
        Summarize a list of code chunks with parallel processing and intelligent batching

        Args:
            chunks: List of CodeChunk objects to summarize
            use_ai: Whether to use AI for summarization (if False, returns empty summaries)
            fallback_to_existing: Whether to use existing summaries if available

        Returns:
            List of SummaryResult objects
        """
        start_time = time.time()
        total_chunks = len(chunks)

        logger.info(f"🚀 Starting parallel AI summarization for {total_chunks} chunks")

        if not use_ai:
            logger.info("❌ AI summarization disabled, returning empty summaries")
            return [
                SummaryResult(
                    chunk_id=chunk.chunk_id,
                    title="AI Disabled",
                    summary="",
                    key_concepts=[],
                    reasoning="AI summarization was disabled",
                )
                for chunk in chunks
            ]  # Use existing summaries if requested
        if fallback_to_existing:
            logger.info(f"📚 Checking for existing summaries...")
            chunks = self._filter_chunks_needing_summaries(chunks)
            if len(chunks) != total_chunks:
                logger.info(
                    f"✅ Found {total_chunks - len(chunks)} existing summaries, processing {len(chunks)} new chunks"
                )

        if not chunks:
            logger.info(
                "✅ All chunks already have summaries, returning cached results"
            )
            return []

        # Parallel complexity analysis (CPU-intensive) - disabled for stability
        logger.info(f"🧠 Calculating complexity scores...")
        try:
            # Use sequential complexity analysis for better stability
            chunk_complexity_pairs = [
                (
                    chunk,
                    self._calculate_complexity(
                        chunk.processed_content.lower(),
                        chunk.processed_content.split("\n"),
                    ),
                )
                for chunk in chunks
            ]
        except Exception as e:
            logger.warning(f"⚠️ Complexity analysis failed: {e}, using default scores")
            chunk_complexity_pairs = [
                (chunk, 10) for chunk in chunks
            ]  # Default moderate complexity

        # Filter by complexity threshold
        filtered_pairs = [
            (chunk, score)
            for chunk, score in chunk_complexity_pairs
            if score >= self.complexity_threshold
        ]

        if len(filtered_pairs) != len(chunks):
            skipped_count = len(chunks) - len(filtered_pairs)
            logger.info(
                f"🧠 Filtered out {skipped_count} low-complexity chunks (threshold: {self.complexity_threshold}+), processing {len(filtered_pairs)} chunks"
            )

        if not filtered_pairs:
            logger.info(
                "⚠️ No chunks meet complexity threshold, returning empty summaries for all"
            )
            return [
                SummaryResult(
                    chunk_id=chunk.chunk_id,
                    title="Low Complexity",
                    summary="",
                    key_concepts=[],
                    reasoning=f"Complexity score {score} below threshold {self.complexity_threshold}",
                )
                for chunk, score in chunk_complexity_pairs
            ]

        # Group chunks by source file for better context
        grouped_chunks = self._group_chunks_by_file_with_complexity(filtered_pairs)
        logger.info(
            f"📂 Grouped {len(filtered_pairs)} chunks into {len(grouped_chunks)} file groups"
        )

        # Process file groups in parallel batches
        all_results = []
        batch_tasks = []

        # Process sequentially to avoid GPU memory overload
        logger.info(
            f"🔄 Processing {len(grouped_chunks)} file groups sequentially for optimal GPU performance"
        )

        # Process each file group one by one
        for file_path, file_chunks in grouped_chunks.items():
            try:
                logger.debug(
                    f"📁 Processing file: {file_path} ({len(file_chunks)} chunks)"
                )

                file_results = await self._process_file_group_batched(
                    file_path, file_chunks
                )
                all_results.extend(file_results)

                logger.debug(f"✅ Completed file: {file_path}")

            except Exception as e:
                logger.error(f"❌ Failed to process file group {file_path}: {e}")
                # Add empty results for failed file group
                failed_results = [
                    SummaryResult(
                        chunk_id=chunk.chunk_id,
                        title="Processing Failed",
                        summary="",
                        key_concepts=[],
                        reasoning=f"Failed to process: {str(e)}",
                    )
                    for chunk, _ in file_chunks
                ]
                all_results.extend(failed_results)
            logger.error(f"❌ Critical error in parallel processing: {e}")
            return [
                SummaryResult(
                    chunk_id=chunk.chunk_id,
                    title="Critical Error",
                    summary="",
                    key_concepts=[],
                    reasoning=f"Critical processing error: {str(e)}",
                )
                for chunk, _ in filtered_pairs
            ]

        # Add empty results for chunks that were filtered out
        filtered_chunk_ids = {chunk.chunk_id for chunk, _ in filtered_pairs}
        for chunk, score in chunk_complexity_pairs:
            if chunk.chunk_id not in filtered_chunk_ids:
                all_results.append(
                    SummaryResult(
                        chunk_id=chunk.chunk_id,
                        title="Low Complexity",
                        summary="",
                        key_concepts=[],
                        reasoning=f"Complexity score {score} below threshold {self.complexity_threshold}",
                    )
                )

        # Final statistics
        end_time = time.time()
        total_time = end_time - start_time
        successful_summaries = sum(
            1 for result in all_results if result.summary.strip()
        )

        logger.info(
            f"🏁 Parallel summarization complete: {successful_summaries}/{total_chunks} successful in {total_time:.2f}s"
        )

        if total_time > 0:
            rate = total_chunks / total_time
            logger.info(f"⚡ Processing rate: {rate:.1f} chunks/second")

        return all_results

    def finalize_cache(self):
        """
        Perform final cache cleanup and consolidation.
        Call this at the end of the entire embedding process.
        """
        try:
            logger.info("🔄 Performing final cache consolidation...")
            self.cache.save_cache(force_full_rebuild=True)
            logger.info("✅ Cache consolidation complete")
        except Exception as e:
            logger.error(f"❌ Failed to finalize cache: {e}")

    def enrich_chunk_with_summary(
        self, chunk: CodeChunk, summary_data: Dict
    ) -> CodeChunk:
        """
        Enrich a CodeChunk with AI-generated summary for better embeddings
        """
        # Add the summary to the chunk's processed content for embedding
        enhanced_content = chunk.processed_content

        if summary_data.get("summary"):
            # Prepend natural language summary to improve embedding quality
            enhanced_content = f"""SUMMARY: {summary_data['summary']}

PURPOSE: {summary_data.get('purpose', '')}

KEYWORDS: {', '.join(summary_data.get('keywords', []))}

ORIGINAL CODE:
{chunk.processed_content}"""

        # Update the chunk
        chunk.processed_content = enhanced_content

        # Store the summary in business_terms for searching
        if summary_data.get("keywords"):
            chunk.business_terms.extend(summary_data["keywords"])
            # Remove duplicates
            chunk.business_terms = list(set(chunk.business_terms))

        # Store raw summary for display
        if not hasattr(chunk, "ai_summary"):
            chunk.ai_summary = summary_data

        return chunk


# Global instance for reuse
_summarizer_instance: Optional[AISummarizer] = None


def get_ai_summarizer(cache_dir: Path = None) -> AISummarizer:
    """Get or create the global AI summarizer instance"""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = AISummarizer(cache_dir)
    return _summarizer_instance


async def enrich_chunks_with_ai_summaries(
    chunks: List[CodeChunk], cache_dir: Path = None, batch_size: int = 5
) -> List[CodeChunk]:
    """
    Convenience function to enrich chunks with AI summaries
    """
    summarizer = get_ai_summarizer(cache_dir)

    # Generate summaries
    summaries = await summarizer.summarize_chunks(chunks, batch_size)

    # Enrich chunks
    enriched_chunks = []
    for chunk in chunks:
        if chunk.chunk_id in summaries:
            enriched_chunk = summarizer.enrich_chunk_with_summary(
                chunk, summaries[chunk.chunk_id]
            )
            enriched_chunks.append(enriched_chunk)
        else:
            enriched_chunks.append(chunk)

    return enriched_chunks
