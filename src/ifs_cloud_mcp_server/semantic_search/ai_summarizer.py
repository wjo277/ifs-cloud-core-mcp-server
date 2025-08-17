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
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import aiofiles

# Optional dependency for development-time summarization
try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

from .data_structures import CodeChunk

logger = logging.getLogger(__name__)


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
    _written_hashes: set = field(default_factory=set)  # Track what's been written to disk

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
        return len(json.dumps(line_data, ensure_ascii=False).encode('utf-8')) + 1  # +1 for newline

    def _load_legacy_cache(self) -> int:
        """Load and migrate legacy JSON cache format"""
        legacy_path = self._get_legacy_cache_path()
        if not legacy_path.exists():
            return 0
        
        try:
            with open(legacy_path, "r", encoding="utf-8") as f:
                legacy_summaries = json.load(f)
            
            logger.info(f"� Migrating {len(legacy_summaries)} summaries from legacy format...")
            
            # Add all legacy summaries to current cache
            for content_hash, summary_data in legacy_summaries.items():
                self.summaries[content_hash] = summary_data
            
            # Save in new format
            self._save_all_summaries()
            
            # Backup and remove legacy file
            backup_path = legacy_path.with_suffix('.json.backup')
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
                logger.info(f"📦 Backing up {len(existing_files)} existing cache files...")
                
                for file_path in existing_files:
                    backup_path = backup_dir / file_path.name
                    # If backup already exists, add timestamp
                    if backup_path.exists():
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = backup_dir / f"{file_path.stem}_{timestamp}.ndjson"
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
                    line_size = len(line.encode('utf-8'))
                    
                    # Check if we need a new file
                    if current_file is None or (current_file_size + line_size > max_size_bytes):
                        if current_file:
                            current_file.close()
                        
                        file_path = self._get_cache_file_path(current_file_index)
                        files_written.append(file_path)
                        current_file = open(file_path, 'w', encoding='utf-8')
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
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = sum(1 for line in f if line.strip())
                        total_written_lines += lines
                
                if total_written_lines != len(self.summaries):
                    raise Exception(f"Data integrity check failed: wrote {total_written_lines} lines but expected {len(self.summaries)}")
                
                # Step 4: Update tracking variables
                self._current_file_index = current_file_index - 1 if current_file_index > 0 else 0
                self._current_file_size = current_file_size
                
                # Mark all summaries as written to disk
                self._written_hashes = set(self.summaries.keys())
                
                logger.info(f"✅ Successfully wrote {len(self.summaries)} summaries to {len(files_written)} NDJSON files")
                
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
                        restore_path = self.cache_dir / backup_file.name.split('_')[0] + '_' + backup_file.name.split('_')[1] + '.ndjson'
                        # Handle timestamped backups
                        if len(backup_file.name.split('_')) > 2:
                            restore_path = self.cache_dir / f"summaries_{backup_file.name.split('_')[1]}.ndjson"
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
                                self._written_hashes.add(content_hash)  # Mark as written to disk
                                total_loaded += 1
                                
                            except (json.JSONDecodeError, KeyError) as e:
                                logger.warning(f"⚠️ Skipping invalid line {line_num} in {file_path.name}: {e}")
                                continue
                                
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load cache file {file_path}: {e}")
                    continue
            
            # Track the highest file index and size
            if cache_files:
                last_file_path = cache_files[-1]
                self._current_file_index = int(last_file_path.stem.split('_')[-1])
                self._current_file_size = last_file_path.stat().st_size
            else:
                self._current_file_index = 0
                self._current_file_size = 0
                        
            if total_loaded > 0:
                logger.info(f"📂 Loaded {total_loaded} cached summaries from {len(cache_files)} NDJSON files")
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
                logger.info(f"💾 Full rebuild: {len(self.summaries)} summaries to NDJSON cache")
            else:
                # Incremental save - only append new summaries
                unwritten_count = self._save_new_summaries()
                if unwritten_count > 0:
                    logger.debug(f"💾 Appended {unwritten_count} new summaries to cache")
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
                line_size = len(line.encode('utf-8'))
                
                # Check if we need a new file
                if self._current_file_size + line_size > max_size_bytes:
                    self._current_file_index += 1
                    self._current_file_size = 0
                
                file_path = self._get_cache_file_path()
                
                # Append to file
                with open(file_path, 'a', encoding='utf-8') as f:
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
    AI-powered code summarization using Qwen3-8B for enhanced semantic search
    """

    def __init__(self, cache_dir: Path = None, model_name: str = "qwen3:8b"):
        self.model_name = model_name
        self.cache_dir = cache_dir or Path("cache/ai_summaries")
        self.cache = SummaryCache(self.cache_dir)  # Pass directory, not file
        self.ollama_available = False
        self.async_client = None

        # Initialize Ollama connection
        self._check_ollama()

    def _check_ollama(self):
        """Check if Ollama is available and the model is installed"""
        if not OLLAMA_AVAILABLE:
            logger.info("💡 Ollama not installed. AI summaries disabled.")
            logger.info("   To enable AI summaries in development:")
            logger.info("   1. Install dev dependencies: uv sync --dev")
            logger.info("   2. Install Ollama: https://ollama.ai")
            logger.info("   3. Pull model: ollama pull qwen2.5:8b")
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
                logger.info(f"✅ Ollama available with {self.model_name} (async client)")
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
                "💡 To enable AI summaries, install Ollama and run: ollama pull qwen2.5:8b"
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

        # Create the prompt with less boilerplate-generating language
        prompt = f"""You are an expert in IFS Cloud ERP system. Analyze this {function_context} from the {context}.

CODE TO ANALYZE:
```sql
{chunk.processed_content[:2000]}  
```

CONTEXT: {business_context}

Provide a summary in this exact format (no thinking process, just the direct response):

SUMMARY: [Direct business action/functionality - avoid starting with "This procedure/function/code"]
PURPOSE: [Specific business scenario when this would be used]  
KEYWORDS: [5-8 relevant business and technical keywords, comma-separated]

Guidelines:
- Start summaries with action verbs (e.g., "Updates customer pricing", "Calculates commission rates")
- Focus on unique business value, not generic operations
- Avoid repetitive phrases like "ensures business rules" or "maintains data integrity"
- Skip obvious IFS/ERP terminology unless it's the core function
- Be specific about what makes this different from similar functions

Keep it concise but informative. Do not include any thinking process or metadata."""

        return prompt

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
                options={
                    "temperature": 0.2,  # Lower temperature for more focused summaries
                    "top_p": 0.7,
                    "num_ctx": 4096,  # Context window
                    "num_predict": 300,  # Max tokens for response
                    # Remove stop tokens that were causing empty responses
                },
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
        Determine if chunk should be skipped due to low-value repetitive content
        """
        content_lower = chunk.processed_content.lower()
        
        # Skip very short chunks that are likely just declarations
        if len(chunk.processed_content.strip()) < 100:
            return True
            
        # Skip chunks that are mostly comments or copyright notices
        lines = chunk.processed_content.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('--') or line.strip().startswith('/*') or line.strip().startswith('*'))
        if comment_lines > len(lines) * 0.7:  # More than 70% comments
            return True
        
        # Skip generic validation procedures that follow common patterns
        generic_patterns = [
            'check_exist___', 'exist_control___', 'validate___',
            'check_insert___', 'check_update___', 'check_delete___',
            'unpack___', 'pack___', 'get_obj_state___',
            'finite_state___', 'do_', 'get_', 'set_'
        ]
        
        function_name = getattr(chunk, 'function_name', '').lower()
        if any(pattern in function_name for pattern in generic_patterns):
            return True
            
        # Skip chunks that contain mostly boilerplate IFS patterns
        boilerplate_patterns = [
            'error_sys.record_not_exist',
            'client_sys.add_to_attr',
            'client_sys.set_item_value', 
            'finite_state_machine___',
            'objversion control',
            'rowversion check',
            'user_allowed_site_api'
        ]
        
        if any(pattern in content_lower for pattern in boilerplate_patterns):
            return True
            
        # Skip chunks that are mostly just simple CRUD operations without business logic
        crud_indicators = ['select', 'insert', 'update', 'delete', 'from', 'where']
        crud_count = sum(1 for indicator in crud_indicators if indicator in content_lower)
        other_logic = any(keyword in content_lower for keyword in [
            'if', 'case', 'when', 'loop', 'while', 'for', 'cursor',
            'exception', 'raise', 'commit', 'rollback', 'function', 'procedure'
        ])
        
        # Skip if mostly CRUD with little business logic
        if crud_count >= 3 and not other_logic and len(content_lower) < 500:
            return True
            
        return False

    async def summarize_chunks(
        self, chunks: List[CodeChunk], batch_size: int = 20
    ) -> Dict[str, Dict]:
        """
        Summarize multiple chunks efficiently - Ollama handles internal queuing
        """
        results = {}
        total_new_summaries = 0
        total_cached_summaries = 0

        logger.info(f"🤖 Starting AI summarization of {len(chunks)} chunks...")
        
        # Filter out low-value chunks to reduce noise
        valuable_chunks = [chunk for chunk in chunks if not self._should_skip_chunk(chunk)]
        skipped_count = len(chunks) - len(valuable_chunks)
        
        if skipped_count > 0:
            logger.info(f"⏭️ Skipped {skipped_count} low-value chunks (comments, simple CRUD, boilerplate)")

        # Process in larger batches - Ollama can handle internal queuing
        for i in range(0, len(valuable_chunks), batch_size):
            batch = valuable_chunks[i : i + batch_size]

            # Send all chunks in batch simultaneously to Ollama - it will queue internally
            tasks = [self.summarize_chunk(chunk) for chunk in batch]
            summary_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Store results and track cache hits
            for chunk, result in zip(batch, summary_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"❌ Failed to summarize {chunk.function_name}: {result}"
                    )
                    continue

                summary_data, was_cached = result
                results[chunk.chunk_id] = summary_data

                if was_cached:
                    total_cached_summaries += 1
                else:
                    total_new_summaries += 1

            # Use print with carriage return to overwrite same line
            batch_progress = f"✅ Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} (new: {total_new_summaries}, cached: {total_cached_summaries})"
            print(f"\r{batch_progress}", end="", flush=True)

            # No delay needed - Ollama handles internal queuing

        # Incremental save after processing batch - only append new summaries
        self.cache.save_cache(force_full_rebuild=False)

        # Print newline to finish the progress line, then final summary
        print()  # Move to next line after progress updates
        
        logger.info(
            f"🎉 Completed AI summarization: {len(results)} summaries total "
            f"({total_new_summaries} new, {total_cached_summaries} cached)"
        )
        return results

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
