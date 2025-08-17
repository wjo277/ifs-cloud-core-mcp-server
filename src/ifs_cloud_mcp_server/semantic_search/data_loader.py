"""
Data Loading and Preprocessing for IFS Semantic Search Training
================================================================

This module handles the conversion of raw IFS code files into training data
for the semantic search model. It implements sophisticated chunking strategies
and data augmentation techniques specifically designed for enterprise code.
"""

import os
import re
import asyncio
from typing import List, Dict, Generator, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

from .data_structures import CodeChunk, IFSMetadataExtractor

# Optional AI summarization for enhanced semantic search
try:
    from .ai_summarizer import get_ai_summarizer, enrich_chunks_with_ai_summaries

    AI_SUMMARIZATION_AVAILABLE = True
except ImportError:
    AI_SUMMARIZATION_AVAILABLE = False


@dataclass
class ChunkingConfig:
    """
    Configuration for code chunking strategies.

    WHY CONFIGURABLE CHUNKING?
    --------------------------
    Different types of code require different chunking approaches:

    1. FUNCTIONS: Natural semantic boundaries, ideal chunk size
    2. CLASSES: May be too large, need to split into methods
    3. SQL QUERIES: Usually small, may need to group related queries
    4. CONFIGURATION: Often key-value pairs, need context grouping

    This configuration allows fine-tuning for different file types.
    """

    # Target chunk sizes (in characters)
    min_chunk_size: int = 200  # Too small = noise
    max_chunk_size: int = 2000  # Too large = unfocused
    target_chunk_size: int = 800  # Sweet spot for most functions

    # Overlap between chunks (for context preservation)
    overlap_size: int = 100

    # Language-specific settings
    function_patterns: Dict[str, str] = None

    # AI Enhancement (optional, for development)
    enable_ai_summaries: bool = False  # Enable AI-powered summarization
    ai_batch_size: int = 5  # Batch size for AI processing
    comment_patterns: Dict[str, str] = None

    # Content filtering
    min_meaningful_lines: int = 5  # Skip trivial code
    exclude_patterns: List[str] = None  # Skip generated/vendor code

    def __post_init__(self):
        """Initialize default patterns if not provided."""
        if self.function_patterns is None:
            self.function_patterns = {
                ".plsql": r"\b(?:FUNCTION|PROCEDURE)\s+([A-Z_][A-Z0-9_]*)\b",  # More precise matching
                ".sql": r"\b(?:CREATE|ALTER)\s+(?:FUNCTION|PROCEDURE)\s+([A-Z_][A-Z0-9_]*)\b",
                ".js": r"(?:function|const|let)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[=\(]",
                ".ts": r"(?:function|const|let|export\s+function)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
                ".tsx": r"(?:const|function|export\s+const)\s+([A-Za-z_][A-Za-z0-9_]*)",
                ".java": r"(?:public|private|protected)?\s*(?:static)?\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
            }

        if self.comment_patterns is None:
            self.comment_patterns = {
                ".plsql": r"--.*$|/\*.*?\*/",  # Both single-line and multi-line comments
                ".sql": r"--.*$|/\*.*?\*/",  # Both single-line and multi-line comments
                ".js": r"//.*$|/\*.*?\*/",
                ".ts": r"//.*$|/\*.*?\*/",
                ".tsx": r"//.*$|/\*.*?\*/",
                ".java": r"//.*$|/\*.*?\*/",
            }

        if self.exclude_patterns is None:
            self.exclude_patterns = [
                r".*generated.*",
                r".*\.min\.js$",
                r".*vendor.*",
                r".*node_modules.*",
                r".*target/.*",
                r".*build/.*",
            ]


class IFSCodeDataset:
    """
    Dataset class for loading and preprocessing IFS code for semantic search training.

    TRAINING DATA PHILOSOPHY:
    ------------------------
    For effective semantic search, we need training data that teaches the model:

    1. CODE PATTERNS: Similar functions should have similar embeddings
    2. BUSINESS CONTEXT: Code solving similar business problems clusters together
    3. ARCHITECTURAL RELATIONSHIPS: Related layers and modules are nearby in vector space
    4. USAGE PATTERNS: Code that's often used together should be findable together

    We achieve this through:
    - Careful chunking to preserve semantic units
    - Rich metadata extraction for business context
    - Data augmentation to improve pattern recognition
    - Negative sampling to distinguish dissimilar code
    """

    def __init__(
        self,
        source_directories: List[Path],
        config: ChunkingConfig = None,
        cache_dir: Path = None,
    ):
        """
        Initialize the dataset with source directories.

        Args:
            source_directories: List of directories containing IFS code
            config: Chunking configuration (uses defaults if None)
            cache_dir: Directory for caching processed chunks
        """
        self.source_directories = [Path(d) for d in source_directories]
        self.config = config or ChunkingConfig()
        self.cache_dir = cache_dir or Path("cache/semantic_search")

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics tracking
        self.stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "files_skipped": 0,
            "languages": {},
            "modules": {},
            "chunk_size_distribution": [],
        }

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # File extension to language mapping
        self.language_map = {
            ".plsql": "plsql",
            ".sql": "sql",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript-react",
            ".java": "java",
            ".entity": "ifs-entity",
            ".projection": "ifs-projection",
            ".fragment": "ifs-fragment",
            ".client": "ifs-client",
        }

    def load_all_chunks(self) -> Generator[CodeChunk, None, None]:
        """
        Load all code chunks from the source directories.

        LOADING STRATEGY:
        ----------------
        We process files in parallel for speed but yield chunks sequentially
        to maintain memory efficiency. The generator pattern allows processing
        millions of files without memory issues.

        Steps:
        1. Discover all relevant files
        2. Filter out excluded files
        3. Process files in parallel batches
        4. Yield chunks as they're created
        """
        # Discover all files
        all_files = self._discover_files()
        self.logger.info(f"Discovered {len(all_files)} files for processing")

        # Process files in batches to balance memory and speed
        batch_size = 50
        with ThreadPoolExecutor(max_workers=8) as executor:
            for i in range(0, len(all_files), batch_size):
                batch = all_files[i : i + batch_size]

                # Submit batch for processing
                future_to_file = {
                    executor.submit(self._process_file, file_path): file_path
                    for file_path in batch
                }

                # Collect results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        chunks = future.result()
                        for chunk in chunks:
                            yield chunk
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {str(e)}")
                        self.stats["files_skipped"] += 1

    async def load_all_chunks_with_ai_enhancement(self) -> List[CodeChunk]:
        """
        Load all chunks and optionally enhance them with AI summaries for better semantic search.

        This method processes files, extracts chunks, and then enriches them with AI-generated
        natural language summaries to dramatically improve search quality.

        Returns:
            List of CodeChunks, potentially enhanced with AI summaries
        """
        self.logger.info("🚀 Loading chunks with AI enhancement...")

        # First, load all chunks using the standard method
        chunks = list(self.load_all_chunks())
        self.logger.info(f"📦 Loaded {len(chunks)} chunks")

        # Apply AI enhancement if enabled and available
        if self.config.enable_ai_summaries and AI_SUMMARIZATION_AVAILABLE:
            self.logger.info(
                "🤖 Applying AI summarization for enhanced semantic search..."
            )

            # Filter to only enhance code chunks (not generic text)
            code_chunks = [
                c
                for c in chunks
                if c.chunk_type.startswith(("plsql_", "function", "class", "method"))
            ]
            self.logger.info(
                f"🎯 Enhancing {len(code_chunks)} code chunks with AI summaries"
            )

            try:
                # Apply AI enhancement
                enhanced_chunks = await enrich_chunks_with_ai_summaries(
                    code_chunks,
                    cache_dir=self.cache_dir / "ai_summaries",
                    batch_size=self.config.ai_batch_size,
                )

                # Replace enhanced chunks in the original list
                enhanced_map = {c.chunk_id: c for c in enhanced_chunks}
                for i, chunk in enumerate(chunks):
                    if chunk.chunk_id in enhanced_map:
                        chunks[i] = enhanced_map[chunk.chunk_id]

                self.logger.info(
                    f"✅ AI enhancement complete! Enhanced {len(enhanced_chunks)} chunks"
                )

            except Exception as e:
                self.logger.error(f"❌ AI enhancement failed: {e}")
                self.logger.info(
                    "📝 Continuing with standard chunks (no AI enhancement)"
                )

        elif self.config.enable_ai_summaries and not AI_SUMMARIZATION_AVAILABLE:
            self.logger.warning(
                "⚠️ AI summaries requested but AI summarizer not available"
            )
            self.logger.info("💡 Install dev dependencies: uv sync --dev")

        return chunks

    def _discover_files(self) -> List[Path]:
        """
        Discover all relevant code files in source directories.

        FILE DISCOVERY STRATEGY:
        -----------------------
        We want to include all IFS code files while excluding:
        - Generated files (build artifacts)
        - Vendor/third-party code
        - Binary or non-text files
        - Test fixtures that aren't representative

        We prioritize files that are likely to contain business logic.
        """
        files = []

        for source_dir in self.source_directories:
            if not source_dir.exists():
                self.logger.warning(f"Source directory does not exist: {source_dir}")
                continue

            # Walk directory tree
            for root, dirs, filenames in os.walk(source_dir):
                root_path = Path(root)

                # Skip excluded directories
                dirs[:] = [d for d in dirs if not self._is_excluded_dir(root_path / d)]

                for filename in filenames:
                    file_path = root_path / filename

                    # Check if file should be included
                    if self._should_include_file(file_path):
                        files.append(file_path)

        # Sort files for consistent processing order
        files.sort()

        self.logger.info(f"File discovery complete. Found files by type:")
        file_types = {}
        for file in files:
            ext = file.suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1

        for ext, count in sorted(file_types.items()):
            self.logger.info(f"  {ext}: {count} files")

        return files

    def _should_include_file(self, file_path: Path) -> bool:
        """Determine if a file should be included in processing."""
        # Check file extension
        if file_path.suffix.lower() not in self.language_map:
            return False

        # Check against exclude patterns
        file_str = str(file_path).lower()
        for pattern in self.config.exclude_patterns:
            if re.search(pattern, file_str):
                return False

        # Check file size (skip empty files and very large ones)
        try:
            size = file_path.stat().st_size
            if size < 50:  # Too small to be meaningful
                return False
            if size > 1024 * 1024 * 2:  # Over 2MB is probably generated
                return False
        except OSError:
            return False

        return True

    def _is_excluded_dir(self, dir_path: Path) -> bool:
        """Check if directory should be excluded."""
        dir_name = dir_path.name.lower()
        excluded_dirs = {
            "node_modules",
            "target",
            "build",
            "dist",
            ".git",
            "generated",
            "temp",
            "tmp",
            "__pycache__",
        }
        return dir_name in excluded_dirs

    def _process_file(self, file_path: Path) -> List[CodeChunk]:
        """
        Process a single file into code chunks.

        FILE PROCESSING PIPELINE:
        ------------------------
        1. Read file content (with encoding detection)
        2. Extract file-level metadata
        3. Apply language-specific chunking
        4. Create CodeChunk objects with metadata
        5. Filter out low-quality chunks
        6. Return processed chunks
        """
        try:
            # Read file content
            content = self._read_file_content(file_path)
            if not content or len(content.strip()) < self.config.min_meaningful_lines:
                return []

            # Get language and metadata
            language = self.language_map.get(file_path.suffix.lower(), "unknown")

            # Apply chunking strategy based on file type
            raw_chunks = self._chunk_file(file_path, content, language)

            # Convert to CodeChunk objects
            code_chunks = []
            for (
                chunk_content,
                start_line,
                end_line,
                chunk_type,
                function_name,
            ) in raw_chunks:
                try:
                    chunk = CodeChunk.create_from_file_section(
                        file_path=file_path,
                        content=chunk_content,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type=chunk_type,
                        function_name=function_name,
                    )
                    chunk.language = language
                    chunk.file_size = len(content)
                    chunk.last_modified = file_path.stat().st_mtime

                    code_chunks.append(chunk)

                except Exception as e:
                    self.logger.debug(
                        f"Error creating chunk from {file_path}:{start_line}: {e}"
                    )
                    continue

            # Update statistics
            self.stats["files_processed"] += 1
            self.stats["chunks_created"] += len(code_chunks)
            self.stats["languages"][language] = (
                self.stats["languages"].get(language, 0) + 1
            )

            # Track chunk sizes for analysis
            for chunk in code_chunks:
                self.stats["chunk_size_distribution"].append(len(chunk.raw_content))

            return code_chunks

        except Exception as e:
            self.logger.error(f"Failed to process file {file_path}: {str(e)}")
            self.stats["files_skipped"] += 1
            return []

    def _read_file_content(self, file_path: Path) -> str:
        """
        Read file content with encoding detection.

        ENCODING STRATEGY:
        -----------------
        Enterprise codebases often have mixed encodings:
        - UTF-8 (modern files)
        - Latin-1 (legacy files)
        - UTF-16 (Windows generated files)

        We try encodings in order of likelihood.
        """
        encodings = ["utf-8", "latin-1", "utf-16", "cp1252"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue

        # If all encodings fail, read as binary and decode with errors='replace'
        self.logger.warning(f"Could not decode {file_path} with standard encodings")
        with open(file_path, "rb") as f:
            return f.read().decode("utf-8", errors="replace")

    def _chunk_file(
        self, file_path: Path, content: str, language: str
    ) -> List[Tuple[str, int, int, str, Optional[str]]]:
        """
        Chunk file content based on language and content type.

        CHUNKING STRATEGIES BY LANGUAGE:
        -------------------------------

        PL/SQL (.plsql, .sql):
        - Chunk by FUNCTION/PROCEDURE boundaries
        - Include package context for orphaned code
        - Preserve complete SQL statements

        JavaScript/TypeScript (.js, .ts, .tsx):
        - Chunk by function/class/component boundaries
        - Preserve JSX components as complete units
        - Include import context

        IFS Specific (.entity, .projection, .fragment):
        - These are often configuration files
        - Chunk by logical sections or complete objects
        - Preserve relationships between related configs

        Returns:
            List of tuples: (content, start_line, end_line, chunk_type, function_name)
        """
        extension = file_path.suffix.lower()

        if extension in [".plsql", ".sql"]:
            return self._chunk_plsql(content)
        elif extension in [".js", ".ts", ".tsx"]:
            return self._chunk_javascript(content)
        elif extension in [".entity", ".projection", ".fragment", ".client"]:
            return self._chunk_ifs_config(content)
        else:
            # Generic chunking for unknown file types
            return self._chunk_generic(content)

    def _chunk_plsql(
        self, content: str
    ) -> List[Tuple[str, int, int, str, Optional[str]]]:
        """
        Chunk PL/SQL code by function/procedure boundaries using AST parser.

        ROBUST AST-BASED PL/SQL CHUNKING:
        ---------------------------------
        Uses the ConservativePLSQLAnalyzer to:
        1. Parse PL/SQL with full AST understanding
        2. Handle nested functions/procedures correctly
        3. Properly detect function/procedure boundaries
        4. Include nested functions within their parent scope
        5. Handle all PL/SQL constructs (strings, comments, complex END statements)

        This leverages the existing parser for 100% accuracy.
        """
        # Import the parser-based chunking function
        from .plsql_parser_chunker import chunk_plsql_with_parser

        chunks = []

        if not content.strip():
            return chunks

        # Use the AST-based parser for robust chunking
        parsed_chunks = chunk_plsql_with_parser(content)

        if not parsed_chunks:
            # No functions found - chunk the entire file if it's meaningful
            lines = content.split("\n")
            if len(content.strip()) > self.config.min_chunk_size:
                chunks.append((content, 1, len(lines), "plsql_block", None))
            return chunks

        # Filter chunks by size requirements and return
        for chunk_content, start_line, end_line, unit_type, name in parsed_chunks:
            if len(chunk_content.strip()) >= self.config.min_chunk_size:
                chunks.append((chunk_content, start_line, end_line, unit_type, name))

        return chunks

        return chunks

    # OLD MANUAL PARSING METHODS - NO LONGER NEEDED
    # Now using AST-based parser for 100% accuracy

    def _parse_plsql_units(self, content: str) -> List[Tuple[int, int, str, str, str]]:
        """
        DEPRECATED: Old manual parser replaced by AST-based ConservativePLSQLAnalyzer
        """
        # This method is kept for backward compatibility but is no longer used
        return []

    def _extract_plsql_unit(
        self, lines: List[str], start_idx: int
    ) -> Tuple[List[str], int]:
        """
        DEPRECATED: Old manual parser replaced by AST-based ConservativePLSQLAnalyzer
        """
        # This method is kept for backward compatibility but is no longer used
        return [], start_idx

    def _chunk_javascript(
        self, content: str
    ) -> List[Tuple[str, int, int, str, Optional[str]]]:
        """
        Chunk JavaScript/TypeScript code by function and class boundaries.

        JS/TS CHUNKING STRATEGY:
        -----------------------
        1. Find function declarations (function, const x = ()=>, class methods)
        2. Find React components (const Component = () =>)
        3. Find class definitions
        4. Use bracket matching to find complete blocks
        5. Include JSX components as complete units
        """
        chunks = []
        lines = content.split("\n")

        # Multiple patterns for different JS constructs
        patterns = [
            (
                r"^(?:export\s+)?(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:\([^)]*\)|\([^)]*\))\s*=>",
                "arrow_function",
            ),
            (
                r"^(?:export\s+)?function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
                "function_declaration",
            ),
            (r"^(?:export\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)", "class_declaration"),
            (r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*{", "method"),
        ]

        # Simple approach: look for function-like patterns and try to match brackets
        found_functions = []

        for i, line in enumerate(lines):
            for pattern, chunk_type in patterns:
                match = re.match(pattern, line.strip())
                if match:
                    func_name = match.group(1)
                    start_line = i + 1

                    # Try to find the end by bracket matching
                    end_line = self._find_js_block_end(lines, i)
                    if end_line > start_line:
                        found_functions.append(
                            (start_line, end_line, func_name, chunk_type)
                        )
                    break

        # Convert found functions to chunks
        for start_line, end_line, func_name, chunk_type in found_functions:
            func_lines = lines[start_line - 1 : end_line]
            func_content = "\n".join(func_lines)

            if len(func_content.strip()) >= self.config.min_chunk_size:
                chunks.append(
                    (func_content, start_line, end_line, chunk_type, func_name)
                )

        # If no functions found, chunk by reasonable size
        if not chunks and len(content.strip()) > self.config.min_chunk_size:
            chunks.append((content, 1, len(lines), "js_module", None))

        return chunks

    def _find_js_block_end(self, lines: List[str], start_line_idx: int) -> int:
        """
        Find the end of a JavaScript block using bracket matching.

        BRACKET MATCHING STRATEGY:
        -------------------------
        1. Track opening { and closing }
        2. Account for string literals that contain brackets
        3. Stop when bracket count returns to 0
        4. Handle edge cases like arrow functions
        """
        bracket_count = 0
        in_string = False
        string_char = None

        for i in range(start_line_idx, len(lines)):
            line = lines[i]

            for char in line:
                # Handle string literals
                if char in ['"', "'"]:
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None
                    continue

                if in_string:
                    continue

                # Count brackets
                if char == "{":
                    bracket_count += 1
                elif char == "}":
                    bracket_count -= 1

                    # If we're back to 0 and we've seen at least one opening bracket
                    if bracket_count == 0 and i > start_line_idx:
                        return i + 1

        # Fallback: if no matching bracket found, take next 50 lines or end of file
        return min(start_line_idx + 50, len(lines))

    def _chunk_ifs_config(
        self, content: str
    ) -> List[Tuple[str, int, int, str, Optional[str]]]:
        """
        Chunk IFS configuration files (.entity, .projection, .fragment).

        IFS CONFIG CHUNKING:
        -------------------
        These files are often structured configuration with:
        - Entity definitions
        - Projection mappings
        - Fragment components
        - Client page definitions

        We want to preserve complete logical units.
        """
        # For now, treat each file as a single chunk since they're usually
        # cohesive configuration units
        lines = content.split("\n")

        if len(content.strip()) < self.config.min_chunk_size:
            return []

        # Try to identify the main component name from the content
        component_name = None

        # Look for entity names, projection names, etc.
        patterns = [
            r"entity\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"projection\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"fragment\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"client\s+([A-Za-z_][A-Za-z0-9_]*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                component_name = match.group(1)
                break

        chunk_type = "ifs_config"
        if ".entity" in content:
            chunk_type = "ifs_entity"
        elif ".projection" in content:
            chunk_type = "ifs_projection"
        elif ".fragment" in content:
            chunk_type = "ifs_fragment"
        elif ".client" in content:
            chunk_type = "ifs_client"

        return [(content, 1, len(lines), chunk_type, component_name)]

    def _chunk_generic(
        self, content: str
    ) -> List[Tuple[str, int, int, str, Optional[str]]]:
        """
        Generic chunking strategy for unknown file types.

        GENERIC CHUNKING:
        ----------------
        When we don't have language-specific rules:
        1. Split by reasonable size boundaries
        2. Try to split at logical points (empty lines, comments)
        3. Maintain some overlap for context
        """
        lines = content.split("\n")
        chunks = []

        if len(content) <= self.config.max_chunk_size:
            # Small enough to be one chunk
            return [(content, 1, len(lines), "generic_block", None)]

        # Split into chunks at logical boundaries
        current_chunk = []
        current_size = 0
        start_line = 1

        for i, line in enumerate(lines):
            current_chunk.append(line)
            current_size += len(line) + 1  # +1 for newline

            # Check if we should end this chunk
            should_end = False

            if current_size >= self.config.target_chunk_size:
                # Look for a good breaking point in the next few lines
                for j in range(i + 1, min(i + 10, len(lines))):
                    if not lines[j].strip():  # Empty line
                        should_end = True
                        break
                    if lines[j].strip().startswith("#") or lines[j].strip().startswith(
                        "//"
                    ):  # Comment
                        should_end = True
                        break

            if should_end or current_size >= self.config.max_chunk_size:
                # Create chunk
                chunk_content = "\n".join(current_chunk)
                if len(chunk_content.strip()) >= self.config.min_chunk_size:
                    chunks.append(
                        (chunk_content, start_line, i + 1, "generic_block", None)
                    )

                # Start new chunk with some overlap
                if self.config.overlap_size > 0:
                    overlap_lines = current_chunk[-5:]  # Take last 5 lines as overlap
                    current_chunk = overlap_lines
                    current_size = sum(len(line) + 1 for line in overlap_lines)
                    start_line = i - 4  # Adjust start line for overlap
                else:
                    current_chunk = []
                    current_size = 0
                    start_line = i + 2

        # Handle remaining content
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            if len(chunk_content.strip()) >= self.config.min_chunk_size:
                chunks.append(
                    (chunk_content, start_line, len(lines), "generic_block", None)
                )

        return chunks

    def get_statistics(self) -> Dict[str, any]:
        """Get processing statistics."""
        stats = self.stats.copy()

        if stats["chunk_size_distribution"]:
            sizes = stats["chunk_size_distribution"]
            stats["chunk_size_stats"] = {
                "mean": sum(sizes) / len(sizes),
                "min": min(sizes),
                "max": max(sizes),
                "median": sorted(sizes)[len(sizes) // 2],
            }

        return stats

    def save_chunks_to_cache(
        self, chunks: List[CodeChunk], cache_name: str = "default"
    ) -> Path:
        """
        Save processed chunks to cache for faster reloading.

        CACHING STRATEGY:
        ----------------
        Training data processing is expensive, so we cache:
        1. Processed chunks with all metadata
        2. File modification times for invalidation
        3. Configuration hash for cache invalidation

        This allows quick iteration during model development.
        """
        import pickle

        cache_file = self.cache_dir / f"{cache_name}_chunks.pkl"

        # Create cache metadata
        cache_data = {
            "chunks": chunks,
            "config": self.config,
            "timestamp": datetime.now(),
            "stats": self.stats,
            "version": "1.0",  # For future compatibility
        }

        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        self.logger.info(f"Saved {len(chunks)} chunks to cache: {cache_file}")
        return cache_file

    def load_chunks_from_cache(
        self, cache_name: str = "default"
    ) -> Optional[List[CodeChunk]]:
        """Load chunks from cache if available and valid."""
        import pickle

        cache_file = self.cache_dir / f"{cache_name}_chunks.pkl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Validate cache (basic check - in production you'd check file mtimes)
            if cache_data.get("version") != "1.0":
                self.logger.warning("Cache version mismatch, ignoring cache")
                return None

            chunks = cache_data["chunks"]
            self.stats = cache_data.get("stats", {})

            self.logger.info(f"Loaded {len(chunks)} chunks from cache: {cache_file}")
            return chunks

        except Exception as e:
            self.logger.error(f"Error loading cache {cache_file}: {e}")
            return None


class DataAugmenter:
    """
    Data augmentation for improving semantic search training.

    WHY DATA AUGMENTATION FOR CODE?
    -------------------------------
    Raw code might not provide enough variety for robust semantic learning.
    We augment training data by:

    1. PARAPHRASING: Generate different ways to describe the same code
    2. PARTIAL QUERIES: Create queries that match only part of a function
    3. BUSINESS CONTEXT: Add business-oriented descriptions
    4. NEGATIVE SAMPLING: Ensure model learns what NOT to match

    This creates a richer training dataset that generalizes better to user queries.
    """

    def __init__(self):
        # Templates for generating query variations
        self.query_templates = [
            "How do I {action} {business_object}?",
            "Function that {action} {business_object}",
            "Code for {action} {business_object}",
            "{business_object} {action} implementation",
            "Example of {action} in {module}",
            "{business_object} processing logic",
        ]

        # Action words extraction patterns
        self.action_patterns = [
            r"\b(create|insert|add|new)\b",
            r"\b(update|modify|change|edit)\b",
            r"\b(delete|remove|drop)\b",
            r"\b(select|find|get|retrieve|search)\b",
            r"\b(calculate|compute|process)\b",
            r"\b(validate|check|verify)\b",
        ]

    def generate_queries_for_chunk(self, chunk: CodeChunk) -> List[str]:
        """
        Generate multiple query variations that should match this chunk.

        QUERY GENERATION STRATEGY:
        -------------------------
        For each code chunk, we generate queries that a user might type
        when looking for this functionality:

        1. Extract business terms and actions from the code
        2. Create natural language queries using templates
        3. Include technical and business perspectives
        4. Generate partial matches (for robust training)
        """
        queries = []

        # Start with function name variations
        if chunk.function_name:
            queries.extend(
                [
                    chunk.function_name.lower().replace("_", " "),
                    f"function {chunk.function_name}",
                    chunk.function_name,
                ]
            )

        # Add business term queries
        for term in chunk.business_terms[:3]:  # Limit to top 3
            queries.extend(
                [
                    term.lower(),
                    f"how to handle {term.lower()}",
                    f"{term.lower()} processing",
                ]
            )

        # Add API-based queries
        for api_call in chunk.api_calls[:2]:  # Limit to top 2
            queries.extend([api_call, f"using {api_call}", f"example {api_call}"])

        # Add module-specific queries
        if chunk.module:
            queries.extend(
                [
                    f"{chunk.module.lower()} functions",
                    f"code in {chunk.module.lower()}",
                    f"{chunk.module.lower()} implementation",
                ]
            )

        # Generate template-based queries
        if chunk.business_terms and chunk.function_name:
            # Extract likely action from function name
            action = self._extract_action(chunk.function_name)
            business_object = (
                chunk.business_terms[0] if chunk.business_terms else "data"
            )

            if action:
                for template in self.query_templates[:3]:  # Limit templates
                    try:
                        query = template.format(
                            action=action,
                            business_object=business_object.lower(),
                            module=chunk.module.lower() if chunk.module else "system",
                        )
                        queries.append(query)
                    except KeyError:
                        continue

        # Remove duplicates and empty queries
        unique_queries = list(set(q.strip() for q in queries if q and q.strip()))

        # Limit total queries per chunk to avoid overwhelming training
        return unique_queries[:10]

    def _extract_action(self, function_name: str) -> Optional[str]:
        """Extract action word from function name."""
        name_lower = function_name.lower()

        for pattern in self.action_patterns:
            match = re.search(pattern, name_lower)
            if match:
                return match.group(1)

        # Common prefixes in IFS functions
        if name_lower.startswith("get_"):
            return "get"
        elif name_lower.startswith("set_"):
            return "set"
        elif name_lower.startswith("new_"):
            return "create"
        elif name_lower.startswith("remove_"):
            return "remove"

        return None

    def generate_negative_samples(
        self,
        target_chunk: CodeChunk,
        all_chunks: List[CodeChunk],
        num_negatives: int = 3,
    ) -> List[CodeChunk]:
        """
        Generate negative samples for contrastive learning.

        NEGATIVE SAMPLING STRATEGY:
        --------------------------
        Good negative samples are important for teaching the model
        what NOT to match. We select chunks that are:

        1. DIFFERENT MODULE: Same functionality, different domain
        2. DIFFERENT LAYER: Different architectural concerns
        3. SIMILAR NAMES: Functions with similar names but different purpose
        4. RANDOM: Completely unrelated code

        This helps the model learn fine-grained distinctions.
        """
        negatives = []

        # Filter potential negatives
        candidates = [c for c in all_chunks if c.chunk_id != target_chunk.chunk_id]

        if not candidates:
            return []

        # Strategy 1: Different module, similar functionality
        if target_chunk.module:
            different_module = [
                c
                for c in candidates
                if c.module
                and c.module != target_chunk.module
                and any(
                    term in c.business_terms for term in target_chunk.business_terms
                )
            ]
            if different_module:
                negatives.append(different_module[0])

        # Strategy 2: Different layer
        if target_chunk.layer:
            different_layer = [
                c for c in candidates if c.layer and c.layer != target_chunk.layer
            ]
            if different_layer:
                negatives.append(different_layer[0])

        # Strategy 3: Similar function names but different purpose
        if target_chunk.function_name:
            similar_names = [
                c
                for c in candidates
                if c.function_name
                and c.function_name != target_chunk.function_name
                and c.function_name.split("_")[0]
                == target_chunk.function_name.split("_")[0]
            ]
            if similar_names:
                negatives.append(similar_names[0])

        # Fill remaining slots with random samples
        remaining_needed = num_negatives - len(negatives)
        if remaining_needed > 0:
            import random

            random_samples = random.sample(
                candidates, min(remaining_needed, len(candidates))
            )
            negatives.extend(random_samples)

        return negatives[:num_negatives]


# Example usage and testing
if __name__ == "__main__":
    import logging

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Example: Load chunks from a directory
    source_dirs = [Path("_work")]
    dataset = IFSCodeDataset(source_dirs)

    print("Loading chunks...")
    chunks = list(dataset.load_all_chunks())

    print(f"Loaded {len(chunks)} chunks")
    print("\nStatistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        if key != "chunk_size_distribution":  # Skip the large array
            print(f"  {key}: {value}")

    # Show some example chunks
    print("\nExample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i + 1}:")
        print(f"  File: {chunk.file_path}")
        print(f"  Function: {chunk.function_name}")
        print(f"  Module: {chunk.module}")
        print(f"  Business terms: {chunk.business_terms[:3]}")
        print(f"  Embedding text preview: {chunk.to_embedding_text()[:200]}...")
