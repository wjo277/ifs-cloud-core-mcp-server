"""Tantivy-based indexer for IFS Cloud files with enhanced caching."""

import os
import hashlib
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass
from datetime import datetime

import tantivy
import aiofiles
from pydantic import BaseModel

from .parsers import IFSFileParser


logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Metadata for tracking file changes and caching."""
    
    path: str
    size: int
    modified_time: float
    hash: str
    indexed_at: float


class SearchResult(BaseModel):
    """Search result model."""

    path: str
    name: str
    type: str
    content_preview: str
    score: float
    entities: List[str]
    line_count: int
    complexity_score: float
    modified_time: datetime


class IFSCloudTantivyIndexer:
    """High-performance Tantivy-based indexer for IFS Cloud files."""

    # Supported IFS Cloud file extensions
    SUPPORTED_EXTENSIONS = {
        ".entity",
        ".plsql",
        ".views",
        ".storage",
        ".fragment",
        ".client",
        ".projection",
        ".plsvc",
    }

    def __init__(self, index_path: Union[str, Path], create_new: bool = False):
        """Initialize the Tantivy indexer with enhanced caching.

        Args:
            index_path: Path to store the Tantivy index
            create_new: Whether to create a new index (overwriting existing)
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata file path
        self.cache_metadata_path = self.index_path / "cache_metadata.json"
        self._file_cache: Dict[str, FileMetadata] = {}
        self._load_cache_metadata()

        self._schema = self._create_schema()
        self._index = self._create_or_open_index(create_new)
        self._writer = self._index.writer(heap_size=50_000_000)  # 50MB heap
        self._parser = IFSFileParser()

        logger.info(f"Initialized Tantivy indexer at {self.index_path}")
        logger.info(f"Cache contains {len(self._file_cache)} file entries")

    def _create_schema(self) -> tantivy.Schema:
        """Create the Tantivy schema for IFS Cloud files."""
        schema_builder = tantivy.SchemaBuilder()

        # File metadata fields
        schema_builder.add_text_field("path", stored=True)
        schema_builder.add_text_field("name", stored=True)
        schema_builder.add_text_field("type", stored=True)
        schema_builder.add_integer_field("size", stored=True, indexed=True)
        schema_builder.add_date_field("modified_time", stored=True, indexed=True)

        # Content fields
        schema_builder.add_text_field("content", stored=False)
        schema_builder.add_text_field("content_preview", stored=True)

        # IFS-specific fields
        schema_builder.add_text_field("entities", stored=True)
        schema_builder.add_text_field("dependencies", stored=True)
        schema_builder.add_text_field("functions", stored=True)
        schema_builder.add_text_field("imports", stored=True)

        # Metrics fields
        schema_builder.add_float_field("complexity_score", stored=True, indexed=True)
        schema_builder.add_integer_field("line_count", stored=True, indexed=True)
        schema_builder.add_text_field("hash", stored=True)

        return schema_builder.build()

    def _create_or_open_index(self, create_new: bool) -> tantivy.Index:
        """Create or open a Tantivy index."""
        if create_new or not (self.index_path / "meta.json").exists():
            return tantivy.Index(self._schema, path=str(self.index_path))
        else:
            return tantivy.Index.open(str(self.index_path))
    
    def _load_cache_metadata(self):
        """Load cache metadata from disk."""
        if not self.cache_metadata_path.exists():
            self._file_cache = {}
            return
            
        try:
            with open(self.cache_metadata_path, 'r') as f:
                data = json.load(f)
                self._file_cache = {
                    path: FileMetadata(**metadata) 
                    for path, metadata in data.items()
                }
            logger.info(f"Loaded cache metadata for {len(self._file_cache)} files")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load cache metadata: {e}. Starting fresh.")
            self._file_cache = {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        try:
            data = {
                path: {
                    'path': metadata.path,
                    'size': metadata.size,
                    'modified_time': metadata.modified_time,
                    'hash': metadata.hash,
                    'indexed_at': metadata.indexed_at
                }
                for path, metadata in self._file_cache.items()
            }
            with open(self.cache_metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved cache metadata for {len(self._file_cache)} files")
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _is_file_cached_and_current(self, file_path: Path) -> bool:
        """Check if file is cached and hasn't changed since last indexing.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file is cached and current
        """
        if not file_path.exists():
            return False
            
        file_key = str(file_path)
        if file_key not in self._file_cache:
            return False
            
        try:
            stat = file_path.stat()
            cached = self._file_cache[file_key]
            
            # Check if file has changed
            return (
                cached.size == stat.st_size and
                cached.modified_time == stat.st_mtime
            )
        except Exception:
            return False
    
    def _update_file_cache(self, file_path: Path, file_hash: str):
        """Update cache metadata for a file.
        
        Args:
            file_path: Path to the file
            file_hash: Content hash of the file
        """
        try:
            stat = file_path.stat()
            self._file_cache[str(file_path)] = FileMetadata(
                path=str(file_path),
                size=stat.st_size,
                modified_time=stat.st_mtime,
                hash=file_hash,
                indexed_at=datetime.now().timestamp()
            )
        except Exception as e:
            logger.error(f"Failed to update cache for {file_path}: {e}")
    
    def cleanup_cache(self) -> int:
        """Remove cache entries for files that no longer exist.
        
        Returns:
            Number of stale entries removed
        """
        stale_files = []
        
        for file_path in self._file_cache.keys():
            if not Path(file_path).exists():
                stale_files.append(file_path)
        
        for file_path in stale_files:
            del self._file_cache[file_path]
            
        if stale_files:
            self._save_cache_metadata()
            logger.info(f"Removed {len(stale_files)} stale cache entries")
            
        return len(stale_files)

    def _create_or_open_index(self, create_new: bool) -> tantivy.Index:
        """Create or open a Tantivy index."""
        if create_new or not (self.index_path / "meta.json").exists():
            # Clear cache when creating new index
            self._file_cache = {}
            return tantivy.Index(self._schema, path=str(self.index_path))

    def calculate_complexity_score(self, content: str, file_type: str) -> float:
        """Calculate complexity score for a file based on its content and type.

        Args:
            content: File content
            file_type: File extension/type

        Returns:
            Complexity score (0.0 - 1.0)
        """
        if not content:
            return 0.0

        # Use the parser to get complexity indicators
        parsed = self._parser.parse(content, file_type)

        lines = content.split("\n")
        line_count = len(lines)

        # Base complexity from line count (normalized)
        line_complexity = min(line_count / 1000.0, 0.3)

        # Type-specific complexity weights
        type_weights = {
            ".plsql": 1.0,  # PL/SQL is inherently complex
            ".plsvc": 1.0,  # PL/SQL service layer
            ".entity": 0.7,  # Entity definitions are moderately complex
            ".views": 0.6,  # Views are moderately complex
            ".projection": 0.8,  # Projections can be complex
            ".client": 0.6,  # Client code moderate complexity
            ".fragment": 0.9,  # Fragments can be very complex (mixed content)
            ".storage": 0.4,  # Storage configs are less complex
        }

        type_weight = type_weights.get(file_type, 0.5)

        # Complexity from parsed indicators (normalized)
        indicator_complexity = min(parsed.complexity_indicators / 100.0, 0.7)

        # Combine all factors
        total_complexity = (line_complexity + indicator_complexity) * type_weight

        return min(total_complexity, 1.0)

    def extract_entities(self, content: str, file_type: str) -> List[str]:
        """Extract IFS entities from file content using specialized parsers.

        Args:
            content: File content
            file_type: File extension/type

        Returns:
            List of extracted entities
        """
        if not content:
            return []

        parsed = self._parser.parse(content, file_type)
        return parsed.entities

    def extract_dependencies(self, content: str, file_type: str) -> List[str]:
        """Extract dependencies from file content.

        Args:
            content: File content
            file_type: File extension/type

        Returns:
            List of dependencies
        """
        if not content:
            return []

        parsed = self._parser.parse(content, file_type)
        return parsed.dependencies

    def extract_functions(self, content: str, file_type: str) -> List[str]:
        """Extract functions/procedures from file content.

        Args:
            content: File content
            file_type: File extension/type

        Returns:
            List of functions
        """
        if not content:
            return []

        parsed = self._parser.parse(content, file_type)
        return parsed.functions

    def extract_imports(self, content: str, file_type: str) -> List[str]:
        """Extract imports/includes from file content.

        Args:
            content: File content
            file_type: File extension/type

        Returns:
            List of imports
        """
        if not content:
            return []

        parsed = self._parser.parse(content, file_type)
        return parsed.imports

    async def index_file(self, file_path: Union[str, Path], force_reindex: bool = False) -> bool:
        """Index a single file with intelligent caching.

        Args:
            file_path: Path to the file to index
            force_reindex: Force re-indexing even if file hasn't changed

        Returns:
            True if file was indexed successfully
        """
        file_path = Path(file_path)

        if not file_path.exists() or file_path.suffix not in self.SUPPORTED_EXTENSIONS:
            return False

        # Check cache first (unless forced)
        if not force_reindex and self._is_file_cached_and_current(file_path):
            logger.debug(f"File unchanged, using cache: {file_path}")
            return True

        try:
            # Read file content
            async with aiofiles.open(
                file_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                content = await f.read()

            # Get file metadata
            stat = file_path.stat()
            file_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

            # Check if we need to remove old version first
            if str(file_path) in self._file_cache:
                # Delete old document by path
                self._writer.delete_documents("path", str(file_path))
                logger.debug(f"Removed old version of {file_path}")

            # Extract IFS-specific data using enhanced parsers
            file_type = file_path.suffix
            entities = self.extract_entities(content, file_type)
            dependencies = self.extract_dependencies(content, file_type)
            functions = self.extract_functions(content, file_type)
            imports = self.extract_imports(content, file_type)

            # Calculate metrics
            line_count = len(content.split("\n"))
            complexity_score = self.calculate_complexity_score(content, file_type)
            content_preview = content[:500] if content else ""

            # Create document
            doc = {
                "path": str(file_path),
                "name": file_path.name,
                "type": file_type,
                "size": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime),
                "content": content,
                "content_preview": content_preview,
                "entities": " ".join(entities),
                "dependencies": " ".join(dependencies),
                "functions": " ".join(functions),
                "imports": " ".join(imports),
                "complexity_score": complexity_score,
                "line_count": line_count,
                "hash": file_hash,
            }

            # Add document to index
            self._writer.add_document(tantivy.Document(**doc))
            
            # Update cache
            self._update_file_cache(file_path, file_hash)

            logger.debug(
                f"Indexed file: {file_path} (entities: {len(entities)}, "
                f"complexity: {complexity_score:.2f})"
            )

            return True

        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            return False

    async def index_directory(
        self, directory_path: Union[str, Path], recursive: bool = True, force_reindex: bool = False
    ) -> Dict[str, int]:
        """Index all supported files in a directory with intelligent caching.

        Args:
            directory_path: Path to directory to index
            recursive: Whether to index subdirectories
            force_reindex: Force re-indexing even if files haven't changed

        Returns:
            Dictionary with indexing statistics
        """
        directory_path = Path(directory_path)
        stats = {"indexed": 0, "skipped": 0, "errors": 0, "cached": 0}

        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return stats

        # Find all supported files
        pattern = "**/*" if recursive else "*"
        files = []

        for extension in self.SUPPORTED_EXTENSIONS:
            files.extend(directory_path.glob(f"{pattern}{extension}"))

        logger.info(f"Found {len(files)} files to index in {directory_path}")

        # Index files
        for file_path in files:
            try:
                # Check cache first
                if not force_reindex and self._is_file_cached_and_current(file_path):
                    stats["cached"] += 1
                    continue
                    
                success = await self.index_file(file_path, force_reindex)
                if success:
                    stats["indexed"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
                stats["errors"] += 1

        # Commit changes and save cache metadata
        self._writer.commit()
        self._save_cache_metadata()

        logger.info(
            f"Indexing complete: {stats['indexed']} indexed, "
            f"{stats['cached']} cached, {stats['skipped']} skipped, "
            f"{stats['errors']} errors"
        )

        return stats

        logger.info(f"Indexing complete: {stats}")
        return stats

    def search(
        self,
        query: str,
        limit: int = 10,
        file_type: Optional[str] = None,
        min_complexity: Optional[float] = None,
        max_complexity: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search the index with various filters.

        Args:
            query: Search query
            limit: Maximum number of results
            file_type: Filter by file type (optional)
            min_complexity: Minimum complexity score (optional)
            max_complexity: Maximum complexity score (optional)

        Returns:
            List of search results
        """
        searcher = self._index.searcher()

        # Build query
        query_parser = tantivy.QueryParser.for_index(
            self._index, ["content", "entities", "functions"]
        )

        try:
            parsed_query = query_parser.parse_query(query)

            # Apply filters
            filters = []
            if file_type:
                type_query = query_parser.parse_query(f"type:{file_type}")
                filters.append(type_query)

            if min_complexity is not None:
                filters.append(f"complexity_score:[{min_complexity} TO *]")

            if max_complexity is not None:
                filters.append(f"complexity_score:[* TO {max_complexity}]")

            # Execute search
            results = searcher.search(parsed_query, limit=limit)

            # Convert to SearchResult objects
            search_results = []
            for score, doc_address in results.hits:
                doc = searcher.doc(doc_address)

                search_result = SearchResult(
                    path=doc["path"][0],
                    name=doc["name"][0],
                    type=doc["type"][0],
                    content_preview=doc.get("content_preview", [""])[0],
                    score=score,
                    entities=doc.get("entities", [""])[0].split(),
                    line_count=doc.get("line_count", [0])[0],
                    complexity_score=doc.get("complexity_score", [0.0])[0],
                    modified_time=datetime.fromisoformat(
                        doc.get("modified_time", ["1970-01-01T00:00:00"])[0]
                    ),
                )
                search_results.append(search_result)

            return search_results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def find_similar_files(
        self, file_path: Union[str, Path], limit: int = 5
    ) -> List[SearchResult]:
        """Find files similar to the given file based on entities and content.

        Args:
            file_path: Path to the reference file
            limit: Maximum number of similar files to return

        Returns:
            List of similar files
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return []

        try:
            # Read the reference file
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract entities and create similarity query
            entities = self.extract_entities(content, file_path.suffix)

            if not entities:
                return []

            # Create query from entities
            entity_query = " OR ".join(entities[:10])  # Use top 10 entities

            return self.search(entity_query, limit=limit + 1)  # +1 to exclude self

        except Exception as e:
            logger.error(f"Error finding similar files for {file_path}: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics including cache information."""
        searcher = self._index.searcher()
        
        cache_size = 0
        if self.cache_metadata_path.exists():
            cache_size = self.cache_metadata_path.stat().st_size

        return {
            "total_documents": searcher.num_docs,
            "index_size": sum(
                f.stat().st_size for f in self.index_path.glob("*") if f.is_file()
            ),
            "cache_size": cache_size,
            "cached_files": len(self._file_cache),
            "index_path": str(self.index_path),
            "cache_metadata_path": str(self.cache_metadata_path),
            "supported_extensions": list(self.SUPPORTED_EXTENSIONS),
        }

    def close(self):
        """Close the indexer and release resources."""
        try:
            if hasattr(self, "_writer"):
                self._writer.commit()
            # Save cache metadata before closing
            self._save_cache_metadata()
            logger.info("Indexer closed successfully")
        except Exception as e:
            logger.error(f"Error closing indexer: {e}")
        except Exception as e:
            logger.error(f"Error closing indexer: {e}")
