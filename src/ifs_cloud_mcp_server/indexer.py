"""Tantivy-based indexer for IFS Cloud files."""

import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass
from datetime import datetime

import tantivy
import aiofiles
from pydantic import BaseModel


logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Metadata for an indexed file."""
    path: str
    name: str
    type: str
    size: int
    modified_time: datetime
    complexity_score: float
    line_count: int
    hash: str


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
        '.entity', '.plsql', '.views', '.storage', 
        '.fragment', '.client', '.projection'
    }
    
    def __init__(self, index_path: Union[str, Path], create_new: bool = False):
        """Initialize the Tantivy indexer.
        
        Args:
            index_path: Path to store the Tantivy index
            create_new: Whether to create a new index (overwriting existing)
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self._schema = self._create_schema()
        self._index = self._create_or_open_index(create_new)
        self._writer = self._index.writer(heap_size=50_000_000)  # 50MB heap
        
        logger.info(f"Initialized Tantivy indexer at {self.index_path}")
    
    def _create_schema(self) -> tantivy.Schema:
        """Create the Tantivy schema for IFS Cloud files."""
        schema_builder = tantivy.SchemaBuilder()
        
        # File metadata fields
        schema_builder.add_text_field("path", stored=True, tokenizer_name="keyword")
        schema_builder.add_text_field("name", stored=True, tokenizer_name="default")
        schema_builder.add_text_field("type", stored=True, tokenizer_name="keyword")
        schema_builder.add_integer_field("size", stored=True, indexed=True)
        schema_builder.add_date_field("modified_time", stored=True, indexed=True)
        
        # Content fields
        schema_builder.add_text_field("content", stored=False, tokenizer_name="default")
        schema_builder.add_text_field("content_preview", stored=True, tokenizer_name="default")
        
        # IFS-specific fields
        schema_builder.add_text_field("entities", stored=True, tokenizer_name="default")
        schema_builder.add_text_field("dependencies", stored=True, tokenizer_name="default")
        schema_builder.add_text_field("functions", stored=True, tokenizer_name="default")
        schema_builder.add_text_field("imports", stored=True, tokenizer_name="default")
        
        # Metrics fields
        schema_builder.add_f64_field("complexity_score", stored=True, indexed=True)
        schema_builder.add_integer_field("line_count", stored=True, indexed=True)
        schema_builder.add_text_field("hash", stored=True, tokenizer_name="keyword")
        
        return schema_builder.build()
    
    def _create_or_open_index(self, create_new: bool) -> tantivy.Index:
        """Create or open a Tantivy index."""
        if create_new or not (self.index_path / "meta.json").exists():
            return tantivy.Index(self._schema, path=str(self.index_path))
        else:
            return tantivy.Index.open(str(self.index_path))
    
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
        
        lines = content.split('\n')
        line_count = len(lines)
        
        # Base complexity from line count
        complexity = min(line_count / 1000.0, 0.3)  # Max 0.3 from line count
        
        # Type-specific complexity factors
        type_weights = {
            '.plsql': 0.8,      # PL/SQL is inherently complex
            '.entity': 0.6,     # Entity definitions are moderately complex
            '.views': 0.5,      # Views are moderately complex
            '.storage': 0.4,    # Storage configs are less complex
            '.fragment': 0.7,   # Fragments can be complex
            '.client': 0.6,     # Client code moderate complexity
            '.projection': 0.5  # Projections are moderate
        }
        
        type_weight = type_weights.get(file_type, 0.5)
        complexity *= type_weight
        
        # Content-based complexity indicators
        complexity_indicators = [
            'PROCEDURE', 'FUNCTION', 'PACKAGE', 'TRIGGER',
            'IF', 'WHILE', 'FOR', 'LOOP', 'CASE', 'WHEN',
            'EXCEPTION', 'CURSOR', 'SELECT', 'INSERT', 'UPDATE', 'DELETE',
            'JOIN', 'UNION', 'EXISTS', 'NOT EXISTS'
        ]
        
        indicator_count = sum(1 for indicator in complexity_indicators 
                            if indicator in content.upper())
        complexity += min(indicator_count / 50.0, 0.7)  # Max 0.7 from indicators
        
        return min(complexity, 1.0)
    
    def extract_entities(self, content: str, file_type: str) -> List[str]:
        """Extract IFS entities from file content.
        
        Args:
            content: File content
            file_type: File extension/type
            
        Returns:
            List of extracted entities
        """
        entities = set()
        
        if not content:
            return list(entities)
        
        lines = content.split('\n')
        
        # Entity-specific extraction based on file type
        if file_type == '.entity':
            # Extract entity names from entity files
            for line in lines:
                line = line.strip()
                if line.startswith('entity '):
                    entity_name = line.split()[1].split('(')[0]
                    entities.add(entity_name)
        
        elif file_type == '.plsql':
            # Extract procedure/function names and referenced entities
            for line in lines:
                line = line.strip().upper()
                if any(keyword in line for keyword in ['PROCEDURE ', 'FUNCTION ', 'PACKAGE ']):
                    # Extract name after keyword
                    for keyword in ['PROCEDURE ', 'FUNCTION ', 'PACKAGE ']:
                        if keyword in line:
                            parts = line.split(keyword, 1)
                            if len(parts) > 1:
                                name = parts[1].split()[0].split('(')[0]
                                entities.add(name)
        
        elif file_type == '.views':
            # Extract view names and referenced tables
            for line in lines:
                line = line.strip()
                if line.upper().startswith('VIEW '):
                    view_name = line.split()[1]
                    entities.add(view_name)
        
        # Common entity patterns across all file types
        # Look for typical IFS entity patterns (CamelCase words)
        import re
        camel_case_pattern = r'\b[A-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b'
        matches = re.findall(camel_case_pattern, content)
        entities.update(matches[:20])  # Limit to first 20 matches
        
        return list(entities)
    
    def extract_dependencies(self, content: str, file_type: str) -> List[str]:
        """Extract dependencies from file content."""
        dependencies = set()
        
        if not content:
            return list(dependencies)
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip().upper()
            
            # Common dependency patterns
            if any(keyword in line for keyword in ['IMPORT ', 'INCLUDE ', 'USES ', 'FROM ']):
                # Extract dependency names
                words = line.split()
                for i, word in enumerate(words):
                    if word in ['IMPORT', 'INCLUDE', 'USES', 'FROM'] and i + 1 < len(words):
                        dep = words[i + 1].strip('();,')
                        if dep:
                            dependencies.add(dep)
        
        return list(dependencies)
    
    def extract_functions(self, content: str, file_type: str) -> List[str]:
        """Extract function/procedure names from file content."""
        functions = set()
        
        if not content:
            return list(functions)
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            upper_line = line.upper()
            
            # Extract function/procedure names
            for keyword in ['FUNCTION ', 'PROCEDURE ', 'METHOD ']:
                if keyword in upper_line:
                    parts = upper_line.split(keyword, 1)
                    if len(parts) > 1:
                        name = parts[1].split()[0].split('(')[0]
                        if name:
                            functions.add(name)
        
        return list(functions)
    
    async def index_file(self, file_path: Union[str, Path]) -> bool:
        """Index a single file.
        
        Args:
            file_path: Path to the file to index
            
        Returns:
            True if file was indexed successfully
        """
        file_path = Path(file_path)
        
        if not file_path.exists() or file_path.suffix not in self.SUPPORTED_EXTENSIONS:
            return False
        
        try:
            # Read file content
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
            
            # Get file metadata
            stat = file_path.stat()
            file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            # Extract IFS-specific data
            file_type = file_path.suffix
            entities = self.extract_entities(content, file_type)
            dependencies = self.extract_dependencies(content, file_type)
            functions = self.extract_functions(content, file_type)
            
            # Calculate metrics
            line_count = len(content.split('\n'))
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
                "imports": "",  # TODO: Extract imports based on file type
                "complexity_score": complexity_score,
                "line_count": line_count,
                "hash": file_hash,
            }
            
            # Add document to index
            self._writer.add_document(tantivy.Document(**doc))
            
            logger.debug(f"Indexed file: {file_path} (entities: {len(entities)}, "
                        f"complexity: {complexity_score:.2f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            return False
    
    async def index_directory(self, directory_path: Union[str, Path], 
                            recursive: bool = True) -> Dict[str, int]:
        """Index all supported files in a directory.
        
        Args:
            directory_path: Path to directory to index
            recursive: Whether to index subdirectories
            
        Returns:
            Dictionary with indexing statistics
        """
        directory_path = Path(directory_path)
        stats = {"indexed": 0, "skipped": 0, "errors": 0}
        
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
                success = await self.index_file(file_path)
                if success:
                    stats["indexed"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
                stats["errors"] += 1
        
        # Commit changes
        self._writer.commit()
        
        logger.info(f"Indexing complete: {stats}")
        return stats
    
    def search(self, query: str, limit: int = 10, 
              file_type: Optional[str] = None,
              min_complexity: Optional[float] = None,
              max_complexity: Optional[float] = None) -> List[SearchResult]:
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
        query_parser = tantivy.QueryParser.for_index(self._index, ["content", "entities", "functions"])
        
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
                    modified_time=datetime.fromisoformat(doc.get("modified_time", ["1970-01-01T00:00:00"])[0])
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def find_similar_files(self, file_path: Union[str, Path], limit: int = 5) -> List[SearchResult]:
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
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
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
        """Get index statistics."""
        searcher = self._index.searcher()
        
        return {
            "total_documents": searcher.num_docs(),
            "index_size": sum(f.stat().st_size for f in self.index_path.glob("*") if f.is_file()),
            "index_path": str(self.index_path),
            "supported_extensions": list(self.SUPPORTED_EXTENSIONS),
        }
    
    def close(self):
        """Close the indexer and release resources."""
        try:
            if hasattr(self, '_writer'):
                self._writer.commit()
        except Exception as e:
            logger.error(f"Error closing indexer: {e}")