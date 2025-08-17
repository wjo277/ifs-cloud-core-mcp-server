"""
Production-Safe Data Structures for IFS Semantic Search
=======================================================

This module defines production-safe data structures that do NOT store copyrighted
source code content, only references and metadata for compliance with copyright laws.

Key Principle: Store REFERENCES, not CONTENT
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import numpy as np


@dataclass
class ProductionCodeChunk:
    """
    Production-safe code chunk that stores only metadata and references.

    IMPORTANT: This class does NOT store actual source code content to comply
    with copyright restrictions in production deployments.

    Instead, it stores:
    - File location references
    - Code structure metadata
    - AI-generated summaries (derived work, not copyright)
    - Business context information
    """

    # Unique identification
    chunk_id: str  # Hash-based unique identifier

    # File reference (NOT content)
    relative_file_path: str  # Relative path from user's workspace root
    file_hash: str  # Hash for integrity checking
    start_line: int  # Starting line number (1-indexed)
    end_line: int  # Ending line number (inclusive)

    # Structure metadata (extracted, not copyrighted)
    chunk_type: str  # 'function', 'procedure', 'package', etc.
    function_name: Optional[str] = None  # Extracted identifier name
    language: str = "plsql"  # Programming language
    module_name: Optional[str] = None  # IFS module name

    # AI-generated content (derived work, not copyright)
    ai_summary: Optional[str] = None  # AI-generated business summary
    ai_purpose: Optional[str] = None  # AI-extracted purpose
    ai_keywords: Optional[List[str]] = None  # AI-extracted keywords

    # Business context (metadata, not copyrighted)
    business_domain: Optional[str] = None  # e.g., "finance", "inventory"
    api_calls: Optional[List[str]] = None  # Called APIs (structural info)
    complexity_score: Optional[float] = None  # Calculated complexity

    # Processing metadata
    created_at: datetime = field(default_factory=datetime.now)
    embedding_model: Optional[str] = None  # Model used for embedding
    parsing_method: str = "ast"  # How it was extracted

    # Privacy and compliance
    contains_sensitive_data: bool = False  # Flag for sensitive content
    copyright_owner: Optional[str] = None  # Copyright holder info

    def get_content_reference(self) -> Dict[str, Any]:
        """
        Get a reference to the content location for runtime loading.
        This allows the user to provide their own copy of the code.
        """
        return {
            "relative_path": self.relative_file_path,
            "file_hash": self.file_hash,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "function_name": self.function_name,
        }

    def get_searchable_text(self) -> str:
        """
        Get text content that can be safely used for search embeddings.
        This includes only AI-generated summaries and metadata, not source code.
        """
        searchable_parts = []

        # AI-generated content (safe to use)
        if self.ai_summary:
            searchable_parts.append(f"SUMMARY: {self.ai_summary}")

        if self.ai_purpose:
            searchable_parts.append(f"PURPOSE: {self.ai_purpose}")

        if self.ai_keywords:
            searchable_parts.append(f"KEYWORDS: {', '.join(self.ai_keywords)}")

        # Structural metadata (not copyrighted)
        if self.function_name:
            searchable_parts.append(f"FUNCTION: {self.function_name}")

        if self.module_name:
            searchable_parts.append(f"MODULE: {self.module_name}")

        if self.business_domain:
            searchable_parts.append(f"DOMAIN: {self.business_domain}")

        if self.api_calls:
            searchable_parts.append(
                f"APIs: {', '.join(self.api_calls[:5])}"
            )  # Limit for size

        return "\n".join(searchable_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (no source code)"""
        return {
            "chunk_id": self.chunk_id,
            "relative_file_path": self.relative_file_path,
            "file_hash": self.file_hash,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "function_name": self.function_name,
            "language": self.language,
            "module_name": self.module_name,
            "ai_summary": self.ai_summary,
            "ai_purpose": self.ai_purpose,
            "ai_keywords": self.ai_keywords,
            "business_domain": self.business_domain,
            "api_calls": self.api_calls,
            "complexity_score": self.complexity_score,
            "created_at": self.created_at.isoformat(),
            "embedding_model": self.embedding_model,
            "parsing_method": self.parsing_method,
            "contains_sensitive_data": self.contains_sensitive_data,
            "copyright_owner": self.copyright_owner,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductionCodeChunk":
        """Create from dictionary"""
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class CodeContentLoader:
    """
    Handles loading actual code content at runtime using user-provided files.
    This separates the metadata storage from content access.
    """

    workspace_root: Path

    def load_chunk_content(self, chunk: ProductionCodeChunk) -> Optional[str]:
        """
        Load the actual code content for a chunk from user's local files.

        Args:
            chunk: ProductionCodeChunk with file reference

        Returns:
            The actual source code content, or None if not available

        This method allows users to provide their own copy of the code
        while the production system only stores metadata references.
        """
        try:
            # Construct full path from relative path
            full_path = self.workspace_root / chunk.relative_file_path

            if not full_path.exists():
                return None

            # Read file content
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Verify file hasn't changed using hash
            current_content = "".join(lines)
            current_hash = hashlib.md5(current_content.encode()).hexdigest()

            if current_hash != chunk.file_hash:
                # File has changed since indexing
                print(
                    f"⚠️ Warning: File {chunk.relative_file_path} has changed since indexing"
                )

            # Extract the specific lines for this chunk
            if chunk.end_line <= len(lines):
                chunk_lines = lines[chunk.start_line - 1 : chunk.end_line]
                return "".join(chunk_lines)
            else:
                return None

        except Exception as e:
            print(f"❌ Error loading content for {chunk.relative_file_path}: {e}")
            return None

    def verify_chunk_integrity(self, chunk: ProductionCodeChunk) -> bool:
        """
        Verify that the chunk reference still points to valid content.
        """
        try:
            full_path = self.workspace_root / chunk.relative_file_path

            if not full_path.exists():
                return False

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            current_hash = hashlib.md5(content.encode()).hexdigest()
            return current_hash == chunk.file_hash

        except Exception:
            return False


@dataclass
class ProductionSearchResult:
    """
    Search result that provides content loading capabilities.
    """

    chunk: ProductionCodeChunk
    similarity_score: float
    rank: int

    def load_content(self, content_loader: CodeContentLoader) -> Optional[str]:
        """Load the actual source code content for this result"""
        return content_loader.load_chunk_content(self.chunk)

    def get_display_info(self) -> Dict[str, Any]:
        """Get display information without exposing source code"""
        return {
            "function_name": self.chunk.function_name,
            "module_name": self.chunk.module_name,
            "file_path": self.chunk.relative_file_path,
            "line_range": f"{self.chunk.start_line}-{self.chunk.end_line}",
            "similarity_score": self.similarity_score,
            "rank": self.rank,
            "ai_summary": self.chunk.ai_summary,
            "business_domain": self.chunk.business_domain,
            "chunk_type": self.chunk.chunk_type,
        }


# Migration utilities for converting existing chunks
def convert_to_production_chunk(
    original_chunk,  # Original CodeChunk with content
    workspace_root: Path,
    copyright_owner: str = "IFS",
) -> ProductionCodeChunk:
    """
    Convert an existing CodeChunk (with content) to ProductionCodeChunk (reference only).

    This is used during the migration process to create production-safe indices.
    """

    # Calculate relative path
    abs_path = Path(original_chunk.file_path)
    try:
        relative_path = abs_path.relative_to(workspace_root)
    except ValueError:
        # If can't make relative, use the absolute path as fallback
        relative_path = abs_path

    # Calculate file hash
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            file_content = f.read()
        file_hash = hashlib.md5(file_content.encode()).hexdigest()
    except Exception:
        # Fallback hash based on content
        file_hash = hashlib.md5(original_chunk.raw_content.encode()).hexdigest()

    # Extract API calls from content (structural analysis, not copyright)
    api_calls = []
    if hasattr(original_chunk, "raw_content"):
        # Extract API pattern calls (this is structural analysis, not copyrighted content)
        api_pattern = r"(\w+_API\.\w+)"
        api_calls = list(set(re.findall(api_pattern, original_chunk.raw_content)))

    return ProductionCodeChunk(
        chunk_id=getattr(
            original_chunk,
            "chunk_id",
            hashlib.md5(
                f"{relative_path}:{original_chunk.start_line}".encode()
            ).hexdigest(),
        ),
        relative_file_path=str(relative_path),
        file_hash=file_hash,
        start_line=original_chunk.start_line,
        end_line=original_chunk.end_line,
        chunk_type=original_chunk.chunk_type,
        function_name=original_chunk.function_name,
        language=getattr(original_chunk, "language", "plsql"),
        module_name=getattr(original_chunk, "module_name", None),
        ai_summary=getattr(original_chunk, "ai_summary", None),
        ai_purpose=getattr(original_chunk, "ai_purpose", None),
        ai_keywords=getattr(original_chunk, "ai_keywords", None),
        api_calls=api_calls,
        copyright_owner=copyright_owner,
    )
