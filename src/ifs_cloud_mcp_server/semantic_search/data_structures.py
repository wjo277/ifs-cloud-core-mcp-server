"""
Core Data Structures for IFS Semantic Search
============================================

This module defines the core data structures used throughout the semantic search system.
Every design decision is explained for AI agent understanding.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import numpy as np


@dataclass
class CodeChunk:
    """
    Represents a semantically meaningful piece of code for embedding and search.

    WHY THIS STRUCTURE?
    -------------------
    1. GRANULAR SEARCH: Instead of returning entire files, we return specific
       functions/procedures that are relevant to the query

    2. RICH METADATA: Each chunk carries business context, dependencies, and
       architectural information that helps AI agents understand how to use it

    3. EMBEDDING READY: The structure is optimized for converting to embeddings
       while preserving all necessary context for reconstruction

    4. IFS SPECIFIC: Includes IFS-specific fields like module, API calls, and
       business terminology that are crucial for enterprise search
    """

    # Core identification
    chunk_id: str  # Unique hash-based identifier
    file_path: str  # Source file absolute path
    start_line: int  # Starting line number (1-indexed)
    end_line: int  # Ending line number (inclusive)

    # Content fields
    raw_content: str  # Original code exactly as written
    processed_content: str  # Cleaned/normalized for embedding
    chunk_type: str  # Type: 'function', 'procedure', 'class', 'module', etc.

    # Semantic metadata for enhanced search
    function_name: Optional[str] = None  # Extracted function/procedure name
    class_name: Optional[str] = None  # Containing class/package name
    docstring: Optional[str] = None  # Documentation string

    # Dependency information
    imports: List[str] = field(default_factory=list)  # Import statements
    function_calls: List[str] = field(default_factory=list)  # Called functions
    sql_queries: List[str] = field(default_factory=list)  # Embedded SQL

    # IFS-specific business context
    module: Optional[str] = None  # IFS module (ORDER, INVOICE, INVENTORY, etc.)
    business_terms: List[str] = field(default_factory=list)  # Domain vocabulary
    api_calls: List[str] = field(default_factory=list)  # IFS API usage
    database_tables: List[str] = field(default_factory=list)  # Referenced tables

    # Architectural context
    layer: Optional[str] = None  # presentation/business/data/integration

    # Code quality indicators for ranking
    has_error_handling: bool = False  # Contains try-catch or error handling
    has_transactions: bool = False  # Contains database transactions
    complexity_score: float = 0.0  # Cyclomatic complexity (0-1 scale)

    # File metadata
    last_modified: Optional[datetime] = None
    file_size: int = 0
    language: Optional[str] = None  # Programming language

    # Embedding storage (populated after training)
    embedding: Optional[np.ndarray] = None  # Vector representation

    def to_embedding_text(self) -> str:
        """
        Convert chunk to optimized text for embedding generation.

        EMBEDDING TEXT STRATEGY:
        ------------------------
        We create a rich text representation that combines multiple signals:

        1. NATURAL LANGUAGE: Function names, docstrings, business terms
        2. CODE STRUCTURE: Simplified code patterns and API calls
        3. DOMAIN CONTEXT: Module, layer, and business domain information
        4. TECHNICAL CONTEXT: Database tables, dependencies, error handling

        This multi-signal approach maximizes the chances of semantic matching
        regardless of how the user phrases their query.

        Why this format works:
        - Pre-trained models expect natural language-like input
        - We structure technical info as pseudo-sentences
        - Business context bridges technical and domain knowledge
        - Code snippets provide structural pattern matching
        """
        parts = []

        # Start with identity and purpose
        if self.function_name:
            parts.append(f"Function named {self.function_name}")
        if self.class_name:
            parts.append(f"in {self.class_name} class")
        if self.docstring:
            # Limit docstring to prevent overwhelming the embedding
            doc_preview = self.docstring[:200].replace("\n", " ")
            parts.append(f"Description: {doc_preview}")

        # Add business context (crucial for enterprise search)
        if self.module:
            parts.append(f"Module: {self.module}")
        if self.business_terms:
            # Include most important business terms
            terms = ", ".join(self.business_terms[:8])
            parts.append(f"Business concepts: {terms}")

        # Add technical context
        if self.api_calls:
            apis = ", ".join(self.api_calls[:5])
            parts.append(f"Uses APIs: {apis}")

        if self.database_tables:
            tables = ", ".join(self.database_tables[:5])
            parts.append(f"Database tables: {tables}")

        # Add architectural context
        if self.layer:
            parts.append(f"Architecture layer: {self.layer}")

        # Add quality indicators
        quality_indicators = []
        if self.has_error_handling:
            quality_indicators.append("error handling")
        if self.has_transactions:
            quality_indicators.append("database transactions")
        if quality_indicators:
            parts.append(f"Includes: {', '.join(quality_indicators)}")

        # Add simplified code snippet for pattern matching
        code_snippet = self._create_code_snippet()
        if code_snippet:
            parts.append(f"Code pattern: {code_snippet}")

        # Join all parts with separators that help the model understand structure
        return " | ".join(parts)

    def _create_code_snippet(self) -> str:
        """
        Create a simplified, embedding-friendly code snippet.

        SIMPLIFICATION STRATEGY:
        ------------------------
        1. Remove comments (handled separately in docstring)
        2. Normalize whitespace and indentation
        3. Keep function signatures and key statements
        4. Remove string literals that don't add semantic value
        5. Preserve keywords that indicate functionality

        The goal is to capture the semantic essence of the code
        while removing noise that doesn't help with matching.
        """
        if not self.processed_content:
            return ""

        simplified = self.processed_content

        # Remove comments
        simplified = re.sub(r"--.*$", "", simplified, flags=re.MULTILINE)  # SQL
        simplified = re.sub(r"//.*$", "", simplified, flags=re.MULTILINE)  # JS/TS
        simplified = re.sub(r"/\*.*?\*/", "", simplified, flags=re.DOTALL)  # Block

        # Normalize whitespace
        simplified = " ".join(simplified.split())

        # Keep only the most important parts (function signature + key statements)
        # For long code, take beginning (signature) and important keywords
        if len(simplified) > 300:
            # Keep the beginning (usually contains signature/declaration)
            start = simplified[:150]

            # Find important keywords in the rest
            important_patterns = [
                r"\b(?:SELECT|INSERT|UPDATE|DELETE)\s+[^;]+",  # SQL operations
                r"\b(?:IF|WHILE|FOR|TRY|CATCH)\s+[^{]+",  # Control flow
                r"\b(?:RETURN|THROW|RAISE)\s+[^;]+",  # Control statements
                r"\w+_API\.\w+",  # IFS API calls
            ]

            important_parts = []
            for pattern in important_patterns:
                matches = re.findall(pattern, simplified, re.IGNORECASE)
                important_parts.extend(matches[:2])  # Max 2 matches per pattern

            if important_parts:
                simplified = f"{start} ... {' '.join(important_parts[:3])}"
            else:
                simplified = start

        return simplified

    @classmethod
    def create_from_file_section(
        cls,
        file_path: Path,
        content: str,
        start_line: int,
        end_line: int,
        chunk_type: str,
        function_name: str = None,
    ) -> "CodeChunk":
        """
        Factory method to create CodeChunk from file section.

        This method handles all the metadata extraction and processing
        needed to create a properly initialized CodeChunk.
        """
        # Generate unique ID
        chunk_id = cls._generate_chunk_id(file_path, function_name or str(start_line))

        # Extract metadata using various parsers
        metadata = IFSMetadataExtractor.extract_metadata(content, file_path)

        # Override extracted function_name with provided one if available
        if function_name:
            metadata["function_name"] = function_name

        return cls(
            chunk_id=chunk_id,
            file_path=str(file_path),
            start_line=start_line,
            end_line=end_line,
            raw_content=content,
            processed_content=content,  # Will be cleaned by preprocessor
            chunk_type=chunk_type,
            **metadata,  # Unpack extracted metadata
        )

    @staticmethod
    def _generate_chunk_id(file_path: Path, identifier: str) -> str:
        """
        Generate unique, deterministic chunk ID.

        WHY MD5 HASH?
        -------------
        1. Deterministic: Same input always produces same ID
        2. Unique: Virtually no collisions for our use case
        3. Fixed length: Consistent storage and indexing
        4. Fast: MD5 is computationally efficient

        We use file path + identifier to ensure uniqueness across the codebase.
        """
        content = f"{file_path}:{identifier}"
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class SearchResult:
    """
    Represents a search result with rich context for AI agents.

    WHY RICH RESULTS?
    -----------------
    AI agents need more than just code - they need context to understand:
    1. Why this result is relevant (explanation)
    2. How to use it (dependencies, APIs)
    3. What business problem it solves (domain context)
    4. How it fits in the system (architecture)
    """

    chunk: CodeChunk
    similarity_score: float  # 0-1, higher = more similar
    rank: int  # Position in search results (1-based)

    # Explanation for AI agents
    relevance_explanation: str  # Why this result matches the query

    # Context for implementation
    implementation_context: Dict[str, Any] = field(default_factory=dict)

    # Related chunks (for broader context)
    related_chunks: List["CodeChunk"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "file_path": self.chunk.file_path,
            "function_name": self.chunk.function_name,
            "chunk_type": self.chunk.chunk_type,
            "module": self.chunk.module,
            "layer": self.chunk.layer,
            "similarity_score": self.similarity_score,
            "rank": self.rank,
            "relevance_explanation": self.relevance_explanation,
            "start_line": self.chunk.start_line,
            "end_line": self.chunk.end_line,
            "business_terms": self.chunk.business_terms,
            "api_calls": self.chunk.api_calls,
            "database_tables": self.chunk.database_tables,
            "has_error_handling": self.chunk.has_error_handling,
            "has_transactions": self.chunk.has_transactions,
            "complexity_score": self.chunk.complexity_score,
            "language": self.chunk.language,
            "implementation_context": self.implementation_context,
            "code_preview": self.chunk.raw_content[:500]
            + ("..." if len(self.chunk.raw_content) > 500 else ""),
        }


class IFSMetadataExtractor:
    """
    Extracts IFS-specific metadata from code chunks.

    WHY SEPARATE EXTRACTOR?
    ----------------------
    1. MODULARITY: Keeps extraction logic separate from data structures
    2. TESTABILITY: Easy to test extraction logic independently
    3. EXTENSIBILITY: Easy to add new extraction patterns
    4. MAINTAINABILITY: Single place to update IFS patterns
    """

    # Precompiled regex patterns for performance
    PATTERNS = {
        "api_call": re.compile(r"(\w+_(?:API|RPI|SYS|CFP))\.(\w+)"),
        "module": re.compile(r"^([A-Z]+)_"),
        "business_term": re.compile(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+"),
        "sql_table": re.compile(r"\b(?:FROM|JOIN|INTO|UPDATE)\s+([A-Z_]+)"),
        "error_handling": re.compile(
            r"\b(?:EXCEPTION|Error_SYS|RAISE_APPLICATION_ERROR|try|catch)\b"
        ),
        "transaction": re.compile(
            r"\b(?:COMMIT|ROLLBACK|SAVEPOINT|@ApproveTransactionStatement)\b"
        ),
        "plsql_function": re.compile(
            r"(?:FUNCTION|PROCEDURE)\s+([A-Z_]+)", re.IGNORECASE
        ),
        "js_function": re.compile(r"(?:function|const|let)\s+([a-zA-Z_][a-zA-Z0-9_]*)"),
    }

    @classmethod
    def extract_metadata(cls, content: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract all metadata from a code chunk.

        EXTRACTION STRATEGY:
        -------------------
        We extract multiple types of metadata:
        1. IFS API calls and modules
        2. Business terminology
        3. Database operations
        4. Code quality indicators
        5. Architectural layer information

        This metadata is crucial for:
        - Improving search relevance
        - Providing context to AI agents
        - Enabling filtered searches
        - Supporting impact analysis
        """
        metadata = {}

        # Extract module from file path or content
        metadata["module"] = cls._extract_module(file_path)

        # Extract API calls
        api_matches = cls.PATTERNS["api_call"].findall(content)
        metadata["api_calls"] = [f"{pkg}.{method}" for pkg, method in api_matches[:10]]

        # Extract database tables
        table_matches = cls.PATTERNS["sql_table"].findall(content)
        metadata["database_tables"] = list(set(table_matches))[:10]

        # Extract business terms (filter out common programming terms)
        business_matches = cls.PATTERNS["business_term"].findall(content)
        programming_terms = {
            "String",
            "Integer",
            "Boolean",
            "Array",
            "Object",
            "Function",
        }
        metadata["business_terms"] = [
            term for term in set(business_matches) if term not in programming_terms
        ][:10]

        # Extract quality indicators
        metadata["has_error_handling"] = bool(
            cls.PATTERNS["error_handling"].search(content)
        )
        metadata["has_transactions"] = bool(cls.PATTERNS["transaction"].search(content))

        # Calculate complexity score (simplified cyclomatic complexity)
        metadata["complexity_score"] = cls._calculate_complexity(content)

        # Determine architectural layer
        metadata["layer"] = cls._determine_layer(file_path)

        # Extract function/class names based on language
        metadata.update(cls._extract_names(content, file_path.suffix))

        return metadata

    @classmethod
    def _extract_module(cls, file_path: Path) -> Optional[str]:
        """
        Extract IFS module from file path or name.

        MODULE DETECTION STRATEGY:
        -------------------------
        1. Look for uppercase directories (ORDER, INVOICE, etc.)
        2. Check file name prefixes (Order_, Invoice_, etc.)
        3. Fall back to parent directory names

        Modules are important because they indicate business domains.
        """
        # Check path components for uppercase module names
        for part in file_path.parts:
            if part.isupper() and len(part) > 2 and "_" not in part:
                return part

        # Check filename prefix
        module_match = cls.PATTERNS["module"].match(file_path.stem)
        if module_match:
            return module_match.group(1)

        return None

    @classmethod
    def _determine_layer(cls, file_path: Path) -> str:
        """
        Determine architectural layer from file path.

        LAYER CLASSIFICATION:
        --------------------
        - presentation: Client-side UI code
        - business: Server-side business logic
        - data: Database and entity definitions
        - integration: APIs and external interfaces

        This classification helps with:
        - Targeted searches by layer
        - Impact analysis across layers
        - Architectural understanding
        """
        path_str = str(file_path).lower()

        # Presentation layer indicators
        if any(
            indicator in path_str for indicator in ["client", "aurena", "web", "ui"]
        ):
            return "presentation"

        # Data layer indicators
        if any(indicator in path_str for indicator in ["entity", "model", "database"]):
            return "data"

        # Integration layer indicators
        if any(indicator in path_str for indicator in ["projection", "api", "service"]):
            return "integration"

        # Business layer (default for server-side code)
        if file_path.suffix in [".plsql", ".sql"]:
            return "business"

        return "business"  # Default assumption

    @classmethod
    def _calculate_complexity(cls, content: str) -> float:
        """
        Calculate simplified complexity score (0-1).

        COMPLEXITY FACTORS:
        ------------------
        - Conditional statements (IF, CASE)
        - Loops (FOR, WHILE)
        - Exception handlers
        - Nesting depth

        Why measure complexity?
        - Helps prioritize simpler code for examples
        - Indicates code that may need careful modification
        - Useful for ranking similar results
        """
        score = 0.0

        # Count control flow statements
        if_count = len(re.findall(r"\bIF\b", content, re.IGNORECASE))
        loop_count = len(re.findall(r"\b(?:FOR|WHILE|LOOP)\b", content, re.IGNORECASE))
        case_count = len(re.findall(r"\bCASE\b", content, re.IGNORECASE))
        exception_count = len(
            re.findall(r"\b(?:EXCEPTION|TRY)\b", content, re.IGNORECASE)
        )

        # Weight different constructs
        score += if_count * 0.1
        score += loop_count * 0.2
        score += case_count * 0.15
        score += exception_count * 0.1

        # Add penalty for deep nesting (rough approximation)
        lines = content.split("\n")
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)

        if max_indent > 40:  # Very deep nesting
            score += 0.3
        elif max_indent > 20:  # Moderate nesting
            score += 0.15

        # Normalize to 0-1 range
        return min(1.0, score)

    @classmethod
    def _extract_names(
        cls, content: str, file_extension: str
    ) -> Dict[str, Optional[str]]:
        """Extract function and class names based on file type."""
        result = {"function_name": None, "class_name": None}

        if file_extension in [".plsql", ".sql"]:
            # Extract PL/SQL function/procedure names
            match = cls.PATTERNS["plsql_function"].search(content)
            if match:
                result["function_name"] = match.group(1)

        elif file_extension in [".js", ".ts", ".tsx"]:
            # Extract JavaScript/TypeScript function names
            match = cls.PATTERNS["js_function"].search(content)
            if match:
                result["function_name"] = match.group(1)

        return result
