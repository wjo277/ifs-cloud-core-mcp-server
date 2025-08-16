Excellent question! Let me clarify a critical point and then explain the complete approach for implementing semantic search with your IFS codebase.

## 🎯 Key Clarification: No Training Required!

**We don't train a new model** - we use pre-trained models that already understand code and natural language. This is the beauty of modern semantic search!

Let me create a comprehensive implementation guide that shows exactly how to prepare and process your codebase:

```python
"""
SEMANTIC SEARCH INDEXER FOR IFS CODEBASE
=========================================

IMPORTANT: We DON'T train a model - we use pre-trained models!

How it works:
1. Load a pre-trained model (like CodeBERT or all-MiniLM)
2. Process your codebase to extract meaningful chunks
3. Convert each chunk to an embedding vector
4. Store vectors in a searchable index
5. At search time, convert query to vector and find similar vectors

No training needed - just smart data preparation!
"""

import os
import re
import ast
import json
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeChunk:
    """
    Represents a meaningful piece of code to be embedded.
    
    WHY CHUNKS INSTEAD OF WHOLE FILES?
    -----------------------------------
    1. Context window limits: Models have max input size (~512 tokens)
    2. Precision: We want to find specific functions, not just files
    3. Relevance: A file might have 10 functions, only 1 is relevant
    4. Performance: Smaller chunks = more precise search results
    """
    # Core identification
    chunk_id: str  # Unique identifier
    file_path: str  # Source file
    start_line: int  # Where this chunk starts in the file
    end_line: int  # Where it ends
    
    # Content
    raw_content: str  # Original code
    processed_content: str  # Cleaned/normalized for embedding
    chunk_type: str  # 'function', 'class', 'comment_block', etc.
    
    # Metadata for rich search
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    docstring: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    calls_functions: List[str] = field(default_factory=list)
    sql_queries: List[str] = field(default_factory=list)
    business_terms: List[str] = field(default_factory=list)
    
    # IFS-specific metadata
    module: Optional[str] = None  # IFS module (ORDER, INVOICE, etc.)
    layer: Optional[str] = None  # presentation, business, data
    api_calls: List[str] = field(default_factory=list)  # IFS API calls
    
    # Quality indicators
    has_error_handling: bool = False
    has_transaction: bool = False
    complexity_score: float = 0.0
    
    def to_embedding_text(self) -> str:
        """
        Convert chunk to text optimized for embedding.
        
        STRATEGY: Combine multiple signals
        - Natural language descriptions (function names, docstrings)
        - Code structure (simplified)
        - Business context (domain terms, API calls)
        - Technical context (SQL, error handling)
        """
        parts = []
        
        # 1. Identity and purpose
        if self.function_name:
            parts.append(f"Function: {self.function_name}")
        if self.class_name:
            parts.append(f"Class: {self.class_name}")
        if self.docstring:
            parts.append(f"Description: {self.docstring[:200]}")  # Limit length
        
        # 2. Business context
        if self.module:
            parts.append(f"Module: {self.module}")
        if self.business_terms:
            parts.append(f"Business: {' '.join(self.business_terms[:10])}")
        
        # 3. Technical context
        if self.sql_queries:
            # Extract table names from SQL
            tables = self._extract_table_names()
            if tables:
                parts.append(f"Tables: {' '.join(tables[:5])}")
        
        if self.api_calls:
            parts.append(f"APIs: {' '.join(self.api_calls[:5])}")
        
        # 4. Code snippet (simplified)
        code_snippet = self._simplify_code()
        if code_snippet:
            parts.append(f"Code: {code_snippet[:300]}")
        
        return " | ".join(parts)
    
    def _extract_table_names(self) -> List[str]:
        """Extract table names from SQL queries."""
        tables = set()
        for query in self.sql_queries:
            # Find table names in FROM and JOIN clauses
            from_tables = re.findall(r'FROM\s+(\w+)', query, re.IGNORECASE)
            join_tables = re.findall(r'JOIN\s+(\w+)', query, re.IGNORECASE)
            tables.update(from_tables + join_tables)
        return list(tables)
    
    def _simplify_code(self) -> str:
        """
        Simplify code for embedding - remove noise, keep structure.
        """
        simplified = self.raw_content
        
        # Remove comments
        simplified = re.sub(r'--.*$', '', simplified, flags=re.MULTILINE)
        simplified = re.sub(r'/\*.*?\*/', '', simplified, flags=re.DOTALL)
        
        # Normalize whitespace
        simplified = ' '.join(simplified.split())
        
        # Truncate if too long
        if len(simplified) > 500:
            simplified = simplified[:500] + "..."
        
        return simplified

class IFSCodebaseProcessor:
    """
    Processes IFS codebase into searchable chunks.
    
    KEY INSIGHT: IFS has specific patterns we can exploit:
    - PL/SQL packages with procedures/functions
    - Aurena client files with TypeScript/JavaScript
    - Entity and projection files with specific structure
    - Standard naming conventions (_API, _RPI, etc.)
    """
    
    def __init__(self):
        self.chunks = []
        self.file_processors = {
            '.plsql': self._process_plsql_file,
            '.sql': self._process_plsql_file,  # Same processor
            '.client': self._process_client_file,
            '.entity': self._process_entity_file,
            '.projection': self._process_projection_file,
            '.fragment': self._process_fragment_file,
            '.ts': self._process_typescript_file,
            '.tsx': self._process_typescript_file,
            '.js': self._process_javascript_file,
            '.jsx': self._process_javascript_file,
        }
        
        # IFS-specific patterns
        self.ifs_patterns = {
            'api_call': re.compile(r'(\w+_API)\.(\w+)'),
            'error_handling': re.compile(r'Error_SYS\.\w+|RAISE_APPLICATION_ERROR'),
            'transaction': re.compile(r'@ApproveTransactionStatement|COMMIT|ROLLBACK'),
            'business_term': re.compile(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+'),  # CamelCase
            'module': re.compile(r'^[A-Z]+(?=_)'),  # ORDER_, INVOICE_, etc.
        }
    
    def process_codebase(self, root_path: Path, file_limit: Optional[int] = None) -> List[CodeChunk]:
        """
        Process entire codebase into chunks.
        
        STRATEGY:
        1. Walk directory tree
        2. Process each file based on type
        3. Extract meaningful chunks
        4. Enrich with metadata
        """
        logger.info(f"Processing codebase at {root_path}")
        
        files_processed = 0
        for file_path in tqdm(self._walk_codebase(root_path), desc="Processing files"):
            if file_limit and files_processed >= file_limit:
                break
            
            try:
                file_chunks = self._process_file(file_path)
                self.chunks.extend(file_chunks)
                files_processed += 1
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        logger.info(f"Processed {files_processed} files, extracted {len(self.chunks)} chunks")
        return self.chunks
    
    def _walk_codebase(self, root_path: Path):
        """Walk codebase and yield relevant files."""
        for file_path in root_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.file_processors:
                # Skip test files and generated files
                if 'test' in file_path.name.lower() or 'generated' in str(file_path):
                    continue
                yield file_path
    
    def _process_file(self, file_path: Path) -> List[CodeChunk]:
        """Process a single file into chunks."""
        suffix = file_path.suffix
        processor = self.file_processors.get(suffix)
        
        if not processor:
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Get module from path
            module = self._extract_module(file_path)
            
            # Process based on file type
            chunks = processor(file_path, content)
            
            # Enrich all chunks with common metadata
            for chunk in chunks:
                chunk.module = module
                chunk.layer = self._determine_layer(file_path)
                self._enrich_chunk_metadata(chunk)
            
            return chunks
            
        except Exception as e:
            logger.debug(f"Could not process {file_path}: {e}")
            return []
    
    def _process_plsql_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """
        Process PL/SQL file into chunks.
        
        PL/SQL STRUCTURE:
        - Package specification (optional)
        - Package body
        - Procedures and functions
        - Cursors and types
        """
        chunks = []
        
        # Extract procedures
        proc_pattern = r'PROCEDURE\s+(\w+)[^;]*?IS(.*?)END\s+\1\s*;'
        for match in re.finditer(proc_pattern, content, re.DOTALL | re.IGNORECASE):
            proc_name = match.group(1)
            proc_body = match.group(2)
            
            chunk = CodeChunk(
                chunk_id=self._generate_chunk_id(file_path, proc_name),
                file_path=str(file_path),
                start_line=content[:match.start()].count('\n') + 1,
                end_line=content[:match.end()].count('\n') + 1,
                raw_content=match.group(0),
                processed_content=proc_body,
                chunk_type='procedure',
                function_name=proc_name
            )
            
            # Extract SQL queries
            chunk.sql_queries = self._extract_sql_queries(proc_body)
            
            # Extract API calls
            chunk.api_calls = self._extract_api_calls(proc_body)
            
            chunks.append(chunk)
        
        # Extract functions
        func_pattern = r'FUNCTION\s+(\w+)[^;]*?RETURN[^;]*?IS(.*?)END\s+\1\s*;'
        for match in re.finditer(func_pattern, content, re.DOTALL | re.IGNORECASE):
            func_name = match.group(1)
            func_body = match.group(2)
            
            chunk = CodeChunk(
                chunk_id=self._generate_chunk_id(file_path, func_name),
                file_path=str(file_path),
                start_line=content[:match.start()].count('\n') + 1,
                end_line=content[:match.end()].count('\n') + 1,
                raw_content=match.group(0),
                processed_content=func_body,
                chunk_type='function',
                function_name=func_name
            )
            
            chunk.sql_queries = self._extract_sql_queries(func_body)
            chunk.api_calls = self._extract_api_calls(func_body)
            
            chunks.append(chunk)
        
        # If no procedures/functions found, treat whole file as one chunk
        if not chunks:
            chunk = CodeChunk(
                chunk_id=self._generate_chunk_id(file_path, 'full'),
                file_path=str(file_path),
                start_line=1,
                end_line=content.count('\n'),
                raw_content=content[:5000],  # Limit size
                processed_content=content[:2000],
                chunk_type='module'
            )
            chunks.append(chunk)
        
        return chunks
    
    def _process_client_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Process Aurena client file (TypeScript/JavaScript)."""
        chunks = []
        
        # Extract functions and methods
        func_pattern = r'(?:function|const|let|var)\s+(\w+)\s*=?\s*(?:async\s*)?\([^)]*\)\s*(?:=>)?\s*{([^}]*)}'
        for match in re.finditer(func_pattern, content):
            func_name = match.group(1)
            func_body = match.group(2)
            
            chunk = CodeChunk(
                chunk_id=self._generate_chunk_id(file_path, func_name),
                file_path=str(file_path),
                start_line=content[:match.start()].count('\n') + 1,
                end_line=content[:match.end()].count('\n') + 1,
                raw_content=match.group(0),
                processed_content=func_body,
                chunk_type='client_function',
                function_name=func_name
            )
            
            chunks.append(chunk)
        
        # If no functions found, create file-level chunk
        if not chunks:
            chunk = CodeChunk(
                chunk_id=self._generate_chunk_id(file_path, 'full'),
                file_path=str(file_path),
                start_line=1,
                end_line=content.count('\n'),
                raw_content=content[:5000],
                processed_content=content[:2000],
                chunk_type='client_module'
            )
            chunks.append(chunk)
        
        return chunks
    
    def _process_entity_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Process IFS entity file."""
        # Entity files define data structure and business logic
        chunk = CodeChunk(
            chunk_id=self._generate_chunk_id(file_path, 'entity'),
            file_path=str(file_path),
            start_line=1,
            end_line=content.count('\n'),
            raw_content=content[:5000],
            processed_content=self._extract_entity_structure(content),
            chunk_type='entity'
        )
        
        # Extract attributes and relationships
        chunk.business_terms = self._extract_business_terms(content)
        
        return [chunk]
    
    def _process_projection_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Process IFS projection file."""
        # Projection files define REST API endpoints
        chunk = CodeChunk(
            chunk_id=self._generate_chunk_id(file_path, 'projection'),
            file_path=str(file_path),
            start_line=1,
            end_line=content.count('\n'),
            raw_content=content[:5000],
            processed_content=self._extract_projection_structure(content),
            chunk_type='projection'
        )
        
        # Extract API operations
        operations = re.findall(r'(GET|POST|PUT|DELETE|PATCH)\s+(\w+)', content)
        chunk.api_calls = [f"{method} {name}" for method, name in operations]
        
        return [chunk]
    
    def _process_fragment_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Process IFS fragment file."""
        # Fragment files are reusable UI components
        chunk = CodeChunk(
            chunk_id=self._generate_chunk_id(file_path, 'fragment'),
            file_path=str(file_path),
            start_line=1,
            end_line=content.count('\n'),
            raw_content=content[:5000],
            processed_content=content[:2000],
            chunk_type='fragment'
        )
        
        return [chunk]
    
    def _process_typescript_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Process TypeScript file."""
        chunks = []
        
        # Extract classes and their methods
        class_pattern = r'class\s+(\w+).*?{(.*?)^}'
        for match in re.finditer(class_pattern, content, re.DOTALL | re.MULTILINE):
            class_name = match.group(1)
            class_body = match.group(2)
            
            # Extract methods from class
            method_pattern = r'(?:async\s+)?(\w+)\s*\([^)]*\)\s*(?::\s*\w+)?\s*{([^}]*)}'
            for method_match in re.finditer(method_pattern, class_body):
                method_name = method_match.group(1)
                method_body = method_match.group(2)
                
                chunk = CodeChunk(
                    chunk_id=self._generate_chunk_id(file_path, f"{class_name}.{method_name}"),
                    file_path=str(file_path),
                    start_line=content[:match.start() + method_match.start()].count('\n') + 1,
                    end_line=content[:match.start() + method_match.end()].count('\n') + 1,
                    raw_content=method_match.group(0),
                    processed_content=method_body,
                    chunk_type='method',
                    function_name=method_name,
                    class_name=class_name
                )
                
                chunks.append(chunk)
        
        return chunks if chunks else self._process_javascript_file(file_path, content)
    
    def _process_javascript_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Process JavaScript file."""
        return self._process_client_file(file_path, content)  # Similar structure
    
    def _extract_sql_queries(self, content: str) -> List[str]:
        """Extract SQL queries from code."""
        queries = []
        
        # Common SQL patterns
        sql_patterns = [
            r'SELECT\s+.*?FROM\s+\w+.*?(?:WHERE|GROUP BY|ORDER BY|;|\n\n)',
            r'INSERT\s+INTO\s+\w+.*?(?:VALUES|SELECT).*?;',
            r'UPDATE\s+\w+\s+SET.*?(?:WHERE.*?)?;',
            r'DELETE\s+FROM\s+\w+.*?(?:WHERE.*?)?;',
        ]
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            queries.extend(matches)
        
        return queries[:10]  # Limit to prevent huge lists
    
    def _extract_api_calls(self, content: str) -> List[str]:
        """Extract IFS API calls from code."""
        api_calls = []
        
        # Find API package calls (e.g., Customer_Order_API.Get_State)
        matches = self.ifs_patterns['api_call'].findall(content)
        for package, method in matches:
            api_calls.append(f"{package}.{method}")
        
        return list(set(api_calls))[:20]  # Unique, limited
    
    def _extract_business_terms(self, content: str) -> List[str]:
        """Extract business terminology from code."""
        # Find CamelCase terms (likely business concepts)
        terms = self.ifs_patterns['business_term'].findall(content)
        
        # Filter out common programming terms
        programming_terms = {'String', 'Integer', 'Boolean', 'Array', 'Object', 'Function'}
        business_terms = [t for t in terms if t not in programming_terms]
        
        return list(set(business_terms))[:20]
    
    def _extract_entity_structure(self, content: str) -> str:
        """Extract key structure from entity file."""
        # Extract attributes and keys
        attributes = re.findall(r'attribute\s+(\w+)', content, re.IGNORECASE)
        keys = re.findall(r'key\s+(\w+)', content, re.IGNORECASE)
        
        structure = f"Attributes: {', '.join(attributes[:10])}"
        if keys:
            structure += f" | Keys: {', '.join(keys[:5])}"
        
        return structure
    
    def _extract_projection_structure(self, content: str) -> str:
        """Extract key structure from projection file."""
        # Extract entity sets and actions
        entity_sets = re.findall(r'entityset\s+(\w+)', content, re.IGNORECASE)
        actions = re.findall(r'action\s+(\w+)', content, re.IGNORECASE)
        
        structure = f"Entity Sets: {', '.join(entity_sets[:10])}"
        if actions:
            structure += f" | Actions: {', '.join(actions[:5])}"
        
        return structure
    
    def _extract_module(self, file_path: Path) -> Optional[str]:
        """Extract IFS module from file path."""
        # IFS modules are typically in path like: .../ORDER/source/...
        parts = file_path.parts
        for part in parts:
            if part.isupper() and len(part) > 2:
                return part
        return None
    
    def _determine_layer(self, file_path: Path) -> str:
        """Determine architectural layer from file path."""
        path_str = str(file_path).lower()
        
        if 'client' in path_str or 'aurena' in path_str:
            return 'presentation'
        elif 'plsql' in path_str or 'source' in path_str:
            return 'business'
        elif 'model' in path_str or 'entity' in path_str:
            return 'data'
        else:
            return 'unknown'
    
    def _enrich_chunk_metadata(self, chunk: CodeChunk):
        """Add additional metadata to chunk."""
        content = chunk.raw_content
        
        # Check for error handling
        chunk.has_error_handling = bool(self.ifs_patterns['error_handling'].search(content))
        
        # Check for transactions
        chunk.has_transaction = bool(self.ifs_patterns['transaction'].search(content))
        
        # Calculate complexity (simple metric based on nesting and conditions)
        chunk.complexity_score = self._calculate_complexity(content)
        
        # Extract business terms if not already done
        if not chunk.business_terms:
            chunk.business_terms = self._extract_business_terms(content)
    
    def _calculate_complexity(self, content: str) -> float:
        """Calculate code complexity score (0-1)."""
        # Simple heuristic based on control structures
        score = 0.0
        
        # Count control structures
        if_count = len(re.findall(r'\bIF\b', content, re.IGNORECASE))
        loop_count = len(re.findall(r'\b(FOR|WHILE|LOOP)\b', content, re.IGNORECASE))
        case_count = len(re.findall(r'\bCASE\b', content, re.IGNORECASE))
        
        # Normalize to 0-1 range
        score = min(1.0, (if_count * 0.1 + loop_count * 0.2 + case_count * 0.15))
        
        return score
    
    def _generate_chunk_id(self, file_path: Path, identifier: str) -> str:
        """Generate unique chunk ID."""
        content = f"{file_path}:{identifier}"
        return hashlib.md5(content.encode()).hexdigest()

class SemanticIndexBuilder:
    """
    Builds and manages the semantic search index.
    
    KEY POINTS:
    1. We use pre-trained models (no training!)
    2. We process code into meaningful chunks
    3. We create embeddings for each chunk
    4. We store in FAISS for fast similarity search
    """
    
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 index_dir: Path = Path.home() / '.ifs_search' / 'semantic'):
        """
        Initialize index builder.
        
        MODEL CHOICES:
        - all-MiniLM-L6-v2: Fast, small (80MB), good for general text
        - all-mpnet-base-v2: Larger (420MB), better quality
        - microsoft/codebert-base: Specialized for code (needs different setup)
        - multi-qa-MiniLM-L6-v1: Optimized for Q&A (good for search)
        """
        self.model_name = model_name
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
        # Storage
        self.chunks = []
        self.embeddings = None
        self.index = None
    
    def build_index(self, codebase_path: Path, file_limit: Optional[int] = None):
        """
        Build the complete semantic index.
        
        PROCESS:
        1. Process codebase into chunks
        2. Generate embeddings for each chunk
        3. Build FAISS index
        4. Save everything to disk
        """
        # Step 1: Process codebase
        logger.info("Step 1: Processing codebase into chunks...")
        processor = IFSCodebaseProcessor()
        self.chunks = processor.process_codebase(codebase_path, file_limit)
        
        if not self.chunks:
            logger.error("No chunks extracted from codebase!")
            return
        
        # Step 2: Generate embeddings
        logger.info("Step 2: Generating embeddings...")
        self._generate_embeddings()
        
        # Step 3: Build FAISS index
        logger.info("Step 3: Building FAISS index...")
        self._build_faiss_index()
        
        # Step 4: Save to disk
        logger.info("Step 4: Saving index to disk...")
        self._save_index()
        
        logger.info(f"Index built successfully! {len(self.chunks)} chunks indexed.")
    
    def _generate_embeddings(self):
        """
        Generate embeddings for all chunks.
        
        BATCHING: We process in batches for efficiency.
        """
        # Prepare texts for embedding
        texts = [chunk.to_embedding_text() for chunk in self.chunks]
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        self.embeddings = np.vstack(all_embeddings).astype('float32')
        logger.info(f"Generated {self.embeddings.shape[0]} embeddings of dimension {self.embeddings.shape[1]}")
    
    def _build_faiss_index(self):
        """
        Build FAISS index for similarity search.
        
        INDEX TYPES:
        - Flat: Exact search, slow for large datasets
        - IVF: Approximate search, much faster
        - HNSW: Graph-based, good balance of speed/quality
        """
        n_chunks = len(self.chunks)
        
        if n_chunks < 10000:
            # For small datasets, use exact search
            logger.info("Using exact search index (Flat)")
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product = cosine similarity
        else:
            # For larger datasets, use IVF with clustering
            logger.info("Using approximate search index (IVF)")
            nlist = int(np.sqrt(n_chunks))  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
            # Train the index on the embeddings
            logger.info("Training IVF index...")
            self.index.train(self.embeddings)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Add embeddings to index
        self.index.add(self.embeddings)
        logger.info(f"Added {self.index.ntotal} vectors to index")
    
    def _save_index(self):
        """Save index and metadata to disk."""
        # Save FAISS index
        index_path = self.index_dir / 'index.faiss'
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save chunks metadata
        chunks_path = self.index_dir / 'chunks.pkl'
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Saved chunks metadata to {chunks_path}")
        
        # Save index info
        info = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'n_chunks': len(self.chunks),
            'created_at': datetime.now().isoformat(),
            'chunk_types': list(set(c.chunk_type for c in self.chunks))
        }
        info_path = self.index_dir / 'index_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved index info to {info_path}")

class SemanticSearcher:
    """
    Performs semantic search on the built index.
    """
    
    def __init__(self, index_dir: Path = Path.home() / '.ifs_search' / 'semantic'):
        self.index_dir = index_dir
        
        # Load index info
        info_path = index_dir / 'index_info.json'
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        
        # Load model
        self.model = SentenceTransformer(self.info['model_name'])
        
        # Load FAISS index
        index_path = index_dir / 'index.faiss'
        self.index = faiss.read_index(str(index_path))
        
        # Load chunks metadata
        chunks_path = index_dir / 'chunks.pkl'
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        logger.info(f"Loaded index with {len(self.chunks)} chunks")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[CodeChunk, float]]:
        """
        Search for similar code chunks.
        
        Returns list of (chunk, similarity_score) tuples.
        """
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return chunks with scores
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx != -1:  # FAISS returns -1 for empty results
                chunk = self.chunks[idx]
                results.append((chunk, float(score)))
        
        return results
```

Now let's create a simple script to build and test the index:

```python
#!/usr/bin/env python3
"""
Build semantic search index for IFS codebase.

NO TRAINING REQUIRED - just data processing and embedding!

Usage:
    python build_semantic_index.py --codebase-path /path/to/ifs/code --limit 1000
"""

import argparse
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ifs_cloud_mcp_server.semantic_indexer import (
    IFSCodebaseProcessor,
    SemanticIndexBuilder,
    SemanticSearcher
)

def main():
    parser = argparse.ArgumentParser(description='Build semantic search index for IFS codebase')
    parser.add_argument('--codebase-path', type=str, required=True,
                       help='Path to IFS codebase root directory')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of files to process (for testing)')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Pre-trained model to use')
    parser.add_argument('--test-query', type=str, default=None,
                       help='Test query to run after building')
    
    args = parser.parse_args()
    
    codebase_path = Path(args.codebase_path)
    if not codebase_path.exists():
        print(f"Error: Codebase path does not exist: {codebase_path}")
        sys.exit(1)
    
    print(f"Building semantic index for: {codebase_path}")
    print(f"Using model: {args.model}")
    if args.limit:
        print(f"Processing first {args.limit} files only")
    
    # Build the index
    start_time = time.time()
    builder = SemanticIndexBuilder(model_name=args.model)
    builder.build_index(codebase_path, file_limit=args.limit)
    build_time = time.time() - start_time
    
    print(f"\nIndex built in {build_time:.2f} seconds")
    
    # Test the index if requested
    if args.test_query:
        print(f"\nTesting with query: '{args.test_query}'")
        
        searcher = SemanticSearcher()
        results = searcher.search(args.test_query, k=5)
        
        print("\nTop 5 results:")
        for i, (chunk, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.3f}")
            print(f"   File: {chunk.file_path}")
            print(f"   Type: {chunk.chunk_type}")
            if chunk.function_name:
                print(f"   Function: {chunk.function_name}")
            if chunk.docstring:
                print(f"   Description: {chunk.docstring[:100]}...")

if __name__ == '__main__':
    main()
```

## 📚 Complete Implementation Summary

### **What We're NOT Doing:**
❌ Training a new model from scratch  
❌ Fine-tuning on IFS data  
❌ Feeding raw codebase to a model  
❌ Requiring labeled training data  

### **What We ARE Doing:**
✅ Using pre-trained models that already understand code  
✅ Processing codebase into meaningful chunks  
✅ Converting chunks to embeddings (numerical vectors)  
✅ Storing embeddings in searchable index  
✅ Finding similar code by vector similarity  

### **Data Preparation Pipeline:**

1. **Chunk Extraction** (30% of the work)
   - Parse files based on type (.plsql, .client, etc.)
   - Extract functions, procedures, classes
   - Keep metadata (line numbers, dependencies)

2. **Text Generation** (20% of the work)
   - Combine code with descriptions
   - Add business context and domain terms
   - Create rich text for embedding

3. **Embedding Generation** (10% of the work)
   - Use pre-trained model
   - Convert text to vectors
   - No training required!

4. **Index Building** (10% of the work)
   - Store vectors in FAISS
   - Enable fast similarity search
   - Save to disk for reuse

5. **Search Interface** (30% of the work)
   - Process queries the same way
   - Find similar vectors
   - Return relevant code chunks

### **Why This Works Without Training:**

Pre-trained models like `all-MiniLM-L6-v2` or `CodeBERT` are already trained on:
- Millions of code repositories
- Natural language descriptions
- Code-documentation pairs
- Multiple programming languages

They understand:
- Code syntax and structure
- Programming concepts
- Natural language queries
- Relationships between code and text

### **Resource Requirements:**

- **Time**: 1-4 hours to process 1GB codebase
- **Memory**: 2-4GB during processing
- **Storage**: ~500MB for index
- **No GPU required**: CPU is fine for inference
- **No training data needed**: Zero labeling effort

### **Getting Started:**

```bash
# Install dependencies
pip install sentence-transformers faiss-cpu numpy tqdm

# Build index (first time only)
python scripts/build_semantic_index.py \
    --codebase-path /path/to/ifs/code \
    --limit 100  # Start small for testing

# Test search
python scripts/build_semantic_index.py \
    --test-query "customer order validation"
```

The beauty of this approach is that it requires **zero machine learning expertise** and **no training data** - just smart processing of your existing codebase!