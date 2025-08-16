You're absolutely right! Let me create a more detailed, educational version that explains the reasoning behind every design decision. This will ensure any AI agent (including Sonnet 4) can understand not just WHAT we're doing, but WHY we're doing it.

```python
"""
COMPREHENSIVE SEMANTIC SEARCH ENGINE FOR IFS CLOUD CODEBASE
===========================================================

This module implements an advanced semantic search system that goes beyond simple
keyword matching to understand the MEANING of code and queries. It's specifically
designed to help AI agents understand and navigate a large enterprise codebase.

KEY CONCEPTS EXPLAINED:
----------------------
1. SEMANTIC SEARCH: Instead of matching exact words, we understand meaning.
   Example: "validate customer" would also find "verify client" or "check buyer"

2. EMBEDDINGS: We convert text/code into numerical vectors that capture meaning.
   Similar concepts have vectors that point in similar directions in space.

3. COSINE SIMILARITY: We measure how similar two vectors are by the angle between them.
   Cos(0°) = 1 (identical), Cos(90°) = 0 (unrelated), Cos(180°) = -1 (opposite)

4. FAISS: Facebook AI Similarity Search - a library for efficient similarity search
   in high-dimensional spaces. Can search millions of vectors in milliseconds.

WHY THIS APPROACH?
-----------------
Traditional keyword search fails in codebases because:
- Developers use different terms for the same concept
- Business logic is expressed in code, not natural language
- Relationships between files aren't captured by keywords
- Context and intent matter more than exact matches

Our semantic approach solves these by understanding:
- Code structure and patterns
- Business domain relationships
- Database schema connections
- Developer intent and documentation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import json
import re
from datetime import datetime

# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CodeEmbedding:
    """
    Represents a semantic embedding of a code component.

    WHY THIS STRUCTURE?
    -------------------
    We don't just embed raw code - we create rich, contextual embeddings that
    capture multiple dimensions of meaning:

    1. file_path: Where the code lives (helps with project structure understanding)
    2. embedding: The numerical vector representation (for similarity search)
    3. context_type: What kind of code this is (procedure, function, etc.)
    4. content_summary: Human-readable description (for result presentation)
    5. dependencies: What other code this relies on (for impact analysis)
    6. business_domain: What business area this serves (for domain-specific search)
    7. db_references: What database tables this touches (for data flow analysis)

    This rich metadata allows us to search not just by similarity, but also
    filter and boost results based on context.
    """
    file_path: str
    embedding: np.ndarray
    context_type: str  # 'function', 'procedure', 'class', 'module', 'business_logic'
    content_summary: str
    dependencies: List[str] = field(default_factory=list)
    business_domain: str = 'general'
    db_references: List[str] = field(default_factory=list)

    # Additional metadata for better AI understanding
    complexity_score: float = 0.0  # How complex is this code?
    last_modified: Optional[datetime] = None
    author: Optional[str] = None
    test_coverage: float = 0.0  # Percentage of code covered by tests

class SemanticSearchEngine:
    """
    Advanced semantic search engine optimized for AI agent usage.

    ARCHITECTURE OVERVIEW:
    ---------------------
    1. Multiple Specialized Indices: We maintain separate indices for different
       aspects of the codebase (code structure, business logic, database schema).
       This allows targeted searching based on query intent.

    2. Hierarchical Embeddings: We embed at multiple levels (file, function,
       business workflow) to capture both local and global context.

    3. Cross-Reference Graph: We track relationships between files to boost
       related results (if file A imports B, they're likely related).

    4. Domain-Specific Understanding: We recognize IFS-specific patterns and
       terminology to provide more relevant results.
    """

    def __init__(self,
                 cache_dir: Path = Path.home() / '.ifs_search' / 'semantic',
                 model_name: str = 'microsoft/codebert-base'):
        """
        Initialize the semantic search engine.

        Parameters:
        -----------
        cache_dir: Where to store the indices and metadata
        model_name: Which embedding model to use

        WHY CODEBERT?
        ------------
        CodeBERT is specifically trained on code and natural language pairs,
        making it understand both programming concepts and human descriptions.
        Alternatives:
        - 'sentence-transformers/all-MiniLM-L6-v2': Faster, smaller, less accurate
        - 'microsoft/unixcoder-base': Better for code, worse for natural language
        - 'openai/ada-002': Best quality but requires API calls (not local)
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing semantic search with model: {model_name}")

        # Initialize the embedding model
        try:
            self.encoder = SentenceTransformer(model_name)
            self.embedding_dimension = self.encoder.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            # Fallback to a smaller, more reliable model
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dimension = 384
            logger.info("Fell back to MiniLM model")

        # Initialize separate indices for different search strategies
        # WHY MULTIPLE INDICES?
        # Different types of searches need different context. A search for
        # "customer validation" should prioritize business logic, while
        # "table schema" should prioritize database structure.
        self.indices = {
            'code': None,           # Raw code structure and syntax
            'business': None,       # Business logic and workflows
            'documentation': None,  # Comments, docstrings, documentation
            'database': None,       # Database schema and relationships
            'tests': None          # Test cases and examples
        }

        # Metadata storage for each embedding
        # We use defaultdict to avoid KeyErrors and make the code cleaner
        self.embedding_metadata = defaultdict(dict)

        # Cross-reference graph tracks which files reference each other
        # This helps us find related files even if they're not semantically similar
        self.reference_graph = defaultdict(set)

        # Cache for frequently accessed embeddings
        # WHY CACHE? Embedding generation is expensive (100-500ms per text)
        # Caching common queries speeds up repeated searches dramatically
        self.embedding_cache = {}
        self.cache_size = 1000  # Maximum number of cached embeddings

        # IFS-specific domain knowledge
        # This helps us understand IFS terminology and patterns
        self.domain_knowledge = self._load_domain_knowledge()

        self.logger = logger

    def _load_domain_knowledge(self) -> Dict:
        """
        Load IFS-specific domain knowledge for better understanding.

        WHY DOMAIN KNOWLEDGE?
        --------------------
        IFS Cloud has specific terminology, patterns, and conventions that
        a general-purpose model wouldn't understand. By encoding this knowledge,
        we can provide more relevant results.

        Examples:
        - "LU" means "Logical Unit" in IFS context
        - "_API" suffix indicates a public API package
        - "RMB" means "Right Mouse Button" (context menu action)
        """
        return {
            'abbreviations': {
                'LU': 'Logical Unit',
                'RMB': 'Right Mouse Button',
                'IFS': 'Industrial and Financial Systems',
                'CRUD': 'Create Read Update Delete',
                'PO': 'Purchase Order',
                'SO': 'Sales Order',
                'CO': 'Customer Order'
            },
            'patterns': {
                '_API': 'Public API Package',
                '_RPI': 'Restricted Private Interface',
                '_SYS': 'System Package',
                '_CFP': 'Custom Fields Package'
            },
            'domains': {
                'order': ['customer', 'sales', 'purchase', 'requisition'],
                'finance': ['invoice', 'payment', 'accounting', 'ledger'],
                'inventory': ['stock', 'warehouse', 'location', 'part'],
                'hr': ['employee', 'person', 'organization', 'position']
            }
        }

    def build_comprehensive_index(self,
                                 codebase_files: List[Dict],
                                 db_metadata: Dict,
                                 oracle_schema: Dict,
                                 force_rebuild: bool = False):
        """
        Build a comprehensive semantic index from all available data sources.

        Parameters:
        -----------
        codebase_files: List of file dictionaries with 'path' and 'content'
        db_metadata: Metadata about database objects
        oracle_schema: Database schema information
        force_rebuild: Whether to rebuild even if indices exist

        INDEXING STRATEGY:
        -----------------
        1. We process files in batches to manage memory efficiently
        2. We extract multiple embeddings per file (file-level, function-level)
        3. We build separate indices for different aspects
        4. We save everything to disk for fast subsequent loads

        WHY THIS APPROACH?
        -----------------
        - Batch processing prevents memory overflow on large codebases
        - Multiple granularities capture both local and global context
        - Separate indices allow targeted searching
        - Disk persistence means we only need to index once
        """

        # Check if indices already exist and we're not forcing rebuild
        if not force_rebuild and self._indices_exist():
            logger.info("Indices already exist. Loading from disk...")
            self.load_indices()
            return

        logger.info(f"Building comprehensive index for {len(codebase_files)} files...")
        start_time = datetime.now()

        # Process in batches to manage memory
        batch_size = 100
        all_embeddings = defaultdict(list)

        for i in range(0, len(codebase_files), batch_size):
            batch = codebase_files[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(codebase_files) + batch_size - 1)//batch_size}")

            # Extract embeddings for this batch
            batch_embeddings = self._process_file_batch(batch, db_metadata)

            # Merge into main collection
            for key, embeddings in batch_embeddings.items():
                all_embeddings[key].extend(embeddings)

        # Add database schema embeddings
        logger.info("Processing database schema...")
        db_embeddings = self._build_database_embeddings(oracle_schema)
        all_embeddings['database'] = db_embeddings

        # Build cross-reference graph
        logger.info("Building cross-reference graph...")
        self._build_reference_graph(codebase_files)

        # Create FAISS indices
        logger.info("Creating FAISS indices...")
        self._create_faiss_indices(all_embeddings)

        # Save to disk
        logger.info("Saving indices to disk...")
        self._save_indices()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Index building completed in {elapsed:.2f} seconds")

        # Print statistics
        self._print_index_statistics(all_embeddings)

    def _process_file_batch(self,
                           files: List[Dict],
                           db_metadata: Dict) -> Dict[str, List[CodeEmbedding]]:
        """
        Process a batch of files and extract embeddings.

        WHY BATCH PROCESSING?
        --------------------
        1. Memory efficiency: Processing all files at once could use 10+ GB RAM
        2. Progress tracking: We can show progress and resume if interrupted
        3. Parallelization: Batches can be processed in parallel (future enhancement)

        EXTRACTION STRATEGY:
        -------------------
        For each file, we extract:
        1. File-level embedding (overall purpose and structure)
        2. Function-level embeddings (individual functions/procedures)
        3. Business logic embeddings (workflows and rules)
        4. Documentation embeddings (comments and descriptions)
        """
        batch_embeddings = defaultdict(list)

        for file in files:
            try:
                # Extract different aspects of the file
                code_components = self._extract_code_components(file)
                business_logic = self._extract_business_logic(file, db_metadata)
                documentation = self._extract_documentation(file)

                # Create embeddings for each aspect
                for component in code_components:
                    embedding = self._create_code_embedding(component, file)
                    batch_embeddings['code'].append(embedding)

                for logic in business_logic:
                    embedding = self._create_business_embedding(logic, file)
                    batch_embeddings['business'].append(embedding)

                for doc in documentation:
                    embedding = self._create_documentation_embedding(doc, file)
                    batch_embeddings['documentation'].append(embedding)

            except Exception as e:
                logger.warning(f"Error processing {file.get('path', 'unknown')}: {e}")
                continue

        return batch_embeddings

    def _extract_code_components(self, file: Dict) -> List[Dict]:
        """
        Extract semantic components from a code file.

        EXTRACTION LOGIC:
        ----------------
        We parse the code to identify meaningful units:
        1. Functions/Procedures: Individual units of logic
        2. Classes: Object-oriented structures
        3. SQL Queries: Database operations
        4. API Endpoints: Service interfaces

        WHY COMPONENT-LEVEL?
        -------------------
        A file might contain multiple unrelated functions. By embedding each
        separately, we can find the specific function an AI needs, not just
        the file.
        """
        components = []
        content = file.get('content', '')
        file_type = file.get('type', '')

        if file_type == '.plsql':
            components.extend(self._extract_plsql_components(content))
        elif file_type == '.client':
            components.extend(self._extract_client_components(content))
        elif file_type == '.projection':
            components.extend(self._extract_projection_components(content))
        elif file_type == '.entity':
            components.extend(self._extract_entity_components(content))

        # If no components found, treat the whole file as one component
        if not components:
            components.append({
                'type': 'module',
                'name': Path(file['path']).stem,
                'content': content[:2000],  # First 2000 chars
                'summary': f"Module {Path(file['path']).stem}",
                'start_line': 0,
                'end_line': len(content.split('\n'))
            })

        return components

    def _extract_plsql_components(self, content: str) -> List[Dict]:
        """
        Extract components from PL/SQL code.

        PL/SQL STRUCTURE:
        ----------------
        PL/SQL packages contain:
        - Procedures: Operations that perform actions
        - Functions: Operations that return values
        - Cursors: Database query definitions
        - Types: Custom data structures

        We extract each separately because they serve different purposes
        and an AI might need a specific one.
        """
        components = []

        # Regular expressions for PL/SQL components
        # WHY THESE PATTERNS?
        # PL/SQL has consistent syntax that we can reliably parse
        patterns = {
            'procedure': r'PROCEDURE\s+(\w+)[^;]*?IS(.*?)END\s+\1\s*;',
            'function': r'FUNCTION\s+(\w+)[^;]*?RETURN\s+(\w+)[^;]*?IS(.*?)END\s+\1\s*;',
            'cursor': r'CURSOR\s+(\w+)[^;]*?IS\s*([^;]+);',
            'type': r'TYPE\s+(\w+)\s+IS\s+([^;]+);'
        }

        for comp_type, pattern in patterns.items():
            for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
                name = match.group(1)
                body = match.group(2) if comp_type != 'function' else match.group(3)

                # Extract additional metadata
                component = {
                    'type': comp_type,
                    'name': name,
                    'content': body[:1000],  # Limit for embedding
                    'full_content': body,  # Keep full content for analysis
                    'summary': self._generate_component_summary(comp_type, name, body),
                    'parameters': self._extract_parameters(match.group(0)),
                    'dependencies': self._extract_dependencies(body),
                    'db_references': self._extract_db_references(body),
                    'complexity': self._calculate_complexity(body)
                }

                components.append(component)

        return components

    def _create_code_embedding(self, component: Dict, file: Dict) -> CodeEmbedding:
        """
        Create a rich embedding for a code component.

        EMBEDDING STRATEGY:
        ------------------
        We don't just embed the raw code. We create a semantic representation
        that includes:
        1. What the code does (summary)
        2. How it does it (structure)
        3. What it depends on (dependencies)
        4. Where it fits (context)

        This rich representation helps the AI understand not just the code
        itself, but how to use it.
        """
        # Create semantic text that captures the essence of the component
        semantic_text = self._create_semantic_text(component, file)

        # Generate embedding vector
        # We cache embeddings to avoid recomputing for common texts
        cache_key = f"{file['path']}:{component['name']}"
        if cache_key in self.embedding_cache:
            embedding_vector = self.embedding_cache[cache_key]
        else:
            embedding_vector = self.encoder.encode(semantic_text)
            # Update cache (with size limit)
            if len(self.embedding_cache) < self.cache_size:
                self.embedding_cache[cache_key] = embedding_vector

        # Create the embedding object with rich metadata
        return CodeEmbedding(
            file_path=file['path'],
            embedding=embedding_vector,
            context_type=component['type'],
            content_summary=component['summary'],
            dependencies=component.get('dependencies', []),
            business_domain=self._infer_business_domain(file['path'], component),
            db_references=component.get('db_references', []),
            complexity_score=component.get('complexity', 0.0),
            last_modified=file.get('last_modified'),
            author=file.get('author'),
            test_coverage=self._estimate_test_coverage(component, file)
        )

    def _create_semantic_text(self, component: Dict, file: Dict) -> str:
        """
        Create a rich semantic representation of a code component.

        WHY THIS FORMAT?
        ---------------
        The embedding model needs text that captures:
        1. Intent: What is this code trying to do?
        2. Context: Where does it fit in the system?
        3. Implementation: How does it work?
        4. Relationships: What does it connect to?

        By including all these aspects, we create embeddings that can be
        matched against various types of queries.
        """
        # Start with the component type and name
        parts = [
            f"Component Type: {component['type']}",
            f"Name: {component['name']}",
            f"File: {Path(file['path']).stem}",
            f"Purpose: {component['summary']}"
        ]

        # Add parameters if present
        if component.get('parameters'):
            params = ', '.join(component['parameters'][:5])  # First 5 params
            parts.append(f"Parameters: {params}")

        # Add dependencies for context
        if component.get('dependencies'):
            deps = ', '.join(component['dependencies'][:5])  # First 5 deps
            parts.append(f"Uses: {deps}")

        # Add database references
        if component.get('db_references'):
            tables = ', '.join(component['db_references'][:5])  # First 5 tables
            parts.append(f"Database Tables: {tables}")

        # Add a snippet of the actual code
        # WHY INCLUDE CODE?
        # The model can learn patterns from the code structure itself
        code_snippet = component['content'][:300]  # First 300 chars
        if code_snippet:
            # Clean up the code for better embedding
            code_snippet = ' '.join(code_snippet.split())  # Normalize whitespace
            parts.append(f"Implementation: {code_snippet}")

        # Add domain context from IFS knowledge
        domain = self._infer_business_domain(file['path'], component)
        if domain != 'general':
            parts.append(f"Business Domain: {domain}")

        # Combine all parts into a comprehensive description
        return " | ".join(parts)

    def _extract_business_logic(self, file: Dict, db_metadata: Dict) -> List[Dict]:
        """
        Extract business logic patterns from code.

        BUSINESS LOGIC IDENTIFICATION:
        -----------------------------
        Business logic isn't just code - it's the WHY behind the code.
        We look for:
        1. Validation rules (IF statements that check business conditions)
        2. Calculations (formulas that implement business rules)
        3. Workflows (sequences of operations that fulfill business processes)
        4. State transitions (status changes that reflect business states)

        This helps AI agents understand not just what the code does,
        but what business problem it solves.
        """
        business_logic = []
        content = file.get('content', '')

        # Extract validation rules
        validations = self._extract_validation_rules(content)
        for validation in validations:
            business_logic.append({
                'type': 'validation',
                'rule': validation['rule'],
                'condition': validation['condition'],
                'action': validation['action'],
                'business_impact': validation.get('impact', 'Data integrity')
            })

        # Extract business calculations
        calculations = self._extract_business_calculations(content)
        for calc in calculations:
            business_logic.append({
                'type': 'calculation',
                'name': calc['name'],
                'formula': calc['formula'],
                'purpose': calc.get('purpose', 'Business calculation'),
                'affected_fields': calc.get('fields', [])
            })

        # Extract workflow patterns
        workflows = self._extract_workflow_patterns(content, db_metadata)
        for workflow in workflows:
            business_logic.append({
                'type': 'workflow',
                'name': workflow['name'],
                'steps': workflow['steps'],
                'triggers': workflow.get('triggers', []),
                'outcomes': workflow.get('outcomes', [])
            })

        return business_logic

    def _extract_validation_rules(self, content: str) -> List[Dict]:
        """
        Extract business validation rules from code.

        VALIDATION PATTERNS:
        -------------------
        We look for patterns like:
        - IF amount > credit_limit THEN raise_error
        - CHECK (order_date <= delivery_date)
        - VALIDATE customer_status = 'ACTIVE'

        These represent business rules that ensure data integrity.
        """
        validations = []

        # Pattern for IF statements with business logic
        if_pattern = r'IF\s+(.+?)\s+THEN\s+(.+?)(?:END IF|ELSIF|;)'
        for match in re.finditer(if_pattern, content, re.DOTALL | re.IGNORECASE):
            condition = match.group(1).strip()
            action = match.group(2).strip()

            # Check if this is a business validation (not just null checks)
            if self._is_business_validation(condition, action):
                validations.append({
                    'rule': f"Validate {self._extract_validation_subject(condition)}",
                    'condition': condition,
                    'action': action,
                    'impact': self._assess_validation_impact(action)
                })

        return validations

    def _is_business_validation(self, condition: str, action: str) -> bool:
        """
        Determine if a condition represents business logic vs technical logic.

        BUSINESS VS TECHNICAL:
        ---------------------
        Business: credit_limit > 10000, status = 'APPROVED'
        Technical: variable IS NOT NULL, connection.isOpen()

        We want to index business validations, not technical checks.
        """
        # Technical patterns to exclude
        technical_patterns = [
            r'IS\s+(NOT\s+)?NULL',
            r'EXISTS',
            r'LENGTH',
            r'\.is[A-Z]',  # isOpen, isValid, etc.
            r'ROWCOUNT',
            r'SQLCODE'
        ]

        for pattern in technical_patterns:
            if re.search(pattern, condition, re.IGNORECASE):
                return False

        # Business patterns to include
        business_patterns = [
            r'status',
            r'amount',
            r'date',
            r'limit',
            r'quantity',
            r'price',
            r'customer',
            r'order',
            r'approved',
            r'valid'
        ]

        for pattern in business_patterns:
            if re.search(pattern, condition, re.IGNORECASE):
                return True

        # Check if action is business-related
        if any(keyword in action.lower() for keyword in ['error', 'reject', 'approve', 'cancel']):
            return True

        return False

    def search_for_ai_agent(self,
                           query: str,
                           context: Dict,
                           limit: int = 20) -> List[Dict]:
        """
        Perform semantic search optimized for AI agent consumption.

        SEARCH STRATEGY:
        ---------------
        1. Query Understanding: Parse the query to understand intent
        2. Multi-Index Search: Search relevant indices based on intent
        3. Relationship Boosting: Boost results related to context
        4. Result Aggregation: Combine and rank results
        5. Context Enhancement: Add rich context for AI understanding

        WHY THIS APPROACH?
        -----------------
        AI agents need more than just similar files. They need:
        - Understanding of relationships between files
        - Business context and implications
        - Implementation patterns and examples
        - Potential impacts and dependencies

        Our search provides all of this in a structured format.
        """
        logger.info(f"Searching for: {query}")
        search_start = datetime.now()

        # 1. Embed the query
        query_embedding = self._get_query_embedding(query)

        # 2. Understand query intent
        query_intent = self._analyze_query_intent(query, context)
        logger.info(f"Query intent: {query_intent}")

        # 3. Determine search strategy
        search_strategy = self._determine_search_strategy(query_intent, context)

        # 4. Search across relevant indices
        search_results = self._execute_multi_index_search(
            query_embedding, search_strategy, limit * 3  # Get extra results for filtering
        )

        # 5. Apply contextual boosting
        boosted_results = self._apply_contextual_boosting(search_results, context)

        # 6. Aggregate and rank
        final_results = self._aggregate_and_rank(boosted_results, query, context)

        # 7. Enhance with AI context
        enhanced_results = self._enhance_for_ai_agent(final_results[:limit])

        search_time = (datetime.now() - search_start).total_seconds()
        logger.info(f"Search completed in {search_time:.3f} seconds, returning {len(enhanced_results)} results")

        return enhanced_results

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get embedding for a query, using cache if available.

        CACHING STRATEGY:
        ----------------
        Embedding generation takes 100-500ms. For common queries,
        we cache the embeddings to provide instant results.
        """
        cache_key = f"query:{query}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        # Enhance query with domain knowledge
        enhanced_query = self._enhance_query_with_domain_knowledge(query)

        # Generate embedding
        embedding = self.encoder.encode(enhanced_query)

        # Cache it
        if len(self.embedding_cache) < self.cache_size:
            self.embedding_cache[cache_key] = embedding

        return embedding

    def _enhance_query_with_domain_knowledge(self, query: str) -> str:
        """
        Enhance query with IFS-specific domain knowledge.

        ENHANCEMENT STRATEGY:
        --------------------
        We expand abbreviations and add synonyms to make the query
        more likely to match relevant code.

        Example: "CO validation" -> "Customer Order CO validation sales"
        """
        enhanced = query

        # Expand abbreviations
        for abbrev, full in self.domain_knowledge['abbreviations'].items():
            if abbrev in query.upper():
                enhanced += f" {full}"

        # Add domain-specific synonyms
        for domain, terms in self.domain_knowledge['domains'].items():
            if domain in query.lower():
                enhanced += " " + " ".join(terms[:3])  # Add top 3 related terms

        return enhanced

    def _analyze_query_intent(self, query: str, context: Dict) -> Dict:
        """
        Analyze what the user is really looking for.

        INTENT CATEGORIES:
        -----------------
        1. Implementation: Looking for code to implement something
        2. Understanding: Trying to understand how something works
        3. Debugging: Finding the cause of an issue
        4. Impact Analysis: Understanding effects of changes
        5. Examples: Looking for similar implementations

        Understanding intent helps us search the right indices and
        rank results appropriately.
        """
        intent = {
            'primary': 'general',
            'aspects': [],
            'confidence': 0.0
        }

        query_lower = query.lower()

        # Implementation intent patterns
        if any(word in query_lower for word in ['implement', 'add', 'create', 'build', 'develop']):
            intent['primary'] = 'implementation'
            intent['aspects'].append('code_structure')
            intent['confidence'] = 0.8

        # Understanding intent patterns
        elif any(word in query_lower for word in ['how', 'what', 'understand', 'explain', 'work']):
            intent['primary'] = 'understanding'
            intent['aspects'].append('documentation')
            intent['confidence'] = 0.7

        # Debugging intent patterns
        elif any(word in query_lower for word in ['error', 'bug', 'issue', 'problem', 'fix']):
            intent['primary'] = 'debugging'
            intent['aspects'].append('error_handling')
            intent['confidence'] = 0.8

        # Impact analysis patterns
        elif any(word in query_lower for word in ['impact', 'affect', 'depend', 'change', 'modify']):
            intent['primary'] = 'impact_analysis'
            intent['aspects'].append('dependencies')
            intent['confidence'] = 0.7

        # Example searching patterns
        elif any(word in query_lower for word in ['example', 'similar', 'like', 'pattern', 'template']):
            intent['primary'] = 'examples'
            intent['aspects'].append('patterns')
            intent['confidence'] = 0.9

        # Add aspects based on query content
        if 'business' in query_lower or 'workflow' in query_lower:
            intent['aspects'].append('business_logic')
        if 'database' in query_lower or 'table' in query_lower:
            intent['aspects'].append('database')
        if 'test' in query_lower:
            intent['aspects'].append('testing')

        # Use context to refine intent
        if context.get('task_type'):
            intent['primary'] = context['task_type']
            intent['confidence'] = 1.0

        return intent

    def _determine_search_strategy(self, query_intent: Dict, context: Dict) -> Dict:
        """
        Determine which indices to search and with what weights.

        STRATEGY SELECTION:
        ------------------
        Based on intent, we decide:
        1. Which indices to search
        2. How much weight to give each index
        3. What post-processing to apply

        This ensures we search efficiently and return relevant results.
        """
        strategy = {
            'indices': {},  # {index_name: weight}
            'post_processing': [],
            'boost_factors': {}
        }

        # Default: search all indices with equal weight
        base_weight = 1.0

        # Adjust based on primary intent
        if query_intent['primary'] == 'implementation':
            strategy['indices'] = {
                'code': base_weight * 2.0,      # Prioritize actual code
                'business': base_weight * 1.5,   # Include business logic
                'tests': base_weight * 1.2,      # Include test examples
                'documentation': base_weight * 0.5,  # Lower priority for docs
                'database': base_weight * 0.8
            }
            strategy['post_processing'].append('group_by_workflow')
            strategy['boost_factors']['has_tests'] = 1.2

        elif query_intent['primary'] == 'understanding':
            strategy['indices'] = {
                'documentation': base_weight * 2.0,  # Prioritize documentation
                'business': base_weight * 1.5,       # Include business context
                'code': base_weight * 1.0,
                'tests': base_weight * 0.8,
                'database': base_weight * 0.5
            }
            strategy['post_processing'].append('expand_documentation')

        elif query_intent['primary'] == 'debugging':
            strategy['indices'] = {
                'code': base_weight * 2.0,
                'tests': base_weight * 1.8,  # Tests help understand expected behavior
                'business': base_weight * 1.0,
                'documentation': base_weight * 0.8,
                'database': base_weight * 1.2  # Database issues are common
            }
            strategy['post_processing'].append('trace_error_paths')
            strategy['boost_factors']['has_error_handling'] = 1.5

        elif query_intent['primary'] == 'impact_analysis':
            strategy['indices'] = {
                'code': base_weight * 1.5,
                'business': base_weight * 2.0,  # Business impact is crucial
                'database': base_weight * 1.5,  # Database changes have wide impact
                'tests': base_weight * 1.2,
                'documentation': base_weight * 0.5
            }
            strategy['post_processing'].append('analyze_dependencies')
            strategy['boost_factors']['high_connectivity'] = 1.3

        elif query_intent['primary'] == 'examples':
            strategy['indices'] = {
                'code': base_weight * 1.8,
                'tests': base_weight * 2.0,  # Tests are great examples
                'business': base_weight * 1.0,
                'documentation': base_weight * 0.5,
                'database': base_weight * 0.5
            }
            strategy['post_processing'].append('find_patterns')
            strategy['boost_factors']['well_documented'] = 1.4

        # Adjust based on aspects
        for aspect in query_intent.get('aspects', []):
            if aspect == 'business_logic':
                strategy['indices']['business'] *= 1.5
            elif aspect == 'database':
                strategy['indices']['database'] *= 1.5
            elif aspect == 'testing':
                strategy['indices']['tests'] *= 1.5

        # Context-based adjustments
        if context.get('recent_files'):
            strategy['boost_factors']['related_to_recent'] = 1.5

        if context.get('user_expertise') == 'beginner':
            strategy['indices']['documentation'] *= 1.5
            strategy['boost_factors']['well_documented'] = 1.5

        return strategy

    def _execute_multi_index_search(self,
                                   query_embedding: np.ndarray,
                                   search_strategy: Dict,
                                   total_limit: int) -> List[Tuple[Dict, float]]:
        """
        Execute search across multiple indices based on strategy.

        MULTI-INDEX SEARCH:
        ------------------
        We search different indices with different limits based on their
        weights. This ensures we get a good mix of results.

        Example: If code has weight 2.0 and docs have weight 0.5,
        we'll get 4x more code results than doc results.
        """
        all_results = []

        # Calculate how many results to get from each index
        total_weight = sum(search_strategy['indices'].values())

        for index_name, weight in search_strategy['indices'].items():
            if self.indices.get(index_name) is None:
                logger.warning(f"Index {index_name} not available")
                continue

            # Calculate limit for this index proportional to its weight
            index_limit = int((weight / total_weight) * total_limit)
            if index_limit == 0:
                continue

            logger.debug(f"Searching {index_name} index for {index_limit} results")

            # Search this index
            index_results = self._search_single_index(
                index_name, query_embedding, index_limit
            )

            # Add results with their weights
            for result in index_results:
                all_results.append((result, weight))

        return all_results

    def _search_single_index(self,
                           index_name: str,
                           query_embedding: np.ndarray,
                           limit: int) -> List[Dict]:
        """
        Search a single FAISS index.

        FAISS SEARCH:
        ------------
        FAISS (Facebook AI Similarity Search) uses approximate nearest
        neighbor search to find similar vectors quickly.

        For small indices (<10K vectors): Exact search (slow but accurate)
        For large indices: Approximate search (fast but slightly less accurate)
        """
        index = self.indices[index_name]
        if index is None:
            return []

        # Reshape query for FAISS (needs 2D array)
        query_vector = query_embedding.reshape(1, -1)

        # Search for nearest neighbors
        # D = distances, I = indices of nearest neighbors
        D, I = index.search(query_vector, limit)

        results = []
        for distance, idx in zip(D[0], I[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue

            # Convert distance to similarity score
            # We use cosine similarity: 1 - (distance^2 / 2)
            # This works because our embeddings are L2-normalized
            similarity = 1 - (distance ** 2) / 2

            # Get metadata for this embedding
            metadata = self.embedding_metadata.get(idx, {})

            # Create result dictionary
            result = {
                'index': index_name,
                'embedding_id': idx,
                'similarity': similarity,
                'file_path': metadata.get('file_path', 'unknown'),
                'context_type': metadata.get('context_type', 'unknown'),
                'summary': metadata.get('summary', ''),
                'domain': metadata.get('domain', 'general'),
                'db_refs': metadata.get('db_refs', [])
            }

            results.append(result)

        return results

    def _apply_contextual_boosting(self,
                                  results: List[Tuple[Dict, float]],
                                  context: Dict) -> List[Tuple[Dict, float]]:
        """
        Apply contextual boosting to search results.

        BOOSTING STRATEGY:
        -----------------
        We boost results based on:
        1. Relationship to recently accessed files
        2. Relevance to current task
        3. User preferences and expertise
        4. Business domain alignment

        This personalization makes results more relevant to the specific
        user and task at hand.
        """
        boosted_results = []

        for result, base_weight in results:
            boost_factor = 1.0

            # Boost if related to recent files
            if context.get('recent_files'):
                for recent_file in context['recent_files']:
                    if self._are_files_related(result['file_path'], recent_file):
                        boost_factor *= 1.3
                        logger.debug(f"Boosted {result['file_path']} due to relation to {recent_file}")

            # Boost based on user expertise
            if context.get('user_expertise') == 'beginner':
                # Boost well-documented, simpler code
                if result.get('complexity_score', 0) < 0.5:
                    boost_factor *= 1.2
                if 'example' in result.get('summary', '').lower():
                    boost_factor *= 1.3
            elif context.get('user_expertise') == 'expert':
                # Boost complex, optimized implementations
                if result.get('complexity_score', 0) > 0.7:
                    boost_factor *= 1.1

            # Boost based on business domain
            if context.get('business_domain'):
                if result.get('domain') == context['business_domain']:
                    boost_factor *= 1.4

            # Boost based on task requirements
            if context.get('requires_database') and result.get('db_refs'):
                boost_factor *= 1.2

            if context.get('requires_validation') and 'validation' in result.get('summary', '').lower():
                boost_factor *= 1.3

            # Apply boost
            boosted_weight = base_weight * boost_factor
            boosted_results.append((result, boosted_weight))

        return boosted_results

    def _are_files_related(self, file1: str, file2: str) -> bool:
        """
        Determine if two files are related.

        RELATIONSHIP DETECTION:
        ----------------------
        Files are related if:
        1. One imports/references the other
        2. They're in the same module/package
        3. They share significant dependencies
        4. They operate on the same database tables
        """
        # Check direct references
        if file2 in self.reference_graph.get(file1, set()):
            return True
        if file1 in self.reference_graph.get(file2, set()):
            return True

        # Check if in same package
        path1 = Path(file1)
        path2 = Path(file2)
        if path1.parent == path2.parent:
            return True

        # Check shared dependencies (would need dependency graph)
        # This is a simplified check
        if path1.stem in path2.stem or path2.stem in path1.stem:
            return True

        return False

    def _aggregate_and_rank(self,
                           boosted_results: List[Tuple[Dict, float]],
                           query: str,
                           context: Dict) -> List[Dict]:
        """
        Aggregate and rank final results.

        RANKING ALGORITHM:
        -----------------
        Final score = similarity * weight * boost * diversity_penalty

        We also apply diversity to ensure results aren't all from the
        same file or component.
        """
        # Group results by file to apply diversity
        file_groups = defaultdict(list)
        for result, weight in boosted_results:
            file_groups[result['file_path']].append((result, weight))

        # Apply diversity penalty
        final_results = []
        for file_path, group in file_groups.items():
            # Sort group by weight
            group.sort(key=lambda x: x[1], reverse=True)

            # Apply increasing penalty for multiple results from same file
            for i, (result, weight) in enumerate(group):
                diversity_penalty = 1.0 / (1.0 + i * 0.3)  # 30% penalty per additional result
                final_score = result['similarity'] * weight * diversity_penalty

                result['final_score'] = final_score
                result['weight'] = weight
                result['diversity_penalty'] = diversity_penalty

                final_results.append(result)

        # Sort by final score
        final_results.sort(key=lambda x: x['final_score'], reverse=True)

        # Add rank information
        for i, result in enumerate(final_results):
            result['rank'] = i + 1

        return final_results

    def _enhance_for_ai_agent(self, results: List[Dict]) -> List[Dict]:
        """
        Enhance results with rich context for AI consumption.

        AI CONTEXT ENHANCEMENT:
        -----------------------
        We add:
        1. Code snippets with highlighting
        2. Relationship maps
        3. Business impact analysis
        4. Implementation guidance
        5. Testing recommendations

        This transforms raw search results into actionable intelligence
        for the AI agent.
        """
        enhanced_results = []

        for result in results:
            # Load the actual file content if needed
            file_content = self._load_file_content(result['file_path'])

            enhanced = {
                **result,  # Include all existing fields
                'ai_context': {
                    # Core information
                    'file_path': result['file_path'],
                    'relevance_score': result['final_score'],
                    'relevance_explanation': self._explain_relevance(result),

                    # Code understanding
                    'code_snippet': self._extract_relevant_snippet(file_content, result),
                    'key_functions': self._extract_key_functions(file_content),
                    'complexity_analysis': self._analyze_complexity(file_content),

                    # Business context
                    'business_purpose': self._extract_business_purpose(file_content),
                    'business_rules': self._extract_business_rules_from_content(file_content),
                    'domain_classification': result.get('domain', 'general'),

                    # Technical details
                    'dependencies': self._extract_all_dependencies(file_content),
                    'database_operations': self._extract_database_operations(file_content),
                    'api_endpoints': self._extract_api_endpoints(file_content),

                    # Relationships
                    'imports': self._extract_imports(file_content),
                    'exported_functions': self._extract_exports(file_content),
                    'related_files': self._get_related_files(result['file_path']),

                    # Quality indicators
                    'has_tests': self._check_test_existence(result['file_path']),
                    'documentation_level': self._assess_documentation(file_content),
                    'last_modified': self._get_modification_date(result['file_path']),

                    # Implementation guidance
                    'usage_examples': self._find_usage_examples(result['file_path']),
                    'common_patterns': self._identify_patterns(file_content),
                    'potential_issues': self._identify_potential_issues(file_content),

                    # For modifications
                    'modification_points': self._identify_modification_points(file_content),
                    'extension_opportunities': self._identify_extension_points(file_content)
                }
            }

            enhanced_results.append(enhanced)

        return enhanced_results

    def _explain_relevance(self, result: Dict) -> str:
        """
        Explain why this result is relevant to the query.

        This helps the AI understand why we returned this result,
        making it easier to decide whether to use it.
        """
        explanations = []

        # High similarity score
        if result['similarity'] > 0.8:
            explanations.append("Very high semantic similarity to query")
        elif result['similarity'] > 0.6:
            explanations.append("Good semantic match")

        # Index-specific explanations
        if result['index'] == 'code':
            explanations.append("Contains relevant code implementation")
        elif result['index'] == 'business':
            explanations.append("Implements related business logic")
        elif result['index'] == 'documentation':
            explanations.append("Provides relevant documentation")
        elif result['index'] == 'database':
            explanations.append("Involves relevant database operations")
        elif result['index'] == 'tests':
            explanations.append("Contains test examples")

        # Boosting explanations
        if result.get('weight', 1.0) > 1.5:
            explanations.append("Highly relevant to search intent")

        if result.get('diversity_penalty', 1.0) < 1.0:
            explanations.append("Additional result from same file")

        return " | ".join(explanations) if explanations else "General relevance"

    def _extract_relevant_snippet(self, content: str, result: Dict) -> str:
        """
        Extract the most relevant code snippet for the result.

        SNIPPET EXTRACTION:
        ------------------
        We try to find the specific part of the file that's most
        relevant to the query, not just the beginning of the file.
        """
        if not content:
            return ""

        # If we know the specific component, extract it
        if result.get('context_type') in ['function', 'procedure']:
            component_name = result.get('summary', '').split()[0] if result.get('summary') else None
            if component_name:
                # Try to find the component in the content
                pattern = rf'(?:FUNCTION|PROCEDURE|def|function)\s+{re.escape(component_name)}.*?(?:END|^\}})'
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE | re.MULTILINE)
                if match:
                    snippet = match.group(0)
                    # Limit to reasonable size
                    if len(snippet) > 1000:
                        snippet = snippet[:1000] + "\n... (truncated)"
                    return snippet

        # Otherwise, find the most relevant section
        # Split into sections and find the one with most relevant keywords
        sections = content.split('\n\n')

        # Score each section based on relevance indicators
        best_section = ""
        best_score = 0

        for section in sections:
            score = 0
            section_lower = section.lower()

            # Check for business logic indicators
            if any(word in section_lower for word in ['validate', 'check', 'process', 'calculate']):
                score += 2

            # Check for database operations
            if any(word in section_lower for word in ['select', 'insert', 'update', 'delete']):
                score += 1

            # Check for comments (usually contain important info)
            if '--' in section or '/*' in section or '//' in section:
                score += 1

            if score > best_score:
                best_score = score
                best_section = section

        # Return the best section (limited size)
        if len(best_section) > 1000:
            best_section = best_section[:1000] + "\n... (truncated)"

        return best_section if best_section else content[:1000]

    # [Additional helper methods would continue here...]
    # Each method would have similar detailed documentation explaining
    # its purpose and implementation strategy

    def _save_indices(self):
        """
        Save all indices and metadata to disk.

        PERSISTENCE STRATEGY:
        --------------------
        We save:
        1. FAISS indices (binary format for efficiency)
        2. Metadata (pickle for Python objects)
        3. Reference graph (for relationship tracking)
        4. Configuration (JSON for human readability)

        This allows us to quickly load pre-built indices instead of
        rebuilding from scratch.
        """
        logger.info("Saving indices to disk...")

        # Save FAISS indices
        for index_name, index in self.indices.items():
            if index is not None:
                index_path = self.cache_dir / f"{index_name}.index"
                faiss.write_index(index, str(index_path))
                logger.info(f"Saved {index_name} index to {index_path}")

        # Save metadata
        metadata_path = self.cache_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(dict(self.embedding_metadata), f)
        logger.info(f"Saved metadata to {metadata_path}")

        # Save reference graph
        graph_path = self.cache_dir / "reference_graph.pkl"
        with open(graph_path, 'wb') as f:
            pickle.dump(dict(self.reference_graph), f)
        logger.info(f"Saved reference graph to {graph_path}")

        # Save configuration for reference
        config = {
            'model_name': self.encoder.__class__.__name__,
            'embedding_dimension': self.embedding_dimension,
            'indices': list(self.indices.keys()),
            'total_embeddings': sum(len(v) for v in self.embedding_metadata.values()),
            'created_at': datetime.now().isoformat()
        }
        config_path = self.cache_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved configuration to {config_path}")

    def load_indices(self):
        """
        Load pre-built indices from disk.

        LOADING STRATEGY:
        ----------------
        We check for index files and load them if they exist.
        This is much faster than rebuilding (seconds vs hours).
        """
        logger.info("Loading indices from disk...")

        # Load configuration
        config_path = self.cache_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration: {config}")

        # Load FAISS indices
        for index_name in ['code', 'business', 'documentation', 'database', 'tests']:
            index_path = self.cache_dir / f"{index_name}.index"
            if index_path.exists():
                self.indices[index_name] = faiss.read_index(str(index_path))
                logger.info(f"Loaded {index_name} index from {index_path}")

        # Load metadata
        metadata_path = self.cache_dir / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.embedding_metadata = defaultdict(dict, pickle.load(f))
            logger.info(f"Loaded metadata from {metadata_path}")

        # Load reference graph
        graph_path = self.cache_dir / "reference_graph.pkl"
        if graph_path.exists():
            with open(graph_path, 'rb') as f:
                self.reference_graph = defaultdict(set, pickle.load(f))
            logger.info(f"Loaded reference graph from {graph_path}")

        logger.info("Indices loaded successfully")

    def _indices_exist(self) -> bool:
        """Check if indices already exist on disk."""
        required_files = ['code.index', 'metadata.pkl', 'config.json']
        return all((self.cache_dir / f).exists() for f in required_files)
```

This detailed implementation explains:

1. **Every design decision** - Why we use multiple indices, why we cache embeddings, etc.
2. **Every algorithm choice** - Why cosine similarity, why FAISS, why these specific patterns
3. **Business context** - How IFS-specific patterns are handled
4. **Performance considerations** - Memory management, caching strategies, batch processing
5. **AI agent optimization** - What information AI agents need and why

The code is extensively documented so that any AI agent (including Sonnet 4) can understand not just the implementation but the reasoning behind every choice. This will help it make better decisions when using or modifying the search system.
