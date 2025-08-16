"""Tantivy-based indexer for IFS Cloud files with enhanced caching."""

import os
import hashlib
import logging
import json
import asyncio
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, Counter

import tantivy
import aiofiles
from pydantic import BaseModel

from .parsers import IFSFileParser
from .metadata_extractor import MetadataManager, MetadataExtract
from .enhanced_search import (
    MetadataEnhancedSearchEngine,
    SearchContext,
    SearchResult as EnhancedSearchResult,
)


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
    pagerank_score: float
    modified_time: datetime
    hash: str  # Unique content hash for React keys
    module: Optional[str] = None
    logical_unit: Optional[str] = None
    entity_name: Optional[str] = None
    component: Optional[str] = None
    pages: List[str] = []
    lists: List[str] = []
    groups: List[str] = []
    entitysets: List[str] = []
    iconsets: List[str] = []
    trees: List[str] = []
    navigators: List[str] = []
    contexts: List[str] = []

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class IFSCloudIndexer:
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
        ".enumeration",
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

        # GUI to backend navigation mappings
        self._gui_mappings: Dict[str, List[Dict[str, str]]] = {}
        self._load_gui_navigation_mappings()

        # Initialize metadata system
        metadata_dir = self.index_path / "metadata"
        self._metadata_manager = MetadataManager(metadata_dir)
        self._enhanced_search_engine = (
            None  # Will be initialized when metadata is available
        )
        self._current_ifs_version: Optional[str] = None

        self._schema = self._create_schema()
        self._index = self._create_or_open_index(create_new)
        self._writer = None  # Will be created on demand
        self._parser = IFSFileParser()

    def _load_gui_navigation_mappings(self) -> None:
        """Load GUI navigation mappings from generated data file."""
        gui_mappings_file = (
            Path(__file__).parent.parent.parent
            / "data"
            / "gui_navigation_mappings.json"
        )

        if not gui_mappings_file.exists():
            logger.warning(
                f"GUI navigation mappings file not found: {gui_mappings_file}"
            )
            self._gui_mappings = {}
            return

        try:
            import json

            with open(gui_mappings_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Process new format mappings
            self._gui_mappings = {
                "gui_to_entity": data.get("gui_to_entity", {}),
                "entity_synonyms": data.get("entity_synonyms", {}),
                "gui_to_projection": data.get("gui_to_projection", {}),
            }

            logger.info(
                f"Loaded GUI mappings: {len(self._gui_mappings['gui_to_entity'])} GUI labels, {len(self._gui_mappings['entity_synonyms'])} entity synonym groups"
            )

        except Exception as e:
            logger.error(f"Failed to load GUI mappings: {e}")
            self._gui_mappings = {}

        except Exception as e:
            logger.error(f"Failed to load GUI navigation mappings: {e}")
            self._gui_mappings = {}

        logger.info(f"Initialized Tantivy indexer at {self.index_path}")
        logger.info(f"Cache contains {len(self._file_cache)} file entries")

    def _get_writer(self):
        """Get or create a writer instance."""
        if self._writer is None:
            try:
                self._writer = self._index.writer(heap_size=50_000_000)  # 50MB heap
            except Exception as e:
                logger.error(f"Failed to acquire writer: {e}")
                raise
        return self._writer

    def _close_writer(self):
        """Close the current writer instance and clean up lock files."""
        if self._writer is not None:
            try:
                self._writer.rollback()
            except:
                pass  # Ignore rollback errors

            # Explicitly delete the writer to release lock
            del self._writer
            self._writer = None

            # Force garbage collection to ensure cleanup
            import gc

            gc.collect()

            # Longer delay to ensure lock file is fully released
            import time

            time.sleep(1.0)  # Increased from 0.5 to 1.0 for better lock release

    def _commit_writer(self):
        """Commit and close the writer safely."""
        if self._writer is not None:
            try:
                self._writer.commit()
                self._writer = None
                return True
            except Exception as e:
                logger.error(f"Error committing writer: {e}")
                self._close_writer()
                return False
        return False

    def _calculate_logical_unit_rankings(self, all_results) -> Dict[str, float]:
        """Calculate importance rankings for logical units using page rank strategy."""
        logical_unit_stats = {}

        # First pass: collect basic stats
        for result in all_results:
            if not result.logical_unit:
                continue

            lu = result.logical_unit
            if lu not in logical_unit_stats:
                logical_unit_stats[lu] = {
                    "file_count": 0,
                    "entity_count": 0,
                    "file_types": set(),
                    "modules": set(),
                    "total_entities": 0,
                    "complexity_score": 0,
                }

            stats = logical_unit_stats[lu]
            stats["file_count"] += 1
            stats["entity_count"] += len(result.entities)
            stats["total_entities"] += len(result.entities)
            stats["file_types"].add(result.type)
            if result.module:
                stats["modules"].add(result.module)

            # Add complexity based on entity types and file types
            if result.entities:
                unique_types = set(entity.get("type", "") for entity in result.entities)
                stats["complexity_score"] += len(unique_types)

        # Second pass: calculate ranking scores
        rankings = {}
        max_files = max(
            (stats["file_count"] for stats in logical_unit_stats.values()), default=1
        )
        max_entities = max(
            (stats["total_entities"] for stats in logical_unit_stats.values()),
            default=1,
        )
        max_complexity = max(
            (stats["complexity_score"] for stats in logical_unit_stats.values()),
            default=1,
        )

        for lu, stats in logical_unit_stats.items():
            # Normalize factors (0-1 scale)
            file_factor = stats["file_count"] / max_files
            entity_factor = stats["total_entities"] / max_entities
            diversity_factor = len(stats["file_types"]) / 5.0  # Assume max 5 types
            module_factor = (
                len(stats["modules"]) / 3.0
            )  # Multi-module LUs are more important
            complexity_factor = (
                stats["complexity_score"] / max_complexity if max_complexity > 0 else 0
            )

            # Weighted importance score
            # Higher weights for file count and entity count as they indicate usage/importance
            score = (
                file_factor * 0.3  # Number of files using this LU
                + entity_factor * 0.25  # Total entities across all files
                + diversity_factor * 0.2  # File type diversity
                + module_factor * 0.15  # Cross-module usage
                + complexity_factor * 0.1  # Entity type complexity
            )

            rankings[lu] = score

        return rankings

    def _calculate_module_rankings(self, all_results) -> Dict[str, float]:
        """Calculate importance rankings for modules using page rank strategy."""
        module_stats = {}

        # First pass: collect basic stats
        for result in all_results:
            if not result.module:
                continue

            module = result.module
            if module not in module_stats:
                module_stats[module] = {
                    "file_count": 0,
                    "entity_count": 0,
                    "file_types": set(),
                    "logical_units": set(),
                    "total_entities": 0,
                }

            stats = module_stats[module]
            stats["file_count"] += 1
            stats["entity_count"] += len(result.entities)
            stats["total_entities"] += len(result.entities)
            stats["file_types"].add(result.type)
            if result.logical_unit:
                stats["logical_units"].add(result.logical_unit)

        # Second pass: calculate ranking scores
        rankings = {}
        max_files = max(
            (stats["file_count"] for stats in module_stats.values()), default=1
        )
        max_entities = max(
            (stats["total_entities"] for stats in module_stats.values()), default=1
        )
        max_logical_units = max(
            (len(stats["logical_units"]) for stats in module_stats.values()), default=1
        )

        for module, stats in module_stats.items():
            # Normalize factors (0-1 scale)
            file_factor = stats["file_count"] / max_files
            entity_factor = stats["total_entities"] / max_entities
            diversity_factor = len(stats["file_types"]) / 5.0  # File type diversity
            lu_factor = (
                len(stats["logical_units"]) / max_logical_units
            )  # Logical unit diversity

            # Weighted importance score
            score = (
                file_factor * 0.35  # Number of files in this module
                + entity_factor * 0.25  # Total entities in module
                + diversity_factor * 0.2  # File type diversity
                + lu_factor * 0.2  # Logical unit diversity
            )

            rankings[module] = score

        return rankings

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            searcher = self._index.searcher()

            # Get total document count
            total_docs = searcher.num_docs

            # Get cache stats
            cache_files = len(self._file_cache.get("files", {}))

            # Get all documents by searching for documents that have a path field (which all should have)
            try:
                # Use a term query that will match all documents by searching for any path value
                all_query = tantivy.Query.all_query()
                all_docs = searcher.search(all_query, limit=total_docs or 10000)

                total_entities = 0
                file_types = {}
                modules = set()
                logical_units = set()
                all_results = []

                for _score, doc_address in all_docs.hits:
                    try:
                        doc = searcher.doc(doc_address)

                        # Debug: log the actual document type and structure
                        logger.debug(
                            f"Document type: {type(doc)}, doc: {str(doc)[:100]}..."
                        )

                        # Ensure doc is a proper document object
                        if not hasattr(doc, "get_first"):
                            logger.warning(
                                f"Document at address {doc_address} is not a proper Tantivy document: {type(doc)}"
                            )
                            logger.warning(f"Document content: {str(doc)[:200]}...")
                            continue

                        # Create a SearchResult-like object with safe field access
                        path = doc.get_first("path")
                        if not path:
                            logger.debug(
                                f"Document at address {doc_address} has no path field"
                            )
                            continue  # Skip documents without path

                        name = doc.get_first("name") or ""
                        type_val = doc.get_first("type") or ""
                        module_val = doc.get_first("module") or ""
                        logical_unit_val = doc.get_first("logical_unit") or ""
                        entities = doc.get_first("entities")
                        entities_list = entities.split() if entities else []

                        # Get additional fields with safe defaults
                        complexity_score = float(
                            doc.get_first("complexity_score") or 0.0
                        )
                        pagerank_score = float(doc.get_first("pagerank_score") or 0.0)

                    except Exception as doc_error:
                        logger.error(
                            f"Error processing document at address {doc_address}: {doc_error}"
                        )
                        logger.error(
                            f"Document type: {type(doc) if 'doc' in locals() else 'Not retrieved'}"
                        )
                        continue

                    # Count entities
                    total_entities += len(entities_list)

                    # Count by file type
                    if type_val:
                        file_types[type_val] = file_types.get(type_val, 0) + 1

                    # Add to modules and logical units sets
                    if module_val:
                        modules.add(module_val)
                    if logical_unit_val:
                        logical_units.add(logical_unit_val)

                    # Create a minimal SearchResult for ranking calculations
                    from types import SimpleNamespace

                    result = SimpleNamespace()
                    result.path = path
                    result.name = name
                    result.type = type_val
                    result.module = module_val
                    result.logical_unit = logical_unit_val
                    result.entities = entities_list
                    result.complexity_score = complexity_score
                    result.pagerank_score = pagerank_score
                    all_results.append(result)

                # Calculate logical unit and module rankings
                lu_rankings = self._calculate_logical_unit_rankings(all_results)
                module_rankings = self._calculate_module_rankings(all_results)

                # Sort by importance score (descending)
                ranked_logical_units = sorted(
                    logical_units, key=lambda lu: lu_rankings.get(lu, 0), reverse=True
                )

                ranked_modules = sorted(
                    modules,
                    key=lambda module: module_rankings.get(module, 0),
                    reverse=True,
                )

            except Exception as e:
                logger.error(f"Error processing documents for stats: {e}")
                # Fallback if search fails
                total_entities = 0
                file_types = {}
                modules = set()
                logical_units = set()
                ranked_logical_units = []
                ranked_modules = []

            return {
                "total_files": total_docs,
                "total_entities": total_entities,
                "cached_files": cache_files,
                "supported_extensions": list(self.SUPPORTED_EXTENSIONS),
                "file_types": file_types,
                "modules": ranked_modules,  # Now ranked by importance
                "logical_units": ranked_logical_units,  # Now ranked by importance
                "index_path": str(self.index_path),
                "cache_enabled": True,
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_files": 0,
                "total_entities": 0,
                "cached_files": 0,
                "supported_extensions": list(self.SUPPORTED_EXTENSIONS),
                "file_types": {},
                "modules": [],
                "logical_units": [],
                "index_path": str(self.index_path),
                "cache_enabled": True,
                "error": str(e),
            }

    def cleanup(self):
        """Clean up resources."""
        self._close_writer()
        logger.info("Indexer resources cleaned up")

    def _parse_query_terms(self, query: str) -> List[str]:
        """Parse a query into meaningful terms for multi-term matching with synonym mapping.

        Args:
            query: The search query (e.g., "Project Scope and Schedule", "ExpenseSheet authorization")

        Returns:
            List of significant terms with synonyms expanded (e.g., ["Project", "Scope", "Schedule"])
        """
        import re

        # Enhanced entity name synonym mappings with GUI mappings and action patterns
        entity_synonyms = {
            "expensesheet": ["expenseheader", "expense_header", "trvexp"],
            "expense_sheet": ["expenseheader", "expense_header", "trvexp"],
            "expenseheader": ["expensesheet", "expense_sheet"],
            "expense_header": ["expensesheet", "expense_sheet"],
            # Project domain patterns
            "project": ["project", "proj"],
            "transaction": ["transaction", "trans"],
            "posting": ["posting", "post"],
            # Activity domain patterns
            "activity": ["activity", "act"],
            "creation": ["creation", "create", "new", "add"],
            # Detail/line patterns
            "lines": ["detail", "line", "item"],
            "detail": ["lines", "line", "item"],
            "sheet": ["header", "main", "master"],
            # Payment domain patterns
            "payment": ["payment", "pay", "payable"],
            "modification": ["modification", "modify", "change", "update"],
            "authorization": ["authorization", "authorize", "approval", "approve"],
            # Item/Part domain patterns
            "item": ["part", "inventory", "catalog"],
            "part": ["item", "inventory", "catalog"],
            "global": ["catalog", "master", "central"],
            "company": ["org", "organization"],
            "stocking": ["inventory", "stock"],
            "purchase": ["purchasing", "procurement", "buy"],
            "rules": ["rule", "policy", "constraint"],
            # Common mappings
            "customerorder": ["customer_order", "custord"],
            "customer_order": ["customerorder", "custord"],
            "purchaseorder": ["purchase_order", "purord"],
            "purchase_order": ["purchaseorder", "purord"],
        }

        # Add GUI mappings if available
        if hasattr(self, "_gui_mappings") and self._gui_mappings:
            # Add GUI-to-entity mappings
            for gui_label, entities in self._gui_mappings.get(
                "gui_to_entity", {}
            ).items():
                # Add GUI label as synonym for all its entities
                for entity in entities:
                    entity_lower = entity.lower()
                    if entity_lower not in entity_synonyms:
                        entity_synonyms[entity_lower] = []
                    entity_synonyms[entity_lower].extend(
                        [gui_label] + [e.lower() for e in entities if e != entity]
                    )

            # Add entity synonym mappings from GUI data
            for entity_key, synonyms in self._gui_mappings.get(
                "entity_synonyms", {}
            ).items():
                entity_synonyms[entity_key] = list(
                    set(
                        entity_synonyms.get(entity_key, [])
                        + [s.lower() for s in synonyms]
                    )
                )

        # Action-to-entity mappings for compound patterns (including GUI mappings)
        action_entity_patterns = {
            # "entity action" -> prioritize EntityAction or Entity files
            ("project", "transaction"): ["ProjectTransaction", "ProjectTrans"],
            ("project", "posting"): ["ProjectTransPosting", "ProjectPosting"],
            ("employee", "validation"): [
                "CompanyPerson",
                "Employee",
            ],  # GUI: "Employee File" -> CompanyPerson.plsql
            ("employee", "name"): ["CompanyPerson", "Employee"],
            ("activity", "creation"): ["Activity", "ActivityCreation"],
            ("expense", "lines"): ["ExpenseDetail", "ExpenseLine"],
            ("expense", "detail"): ["ExpenseDetail", "ExpenseHeader"],
            ("payment", "authorization"): ["PaymentAddress", "Payment"],
            ("item", "creation"): ["PartCatalog", "ItemCatalog"],
            ("item", "stocking"): ["InventoryPart", "InventoryItem"],
            ("item", "purchase"): ["PurchasePart", "PurchaseItem"],
        }

        # Remove common stop words
        stop_words = {
            "and",
            "or",
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "can",
            "may",
            "might",
            "must",
            "per",
            "global",
        }

        # Split on whitespace and common punctuation
        terms = re.split(r"[\s\-_.,;:!?()]+", query.lower())

        # Filter out empty strings, stop words, and very short terms
        significant_terms = [
            term.strip()
            for term in terms
            if term.strip()
            and len(term.strip()) >= 2
            and term.strip() not in stop_words
        ]

        # Add synonyms for entity names from static mappings
        expanded_terms = list(significant_terms)  # Start with original terms
        for term in significant_terms:
            if term in entity_synonyms:
                expanded_terms.extend(entity_synonyms[term])

        # Add GUI navigation mappings (loaded from generated data)
        for term in significant_terms:
            if term in self._gui_mappings:
                mapping = self._gui_mappings[term]
                entity_name = mapping.get("entity_name", "")
                if entity_name:
                    expanded_terms.append(entity_name.lower())
                    # Add camelCase to words conversion
                    expanded_terms.append(self._camel_to_words(entity_name).lower())

                # Add additional synonyms from GUI mapping
                gui_synonyms = mapping.get("synonyms", [])
                expanded_terms.extend(gui_synonyms)

        # Add compound entity names based on action patterns
        query_terms = [
            t for t in significant_terms if len(t) >= 3
        ]  # Focus on meaningful terms
        for i in range(len(query_terms) - 1):
            term_pair = (query_terms[i], query_terms[i + 1])
            if term_pair in action_entity_patterns:
                expanded_terms.extend(action_entity_patterns[term_pair])

        # Also check reverse order patterns
        for i in range(len(query_terms) - 1):
            term_pair = (query_terms[i + 1], query_terms[i])  # Reversed
            if term_pair in action_entity_patterns:
                expanded_terms.extend(action_entity_patterns[term_pair])

        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in expanded_terms:
            if term not in seen:
                unique_terms.append(term)
                seen.add(term)

        # Also include the original query as a single term for exact matching
        if query.strip():
            unique_terms.insert(0, query.strip())

        return unique_terms

    def _camel_to_words(self, camel_str: str) -> str:
        """Convert CamelCase to space-separated words.

        Args:
            camel_str: CamelCase string

        Returns:
            Space-separated words
        """
        import re

        return re.sub(r"(?<!^)(?=[A-Z])", " ", camel_str)

    def _check_compound_entity_match(
        self, filename: str, query_terms: List[str], original_query: str
    ) -> float:
        """Check for compound entity name matches based on action patterns.

        Args:
            filename: The filename to check against
            query_terms: Parsed query terms
            original_query: Original search query

        Returns:
            Bonus score for compound entity matches (0 if no match)
        """
        filename_lower = filename.lower()
        filename_base = (
            filename_lower.rsplit(".", 1)[0]
            if "." in filename_lower
            else filename_lower
        )

        # High-priority compound patterns with their expected filename matches
        compound_patterns = {
            # Project domain
            ("project", "transaction"): ["projecttransaction"],
            ("project", "posting"): ["projecttransposting", "projectposting"],
            ("transaction", "project"): ["projecttransaction"],
            ("posting", "project"): ["projecttransposting", "projectposting"],
            # Employee domain (GUI mappings included)
            ("employee", "validation"): [
                "companyperson",
                "employee",
            ],  # "Employee File" -> CompanyPerson.plsql
            ("employee", "name"): ["companyperson", "employee"],
            ("validation", "employee"): ["companyperson", "employee"],
            ("name", "employee"): ["companyperson", "employee"],
            ("name", "validation"): [
                "companyperson",
                "employee",
            ],  # "name validation" should find CompanyPerson.plsql
            # Activity domain
            ("activity", "creation"): ["activity", "activitycreation"],
            ("creation", "activity"): ["activity", "activitycreation"],
            # Expense domain - lines/details
            ("expense", "lines"): ["expensedetail", "expenseline"],
            ("expense", "detail"): ["expensedetail", "expenseheader"],
            ("lines", "expense"): ["expensedetail", "expenseline"],
            ("detail", "expense"): ["expensedetail", "expenseheader"],
            ("sheet", "lines"): ["expensedetail"],
            # Payment domain
            ("payment", "authorization"): ["paymentaddress", "payment"],
            ("payment", "modification"): ["paymentaddress", "payment"],
            ("authorization", "payment"): ["paymentaddress", "payment"],
            ("modification", "payment"): ["paymentaddress", "payment"],
            # Item/Part domain
            ("item", "creation"): ["partcatalog", "itemcatalog"],
            ("item", "stocking"): ["inventorypart", "inventoryitem"],
            ("item", "purchase"): ["purchasepart", "purchaseitem"],
            ("part", "creation"): ["partcatalog"],
            ("part", "stocking"): ["inventorypart"],
            ("part", "purchase"): ["purchasepart"],
            ("stocking", "item"): ["inventorypart", "inventoryitem"],
            ("purchase", "item"): ["purchasepart", "purchaseitem"],
            ("creation", "item"): ["partcatalog", "itemcatalog"],
        }

        # Check for compound matches
        max_bonus = 0.0
        query_terms_clean = [
            t for t in query_terms if len(t) >= 3
        ]  # Focus on meaningful terms

        for i in range(len(query_terms_clean) - 1):
            # Check adjacent term pairs
            term_pair = (query_terms_clean[i], query_terms_clean[i + 1])

            if term_pair in compound_patterns:
                expected_filenames = compound_patterns[term_pair]

                for expected in expected_filenames:
                    if expected in filename_base:
                        # Strong bonus for compound entity match - higher for exact matches
                        if filename_base == expected:
                            max_bonus = max(max_bonus, 90.0)  # Near-exact match
                        elif filename_base.startswith(expected):
                            max_bonus = max(max_bonus, 75.0)  # Prefix match
                        elif expected in filename_base:
                            max_bonus = max(max_bonus, 60.0)  # Contains match

        # Also check reverse patterns (e.g., "posting project")
        for i in range(len(query_terms_clean) - 1):
            term_pair = (query_terms_clean[i + 1], query_terms_clean[i])  # Reversed

            if term_pair in compound_patterns:
                expected_filenames = compound_patterns[term_pair]

                for expected in expected_filenames:
                    if expected in filename_base:
                        # Slightly lower bonus for reversed patterns
                        if filename_base == expected:
                            max_bonus = max(max_bonus, 85.0)
                        elif filename_base.startswith(expected):
                            max_bonus = max(max_bonus, 70.0)
                        elif expected in filename_base:
                            max_bonus = max(max_bonus, 55.0)

        return max_bonus

    def _check_simple_entity_priority(
        self, filename: str, query: str, query_terms: List[str]
    ) -> float:
        """Check for simple entity name priority boosts for property/validation queries.

        Args:
            filename: The filename to check
            query: Original search query
            query_terms: Parsed query terms

        Returns:
            Bonus score for simple entity name matches (0 if no match)
        """
        filename_lower = filename.lower()
        filename_base = (
            filename_lower.rsplit(".", 1)[0]
            if "." in filename_lower
            else filename_lower
        )
        query_lower = query.lower()

        # Property/validation query patterns that should prioritize simple entity names
        property_patterns = [
            "name",
            "validation",
            "validate",
            "property",
            "attribute",
            "field",
        ]
        is_property_query = any(pattern in query_lower for pattern in property_patterns)

        if not is_property_query:
            return 0.0

        # Entity name mappings for property queries (including GUI mappings)
        entity_mappings = {
            "employee": [
                "companyperson",
                "employee",
            ],  # GUI: "Employee File" -> CompanyPerson.plsql
            "activity": ["activity"],
            "project": ["project"],
            "customer": ["customer"],
            "item": ["partcatalog", "part"],
            "part": ["partcatalog", "part"],
        }

        bonus = 0.0

        # Check if any query terms match entity patterns and filename matches expected entity files
        for term in query_terms:
            if term in entity_mappings:
                expected_names = entity_mappings[term]

                for expected in expected_names:
                    if expected in filename_base:
                        # Boost simple entity names for property/validation queries
                        if filename_base == expected:
                            bonus = max(
                                bonus, 85.0
                            )  # Strong boost for exact entity match
                        elif filename_base.startswith(expected):
                            bonus = max(bonus, 70.0)  # Good boost for prefix match
                        elif expected in filename_base:
                            bonus = max(
                                bonus, 55.0
                            )  # Moderate boost for contains match

        return bonus

    def _calculate_term_matches(
        self, filename: str, entity_name: str, query_terms: List[str]
    ) -> Dict[str, Any]:
        """Calculate how well query terms match against filename and entity name.

        Args:
            filename: The filename to check
            entity_name: The entity name to check
            query_terms: List of query terms to match

        Returns:
            Dictionary with match statistics
        """
        if not query_terms or len(query_terms) <= 1:
            # Single term or no terms - fallback to simple matching
            return {
                "all_terms_in_filename": False,
                "partial_terms_in_filename": 0.0,
                "any_term_in_filename": False,
                "all_terms_in_entity": False,
                "partial_terms_in_entity": 0.0,
                "any_term_in_entity": False,
            }

        # Skip the first term (full query) and work with individual terms
        individual_terms = query_terms[1:] if len(query_terms) > 1 else []

        if not individual_terms:
            return {
                "all_terms_in_filename": False,
                "partial_terms_in_filename": 0.0,
                "any_term_in_filename": False,
                "all_terms_in_entity": False,
                "partial_terms_in_entity": 0.0,
                "any_term_in_entity": False,
            }

        filename_lower = filename.lower()
        entity_lower = (entity_name or "").lower()

        # Check filename matches
        filename_matches = 0
        for term in individual_terms:
            if term in filename_lower:
                filename_matches += 1

        # Check entity name matches
        entity_matches = 0
        for term in individual_terms:
            if term in entity_lower:
                entity_matches += 1

        total_terms = len(individual_terms)

        return {
            "all_terms_in_filename": filename_matches == total_terms,
            "partial_terms_in_filename": (
                filename_matches / total_terms if total_terms > 0 else 0.0
            ),
            "any_term_in_filename": filename_matches > 0,
            "all_terms_in_entity": entity_matches == total_terms,
            "partial_terms_in_entity": (
                entity_matches / total_terms if total_terms > 0 else 0.0
            ),
            "any_term_in_entity": entity_matches > 0,
        }

    def _get_file_type_modifier(
        self, file_type: str, query: str = "", line_count: int = 0
    ) -> float:
        """Get file type priority modifier with context-aware boosting.

        Args:
            file_type: The file type/extension
            query: The search query for context-aware prioritization
            line_count: Number of lines in the file for business logic boosting

        Returns:
            Modifier value (positive for boost, negative for penalty)
        """
        modifier = 0.0

        # Base file type priorities
        if file_type.endswith(".plsql"):
            modifier = 20.0  # Significantly boosted - core business logic
            # Additional boost for large .plsql files (likely main business logic)
            if line_count > 1000:
                modifier += 15.0  # Major boost for substantial business logic files
            elif line_count > 500:
                modifier += 8.0  # Moderate boost for medium-sized business logic
        elif file_type.endswith(".projection"):
            modifier = 15.0  # High priority for UI integration
        elif file_type.endswith(".views"):
            modifier = 12.0  # High priority for data views
        elif file_type.endswith(".client"):
            modifier = 10.0  # Good priority for client-side logic
        elif file_type.endswith(".entity"):
            modifier = 8.0  # Good priority for data models
        elif file_type.endswith(".fragment"):
            modifier = 6.0  # Medium priority for UI fragments
        elif file_type.endswith(".storage"):
            modifier = -5.0  # Minor penalty for storage files

        # Context-aware boosting based on query content
        query_lower = query.lower()

        # Authorization/approval workflow queries strongly favor .plsql files
        authorization_keywords = [
            "authorization",
            "approve",
            "reject",
            "authorize",
            "workflow",
            "business logic",
        ]
        if any(keyword in query_lower for keyword in authorization_keywords):
            if file_type.endswith(".plsql"):
                modifier += (
                    25.0  # Massive boost for business logic in authorization queries
                )
            elif file_type.endswith(".views"):
                modifier += 12.0  # Good boost for authorization views
            elif file_type.endswith(".projection"):
                modifier += 8.0  # Moderate boost for authorization UI

        # Entity-specific queries should prioritize entity files and main business logic
        entity_keywords = ["entity", "header", "detail", "line", "master"]
        if any(keyword in query_lower for keyword in entity_keywords):
            if file_type.endswith(".entity"):
                modifier += 10.0  # Strong boost for entity definitions
            elif file_type.endswith(".plsql"):
                modifier += 15.0  # Very strong boost for entity business logic

        # UI/frontend queries favor projections and clients
        ui_keywords = ["handling", "management", "client", "ui", "interface", "screen"]
        if any(keyword in query_lower for keyword in ui_keywords):
            if file_type.endswith(".projection"):
                modifier += 18.0  # Strong boost for UI projections
            elif file_type.endswith(".client"):
                modifier += 15.0  # Strong boost for client files

        return modifier

    def _calculate_content_bonus(
        self, doc, query: str, file_type: str, line_count: int
    ) -> float:
        """Calculate bonus points based on content analysis for business logic detection.

        Args:
            doc: Tantivy document
            query: Search query
            file_type: File extension/type
            line_count: Number of lines in the file

        Returns:
            Content bonus score
        """
        bonus = 0.0
        query_lower = query.lower()

        # Get document functions and content for analysis
        functions = doc.get_first("functions") or ""
        content_preview = doc.get_first("content_preview") or ""

        # Authorization/workflow method detection
        authorization_methods = [
            "authorize",
            "approve",
            "reject",
            "validate",
            "check",
            "verify",
            "workflow",
            "state",
            "transition",
            "notify",
            "send",
        ]

        # Check if file contains authorization-related methods
        functions_lower = functions.lower()
        content_lower = content_preview.lower()

        authorization_method_count = 0
        for method in authorization_methods:
            if method in functions_lower or method in content_lower:
                authorization_method_count += 1

        # Bonus for authorization-related queries finding authorization methods
        authorization_query_keywords = [
            "authorization",
            "approve",
            "reject",
            "authorize",
            "workflow",
        ]
        is_authorization_query = any(
            keyword in query_lower for keyword in authorization_query_keywords
        )

        if is_authorization_query and authorization_method_count > 0:
            # Significant bonus for authorization queries finding files with authorization methods
            bonus += min(authorization_method_count * 8.0, 40.0)  # Cap at 40 points

            # Extra bonus for .plsql files with many authorization methods (main business logic)
            if file_type.endswith(".plsql") and authorization_method_count >= 3:
                bonus += 25.0

            # Extra bonus for large files likely containing main business logic
            if line_count > 1000:
                bonus += 15.0

        # Entity-specific query detection
        entity_keywords = ["header", "detail", "line", "master", "entity"]
        is_entity_query = any(keyword in query_lower for keyword in entity_keywords)

        if is_entity_query:
            # Boost files that likely contain the main entity definition
            if file_type.endswith(".plsql") and line_count > 500:
                bonus += 20.0  # Large plsql files are likely main business logic
            elif file_type.endswith(".entity"):
                bonus += 15.0  # Entity files for entity queries

        # Business logic file detection
        business_logic_indicators = [
            "function",
            "procedure",
            "package",
            "begin",
            "end",
            "exception",
        ]
        business_logic_count = sum(
            1
            for indicator in business_logic_indicators
            if indicator in content_lower or indicator in functions_lower
        )

        # General bonus for substantial business logic content
        if business_logic_count >= 3 and line_count > 200:
            bonus += min(
                business_logic_count * 2.0, 15.0
            )  # Up to 15 points for business logic

        # Validation query pattern detection
        validation_keywords = ["validation", "validate", "verify", "check", "name"]
        is_validation_query = any(
            keyword in query_lower for keyword in validation_keywords
        )

        if is_validation_query:
            validation_methods = [
                "validate",
                "verify",
                "check",
                "error",
                "exception",
                "constraint",
            ]
            validation_count = sum(
                1
                for method in validation_methods
                if method in functions_lower or method in content_lower
            )

            if validation_count > 0:
                bonus += min(
                    validation_count * 5.0, 25.0
                )  # Up to 25 points for validation

                # Extra bonus for main entity files doing validation
                if file_type.endswith(".plsql") and line_count > 1000:
                    bonus += (
                        15.0  # Large plsql files likely contain main validation logic
                    )

        # Rules and creation pattern detection
        rules_keywords = [
            "rules",
            "creation",
            "stocking",
            "purchase",
            "global",
            "company",
        ]
        is_rules_query = any(keyword in query_lower for keyword in rules_keywords)

        if is_rules_query:
            rules_methods = [
                "rule",
                "create",
                "new",
                "add",
                "insert",
                "policy",
                "constraint",
            ]
            rules_count = sum(
                1
                for method in rules_methods
                if method in functions_lower or method in content_lower
            )

            if rules_count > 0:
                bonus += min(
                    rules_count * 4.0, 20.0
                )  # Up to 20 points for rules/creation logic

                # Boost catalog/master files for global rules
                name_field = doc.get_first("name") or ""
                if (
                    "global" in query_lower or "catalog" in query_lower
                ) and "catalog" in name_field.lower():
                    bonus += 20.0

        return bonus

    def _calculate_module_context_bonus(self, doc, query: str) -> float:
        """Calculate bonus based on module context matching with query domain.

        Args:
            doc: Tantivy document
            query: Search query

        Returns:
            Module context bonus score
        """
        bonus = 0.0
        query_lower = query.lower()
        module = doc.get_first("module") or ""
        module_lower = module.lower()

        # Module domain mappings based on common business terminology
        domain_module_mappings = {
            # Travel & Expense domain
            "expense": "trvexp",
            "travel": "trvexp",
            "authorization": "trvexp",  # Expense authorization is common
            "receipt": "trvexp",
            "mileage": "trvexp",
            # Order Management domain
            "customer": "order",
            "sales": "order",
            "quote": "order",
            "delivery": "order",
            # Purchase domain
            "purchase": "purch",
            "supplier": "purch",
            "vendor": "purch",
            "requisition": "purch",
            # Financial domain
            "invoice": "accrul",
            "payment": "accrul",
            "accounting": "accrul",
            "ledger": "accrul",
            # Inventory domain
            "inventory": "invent",
            "parts": "invent",
            "warehouse": "invent",
            "stock": "invent",
        }

        # Check if query terms match expected module domains
        for domain_term, expected_module in domain_module_mappings.items():
            if domain_term in query_lower and expected_module in module_lower:
                bonus += 20.0  # Strong bonus for domain-module alignment
                break  # Only apply one domain bonus

        # Specific high-value module context bonuses
        if "expense" in query_lower and "trvexp" in module_lower:
            bonus += (
                25.0  # Very strong bonus for expense queries in travel expense module
            )

        if "authorization" in query_lower and "trvexp" in module_lower:
            bonus += 30.0  # Massive bonus - authorization queries often relate to expense approval

        if "customer" in query_lower and "order" in module_lower:
            bonus += 25.0  # Strong bonus for customer order queries

        return bonus

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False

    def _create_schema(self) -> tantivy.Schema:
        """Create the Tantivy schema for IFS Cloud files."""
        schema_builder = tantivy.SchemaBuilder()

        # File metadata fields
        schema_builder.add_text_field("path", stored=True)
        schema_builder.add_text_field("name", stored=True)
        schema_builder.add_text_field("type", stored=True)
        schema_builder.add_integer_field("size", stored=True, indexed=True)
        schema_builder.add_date_field("modified_time", stored=True, indexed=True)

        # Content fields - make sure these are properly indexed for search
        schema_builder.add_text_field("content", stored=False, index_option="position")
        schema_builder.add_text_field("content_preview", stored=True)

        # IFS-specific fields - ensure they're indexed for fuzzy search
        schema_builder.add_text_field("entities", stored=True, index_option="position")
        schema_builder.add_text_field(
            "dependencies", stored=True, index_option="position"
        )
        schema_builder.add_text_field("functions", stored=True, index_option="position")
        schema_builder.add_text_field("imports", stored=True, index_option="position")

        # Enhanced IFS structure fields
        schema_builder.add_text_field("module", stored=True, index_option="position")
        schema_builder.add_text_field(
            "logical_unit", stored=True, index_option="position"
        )
        schema_builder.add_text_field(
            "entity_name", stored=True, index_option="position"
        )
        schema_builder.add_text_field("component", stored=True, index_option="position")

        # Frontend UI elements - these need to be searchable
        schema_builder.add_text_field("pages", stored=True, index_option="position")
        schema_builder.add_text_field("lists", stored=True, index_option="position")
        schema_builder.add_text_field("groups", stored=True, index_option="position")
        schema_builder.add_text_field(
            "entitysets", stored=True, index_option="position"
        )
        schema_builder.add_text_field("iconsets", stored=True, index_option="position")
        schema_builder.add_text_field("trees", stored=True, index_option="position")
        schema_builder.add_text_field(
            "navigators", stored=True, index_option="position"
        )
        schema_builder.add_text_field("contexts", stored=True, index_option="position")

        # Metrics fields
        schema_builder.add_float_field("complexity_score", stored=True, indexed=True)
        schema_builder.add_float_field("pagerank_score", stored=True, indexed=True)
        schema_builder.add_integer_field("line_count", stored=True, indexed=True)
        schema_builder.add_text_field("hash", stored=True)

        return schema_builder.build()

    def _load_cache_metadata(self):
        """Load cache metadata from disk."""
        if not self.cache_metadata_path.exists():
            self._file_cache = {}
            return

        try:
            with open(self.cache_metadata_path, "r") as f:
                data = json.load(f)
                self._file_cache = {
                    path: FileMetadata(**metadata) for path, metadata in data.items()
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
                    "path": metadata.path,
                    "size": metadata.size,
                    "modified_time": metadata.modified_time,
                    "hash": metadata.hash,
                    "indexed_at": metadata.indexed_at,
                }
                for path, metadata in self._file_cache.items()
            }
            with open(self.cache_metadata_path, "w") as f:
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
            return cached.size == stat.st_size and cached.modified_time == stat.st_mtime
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
                indexed_at=datetime.now().timestamp(),
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

    def calculate_pagerank_scores(self) -> Dict[str, float]:
        """Calculate PageRank scores for all entities in the index.

        This implements a simplified PageRank algorithm where:
        - Each entity gets votes from files that reference it
        - More referenced entities get higher scores
        - Entities that reference important entities also get boosted

        Returns:
            Dictionary mapping entity names to PageRank scores
        """
        if not self._index:
            return {}

        searcher = self._index.searcher()

        # Build entity reference graph
        entity_graph = defaultdict(set)  # entity -> set of entities that reference it
        entity_dependencies = defaultdict(
            set
        )  # entity -> set of entities it depends on
        all_entities = set()

        try:
            # Get all documents to build the graph
            query = tantivy.Query.all_query()
            search_results = searcher.search(query, limit=10000)  # Get all docs

            for score, doc_address in search_results.hits:
                doc = searcher.doc(doc_address)

                # Get the primary entity for this file
                entity_name = doc.get_first("entity_name")
                if entity_name:
                    all_entities.add(entity_name)

                    # Get all entities this file depends on
                    dependencies_str = doc.get_first("dependencies") or ""
                    entities_str = doc.get_first("entities") or ""

                    # Combine dependencies and entities mentioned in the file
                    dependencies = []
                    if dependencies_str:
                        dependencies.extend(dependencies_str.split())
                    if entities_str:
                        dependencies.extend(entities_str.split())

                    # Remove duplicates and self-references
                    dependencies = list(
                        set(dep for dep in dependencies if dep and dep != entity_name)
                    )

                    # Build the graph
                    for dep in dependencies:
                        if dep:
                            all_entities.add(dep)
                            entity_graph[dep].add(
                                entity_name
                            )  # dep is referenced by entity_name
                            entity_dependencies[entity_name].add(
                                dep
                            )  # entity_name depends on dep

            # Initialize PageRank scores
            num_entities = len(all_entities)
            if num_entities == 0:
                return {}

            pagerank_scores = {entity: 1.0 / num_entities for entity in all_entities}

            # PageRank parameters
            damping_factor = 0.85
            iterations = 20
            convergence_threshold = 0.001

            # Run PageRank iterations
            for iteration in range(iterations):
                new_scores = {}

                for entity in all_entities:
                    # Base score (random surfer)
                    score = (1.0 - damping_factor) / num_entities

                    # Add scores from entities that reference this entity
                    for referencing_entity in entity_graph[entity]:
                        # Get the number of entities that the referencing entity depends on
                        out_degree = len(entity_dependencies[referencing_entity])
                        if out_degree > 0:
                            score += damping_factor * (
                                pagerank_scores[referencing_entity] / out_degree
                            )
                        else:
                            # If an entity has no dependencies, distribute its score equally
                            score += damping_factor * (
                                pagerank_scores[referencing_entity] / num_entities
                            )

                    new_scores[entity] = score

                # Check for convergence
                max_change = max(
                    abs(new_scores[entity] - pagerank_scores[entity])
                    for entity in all_entities
                )

                pagerank_scores = new_scores

                if max_change < convergence_threshold:
                    logger.info(f"PageRank converged after {iteration + 1} iterations")
                    break

            # Normalize scores to 0-1 range
            if pagerank_scores:
                max_score = max(pagerank_scores.values())
                min_score = min(pagerank_scores.values())
                score_range = max_score - min_score

                if score_range > 0:
                    normalized_scores = {
                        entity: (score - min_score) / score_range
                        for entity, score in pagerank_scores.items()
                    }
                else:
                    normalized_scores = {entity: 0.5 for entity in pagerank_scores}

                # Log top entities
                top_entities = sorted(
                    normalized_scores.items(), key=lambda x: x[1], reverse=True
                )[:10]
                logger.info("Top PageRank entities:")
                for entity, score in top_entities:
                    logger.info(f"  {entity}: {score:.3f}")

                return normalized_scores

        except Exception as e:
            logger.error(f"Failed to calculate PageRank scores: {e}")

        return {}

    def update_pagerank_scores(self):
        """Update PageRank scores for all documents in the index."""
        if not self._index:
            return

        pagerank_scores = self.calculate_pagerank_scores()
        if not pagerank_scores:
            logger.warning("No PageRank scores calculated, skipping update")
            return

        # Update documents with PageRank scores
        searcher = self._index.searcher()
        writer = self._index.writer()

        try:
            query = tantivy.Query.all_query()
            search_results = searcher.search(query, limit=10000)

            updated_count = 0
            for score, doc_address in search_results.hits:
                doc = searcher.doc(doc_address)

                # Get entity name for this document
                entity_name = doc.get_first("entity_name")
                pagerank_score = 0.0

                if entity_name and entity_name in pagerank_scores:
                    pagerank_score = pagerank_scores[entity_name]

                # Create updated document with PageRank score
                doc_dict = {}
                for field_name in [
                    "path",
                    "name",
                    "type",
                    "content",
                    "content_preview",
                    "entities",
                    "dependencies",
                    "functions",
                    "imports",
                    "module",
                    "logical_unit",
                    "entity_name",
                    "component",
                    "pages",
                    "lists",
                    "groups",
                    "entitysets",
                    "iconsets",
                    "trees",
                    "navigators",
                    "contexts",
                    "hash",
                ]:
                    value = doc.get_first(field_name)
                    if value is not None:
                        doc_dict[field_name] = value

                # Add numeric fields
                for field_name in ["size", "line_count"]:
                    value = doc.get_first(field_name)
                    if value is not None:
                        doc_dict[field_name] = value

                # Add float fields
                complexity_score = doc.get_first("complexity_score")
                if complexity_score is not None:
                    doc_dict["complexity_score"] = complexity_score

                # Add the PageRank score
                doc_dict["pagerank_score"] = pagerank_score

                # Add date field
                modified_time = doc.get_first("modified_time")
                if modified_time is not None:
                    doc_dict["modified_time"] = modified_time

                # Delete old document and add updated one
                writer.delete_term(
                    tantivy.Term.from_field_text(
                        self._schema.get_field("path"), doc_dict["path"]
                    )
                )
                writer.add_document(tantivy.Document.from_dict(doc_dict))
                updated_count += 1

            writer.commit()
            logger.info(f"Updated PageRank scores for {updated_count} documents")

        except Exception as e:
            logger.error(f"Failed to update PageRank scores: {e}")

    async def index_file(
        self, file_path: Union[str, Path], force_reindex: bool = False
    ) -> bool:
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
                writer = self._get_writer()
                writer.delete_documents("path", str(file_path))
                logger.debug(f"Removed old version of {file_path}")

            # Extract IFS-specific data using enhanced parsers
            file_type = file_path.suffix
            parsed = self._parser.parse(content, file_type)

            entities = parsed.entities
            dependencies = parsed.dependencies
            functions = parsed.functions
            imports = parsed.imports

            # Extract new frontend elements (for .client and .fragment files)
            pages = parsed.pages or []
            lists = parsed.lists or []
            groups = parsed.groups or []
            entitysets = parsed.entitysets or []
            iconsets = parsed.iconsets or []
            trees = parsed.trees or []
            navigators = parsed.navigators or []
            contexts = parsed.contexts or []

            # Extract module and logical unit information from file path
            module, logical_unit = self._extract_module_info(file_path)

            # Extract primary entity name and component
            entity_name, component = self._extract_entity_info(
                content, file_type, entities
            )

            # Calculate metrics
            line_count = len(content.split("\n"))
            complexity_score = self.calculate_complexity_score(content, file_type)
            content_preview = content[:500] if content else ""

            # Create document with enhanced frontend elements
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
                "module": module,
                "logical_unit": logical_unit,
                "entity_name": entity_name,
                "component": component,
                "pages": " ".join(pages),
                "lists": " ".join(lists),
                "groups": " ".join(groups),
                "entitysets": " ".join(entitysets),
                "iconsets": " ".join(iconsets),
                "trees": " ".join(trees),
                "navigators": " ".join(navigators),
                "contexts": " ".join(contexts),
                "complexity_score": complexity_score,
                "pagerank_score": 0.0,  # Will be calculated later
                "line_count": line_count,
                "hash": file_hash,
            }

            # Add document to index - let radical retry handle failures
            try:
                writer = self._get_writer()
                writer.add_document(tantivy.Document(**doc))
                logger.debug(f"Successfully added document for {file_path}")
                success = True

            except Exception as e:
                logger.error(f"Error adding document for {file_path}: {e}")

                # Only close writer if it's a corruption or thread error
                error_msg = str(e)
                if (
                    "killed" in error_msg
                    or "worker thread" in error_msg
                    or "thread" in error_msg.lower()
                ):
                    logger.info(
                        f"Writer corruption detected, closing writer for {file_path}"
                    )
                    self._close_writer()
                else:
                    logger.info(
                        f"Non-corruption error for {file_path}, keeping writer open"
                    )

                success = False

            if not success:
                return False

            # IMPORTANT: Don't close writer here in batch processing mode
            # The writer will be closed and committed by index_directory() after each batch

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
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        force_reindex: bool = False,
        batch_size: int = 250,  # Process files in batches to avoid memory issues
    ) -> Dict[str, int]:
        """Index all supported files in a directory with intelligent caching and batch processing.

        Args:
            directory_path: Path to directory to index
            recursive: Whether to index subdirectories
            force_reindex: Force re-indexing even if files haven't changed
            batch_size: Number of files to process in each batch

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

        # Track failed files for retry after successful batches
        failed_files = []
        retry_queue = []

        # Process files in batches to avoid memory/resource exhaustion
        batch_start = 0
        while batch_start < len(files) or retry_queue:
            # Determine current batch: either from main files or retry queue
            if batch_start < len(files):
                batch = files[batch_start : batch_start + batch_size]
                batch_num = (batch_start // batch_size) + 1
                total_batches = (len(files) + batch_size - 1) // batch_size
                batch_start += batch_size
                batch_type = "regular"
            elif retry_queue:
                # Process retry batch after successful regular batch
                batch = retry_queue[:batch_size]
                retry_queue = retry_queue[batch_size:]
                batch_num = f"retry-{len(failed_files)}"
                total_batches = f"(+{len(retry_queue) + len(batch)} retries)"
                batch_type = "retry"
            else:
                break

            logger.info(
                f"Processing {batch_type} batch {batch_num}/{total_batches} ({len(batch)} files)"
            )

            # Process this batch
            batch_errors = 0
            batch_failed_files = []
            batch_success = False

            for file_path in batch:
                try:
                    # Check cache first
                    if not force_reindex and self._is_file_cached_and_current(
                        file_path
                    ):
                        stats["cached"] += 1
                        continue

                    success = await self.index_file(file_path, force_reindex)
                    if success:
                        stats["indexed"] += 1
                    else:
                        stats["skipped"] += 1
                        # Track failed file for potential retry
                        batch_failed_files.append(file_path)
                except Exception as e:
                    logger.error(f"Error indexing {file_path}: {e}")
                    stats["errors"] += 1
                    batch_errors += 1
                    batch_failed_files.append(file_path)

                    # Only close writer on corruption errors, not all errors
                    error_msg = str(e)
                    if (
                        "killed" in error_msg
                        or "worker thread" in error_msg
                        or "thread" in error_msg.lower()
                    ):
                        logger.info(
                            f"Writer corruption detected in batch, closing writer"
                        )
                        self._close_writer()

                    # If too many errors in this batch, skip to next batch
                    if batch_errors > batch_size // 2:
                        logger.warning(
                            f"Too many errors in batch {batch_num}, skipping remaining files in batch"
                        )
                        # Add remaining files in batch to failed list
                        remaining_files = batch[batch.index(file_path) + 1 :]
                        batch_failed_files.extend(remaining_files)
                        break

            # Commit this batch if we have indexed files
            if stats["indexed"] > 0:
                try:
                    if self._commit_writer():
                        logger.info(f"Batch {batch_num} committed successfully")
                        batch_success = True
                        # Reload index to make new documents searchable
                        self._index.reload()
                        self._save_cache_metadata()
                    else:
                        logger.error(f"Failed to commit batch {batch_num}")
                        stats["errors"] += 1
                except Exception as e:
                    logger.error(f"Error committing batch {batch_num}: {e}")
                    stats["errors"] += 1
                    self._close_writer()

            # RADICAL APPROACH: If this batch succeeded and we have failed files from previous batches,
            # add ALL failed files to retry queue for processing after this successful batch
            if batch_success and failed_files and batch_type == "regular":
                retry_queue.extend(failed_files)
                logger.info(
                    f"🔄 Adding {len(failed_files)} failed files to retry queue after successful batch"
                )
                failed_files = (
                    []
                )  # Clear the failed files list since they're now in retry queue

            # Add current batch failures to the failed files list
            if batch_failed_files:
                if batch_type == "retry":
                    # If retry batch failed, put files back at end of failed list
                    failed_files.extend(batch_failed_files)
                    logger.warning(
                        f"⚠️ Retry batch failed, {len(batch_failed_files)} files remain in failed queue"
                    )
                else:
                    # Regular batch failures go to failed list
                    failed_files.extend(batch_failed_files)
                    logger.info(
                        f"📝 {len(batch_failed_files)} files from batch {batch_num} added to failed list"
                    )

        # Report final failed files
        if failed_files:
            logger.warning(
                f"⚠️ {len(failed_files)} files could not be indexed after all retry attempts"
            )
            stats["final_failures"] = len(failed_files)

        logger.info(
            f"Indexing complete: {stats['indexed']} indexed, "
            f"{stats['cached']} cached, {stats['skipped']} skipped, "
            f"{stats['errors']} errors"
        )

        # Calculate PageRank scores after indexing is complete
        if stats["indexed"] > 0:
            logger.info("Calculating PageRank scores for entities...")
            try:
                self.update_pagerank_scores()
                logger.info("PageRank calculation completed")
            except Exception as e:
                logger.error(f"PageRank calculation failed: {e}")

        return stats

    def search_deduplicated(
        self,
        query: str,
        limit: int = 10,
        file_type: Optional[str] = None,
        module: Optional[str] = None,
        logical_unit: Optional[str] = None,
        min_complexity: Optional[float] = None,
        max_complexity: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search the index with deduplication to avoid duplicate results from multiple search strategies.

        This method wraps the base search method and removes duplicates based on path+hash combination.
        This is needed because the search method runs multiple strategies (exact, fuzzy, prefix) that
        can return the same documents.

        Args:
            query: Search query
            limit: Maximum number of results
            file_type: Filter by file type (optional)
            module: Filter by module (optional)
            logical_unit: Filter by logical unit (optional)
            min_complexity: Minimum complexity score (optional)
            max_complexity: Maximum complexity score (optional)

        Returns:
            List of unique search results
        """
        # Get raw results from base search method
        results = self.search(
            query=query,
            limit=limit * 3,  # Request more results to account for deduplication
            file_type=file_type,
            module=module,
            logical_unit=logical_unit,
            min_complexity=min_complexity,
            max_complexity=max_complexity,
        )

        # Deduplicate results by path+hash combination
        seen_keys = set()
        unique_results = []
        for result in results:
            key = f"{result.path}-{result.hash}"
            if key not in seen_keys:
                seen_keys.add(key)
                unique_results.append(result)
                # Stop when we have enough unique results
                if len(unique_results) >= limit:
                    break

        logger.debug(
            f"Search returned {len(results)} results, {len(unique_results)} unique after deduplication"
        )
        return unique_results

    def search(
        self,
        query: str,
        limit: int = 10,
        file_type: Optional[str] = None,
        module: Optional[str] = None,
        logical_unit: Optional[str] = None,
        min_complexity: Optional[float] = None,
        max_complexity: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search the index with multi-term query support and intelligent boosting.

        Args:
            query: Search query (supports multi-term queries like "Project Scope and Schedule")
            limit: Maximum number of results
            file_type: Filter by file type (optional)
            module: Filter by module (optional)
            logical_unit: Filter by logical unit (optional)
            min_complexity: Minimum complexity score (optional)
            max_complexity: Maximum complexity score (optional)

        Returns:
            List of search results with filename matches prioritized
        """
        logger.debug(
            f"Search called with: query='{query}', limit={limit}, file_type='{file_type}', module='{module}', logical_unit='{logical_unit}'"
        )
        searcher = self._index.searcher()

        try:
            # Parse and split query terms for better matching
            query_terms = self._parse_query_terms(query)
            logger.debug(f"Parsed query '{query}' into terms: {query_terms}")

            # Build boosted query that prioritizes filename and entity name matches
            boosted_queries = []

            # 1. Highest priority: Exact full query filename matches
            try:
                filename_query = tantivy.Query.term_query(self._schema, "name", query)
                boosted_queries.append((tantivy.Occur.Should, filename_query))
            except Exception as e:
                logger.debug(f"Failed to create exact filename query: {e}")

            # 2. High priority: Multi-term filename matching
            # For "Project Scope and Schedule" → match "ScopeAndSchedule.client"
            if len(query_terms) > 1:
                try:
                    # Create compound filename queries for multi-term matches
                    term_filename_queries = []
                    for term in query_terms:
                        if len(term) >= 3:  # Skip very short terms
                            term_query = tantivy.Query.term_query(
                                self._schema, "name", term
                            )
                            term_filename_queries.append(
                                (tantivy.Occur.Should, term_query)
                            )

                    if term_filename_queries:
                        compound_filename_query = tantivy.Query.boolean_query(
                            term_filename_queries
                        )
                        boosted_queries.append(
                            (tantivy.Occur.Should, compound_filename_query)
                        )
                except Exception as e:
                    logger.debug(f"Failed to create multi-term filename query: {e}")

            # 3. High priority: Entity name matches (exact and multi-term)
            try:
                entity_name_query = tantivy.Query.term_query(
                    self._schema, "entity_name", query
                )
                boosted_queries.append((tantivy.Occur.Should, entity_name_query))
            except Exception as e:
                logger.debug(f"Failed to create entity name query: {e}")

            # Multi-term entity name matching
            if len(query_terms) > 1:
                try:
                    term_entity_queries = []
                    for term in query_terms:
                        if len(term) >= 3:
                            term_query = tantivy.Query.term_query(
                                self._schema, "entity_name", term
                            )
                            term_entity_queries.append(
                                (tantivy.Occur.Should, term_query)
                            )

                    if term_entity_queries:
                        compound_entity_query = tantivy.Query.boolean_query(
                            term_entity_queries
                        )
                        boosted_queries.append(
                            (tantivy.Occur.Should, compound_entity_query)
                        )
                except Exception as e:
                    logger.debug(f"Failed to create multi-term entity query: {e}")

            # 4. Medium priority: Fuzzy filename matches
            try:
                fuzzy_name_query = tantivy.Query.fuzzy_term_query(
                    self._schema, "name", query.lower(), distance=1, prefix=False
                )
                boosted_queries.append((tantivy.Occur.Should, fuzzy_name_query))
            except Exception as e:
                logger.debug(f"Failed to create fuzzy filename query: {e}")

            # 5. Standard priority: Content and other field searches
            default_fields = [
                "content",
                "entities",
                "functions",
                "module",
                "logical_unit",
                "pages",
                "lists",
                "groups",
                "iconsets",
                "trees",
                "navigators",
                "contexts",
            ]

            # Create standard search across content fields (full query)
            try:
                content_query = self._index.parse_query(
                    query, default_field_names=default_fields
                )
                boosted_queries.append((tantivy.Occur.Should, content_query))
            except Exception as e:
                logger.debug(f"Failed to create content query: {e}")

            # Multi-term content searches for better coverage
            if len(query_terms) > 1:
                try:
                    for term in query_terms:
                        if len(term) >= 3:  # Skip very short terms
                            term_content_query = self._index.parse_query(
                                term, default_field_names=default_fields
                            )
                            boosted_queries.append(
                                (tantivy.Occur.Should, term_content_query)
                            )
                except Exception as e:
                    logger.debug(f"Failed to create multi-term content queries: {e}")

            # 6. Fuzzy search for content fields (lower priority)
            if len(query.strip()) >= 3:
                try:
                    lowercase_query = query.lower()
                    fuzzy_queries = []
                    for field_name in default_fields:
                        try:
                            fuzzy_query = tantivy.Query.fuzzy_term_query(
                                self._schema,
                                field_name,
                                lowercase_query,
                                distance=2,
                                prefix=False,
                            )
                            fuzzy_queries.append((tantivy.Occur.Should, fuzzy_query))
                        except Exception as e:
                            logger.debug(
                                f"Failed to create fuzzy query for field {field_name}: {e}"
                            )
                            continue

                    if fuzzy_queries:
                        combined_fuzzy_query = tantivy.Query.boolean_query(
                            fuzzy_queries
                        )
                        boosted_queries.append(
                            (tantivy.Occur.Should, combined_fuzzy_query)
                        )
                except Exception as e:
                    logger.debug(f"Fuzzy search failed: {e}")

            # Combine all queries into a single boolean query
            if boosted_queries:
                final_query = tantivy.Query.boolean_query(boosted_queries)
                search_results = searcher.search(final_query, limit=limit)
            else:
                # Fallback to simple search if all boosted queries failed
                fallback_query = self._index.parse_query(
                    query, default_field_names=default_fields
                )
                search_results = searcher.search(fallback_query, limit=limit)
                logger.debug(f"Failed to create content query: {e}")

            # 5. Fuzzy search for content fields (lower priority)
            if len(query.strip()) >= 3:
                try:
                    lowercase_query = query.lower()
                    fuzzy_queries = []
                    for field_name in default_fields:
                        try:
                            fuzzy_query = tantivy.Query.fuzzy_term_query(
                                self._schema,
                                field_name,
                                lowercase_query,
                                distance=2,
                                prefix=False,
                            )
                            fuzzy_queries.append((tantivy.Occur.Should, fuzzy_query))
                        except Exception as e:
                            logger.debug(
                                f"Failed to create fuzzy query for field {field_name}: {e}"
                            )
                            continue

                    if fuzzy_queries:
                        combined_fuzzy_query = tantivy.Query.boolean_query(
                            fuzzy_queries
                        )
                        boosted_queries.append(
                            (tantivy.Occur.Should, combined_fuzzy_query)
                        )
                except Exception as e:
                    logger.debug(f"Fuzzy search failed: {e}")

            # Combine all queries into a single boolean query
            if boosted_queries:
                final_query = tantivy.Query.boolean_query(boosted_queries)
                search_results = searcher.search(final_query, limit=limit)
            else:
                # Fallback to simple search if all boosted queries failed
                fallback_query = self._index.parse_query(
                    query, default_field_names=default_fields
                )
                search_results = searcher.search(fallback_query, limit=limit)

            # Convert to SearchResult objects with logarithmic score normalization
            results = []

            for score, doc_address in search_results.hits:
                doc = searcher.doc(doc_address)

                # Get document metadata for scoring
                filename = doc.get_first("name") or ""
                entity_name = doc.get_first("entity_name") or ""
                pagerank_score = doc.get_first("pagerank_score") or 0.0
                line_count = doc.get_first("line_count") or 1
                doc_file_type = doc.get_first("type") or ""

                # Step 1: Apply logarithmic damping to base score to prevent term frequency explosion
                normalized_base_score = math.log(1 + score) / math.log(
                    2
                )  # Log base 2 for reasonable scaling

                # Step 2: Calculate match type bonus with multi-term support
                match_bonus = 0.0

                # Multi-term filename matching analysis
                term_matches = self._calculate_term_matches(
                    filename, entity_name, query_terms
                )

                # Extract filename without extension for exact matching
                filename_base = filename.lower()
                if "." in filename_base:
                    filename_base = filename_base.rsplit(".", 1)[0]

                query_lower = query.lower()

                # Exact filename match (HIGHEST priority - massive bonus)
                exact_filename_match = (
                    filename.lower() == f"{query_lower}.entity"
                    or filename.lower() == f"{query_lower}.plsql"
                    or filename.lower() == f"{query_lower}.storage"
                    or filename.lower() == f"{query_lower}.views"
                    or filename.lower() == f"{query_lower}.client"
                    or filename.lower() == f"{query_lower}.projection"
                    or filename.lower() == f"{query_lower}.fragment"
                    or filename_base
                    == query_lower  # Direct filename match without extension
                )

                # Enhanced compound entity name matching for action patterns
                compound_entity_match = self._check_compound_entity_match(
                    filename, query_terms, query
                )

                # Simple entity name prioritization for validation/property queries
                simple_entity_boost = self._check_simple_entity_priority(
                    filename, query, query_terms
                )

                if exact_filename_match:
                    match_bonus += 100.0  # MASSIVE bonus for exact filename match
                elif compound_entity_match > 0:
                    match_bonus += compound_entity_match  # Strong bonus for compound entity matches
                elif simple_entity_boost > 0:
                    match_bonus += simple_entity_boost  # Boost simple entity names for certain query patterns
                elif term_matches["all_terms_in_filename"]:
                    # All query terms found in filename (e.g., "Scope" and "Schedule" in "ScopeAndSchedule")
                    match_bonus += (
                        80.0  # Very strong bonus for complete term coverage (increased)
                    )
                elif term_matches["partial_terms_in_filename"] >= 0.7:
                    # Most terms found in filename (70%+ coverage)
                    match_bonus += (
                        60.0  # Strong bonus for high term coverage (increased)
                    )
                elif filename.lower().startswith(query.lower()):
                    match_bonus += (
                        40.0  # Good bonus for filename prefix match (increased)
                    )
                elif query.lower() in filename.lower():
                    match_bonus += (
                        25.0  # Moderate bonus for filename contains match (increased)
                    )
                elif term_matches["any_term_in_filename"]:
                    # At least one term found in filename
                    match_bonus += (
                        30.0 * term_matches["partial_terms_in_filename"]
                    )  # Scaled bonus (increased)
                elif entity_name and entity_name.lower() == query.lower():
                    match_bonus += 20.0  # Good bonus for exact entity name match
                elif entity_name and query.lower() in entity_name.lower():
                    match_bonus += 10.0  # Small bonus for entity name contains match
                elif entity_name and term_matches["any_term_in_entity"]:
                    # Multi-term entity name matching
                    match_bonus += 15.0 * term_matches["partial_terms_in_entity"]

                # Step 3: Apply file type priority adjustments with context awareness
                file_type_adjustment = self._get_file_type_modifier(
                    doc_file_type, query, line_count
                )

                # Step 4: Apply document length normalization (prevent long documents from dominating)
                length_factor = 1.0 / math.log(
                    1 + line_count / 100
                )  # Gentle normalization

                # Step 5: Apply PageRank boost (logarithmic to prevent domination)
                pagerank_bonus = (
                    math.log(1 + pagerank_score * 10) if pagerank_score > 0 else 0.0
                )

                # Step 5a: Apply content analysis bonus for business logic keywords
                content_bonus = self._calculate_content_bonus(
                    doc, query, doc_file_type, line_count
                )

                # Step 5b: Apply module context bonus
                module_bonus = self._calculate_module_context_bonus(doc, query)

                # Step 6: Combine all factors with caps to prevent score explosion
                final_score = (
                    (normalized_base_score * length_factor)
                    + match_bonus
                    + file_type_adjustment
                    + pagerank_bonus
                    + content_bonus
                    + module_bonus
                )

                # Cap the maximum score to prevent outliers, but allow higher scores for exact matches and business logic
                final_score = min(
                    final_score, 300.0
                )  # Increased cap to accommodate new bonuses (business logic + module context)

                search_result = SearchResult(
                    path=doc.get_first("path") or "",
                    name=filename,
                    type=doc.get_first("type") or "",
                    content_preview=doc.get_first("content_preview") or "",
                    score=final_score,
                    entities=(
                        doc.get_first("entities").split()
                        if doc.get_first("entities")
                        else []
                    ),
                    line_count=doc.get_first("line_count") or 0,
                    complexity_score=doc.get_first("complexity_score") or 0.0,
                    pagerank_score=doc.get_first("pagerank_score") or 0.0,
                    modified_time=datetime.fromisoformat(
                        str(doc.get_first("modified_time"))
                        if doc.get_first("modified_time")
                        else "1970-01-01T00:00:00"
                    ),
                    hash=doc.get_first("hash") or "",
                    module=doc.get_first("module") or None,
                    logical_unit=doc.get_first("logical_unit") or None,
                    entity_name=entity_name or None,
                    component=doc.get_first("component") or None,
                    pages=(
                        doc.get_first("pages").split() if doc.get_first("pages") else []
                    ),
                    lists=(
                        doc.get_first("lists").split() if doc.get_first("lists") else []
                    ),
                    groups=(
                        doc.get_first("groups").split()
                        if doc.get_first("groups")
                        else []
                    ),
                    entitysets=(
                        doc.get_first("entitysets").split()
                        if doc.get_first("entitysets")
                        else []
                    ),
                    iconsets=(
                        doc.get_first("iconsets").split()
                        if doc.get_first("iconsets")
                        else []
                    ),
                    trees=(
                        doc.get_first("trees").split() if doc.get_first("trees") else []
                    ),
                    navigators=(
                        doc.get_first("navigators").split()
                        if doc.get_first("navigators")
                        else []
                    ),
                    contexts=(
                        doc.get_first("contexts").split()
                        if doc.get_first("contexts")
                        else []
                    ),
                    dependencies=(
                        doc.get_first("dependencies").split()
                        if doc.get_first("dependencies")
                        else []
                    ),
                    functions=(
                        doc.get_first("functions").split()
                        if doc.get_first("functions")
                        else []
                    ),
                    imports=(
                        doc.get_first("imports").split()
                        if doc.get_first("imports")
                        else []
                    ),
                )

                # Apply filters
                logger.debug(
                    f"Checking file {search_result.name} (type: '{search_result.type}') against filter: '{file_type}'"
                )
                if file_type and not search_result.type.endswith(file_type):
                    logger.debug(
                        f"Filtering out {search_result.name} - type '{search_result.type}' doesn't match filter '{file_type}'"
                    )
                    continue

                if (
                    module
                    and search_result.module
                    and search_result.module.lower() != module.lower()
                ):
                    logger.debug(
                        f"Filtering out {search_result.name} - module '{search_result.module}' doesn't match filter '{module}' (case-insensitive)"
                    )
                    continue

                if (
                    logical_unit
                    and search_result.logical_unit
                    and search_result.logical_unit.lower() != logical_unit.lower()
                ):
                    logger.debug(
                        f"Filtering out {search_result.name} - logical_unit '{search_result.logical_unit}' doesn't match filter '{logical_unit}' (case-insensitive)"
                    )
                    continue

                logger.debug(f"File {search_result.name} passed all filter checks")

                if (
                    min_complexity is not None
                    and search_result.complexity_score < min_complexity
                ):
                    continue

                if (
                    max_complexity is not None
                    and search_result.complexity_score > max_complexity
                ):
                    continue

                results.append(search_result)

            # Sort by final score (after our boosting) and return
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Search failed: {e}")
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

    def _extract_module_info(self, file_path: Path) -> tuple[str, str]:
        """Extract module and logical unit information from file path.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (module, logical_unit)

        Example:
            _work/proj/model/proj/Activity.entity -> ("proj", "Activity")
            _work/career/model/career/EmployeeActivity.entity -> ("career", "EmployeeActivity")
        """
        parts = file_path.parts

        # Find _work in the path
        try:
            work_index = parts.index("_work")
            if work_index + 1 < len(parts):
                module = parts[work_index + 1]  # e.g., "proj", "career"

                # Extract entity name from filename (remove extension)
                entity_name = file_path.stem  # e.g., "Activity", "EmployeeActivity"

                return module, entity_name
        except (ValueError, IndexError):
            pass

        # Fallback if _work structure not found
        return "unknown", file_path.stem

    def _extract_entity_info(
        self, content: str, file_type: str, entities: List[str]
    ) -> tuple[str, str]:
        """Extract primary entity name and component from file content.

        Args:
            content: File content
            file_type: File extension
            entities: List of entities found by parser

        Returns:
            Tuple of (primary_entity_name, component)
        """
        primary_entity = ""
        component = ""

        if file_type == ".entity" and content:
            try:
                import xml.etree.ElementTree as ET

                root = ET.fromstring(content)

                # Extract primary entity name
                name_elem = root.find(".//{urn:ifsworld-com:schemas:entity_entity}NAME")
                if name_elem is not None:
                    primary_entity = name_elem.text or ""

                # Extract component
                component_elem = root.find(
                    ".//{urn:ifsworld-com:schemas:entity_entity}COMPONENT"
                )
                if component_elem is not None:
                    component = component_elem.text or ""

            except ET.ParseError:
                # Fallback to using the parsed entities
                if entities:
                    primary_entity = entities[0]
        else:
            # For non-entity files, use the first parsed entity
            if entities:
                primary_entity = entities[0]

        return primary_entity, component

    def set_ifs_version(self, ifs_version: str) -> bool:
        """Set the IFS version for metadata enhancement"""
        self._current_ifs_version = ifs_version

        # Try to load metadata for this version
        if self._metadata_manager.has_metadata(ifs_version):
            if self._enhanced_search_engine is None:
                self._enhanced_search_engine = MetadataEnhancedSearchEngine(
                    self, self._metadata_manager
                )

            success = self._enhanced_search_engine.set_ifs_version(ifs_version)
            if success:
                logger.info(f"Enhanced search enabled for IFS {ifs_version}")
            return success
        else:
            logger.warning(f"No metadata available for IFS {ifs_version}")
            return False

    def get_current_ifs_version(self) -> Optional[str]:
        """Get the currently configured IFS version"""
        return self._current_ifs_version

    def has_metadata_enhancement(self) -> bool:
        """Check if metadata enhancement is available"""
        return (
            self._enhanced_search_engine is not None
            and self._enhanced_search_engine.current_metadata is not None
        )

    def get_metadata_manager(self) -> MetadataManager:
        """Get the metadata manager for external operations"""
        return self._metadata_manager

    def get_available_ifs_versions(self) -> List[str]:
        """Get list of IFS versions with available metadata"""
        return self._metadata_manager.get_available_versions()

    def enhanced_search(self, query: str, **kwargs) -> List[EnhancedSearchResult]:
        """
        Perform enhanced search with metadata integration

        Args:
            query: Search query
            **kwargs: Additional search options (modules_filter, content_types_filter, etc.)

        Returns:
            List of enhanced search results with metadata context
        """
        if not self._enhanced_search_engine:
            # Fallback to basic search
            logger.info("Enhanced search not available - using basic search")
            basic_results = self.search_files(query)

            # Convert to enhanced format
            enhanced_results = []
            for result in basic_results:
                enhanced_results.append(
                    EnhancedSearchResult(
                        file_path=result.path,
                        content_type=result.type,
                        line_number=1,
                        snippet=result.content_preview,
                        confidence=result.score * 100,
                        logical_unit=result.logical_unit,
                        module=result.module,
                        business_description=None,
                        related_entities=[],
                        search_context=["Basic search - no metadata enhancement"],
                    )
                )

            return enhanced_results

        # Use enhanced search
        context = SearchContext(
            query=query,
            modules_filter=kwargs.get("modules_filter"),
            content_types_filter=kwargs.get("content_types_filter"),
            logical_units_filter=kwargs.get("logical_units_filter"),
            fuzzy_threshold=kwargs.get("fuzzy_threshold", 80.0),
            include_related=kwargs.get("include_related", True),
        )

        return self._enhanced_search_engine.enhanced_search(context)

    def get_module_statistics(self) -> Dict[str, Any]:
        """Get statistics about available modules from metadata"""
        if self._enhanced_search_engine:
            return self._enhanced_search_engine.get_module_statistics()
        else:
            return {}

    def suggest_related_searches(self, query: str, limit: int = 5) -> List[str]:
        """Suggest related search terms based on metadata"""
        if self._enhanced_search_engine:
            return self._enhanced_search_engine.suggest_related_searches(query, limit)
        else:
            return []

    def extract_metadata_from_mcp_results(
        self, ifs_version: str, query_results: Dict[str, List[Dict]]
    ) -> bool:
        """
        Extract and save metadata from MCP query results

        Args:
            ifs_version: IFS version identifier
            query_results: Dictionary with MCP query results

        Returns:
            True if extraction was successful
        """
        try:
            from .metadata_extractor import DatabaseMetadataExtractor

            extractor = DatabaseMetadataExtractor()
            metadata_extract = extractor.extract_from_mcp_queries(
                ifs_version, query_results
            )

            # Save metadata
            saved_path = self._metadata_manager.save_metadata(metadata_extract)
            logger.info(f"Metadata saved to {saved_path}")

            # Update search engine if this is the current version
            if ifs_version == self._current_ifs_version:
                self.set_ifs_version(ifs_version)

            return True

        except Exception as e:
            logger.error(f"Failed to extract metadata from MCP results: {e}")
            return False

    def get_metadata_extract_queries(self) -> Dict[str, str]:
        """
        Get the SQL queries needed for metadata extraction

        Returns:
            Dictionary of query names to SQL statements
        """
        from .metadata_extractor import DatabaseMetadataExtractor

        extractor = DatabaseMetadataExtractor()
        return extractor._get_default_queries()

    def close(self):
        """Close the indexer and release resources."""
        try:
            # Commit any pending changes and close writer
            self._commit_writer()
            self._close_writer()
            # Save cache metadata before closing
            self._save_cache_metadata()
            logger.info("Indexer closed successfully")
        except Exception as e:
            logger.error(f"Error closing indexer: {e}")
