"""
IFS Cloud Search Engine

Main search orchestrator

This module provides the primary search functionality for IFS Cloud files,
coordinating between the indexer and various search enhancements.
"""

import logging
from typing import Dict, List, Optional, Set, Any, Union
from pathlib import Path
from collections import Counter

from .indexer import IFSCloudIndexer, SearchResult
from .enhanced_search import (
    MetadataEnhancedSearchEngine,
    SearchContext,
    SearchResult as EnhancedSearchResult,
)
from .metadata_extractor import MetadataManager
from .intent_classifier import IntentClassifier, QueryIntent
from .metadata_indexer import MetadataIndexer

logger = logging.getLogger(__name__)


class IFSCloudSearchEngine:
    """Main search engine for IFS Cloud files with metadata enhancement capabilities."""

    def __init__(self, indexer: "IFSCloudIndexer", metadata_dir: Optional[Path] = None):
        """
        Initialize the search engine.

        Args:
            indexer: IFSCloudIndexer instance for accessing the file index
            metadata_dir: Optional directory for metadata storage (defaults to index_path/metadata)
        """
        self.indexer = indexer

        # Set up metadata directory
        if metadata_dir is None:
            metadata_dir = Path(indexer.index_path) / "metadata"

        self.metadata_manager = MetadataManager(metadata_dir)
        self.enhanced_search_engine: Optional[MetadataEnhancedSearchEngine] = None
        self.current_ifs_version: Optional[str] = None

        # Initialize ML-based intent classifier
        try:
            self.intent_classifier = IntentClassifier()
            logger.info("Intent classifier initialized successfully")
        except Exception as e:
            logger.warning(f"Intent classifier initialization failed: {e}")
            self.intent_classifier = None

        # Initialize dedicated metadata indexer
        try:
            metadata_index_path = Path(indexer.index_path) / "metadata_index"
            self.metadata_indexer = MetadataIndexer(metadata_index_path)

            # Try to build from available metadata export if index is empty
            self._initialize_metadata_index()

            logger.info("Metadata indexer initialized successfully")
        except Exception as e:
            logger.warning(f"Metadata indexer initialization failed: {e}")
            self.metadata_indexer = None

    def _initialize_metadata_index(self):
        """Initialize metadata index from available database exports."""
        if not self.metadata_indexer:
            return

        try:
            # Check if index is already populated
            stats = self.metadata_indexer.get_stats()
            if stats.get("total_documents", 0) > 0:
                logger.info(
                    f"Metadata index already contains {stats['total_documents']} documents"
                )
                return

            # Look for metadata export files
            from .config import get_data_directory

            data_dir = get_data_directory()
            metadata_dir = data_dir / "metadata"

            if not metadata_dir.exists():
                logger.info(
                    "No metadata directory found - metadata indexer will remain empty"
                )
                return

            # Find the most recent metadata export
            metadata_export_file = None
            for version_dir in metadata_dir.iterdir():
                if version_dir.is_dir():
                    export_file = version_dir / "metadata_extract.json"
                    if export_file.exists():
                        metadata_export_file = export_file
                        logger.info(f"Found metadata export: {export_file}")
                        break

            if metadata_export_file:
                # Build metadata index from the export
                count = self.metadata_indexer.build_from_metadata_export(
                    metadata_export_file
                )
                logger.info(
                    f"Built metadata index with {count} documents from database export"
                )
            else:
                logger.info(
                    "No metadata export found - run 'extract' command first to populate metadata index"
                )

        except Exception as e:
            logger.warning(f"Failed to initialize metadata index: {e}")

    def set_ifs_version(self, version: str) -> bool:
        """
        Set the IFS version for metadata-enhanced search.

        Args:
            version: IFS version (e.g., "25.1.0")

        Returns:
            True if metadata was successfully loaded
        """
        try:
            self.current_ifs_version = version
            metadata = self.metadata_manager.get_metadata(version)

            if metadata:
                self.enhanced_search_engine = MetadataEnhancedSearchEngine(
                    base_search_engine=self, metadata_manager=self.metadata_manager
                )
                self.enhanced_search_engine.set_ifs_version(version)
                logger.info(f"Enhanced search enabled for IFS version {version}")
                return True
            else:
                logger.warning(f"No metadata found for IFS version {version}")
                self.enhanced_search_engine = None
                return False

        except Exception as e:
            logger.error(f"Failed to set IFS version {version}: {e}")
            self.enhanced_search_engine = None
            return False

    def search(
        self,
        query: str,
        limit: int = 10,
        file_type: Optional[str] = None,
        min_complexity: Optional[float] = None,
        max_complexity: Optional[float] = None,
        module: Optional[str] = None,
        logical_unit: Optional[str] = None,
        include_related: bool = False,
        max_related_per_lu: int = 5,
    ) -> List[SearchResult]:
        """
        Primary search method with optional related file inclusion.

        Args:
            query: Search query
            limit: Maximum number of results
            file_type: Filter by file type (optional)
            min_complexity: Minimum complexity score (optional)
            max_complexity: Maximum complexity score (optional)
            module: Filter by module (optional)
            logical_unit: Filter by logical unit (optional)
            include_related: Whether to include related files from same logical units
            max_related_per_lu: Maximum related files per logical unit

        Returns:
            List of search results
        """
        if include_related:
            return self.search_with_related_files(
                query=query,
                limit=limit,
                file_type=file_type,
                min_complexity=min_complexity,
                max_complexity=max_complexity,
                max_related_per_lu=max_related_per_lu,
                module=module,
                logical_unit=logical_unit,
            )
        else:
            # Dual-search approach: combine index and metadata results
            all_results = []
            seen_paths = set()

            # 1. Get index-based results (traditional search) - prioritize this
            index_results = self.indexer.search_deduplicated(
                query=query,
                limit=limit * 3,  # Get more results to allow for better merging
                file_type=None,  # Don't filter at indexer level, we'll filter later
                min_complexity=min_complexity,
                max_complexity=max_complexity,
            )

            # 2. Get metadata-enhanced results if metadata indexer is available
            metadata_results = []
            if self.metadata_indexer:
                try:
                    metadata_search_results = self.metadata_indexer.search_metadata(
                        query=query,
                        limit=limit * 2,
                        file_types=[file_type] if file_type else None,
                        modules=[module] if module else None,
                    )

                    # Convert metadata results to SearchResult objects
                    for meta_result in metadata_search_results:
                        if meta_result.get("path") not in seen_paths:
                            # Create a SearchResult from metadata
                            search_result = SearchResult(
                                path=meta_result["path"],
                                name=Path(meta_result["path"]).name,
                                content_preview="",  # Not needed for metadata search
                                score=meta_result.get("score", 1.0)
                                * 0.8,  # Slight penalty for metadata-only
                                type=meta_result.get("file_type", ""),
                                module=meta_result.get("module"),
                                logical_unit=meta_result.get("logical_unit"),
                                line_count=meta_result.get("line_count"),
                                size_kb=meta_result.get("size_kb"),
                                complexity_score=meta_result.get("complexity_score"),
                                highlight="",
                                entities=meta_result.get("entities"),
                                functions=meta_result.get("functions"),
                                pages=meta_result.get("pages"),
                                lists=meta_result.get("lists"),
                                groups=meta_result.get("groups"),
                                trees=meta_result.get("trees"),
                                navigators=meta_result.get("navigators"),
                                dependencies=meta_result.get("dependencies"),
                            )
                            metadata_results.append(search_result)
                            seen_paths.add(meta_result["path"])

                    logger.info(
                        f"Metadata indexer returned {len(metadata_results)} additional results"
                    )

                except Exception as e:
                    logger.warning(f"Metadata search failed: {e}")
                    metadata_results = []

            # Add metadata results first (they have higher priority)
            all_results.extend(metadata_results)

            # Add index results with proper scoring
            for result in index_results:
                if result.path not in seen_paths:
                    all_results.append(result)
                    seen_paths.add(result.path)

            # 2. Get metadata-enhanced results if available (but limit this)
            if self.enhanced_search_engine and len(all_results) < limit * 2:
                try:
                    from .enhanced_search import SearchContext

                    # Map file_type to content_types_filter for metadata search
                    content_types_filter = None
                    if file_type:
                        # Remove the dot and map to content type
                        content_type = file_type.lstrip(".")
                        content_types_filter = [content_type]

                    context = SearchContext(
                        query=query,
                        limit=max(
                            10, limit
                        ),  # Limit metadata search to prevent too many calls
                        modules_filter=[module] if module else None,
                        content_types_filter=content_types_filter,
                        logical_units_filter=[logical_unit] if logical_unit else None,
                        fuzzy_threshold=80.0,  # Higher threshold for more precision
                        include_related=False,
                    )

                    enhanced_results = self.enhanced_search_engine.enhanced_search(
                        context
                    )

                    # Convert enhanced results to SearchResult and merge (limit to prevent spam)
                    for enhanced_result in enhanced_results[:limit]:
                        if enhanced_result.file_path not in seen_paths:
                            # Convert enhanced result to SearchResult
                            search_result = self._convert_enhanced_to_search_result(
                                enhanced_result
                            )
                            # Boost score for metadata-found results
                            search_result.score = (
                                enhanced_result.confidence * 1.5
                            )  # Moderate boost for metadata matches
                            all_results.append(search_result)
                            seen_paths.add(enhanced_result.file_path)

                except Exception as e:
                    logger.debug(f"Metadata search failed: {e}")

            # 3. Re-rank combined results using business logic
            all_results = self._apply_business_ranking(all_results, query)

            # 4. Apply final filters that weren't handled by individual searches
            filtered_results = self._apply_post_search_filters(
                all_results,
                file_type,
                module,
                logical_unit,
                min_complexity,
                max_complexity,
            )

            # 5. Sort by final score and limit
            filtered_results.sort(key=lambda x: x.score, reverse=True)
            return filtered_results[:limit]

    def search_with_related_files(
        self,
        query: str,
        limit: int = 10,
        file_type: Optional[str] = None,
        min_complexity: Optional[float] = None,
        max_complexity: Optional[float] = None,
        max_related_per_lu: int = 5,
        module: Optional[str] = None,
        logical_unit: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search with related files from the same logical units.

        When a logical unit is found (e.g., "Activity"), includes related files
        like Activity.entity, Activity.plsql, Activity.views, etc.
        """
        # Get primary search results (use regular search method to apply filters)
        primary_results = self.search(
            query=query,
            limit=limit,
            file_type=file_type,
            min_complexity=min_complexity,
            max_complexity=max_complexity,
            module=module,
            logical_unit=logical_unit,
            include_related=False,  # Avoid recursion
        )

        if not primary_results:
            return primary_results

        # Collect unique logical units from primary results
        logical_units = set()
        primary_paths = set()

        for result in primary_results:
            primary_paths.add(result.path)
            if result.logical_unit:
                logical_units.add(result.logical_unit)

        # Find related files for each logical unit
        related_results = []

        for lu in logical_units:
            related_files = self.find_related_files(lu)

            # Add related files not already in primary results
            related_count = 0
            for related in related_files:
                if (
                    related.path not in primary_paths
                    and related_count < max_related_per_lu
                ):
                    # Apply file type filter to related files too
                    if file_type and related.type:
                        if not related.type.endswith(file_type):
                            continue
                    elif file_type and not related.type:
                        continue

                    # Reduce score slightly to indicate it's a related result
                    related.score *= 0.8
                    related_results.append(related)
                    related_count += 1

        # Combine and sort results
        all_results = primary_results + related_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results[
            : limit * 2
        ]  # Allow more results when including related files

    def find_related_files(
        self, logical_unit: str, include_extensions: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Find all files related to a specific logical unit.

        Args:
            logical_unit: Name of the logical unit (e.g., "Activity", "CustomerOrder")
            include_extensions: File extensions to include (default: all supported)

        Returns:
            List of related files for the logical unit
        """
        if not logical_unit:
            return []

        include_extensions = include_extensions or list(
            self.indexer.SUPPORTED_EXTENSIONS
        )

        # Search for files matching the logical unit
        results = []

        # Primary patterns to match
        patterns = [
            logical_unit,  # Exact match
            f"{logical_unit}API",  # API files
            f"{logical_unit}Handling",  # Handling files
            f"{logical_unit}Client",  # Client files
        ]

        for pattern in patterns:
            search_results = self.indexer.search_deduplicated(
                query=pattern, limit=50  # Get more results for filtering
            )

            # Filter by file extension
            for result in search_results:
                if any(result.type.endswith(ext) for ext in include_extensions):
                    # Check if this is truly related (not just a coincidental match)
                    if (
                        result.logical_unit
                        and result.logical_unit.lower() == logical_unit.lower()
                    ) or (logical_unit.lower() in result.name.lower()):
                        results.append(result)

        # Deduplicate and sort by relevance
        seen_paths = set()
        unique_results = []

        for result in results:
            if result.path not in seen_paths:
                seen_paths.add(result.path)
                unique_results.append(result)

        # Sort by file type priority and score
        file_type_priority = {
            ".entity": 1,  # Core definition
            ".plsql": 2,  # Business logic
            ".views": 3,  # Data access
            ".projection": 4,  # API layer
            ".client": 5,  # UI layer
            ".fragment": 6,  # Reusable components
            ".storage": 7,  # Database schema
            ".plsvc": 8,  # Service layer
        }

        unique_results.sort(
            key=lambda r: (file_type_priority.get(r.type, 99), -r.score)
        )

        return unique_results

    def enhanced_search(
        self,
        query: str,
        limit: int = 10,
        modules_filter: Optional[List[str]] = None,
        content_types_filter: Optional[List[str]] = None,
        logical_units_filter: Optional[List[str]] = None,
        fuzzy_threshold: float = 80.0,
        include_related: bool = True,
    ) -> List[EnhancedSearchResult]:
        """
        Perform metadata-enhanced search with intelligent ranking and related files.

        Args:
            query: Search query
            limit: Maximum number of results
            modules_filter: Filter by IFS modules (e.g., ['ORDER', 'PERSON'])
            content_types_filter: Filter by file types (e.g., ['entity', 'plsql'])
            logical_units_filter: Filter by logical units
            fuzzy_threshold: Fuzzy matching threshold (0-100)
            include_related: Include related files for matched logical units

        Returns:
            List of enhanced search results with metadata context
        """
        context = SearchContext(
            query=query,
            limit=limit,
            modules_filter=modules_filter or [],
            content_types_filter=content_types_filter or [],
            logical_units_filter=logical_units_filter or [],
            fuzzy_threshold=fuzzy_threshold,
            include_related=include_related,
        )

        if self.enhanced_search_engine:
            # Use enhanced search with metadata
            return self.enhanced_search_engine.enhanced_search(context)
        else:
            # Fallback to basic search and convert results
            return self._fallback_enhanced_search(context)

    def _fallback_enhanced_search(
        self, context: SearchContext
    ) -> List[EnhancedSearchResult]:
        """Fallback enhanced search without metadata."""
        # Get basic search results
        basic_results = self.search(
            query=context.query,
            limit=context.limit,
            include_related=context.include_related,
        )

        # Convert to enhanced results
        enhanced_results = []
        for result in basic_results:
            enhanced_results.append(
                EnhancedSearchResult(
                    file_path=result.path,
                    content_type=result.type.lstrip("."),  # Remove leading dot
                    line_number=1,
                    snippet=result.content_preview,
                    confidence=min(100.0, result.score / 10.0),  # Normalize score
                    logical_unit=result.logical_unit,
                    module=result.module,
                    business_description=None,
                    related_entities=[],
                    search_context=["Basic search - no metadata available"],
                )
            )

        return enhanced_results

    def suggest_related_searches(self, query: str, limit: int = 5) -> List[str]:
        """
        Suggest related search terms based on metadata and index content.

        Args:
            query: Original search query
            limit: Maximum number of suggestions

        Returns:
            List of suggested search terms
        """
        if self.enhanced_search_engine:
            return self.enhanced_search_engine.suggest_related_searches(query, limit)
        else:
            # Basic suggestions from index content
            return self._basic_suggest_related(query, limit)

    def _basic_suggest_related(self, query: str, limit: int) -> List[str]:
        """Basic related search suggestions without metadata."""
        suggestions = []

        # Get some search results to extract related terms
        results = self.search(query, limit=10)

        # Extract entities from results as suggestions
        entity_counts = Counter()

        for result in results:
            # Add entities from the result
            for entity in result.entities:
                if entity.lower() != query.lower():  # Don't suggest the same term
                    entity_counts[entity] += 1

        # Return most common entities as suggestions
        suggestions = [entity for entity, _ in entity_counts.most_common(limit)]

        return suggestions

    # Metadata management methods
    def get_current_ifs_version(self) -> Optional[str]:
        """Get the current IFS version being used for enhanced search."""
        return self.current_ifs_version

    def has_metadata_enhancement(self) -> bool:
        """Check if metadata enhancement is available."""
        return self.enhanced_search_engine is not None

    def get_available_ifs_versions(self) -> List[str]:
        """Get list of available IFS versions with metadata."""
        return self.metadata_manager.get_available_versions()

    def get_module_statistics(self) -> Dict[str, Any]:
        """Get statistics about available modules from metadata."""
        if self.enhanced_search_engine:
            return self.enhanced_search_engine.get_module_statistics()
        else:
            return {"error": "No metadata available - enhanced search not enabled"}

    def extract_metadata_from_mcp_results(
        self, ifs_version: str, query_results: Dict[str, List[Dict[str, Any]]]
    ) -> bool:
        """
        Extract and save metadata from MCP SQLcl query results.

        Args:
            ifs_version: IFS version for the metadata
            query_results: Results from MCP SQLcl queries

        Returns:
            True if metadata was successfully extracted and saved
        """
        try:
            return self.metadata_manager.process_mcp_results(ifs_version, query_results)
        except Exception as e:
            logger.error(f"Failed to extract metadata from MCP results: {e}")
            return False

    def get_metadata_extract_queries(self) -> Dict[str, str]:
        """
        Get the SQL queries needed for metadata extraction via MCP SQLcl.

        Returns:
            Dictionary of query names to SQL statements
        """
        return {
            "logical_units": """
SELECT 
    lu_name,
    module,
    lu_prompt,
    lu_type,
    base_table,
    base_view
FROM dictionary_sys_lu_active 
WHERE lu_type IN ('Entity', 'BasicLU', 'MasterLU')
    AND module IS NOT NULL
ORDER BY module, lu_name
            """.strip(),
            "modules": """
SELECT DISTINCT 
    module,
    COUNT(*) as lu_count
FROM dictionary_sys_lu_active 
WHERE module IS NOT NULL 
    AND lu_type IN ('Entity', 'BasicLU', 'MasterLU')
GROUP BY module
ORDER BY module
            """.strip(),
            "domain_mappings": """
SELECT DISTINCT
    domain_id,
    client_value,
    db_value,
    description
FROM domain_sys_tab
WHERE client_value IS NOT NULL
    AND db_value IS NOT NULL
ORDER BY domain_id, client_value
            """.strip(),
            "views": """
SELECT 
    view_name,
    module,
    lu_name,
    view_comment
FROM dictionary_sys_view_active
WHERE module IS NOT NULL
    AND lu_name IS NOT NULL
ORDER BY module, lu_name, view_name
            """.strip(),
        }

    def _convert_enhanced_to_search_result(self, enhanced_result) -> "SearchResult":
        """Convert an enhanced search result to a standard SearchResult."""
        # Try to find the result in our index to get full metadata
        try:
            index_results = self.indexer.search_deduplicated(
                query=f'path:"{enhanced_result.file_path}"', limit=1
            )
            if index_results:
                # Use the index result but with enhanced score
                return index_results[0]
        except Exception:
            pass

        # Fallback: create a basic SearchResult from enhanced result
        from datetime import datetime
        from .indexer import SearchResult

        return SearchResult(
            path=enhanced_result.file_path,
            name=Path(enhanced_result.file_path).name,
            type=f".{enhanced_result.content_type}",
            content_preview=enhanced_result.snippet,
            score=enhanced_result.confidence,
            entities=[],
            functions=[],
            line_count=0,
            complexity_score=0.0,
            pagerank_score=0.0,
            modified_time=datetime.now(),
            hash="",
            module=enhanced_result.module,
            logical_unit=enhanced_result.logical_unit,
            entity_name=None,
            component=None,
            pages=[],
            lists=[],
            groups=[],
            entitysets=[],
            iconsets=[],
            trees=[],
            navigators=[],
            contexts=[],
            dependencies=[],
            imports=[],
            highlight="",
            tags=[],
        )

    def _apply_business_ranking(
        self, results: List["SearchResult"], query: str
    ) -> List["SearchResult"]:
        """Apply sophisticated business logic ranking with ML-based intent classification."""
        query_lower = query.lower()

        # Use ML intent classifier if available
        if self.intent_classifier:
            try:
                intent_prediction = self.intent_classifier.predict(query)
                intent = intent_prediction.intent
                confidence = intent_prediction.confidence

                logger.info(
                    f"Query intent: {intent.value} (confidence: {confidence:.3f})"
                )

                # Apply intent-specific ranking
                for result in results:
                    original_score = result.score

                    if intent == QueryIntent.BUSINESS_LOGIC:
                        # Business logic queries - heavily favor implementation files
                        if result.type == ".plsql":
                            result.score *= 4.5 * confidence
                        elif result.type == ".client":
                            result.score *= 3.2 * confidence
                        elif result.type == ".views":
                            result.score *= 2.4 * confidence
                        elif result.type == ".projection":
                            result.score *= 1.8 * confidence
                        elif result.type == ".entity":
                            result.score *= 0.1 * confidence

                    elif intent == QueryIntent.ENTITY_DEFINITION:
                        # Entity queries - balanced but practical
                        if result.type == ".plsql":
                            result.score *= 2.8 * confidence
                        elif result.type == ".views":
                            result.score *= 2.2 * confidence
                        elif result.type == ".entity":
                            result.score *= 1.8 * confidence
                        elif result.type == ".client":
                            result.score *= 1.6 * confidence
                        elif result.type == ".projection":
                            result.score *= 1.4 * confidence

                    elif intent == QueryIntent.UI_COMPONENTS:
                        # UI queries - favor client files
                        if result.type == ".client":
                            result.score *= 4.0 * confidence
                        elif result.type == ".plsql":
                            result.score *= 2.5 * confidence
                        elif result.type == ".views":
                            result.score *= 2.0 * confidence
                        elif result.type == ".projection":
                            result.score *= 1.5 * confidence
                        elif result.type == ".entity":
                            result.score *= 0.8 * confidence

                    elif intent == QueryIntent.API_INTEGRATION:
                        # API queries - favor projections
                        if result.type == ".projection":
                            result.score *= 4.0 * confidence
                        elif result.type == ".plsql":
                            result.score *= 2.8 * confidence
                        elif result.type == ".views":
                            result.score *= 2.0 * confidence
                        elif result.type == ".client":
                            result.score *= 1.5 * confidence
                        elif result.type == ".entity":
                            result.score *= 1.0 * confidence

                    elif intent == QueryIntent.DATA_ACCESS:
                        # Data access queries - favor views and reports
                        if result.type == ".views":
                            result.score *= 4.0 * confidence
                        elif result.type == ".plsql":
                            result.score *= 2.5 * confidence
                        elif result.type == ".entity":
                            result.score *= 2.0 * confidence
                        elif result.type == ".projection":
                            result.score *= 1.8 * confidence
                        elif result.type == ".client":
                            result.score *= 1.5 * confidence

                    else:  # GENERAL, TROUBLESHOOTING
                        # Default balanced approach with practical bias
                        if result.type == ".plsql":
                            result.score *= 3.5
                        elif result.type == ".client":
                            result.score *= 2.5
                        elif result.type == ".views":
                            result.score *= 2.0
                        elif result.type == ".projection":
                            result.score *= 1.6
                        elif result.type == ".entity":
                            result.score *= 0.6

                return results

            except Exception as e:
                logger.error(f"Intent classification failed: {e}")
                # Fall back to keyword-based classification

        # Fallback keyword-based classification (original logic)
        business_logic_terms = [
            "authorization",
            "approval",
            "workflow",
            "validation",
            "business",
            "rule",
            "calculation",
            "process",
            "procedure",
            "logic",
            "check",
            "verify",
            "confirm",
            "execute",
            "run",
            "perform",
            "handle",
            "management",
            "control",
            "monitor",
            "track",
        ]

        strong_business_terms = [
            "authorization",
            "validation",
            "calculation",
            "approval",
            "workflow",
            "process",
            "procedure",
            "check",
            "verify",
            "control",
        ]

        entity_focused_terms = [
            "definition",
            "structure",
            "schema",
            "model",
            "attribute",
            "property",
            "field",
            "column",
            "table",
        ]

        has_business_logic = any(term in query_lower for term in business_logic_terms)
        has_strong_business = any(term in query_lower for term in strong_business_terms)
        is_entity_focused = any(term in query_lower for term in entity_focused_terms)

        entity_exact_match = False
        for result in results:
            if (
                result.type == ".entity"
                and result.name.lower().replace(".entity", "") in query_lower
            ):
                entity_exact_match = True
                break

        for result in results:
            original_score = result.score

            # Apply different ranking strategies based on query intent
            if has_strong_business and not is_entity_focused:
                # Strong business logic queries - heavily favor implementation files
                if result.type == ".plsql":
                    result.score *= 4.0  # Very strong boost for PL/SQL
                elif result.type == ".client":
                    result.score *= 3.0  # Strong boost for client-side logic
                elif result.type == ".views":
                    result.score *= 2.2  # Good boost for views
                elif result.type == ".projection":
                    result.score *= 1.8  # Moderate boost for projections
                elif result.type == ".entity":
                    result.score *= 0.15  # Very heavy penalty for entities

            elif has_business_logic and not is_entity_focused:
                # General business queries - moderate favor to implementation
                if result.type == ".plsql":
                    result.score *= 3.5  # Strong boost for PL/SQL
                elif result.type == ".client":
                    result.score *= 2.5  # Good boost for client-side
                elif result.type == ".views":
                    result.score *= 2.0  # Good boost for views
                elif result.type == ".projection":
                    result.score *= 1.6  # Moderate boost for projections
                elif result.type == ".entity":
                    result.score *= 0.25  # Heavy penalty for entities

            elif is_entity_focused or entity_exact_match:
                # Entity-focused queries - balanced but still favor practical files
                if result.type == ".plsql":
                    result.score *= 2.5  # Good boost for PL/SQL
                elif result.type == ".views":
                    result.score *= 2.0  # Good boost for views
                elif result.type == ".entity":
                    result.score *= 1.5  # Moderate boost for entities (only when explicitly requested)
                elif result.type == ".client":
                    result.score *= 1.8  # Boost for client files
                elif result.type == ".projection":
                    result.score *= 1.6  # Boost for projections

            else:
                # General/unknown queries - balanced approach favoring practical files
                if result.type == ".plsql":
                    result.score *= 3.2  # Strong boost for PL/SQL (most practical)
                elif result.type == ".client":
                    result.score *= 2.2  # Good boost for UI logic
                elif result.type == ".views":
                    result.score *= 1.8  # Good boost for views
                elif result.type == ".projection":
                    result.score *= 1.6  # Moderate boost for projections
                elif result.type == ".entity":
                    result.score *= 0.4  # Significant penalty for entities

            # Boost for exact filename matches, but balanced by file type
            result_name_lower = result.name.lower()
            if any(word in result_name_lower for word in query_lower.split()):
                filename_match_boost = 1.4
                if result.type == ".plsql":
                    result.score *= (
                        filename_match_boost * 1.5
                    )  # Extra boost for practical files
                elif result.type == ".client":
                    result.score *= filename_match_boost * 1.4
                elif result.type == ".views":
                    result.score *= filename_match_boost * 1.3
                elif result.type == ".projection":
                    result.score *= filename_match_boost * 1.2
                elif result.type == ".entity":
                    result.score *= (
                        filename_match_boost * 0.6
                    )  # Reduced boost for entities

            # Module relevance - boost core business modules
            if result.module and result.module.lower() in [
                "person",
                "order",
                "accrul",
                "invent",
                "invoic",
                "purch",
                "trvexp",
                "proj",
                "mfgstd",
                "fndadm",
                "fndbas",
            ]:
                result.score *= 1.2

            # Complexity boosting - larger implementation files often more valuable
            if (
                result.type in [".plsql", ".client"]
                and hasattr(result, "line_count")
                and result.line_count
            ):
                if result.line_count > 100:
                    complexity_boost = min(1.4, 1.0 + (result.line_count / 2000))
                    result.score *= complexity_boost

            # Anti-dominance mechanism - reduce scores for overrepresented types
            type_counts = {}
            for r in results:
                type_counts[r.type] = type_counts.get(r.type, 0) + 1

            total_results = len(results)
            if total_results > 10:
                type_ratio = type_counts.get(result.type, 0) / total_results
                if type_ratio > 0.7:  # If one type dominates >70% of results
                    result.score *= 0.8  # Apply penalty to encourage diversity

        return results

    def _apply_post_search_filters(
        self,
        results: List["SearchResult"],
        file_type: Optional[str],
        module: Optional[str],
        logical_unit: Optional[str],
        min_complexity: Optional[float],
        max_complexity: Optional[float],
    ) -> List["SearchResult"]:
        """Apply filters that couldn't be applied during search."""
        filtered_results = []

        for result in results:
            # File type filter
            if file_type and result.type:
                if not result.type.endswith(file_type):
                    continue
            elif file_type and not result.type:
                # Skip results without type if file type filter is specified
                continue

            # Module filter
            if module and result.module:
                if not result.module.lower().startswith(module.lower()):
                    continue
            elif module and not result.module:
                # Skip results without module if module filter is specified
                continue

            # Logical unit filter
            if logical_unit and result.logical_unit:
                if not result.logical_unit.lower().startswith(logical_unit.lower()):
                    continue
            elif logical_unit and not result.logical_unit:
                # Skip results without logical unit if logical unit filter is specified
                continue

            # Complexity filters
            if min_complexity is not None and result.complexity_score < min_complexity:
                continue
            if max_complexity is not None and result.complexity_score > max_complexity:
                continue

            filtered_results.append(result)

        return filtered_results
