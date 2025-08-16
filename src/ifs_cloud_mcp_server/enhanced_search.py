"""
Enhanced Search Engine with IFS Cloud Metadata Integration

Provides intelligent search across IFS Cloud entities using extracted metadata
for business context, cross-module relationships, and semantic understanding.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, Counter
import json

try:
    from rapidfuzz import fuzz, process

    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

from .metadata_extractor import MetadataExtract, LogicalUnit, ModuleInfo, DomainMapping

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Enhanced search result with metadata context"""

    file_path: str
    content_type: str  # 'entity', 'plsql', 'client', 'projection', etc.
    line_number: int
    snippet: str
    confidence: float

    # Metadata enrichment
    logical_unit: Optional[str] = None
    module: Optional[str] = None
    business_description: Optional[str] = None
    related_entities: List[str] = field(default_factory=list)
    search_context: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "file_path": self.file_path,
            "content_type": self.content_type,
            "line_number": self.line_number,
            "snippet": self.snippet,
            "confidence": self.confidence,
            "logical_unit": self.logical_unit,
            "module": self.module,
            "business_description": self.business_description,
            "related_entities": self.related_entities,
            "search_context": self.search_context,
        }


@dataclass
class SearchContext:
    """Context information for enhanced searching"""

    query: str
    modules_filter: Optional[List[str]] = None
    content_types_filter: Optional[List[str]] = None
    logical_units_filter: Optional[List[str]] = None
    fuzzy_threshold: float = 80.0
    include_related: bool = True


class BusinessTermMatcher:
    """Matches business terms to technical entities"""

    def __init__(self, metadata: MetadataExtract):
        self.metadata = metadata
        self._build_term_mappings()

    def _build_term_mappings(self) -> None:
        """Build mappings from business terms to technical entities"""
        self.term_to_lu: Dict[str, List[str]] = defaultdict(list)
        self.lu_to_terms: Dict[str, List[str]] = defaultdict(list)
        self.module_keywords: Dict[str, Set[str]] = defaultdict(set)

        # Process logical units
        for lu in self.metadata.logical_units:
            # Map LU name variations
            lu_terms = [lu.lu_name.lower(), lu.lu_name.lower().replace("_", " ")]

            # Add prompt terms
            if lu.lu_prompt:
                prompt_terms = re.findall(r"\b\w+\b", lu.lu_prompt.lower())
                lu_terms.extend(prompt_terms)

            # Store mappings
            for term in lu_terms:
                if term and len(term) > 2:  # Skip very short terms
                    self.term_to_lu[term].append(lu.lu_name)
                    self.lu_to_terms[lu.lu_name].append(term)
                    self.module_keywords[lu.module].add(term)

        # Process domain mappings
        for mapping in self.metadata.domain_mappings:
            client_terms = re.findall(r"\b\w+\b", mapping.client_value.lower())
            for term in client_terms:
                if term and len(term) > 2:
                    self.term_to_lu[term].append(mapping.lu_name)

        logger.info(
            f"Built term mappings: {len(self.term_to_lu)} terms -> {len(self.lu_to_terms)} LUs"
        )

    def find_logical_units_for_query(
        self, query: str, threshold: float = 70.0
    ) -> List[Tuple[str, float]]:
        """Find logical units that match the query terms"""
        query_terms = re.findall(r"\b\w+\b", query.lower())
        lu_scores: Dict[str, float] = defaultdict(float)

        for term in query_terms:
            # Exact matches
            for lu_name in self.term_to_lu.get(term, []):
                lu_scores[lu_name] += 100.0

            # Fuzzy matches (only if rapidfuzz is available)
            if len(term) > 3 and FUZZY_AVAILABLE:  # Only fuzzy match longer terms
                try:
                    fuzzy_matches = process.extract(
                        term,
                        list(self.term_to_lu.keys()),
                        score_cutoff=threshold,
                        limit=5,
                    )
                    for match_term, score, _ in fuzzy_matches:
                        for lu_name in self.term_to_lu[match_term]:
                            lu_scores[lu_name] += (
                                score * 0.8
                            )  # Reduce fuzzy match weight
                except Exception as e:
                    # Fallback if fuzzy matching fails
                    logger.debug(f"Fuzzy matching failed for term '{term}': {e}")
                    continue

        # Sort by score and return
        return sorted(lu_scores.items(), key=lambda x: x[1], reverse=True)

    def get_related_entities(self, lu_name: str) -> List[str]:
        """Get entities related to the given logical unit"""
        related = set()

        # Find LUs in the same module
        lu_module = None
        for lu in self.metadata.logical_units:
            if lu.lu_name == lu_name:
                lu_module = lu.module
                break

        if lu_module:
            for lu in self.metadata.logical_units:
                if lu.module == lu_module and lu.lu_name != lu_name:
                    related.add(lu.lu_name)

        return list(related)[:10]  # Limit to avoid overwhelming results


class MetadataEnhancedSearchEngine:
    """Enhanced search engine with IFS metadata integration"""

    def __init__(self, base_search_engine, metadata_manager):
        """
        Initialize enhanced search engine

        Args:
            base_search_engine: Base search engine instance
            metadata_manager: MetadataManager instance
        """
        self.base_engine = base_search_engine
        self.metadata_manager = metadata_manager
        self.current_metadata: Optional[MetadataExtract] = None
        self.term_matcher: Optional[BusinessTermMatcher] = None

    def set_ifs_version(self, ifs_version: str) -> bool:
        """Set the IFS version to use for metadata enhancement"""
        metadata = self.metadata_manager.load_metadata(ifs_version)
        if metadata:
            self.current_metadata = metadata
            self.term_matcher = BusinessTermMatcher(metadata)
            logger.info(f"Loaded metadata for IFS {ifs_version}")
            return True
        else:
            logger.warning(f"No metadata available for IFS {ifs_version}")
            return False

    def enhanced_search(self, context: SearchContext) -> List[SearchResult]:
        """Perform enhanced search with metadata integration"""
        results = []

        # Get base search results
        base_results = self._get_base_results(context)

        # Enhance with metadata if available
        if self.current_metadata and self.term_matcher:
            results = self._enhance_with_metadata(base_results, context)
        else:
            # Fallback to basic enhancement
            results = self._basic_enhancement(base_results, context)

        # Sort by confidence and relevance
        results.sort(key=lambda r: r.confidence, reverse=True)

        return results

    def _get_base_results(self, context: SearchContext) -> List[Dict[str, Any]]:
        """Get results from base search engine"""
        try:
            # Get search results from the base search engine
            if hasattr(self.base_engine, "search"):
                # Get basic search results
                search_results = self.base_engine.search(
                    query=context.query,
                    limit=context.limit * 2,  # Get more for metadata filtering
                    include_related=context.include_related,
                )

                # Convert SearchResult objects to dict format
                dict_results = []
                for result in search_results:
                    dict_results.append(
                        {
                            "file_path": result.path,
                            "line_number": 1,
                            "snippet": result.content_preview,
                            "content_type": result.type.lstrip(
                                "."
                            ),  # Remove leading dot
                            "confidence": min(
                                100.0, result.score / 10.0
                            ),  # Normalize score
                            "logical_unit": result.logical_unit,
                            "module": result.module,
                        }
                    )

                return dict_results
            else:
                # Fallback - mock results for testing
                logger.warning(
                    "Base search engine method not found - using mock results"
                )
                return [
                    {
                        "file_path": "CustomerOrder.entity",
                        "line_number": 1,
                        "snippet": f"entity CustomerOrder {context.query}",
                        "content_type": "entity",
                        "confidence": 50.0,
                        "logical_unit": "CustomerOrder",
                        "module": "ORDER",
                    }
                ]
        except Exception as e:
            logger.error(f"Error getting base results: {e}")
            return []

    def _enhance_with_metadata(
        self, base_results: List[Dict], context: SearchContext
    ) -> List[SearchResult]:
        """Enhance base results with metadata"""
        enhanced_results = []

        # Find relevant logical units for the query
        relevant_lus = self.term_matcher.find_logical_units_for_query(context.query)
        lu_context = {lu_name: score for lu_name, score in relevant_lus[:20]}  # Top 20

        for result in base_results:
            enhanced = self._enhance_single_result(result, context, lu_context)
            if enhanced:
                enhanced_results.append(enhanced)

        # Add metadata-driven results if we found relevant LUs
        if relevant_lus and context.include_related:
            metadata_results = self._generate_metadata_results(
                relevant_lus[:10], context
            )
            enhanced_results.extend(metadata_results)

        return enhanced_results

    def _enhance_single_result(
        self, result: Dict, context: SearchContext, lu_context: Dict[str, float]
    ) -> Optional[SearchResult]:
        """Enhance a single search result with metadata"""
        file_path = result.get("file_path", "")

        # Extract logical unit from file path
        logical_unit = self._extract_lu_from_path(file_path)

        # Find matching LU in metadata
        lu_info = None
        module = None
        business_description = None

        if logical_unit:
            for lu in self.current_metadata.logical_units:
                if lu.lu_name.lower() == logical_unit.lower():
                    lu_info = lu
                    module = lu.module
                    business_description = lu.lu_prompt
                    break

        # Calculate confidence based on metadata match
        base_confidence = result.get("confidence", 50.0)
        metadata_boost = 0.0

        if logical_unit in lu_context:
            metadata_boost = lu_context[logical_unit] * 0.3  # Up to 30% boost

        confidence = min(100.0, base_confidence + metadata_boost)

        # Get related entities
        related_entities = []
        if logical_unit and self.term_matcher:
            related_entities = self.term_matcher.get_related_entities(logical_unit)

        # Build search context
        search_context = []
        if module:
            search_context.append(f"Module: {module}")
        if logical_unit:
            search_context.append(f"Logical Unit: {logical_unit}")

        return SearchResult(
            file_path=file_path,
            content_type=result.get("content_type", "unknown"),
            line_number=result.get("line_number", 0),
            snippet=result.get("snippet", ""),
            confidence=confidence,
            logical_unit=logical_unit,
            module=module,
            business_description=business_description,
            related_entities=related_entities,
            search_context=search_context,
        )

    def _generate_metadata_results(
        self, relevant_lus: List[Tuple[str, float]], context: SearchContext
    ) -> List[SearchResult]:
        """Generate additional results based on metadata matches"""
        metadata_results = []

        for lu_name, score in relevant_lus:
            # Find the LU info
            lu_info = None
            for lu in self.current_metadata.logical_units:
                if lu.lu_name == lu_name:
                    lu_info = lu
                    break

            if not lu_info:
                continue

            # Create result for the LU itself
            confidence = min(
                95.0, score * 0.8
            )  # Metadata results get slightly lower confidence

            # Try to find actual files for this LU
            potential_files = [
                f"{lu_name}.entity",
                f"{lu_name}API.plsql",
                f"{lu_name}.client",
                f"{lu_name}.projection",
            ]

            for file_name in potential_files:
                related_entities = (
                    self.term_matcher.get_related_entities(lu_name)
                    if self.term_matcher
                    else []
                )

                search_context = [
                    f"Module: {lu_info.module}",
                    f"Metadata Match: {score:.1f}%",
                    "Suggested based on business terminology",
                ]

                metadata_results.append(
                    SearchResult(
                        file_path=file_name,
                        content_type=(
                            file_name.split(".")[-1] if "." in file_name else "entity"
                        ),
                        line_number=1,
                        snippet=f"// {lu_info.lu_prompt or lu_name} - {context.query}",
                        confidence=confidence,
                        logical_unit=lu_name,
                        module=lu_info.module,
                        business_description=lu_info.lu_prompt,
                        related_entities=related_entities,
                        search_context=search_context,
                    )
                )

                # Only suggest one file per LU to avoid clutter
                break

        return metadata_results

    def _basic_enhancement(
        self, base_results: List[Dict], context: SearchContext
    ) -> List[SearchResult]:
        """Basic enhancement without metadata (fallback)"""
        enhanced_results = []

        for result in base_results:
            file_path = result.get("file_path", "")
            logical_unit = self._extract_lu_from_path(file_path)

            enhanced_results.append(
                SearchResult(
                    file_path=file_path,
                    content_type=result.get("content_type", "unknown"),
                    line_number=result.get("line_number", 0),
                    snippet=result.get("snippet", ""),
                    confidence=result.get("confidence", 50.0),
                    logical_unit=logical_unit,
                    module=self._guess_module_from_path(file_path),
                    business_description=None,
                    related_entities=[],
                    search_context=["Basic search - no metadata available"],
                )
            )

        return enhanced_results

    def _extract_lu_from_path(self, file_path: str) -> Optional[str]:
        """Extract logical unit name from file path"""
        path = Path(file_path)
        base_name = path.stem

        # Remove common suffixes
        suffixes_to_remove = ["API", "Client", "Handling", "Storage", "View"]
        for suffix in suffixes_to_remove:
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]

        return base_name if base_name else None

    def _guess_module_from_path(self, file_path: str) -> Optional[str]:
        """Guess module from file path (basic heuristics)"""
        path_lower = file_path.lower()

        # Simple heuristics based on common patterns
        if "customer" in path_lower or "order" in path_lower:
            return "ORDER"
        elif "person" in path_lower or "employee" in path_lower:
            return "PERSON"
        elif "purchase" in path_lower or "supplier" in path_lower:
            return "PURCH"
        elif "inventory" in path_lower or "part" in path_lower:
            return "INVENT"

        return None

    def get_module_statistics(self) -> Dict[str, Any]:
        """Get statistics about available modules"""
        if not self.current_metadata:
            return {}

        module_stats = defaultdict(
            lambda: {"lu_count": 0, "description": "", "sample_lus": []}
        )

        for lu in self.current_metadata.logical_units:
            module_stats[lu.module]["lu_count"] += 1
            if len(module_stats[lu.module]["sample_lus"]) < 3:
                module_stats[lu.module]["sample_lus"].append(lu.lu_name)

        return dict(module_stats)

    def suggest_related_searches(self, query: str, limit: int = 5) -> List[str]:
        """Suggest related search terms based on metadata"""
        if not self.term_matcher:
            return []

        suggestions = set()

        # Find relevant LUs
        relevant_lus = self.term_matcher.find_logical_units_for_query(
            query, threshold=60.0
        )

        # Get terms from related LUs
        for lu_name, _ in relevant_lus[:3]:  # Top 3 matches
            related_entities = self.term_matcher.get_related_entities(lu_name)
            for entity in related_entities[:2]:  # 2 related per LU
                if entity.lower() not in query.lower():
                    suggestions.add(entity)

        return list(suggestions)[:limit]
