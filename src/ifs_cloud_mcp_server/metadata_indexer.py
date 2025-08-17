"""
Metadata-Specific Indexer using Tantivy

This module creates a dedicated Tantivy index for metadata to enable
faster metadata searches and better query performance.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

import tantivy


logger = logging.getLogger(__name__)


@dataclass
class MetadataExportCache:
    """Cache metadata for metadata export files."""

    export_path: str
    checksum: str
    last_modified: datetime
    document_count: int
    ifs_version: str


@dataclass
class MetadataDocument:
    """Structured metadata document for indexing."""

    path: str
    file_type: str
    module: Optional[str] = None
    logical_unit: Optional[str] = None
    component: Optional[str] = None
    entities: Optional[List[str]] = None
    functions: Optional[List[str]] = None
    pages: Optional[List[str]] = None
    lists: Optional[List[str]] = None
    groups: Optional[List[str]] = None
    trees: Optional[List[str]] = None
    navigators: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None
    size_kb: Optional[float] = None
    line_count: Optional[int] = None
    complexity_score: Optional[float] = None

    def to_search_fields(self) -> Dict[str, str]:
        """Convert metadata to searchable text fields."""
        fields = {}

        # Create searchable text from lists
        if self.entities:
            fields["entities_text"] = " ".join(self.entities)
        if self.functions:
            fields["functions_text"] = " ".join(self.functions)
        if self.pages:
            fields["pages_text"] = " ".join(self.pages)
        if self.lists:
            fields["lists_text"] = " ".join(self.lists)
        if self.groups:
            fields["groups_text"] = " ".join(self.groups)
        if self.trees:
            fields["trees_text"] = " ".join(self.trees)
        if self.navigators:
            fields["navigators_text"] = " ".join(self.navigators)
        if self.dependencies:
            fields["dependencies_text"] = " ".join(self.dependencies)

        # Combine all searchable content
        all_content = []
        all_content.extend(self.entities or [])
        all_content.extend(self.functions or [])
        all_content.extend(self.pages or [])
        all_content.extend(self.lists or [])
        all_content.extend(self.groups or [])
        all_content.extend(self.trees or [])
        all_content.extend(self.navigators or [])

        fields["all_metadata"] = " ".join(all_content)

        return fields


class MetadataIndexer:
    """Dedicated Tantivy indexer for file metadata."""

    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Cache infrastructure (similar to main indexer)
        self.cache_metadata_path = self.index_path / "metadata_cache.json"
        self._export_cache: Dict[str, MetadataExportCache] = {}
        self._load_cache_metadata()

        logger.info(f"Metadata cache contains {len(self._export_cache)} export entries")

        # Create schema
        self.schema = self._create_schema()

        # Create or open index
        try:
            self.index = tantivy.Index.open(str(self.index_path))
            logger.info(f"Opened existing metadata index at {self.index_path}")
        except:
            self.index = tantivy.Index(self.schema, path=str(self.index_path))
            logger.info(f"Created new metadata index at {self.index_path}")

        self.writer = self.index.writer(50_000_000, 1)  # 50MB heap, 1 thread

    def _create_schema(self) -> tantivy.Schema:
        """Create the Tantivy schema for metadata indexing."""
        schema_builder = tantivy.SchemaBuilder()

        # Path and basic info
        schema_builder.add_text_field("path", stored=True)
        schema_builder.add_text_field("file_type", stored=True)
        schema_builder.add_text_field("module", stored=True)
        schema_builder.add_text_field("logical_unit", stored=True)
        schema_builder.add_text_field("component", stored=True)

        # Searchable metadata fields
        schema_builder.add_text_field("entities_text", stored=False)
        schema_builder.add_text_field("functions_text", stored=False)
        schema_builder.add_text_field("pages_text", stored=False)
        schema_builder.add_text_field("lists_text", stored=False)
        schema_builder.add_text_field("groups_text", stored=False)
        schema_builder.add_text_field("trees_text", stored=False)
        schema_builder.add_text_field("navigators_text", stored=False)
        schema_builder.add_text_field("dependencies_text", stored=False)
        schema_builder.add_text_field("all_metadata", stored=False)

        # Structured data (stored for retrieval)
        schema_builder.add_json_field("metadata_json", stored=True)

        # Numerical fields for filtering/sorting
        schema_builder.add_float_field("size_kb", stored=True)
        schema_builder.add_integer_field("line_count", stored=True)
        schema_builder.add_float_field("complexity_score", stored=True)

        return schema_builder.build()

    def add_document(self, metadata: MetadataDocument):
        """Add a metadata document to the index."""
        try:
            # Create document fields dict for compatibility with Tantivy API
            doc_dict = {
                "path": metadata.path,
                "file_type": metadata.file_type,
            }

            # Add optional fields
            if metadata.module:
                doc_dict["module"] = metadata.module
            if metadata.logical_unit:
                doc_dict["logical_unit"] = metadata.logical_unit
            if metadata.component:
                doc_dict["component"] = metadata.component

            # Searchable text fields
            search_fields = metadata.to_search_fields()
            for field_name, content in search_fields.items():
                if content and content.strip():
                    doc_dict[field_name] = content

            # Store full metadata as JSON
            doc_dict["metadata_json"] = json.dumps(asdict(metadata))

            # Numerical fields
            if metadata.size_kb is not None:
                doc_dict["size_kb"] = metadata.size_kb
            if metadata.line_count is not None:
                doc_dict["line_count"] = metadata.line_count
            if metadata.complexity_score is not None:
                doc_dict["complexity_score"] = metadata.complexity_score

            # Add document using dict approach like main indexer
            doc = tantivy.Document(**doc_dict)
            self.writer.add_document(doc)

        except Exception as e:
            logger.error(f"Failed to add metadata document for {metadata.path}: {e}")

    def search(self, query: str, limit: int = 50) -> List[Dict]:
        """Simple search method for compatibility."""
        return self.search_metadata(query, limit=limit)

    def commit(self):
        """Commit changes to the index."""
        try:
            self.writer.commit()
            self.index.reload()
            logger.debug("Committed changes to metadata index")
        except Exception as e:
            logger.error(f"Failed to commit changes: {e}")

    def search_metadata(
        self,
        query: str,
        limit: int = 50,
        file_types: Optional[List[str]] = None,
        modules: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Search metadata using the dedicated index."""
        try:
            self.writer.commit()
            searcher = self.index.searcher()

            # Build query using the same approach as main indexer
            # Parse using multiple fields
            from .indexer import tantivy

            # Create a simple query across multiple fields
            queries = []
            for term in query.split():
                if term.strip():
                    for field in [
                        "all_metadata",
                        "entities_text",
                        "functions_text",
                        "module",
                    ]:
                        try:
                            term_query = tantivy.Query.term_query(
                                self.schema, field, term.lower()
                            )
                            queries.append(term_query)
                        except:
                            continue

            if queries:
                parsed_query = tantivy.Query.boolean_query(
                    [(tantivy.Occur.Should, q) for q in queries]
                )
            else:
                parsed_query = tantivy.Query.all_query()

            # Execute search
            search_results = searcher.search(parsed_query, limit)

            results = []
            for score, doc_address in search_results.hits:
                doc = searcher.doc(doc_address)

                # Parse stored metadata
                metadata_json = doc.get_first("metadata_json")
                if metadata_json:
                    metadata = json.loads(metadata_json)
                    metadata["score"] = float(score)

                    # Apply filters
                    if file_types and metadata.get("file_type") not in file_types:
                        continue
                    if modules and metadata.get("module") not in modules:
                        continue

                    results.append(metadata)

            logger.info(
                f"Metadata search for '{query}' returned {len(results)} results"
            )
            return results

        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []

    def search_by_entity(self, entity_name: str, limit: int = 20) -> List[Dict]:
        """Search for files containing a specific entity."""
        return self.search_metadata(f"entities_text:{entity_name}", limit)

    def search_by_function(self, function_name: str, limit: int = 20) -> List[Dict]:
        """Search for files containing a specific function."""
        return self.search_metadata(f"functions_text:{function_name}", limit)

    def search_by_module(self, module: str, limit: int = 50) -> List[Dict]:
        """Get all files in a specific module."""
        return self.search_metadata(f"module:{module}", limit)

    def get_modules(self) -> Set[str]:
        """Get all unique modules in the index."""
        try:
            self.writer.commit()
            searcher = self.index.searcher()

            # This is a simplified approach - use all documents query
            all_docs_query = tantivy.Query.all_query()

            results = searcher.search(all_docs_query, 10000)  # Get many results

            modules = set()
            for _, doc_address in results.hits:
                doc = searcher.doc(doc_address)
                module = doc.get_first("module")
                if module:
                    modules.add(module)

            return modules

        except Exception as e:
            logger.error(f"Failed to get modules: {e}")
            return set()

    def get_stats(self) -> Dict:
        """Get statistics about the metadata index."""
        try:
            self.writer.commit()
            searcher = self.index.searcher()

            total_docs = searcher.num_docs

            # Handle empty index case
            if total_docs == 0:
                return {
                    "total_files": 0,
                    "total_lines": 0,
                    "file_types": {},
                    "modules": [],
                    "avg_file_size": 0,
                    "largest_file": "",
                    "latest_modified": None,
                }

            # Get file type distribution
            all_docs_query = tantivy.Query.all_query()

            results = searcher.search(all_docs_query, total_docs)

            file_types = {}
            modules = set()

            for _, doc_address in results.hits:
                doc = searcher.doc(doc_address)
                file_type = doc.get_first("file_type")
                module = doc.get_first("module")

                if file_type:
                    file_types[file_type] = file_types.get(file_type, 0) + 1
                if module:
                    modules.add(module)

            return {
                "total_documents": total_docs,
                "file_type_distribution": file_types,
                "unique_modules": len(modules),
                "modules": sorted(modules),
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_documents": 0}

    def commit(self):
        """Commit pending changes to the index."""
        self.writer.commit()
        logger.info("Metadata index changes committed")

    def build_from_metadata_export(self, metadata_extract_path: Path):
        """Build the metadata index from a metadata export JSON file.

        This is the proper way to populate the metadata indexer - from the
        database metadata export, not from source code files.

        Args:
            metadata_extract_path: Path to the metadata_extract.json file
                generated by the CLI export command
        """
        try:
            logger.info(f"Building metadata index from export: {metadata_extract_path}")

            # Check if this export is already cached and current
            if self._is_export_cached_and_current(metadata_extract_path):
                logger.info("Export is cached and current, skipping rebuild")
                return 0

            # Clear existing index before rebuilding
            logger.info("Clearing existing metadata index for rebuild")
            try:
                self.writer.delete_all_documents()
                self.commit()
            except Exception as e:
                logger.warning(f"Failed to clear existing documents: {e}")
                # If we can't clear, recreate the writer
                self.writer = self.index.writer(50_000_000, 1)

            # Load the metadata export
            with open(metadata_extract_path, "r", encoding="utf-8") as f:
                metadata_export = json.load(f)

            # Extract basic info
            ifs_version = metadata_export.get("ifs_version", "unknown")
            extraction_date = metadata_export.get("extraction_date", "")

            logger.info(
                f"Processing metadata export for IFS {ifs_version} from {extraction_date}"
            )

            processed_count = 0

            # Process logical units
            for lu in metadata_export.get("logical_units", []):
                try:
                    # Create metadata document for logical unit
                    metadata_doc = MetadataDocument(
                        path=f"virtual://{lu.get('module', 'unknown')}/{lu.get('lu_name', 'unknown')}.entity",
                        file_type=".entity",
                        module=lu.get("module"),
                        logical_unit=lu.get("lu_name"),
                        component=lu.get("lu_name"),
                        entities=[lu.get("lu_name")] if lu.get("lu_name") else [],
                        functions=[],  # Functions would come from views/projections
                        pages=[],
                        lists=[],
                        groups=[],
                        trees=[],
                        navigators=[],
                        dependencies=[],
                        size_kb=1.0,  # Virtual size
                        line_count=100,  # Estimated
                        complexity_score=50.0,  # Default complexity
                    )

                    self.add_document(metadata_doc)
                    processed_count += 1

                except Exception as e:
                    logger.warning(f"Failed to process logical unit {lu}: {e}")
                    continue

            # Process navigator entries (UI to backend mappings)
            for nav in metadata_export.get("navigator_entries", []):
                try:
                    # Create metadata document for navigator entry
                    metadata_doc = MetadataDocument(
                        path=f"virtual://ui/{nav.get('entity_name', 'unknown')}.client",
                        file_type=".client",
                        module=nav.get("module"),
                        logical_unit=nav.get("entity_name"),
                        component="Navigator",
                        entities=(
                            [nav.get("entity_name")] if nav.get("entity_name") else []
                        ),
                        functions=[],
                        pages=[nav.get("label")] if nav.get("label") else [],
                        lists=[],
                        groups=[],
                        trees=[],
                        navigators=[nav.get("label")] if nav.get("label") else [],
                        dependencies=[],
                        size_kb=0.5,  # Small virtual size
                        line_count=50,  # Estimated
                        complexity_score=25.0,  # Lower complexity for UI
                    )

                    self.add_document(metadata_doc)
                    processed_count += 1

                except Exception as e:
                    logger.warning(f"Failed to process navigator entry {nav}: {e}")
                    continue

            # Process views
            for view in metadata_export.get("views", []):
                try:
                    # Create metadata document for view
                    metadata_doc = MetadataDocument(
                        path=f"virtual://{view.get('module', 'unknown')}/{view.get('view_name', 'unknown')}.views",
                        file_type=".views",
                        module=view.get("module"),
                        logical_unit=view.get("lu_name"),
                        component=view.get("view_name"),
                        entities=[view.get("lu_name")] if view.get("lu_name") else [],
                        functions=(
                            [view.get("view_name")] if view.get("view_name") else []
                        ),
                        pages=[],
                        lists=[],
                        groups=[],
                        trees=[],
                        navigators=[],
                        dependencies=[],
                        size_kb=2.0,  # Estimated size
                        line_count=200,  # Estimated
                        complexity_score=75.0,  # Higher complexity for views
                    )

                    self.add_document(metadata_doc)
                    processed_count += 1

                except Exception as e:
                    logger.warning(f"Failed to process view {view}: {e}")
                    continue

            # Commit all changes
            self.commit()

            logger.info(
                f"Successfully built metadata index with {processed_count} documents"
            )
            logger.info(f"Index populated from IFS {ifs_version} database export")

            # Update cache with the processed export
            self._update_export_cache(
                metadata_extract_path, processed_count, ifs_version
            )

            return processed_count

        except Exception as e:
            logger.error(f"Failed to build metadata index from export: {e}")
            raise

    def _load_cache_metadata(self):
        """Load cache metadata from disk."""
        if not self.cache_metadata_path.exists():
            self._export_cache = {}
            return

        try:
            with open(self.cache_metadata_path, "r") as f:
                data = json.load(f)
                self._export_cache = {
                    path: MetadataExportCache(**metadata)
                    for path, metadata in data.items()
                }
            logger.info(f"Loaded cache metadata for {len(self._export_cache)} exports")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load cache metadata: {e}. Starting fresh.")
            self._export_cache = {}

    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        try:
            # Convert cache to serializable format
            cache_data = {
                path: {
                    "export_path": cache.export_path,
                    "checksum": cache.checksum,
                    "last_modified": cache.last_modified.isoformat(),
                    "document_count": cache.document_count,
                    "ifs_version": cache.ifs_version,
                }
                for path, cache in self._export_cache.items()
            }
            with open(self.cache_metadata_path, "w") as f:
                json.dump(cache_data, f, indent=2)
            logger.debug(f"Saved cache metadata for {len(self._export_cache)} exports")
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _is_export_cached_and_current(self, export_path: Path) -> bool:
        """Check if metadata export is cached and current."""
        export_str = str(export_path)

        if export_str not in self._export_cache:
            return False

        if not export_path.exists():
            return False

        try:
            # Check if file has been modified
            file_stat = export_path.stat()
            file_modified = datetime.fromtimestamp(file_stat.st_mtime)
            cached_modified = self._export_cache[export_str].last_modified

            if file_modified > cached_modified:
                return False

            # Check checksum
            with open(export_path, "rb") as f:
                content = f.read()
                current_checksum = hashlib.md5(content).hexdigest()

            return current_checksum == self._export_cache[export_str].checksum

        except Exception as e:
            logger.debug(f"Failed to check export cache for {export_path}: {e}")
            return False

    def _update_export_cache(
        self, export_path: Path, document_count: int, ifs_version: str
    ):
        """Update cache metadata for a metadata export."""
        try:
            # Calculate checksum
            with open(export_path, "rb") as f:
                content = f.read()
                checksum = hashlib.md5(content).hexdigest()

            # Get file modification time
            file_stat = export_path.stat()
            last_modified = datetime.fromtimestamp(file_stat.st_mtime)

            # Update cache
            self._export_cache[str(export_path)] = MetadataExportCache(
                export_path=str(export_path),
                checksum=checksum,
                last_modified=last_modified,
                document_count=document_count,
                ifs_version=ifs_version,
            )

            self._save_cache_metadata()

        except Exception as e:
            logger.error(f"Failed to update export cache for {export_path}: {e}")

    def cleanup_cache(self) -> int:
        """Remove cache entries for export files that no longer exist.

        Returns:
            Number of stale entries removed
        """
        stale_exports = []

        for export_path in self._export_cache.keys():
            if not Path(export_path).exists():
                stale_exports.append(export_path)

        for export_path in stale_exports:
            del self._export_cache[export_path]

        if stale_exports:
            self._save_cache_metadata()
            logger.info(f"Removed {len(stale_exports)} stale cache entries")

        return len(stale_exports)

    def flush_cache(self):
        """Flush all cache data and force rebuild on next operation."""
        try:
            # Clear in-memory cache
            self._export_cache = {}

            # Remove cache file
            if self.cache_metadata_path.exists():
                self.cache_metadata_path.unlink()
                logger.info("Metadata indexer cache flushed")

            # Clear the index to force rebuild
            if (self.index_path / "meta.json").exists():
                import shutil

                shutil.rmtree(self.index_path)
                self.index_path.mkdir(parents=True, exist_ok=True)

                # Recreate the index
                self.schema = self._create_schema()
                try:
                    self.index = tantivy.Index.open(str(self.index_path))
                except:
                    self.index = tantivy.Index(self.schema, path=str(self.index_path))
                self.writer = self.index.writer(50_000_000, 1)
                logger.info("Metadata index cleared and recreated")

        except Exception as e:
            logger.error(f"Failed to flush metadata cache: {e}")
            raise
