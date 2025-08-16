"""
IFS Cloud Database Metadata Extractor

Extracts metadata from IFS Cloud database for offline search enhancement.
Supports multiple database backends and IFS versions.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class LogicalUnit:
    """Represents an IFS Cloud Logical Unit with metadata"""

    module: str
    lu_name: str
    lu_prompt: Optional[str] = None
    base_table: Optional[str] = None
    base_view: Optional[str] = None
    logical_unit_type: Optional[str] = None
    custom_fields: Optional[str] = None
    search_weight: float = 1.0

    def to_search_terms(self) -> List[str]:
        """Extract searchable terms from this logical unit"""
        terms = []

        # Add primary identifiers
        terms.append(self.lu_name)
        if self.lu_prompt:
            terms.extend(self.lu_prompt.lower().split())

        # Add module context
        terms.append(self.module.lower())

        # Add table/view names if available
        if self.base_table:
            terms.append(self.base_table.lower())
        if self.base_view:
            terms.append(self.base_view.lower())

        return list(set(terms))  # Remove duplicates


@dataclass
class ModuleInfo:
    """Business module information"""

    name: str
    lu_count: int
    description: Optional[str] = None
    category: Optional[str] = None
    search_weight: float = 1.0


@dataclass
class DomainMapping:
    """Database-to-client value mapping"""

    lu_name: str
    package_name: str
    db_value: str
    client_value: str

    def to_search_terms(self) -> List[str]:
        """Extract searchable terms from domain mapping"""
        return [
            self.db_value.lower(),
            self.client_value.lower(),
            self.lu_name.lower(),
            self.package_name.lower(),
        ]


@dataclass
class ViewInfo:
    """View metadata for UI context"""

    lu_name: str
    view_name: str
    view_type: str
    view_prompt: Optional[str] = None
    view_comment: Optional[str] = None


@dataclass
class NavigatorEntry:
    """Navigator entry linking GUI to backend entities"""

    label: str
    projection: str
    entity_name: str
    page_type: Optional[str] = None
    entry_type: Optional[str] = None
    sort_order: Optional[int] = None

    def to_search_terms(self) -> List[str]:
        """Extract searchable terms from navigator entry"""
        terms = []

        # Add GUI label terms
        terms.extend(self.label.lower().split())

        # Add backend entity terms
        terms.append(self.entity_name.lower())
        terms.append(self.projection.lower())

        # Add type information
        if self.page_type:
            terms.append(self.page_type.lower())

        return list(set(terms))  # Remove duplicates


@dataclass
class MetadataExtract:
    """Container for all extracted metadata"""

    ifs_version: str
    extract_date: datetime
    logical_units: List[LogicalUnit]
    modules: List[ModuleInfo]
    domain_mappings: List[DomainMapping]
    views: List[ViewInfo]
    navigator_entries: List[NavigatorEntry]
    checksum: str

    def save_to_file(self, filepath: Path) -> None:
        """Save extract to JSON file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "ifs_version": self.ifs_version,
            "extract_date": self.extract_date.isoformat(),
            "checksum": self.checksum,
            "logical_units": [asdict(lu) for lu in self.logical_units],
            "modules": [asdict(mod) for mod in self.modules],
            "domain_mappings": [asdict(dm) for dm in self.domain_mappings],
            "views": [asdict(view) for view in self.views],
            "navigator_entries": [asdict(nav) for nav in self.navigator_entries],
            "stats": {
                "logical_units_count": len(self.logical_units),
                "modules_count": len(self.modules),
                "domain_mappings_count": len(self.domain_mappings),
                "views_count": len(self.views),
                "navigator_entries_count": len(self.navigator_entries),
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata extract saved to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: Path) -> "MetadataExtract":
        """Load extract from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            ifs_version=data["ifs_version"],
            extract_date=datetime.fromisoformat(data["extract_date"]),
            checksum=data["checksum"],
            logical_units=[LogicalUnit(**lu) for lu in data["logical_units"]],
            modules=[ModuleInfo(**mod) for mod in data["modules"]],
            domain_mappings=[DomainMapping(**dm) for dm in data["domain_mappings"]],
            views=[ViewInfo(**view) for view in data["views"]],
            navigator_entries=[
                NavigatorEntry(**nav) for nav in data.get("navigator_entries", [])
            ],
        )


class DatabaseMetadataExtractor:
    """Extracts metadata from IFS Cloud database"""

    def __init__(self, db_connection=None):
        """
        Initialize extractor

        Args:
            db_connection: Database connection object (optional)
        """
        self.db_connection = db_connection
        self.extraction_queries = self._get_default_queries()

    def _get_default_queries(self) -> Dict[str, str]:
        """Get default SQL queries for metadata extraction"""
        return {
            "logical_units": """
                SELECT module, lu_name, lu_prompt, base_table, base_view, 
                       logical_unit_type, custom_fields
                FROM dictionary_sys_lu_active
                ORDER BY module, lu_name
            """,
            "modules": """
                SELECT module,
                       COUNT(*) as lu_count,
                       COUNT(DISTINCT base_table) as table_count,
                       LISTAGG(DISTINCT lu_type, ',') WITHIN GROUP (ORDER BY lu_type) as lu_types
                FROM dictionary_sys_lu_active 
                GROUP BY module 
                ORDER BY lu_count DESC
            """,
            "domain_mappings": """
                SELECT lu_name, package_name, db_value, client_value
                FROM dictionary_sys_domain_tab
                WHERE ROWNUM <= 10000  -- Limit for performance
                ORDER BY lu_name, package_name
            """,
            "views": """
                SELECT lu_name, view_name, view_type, view_prompt, view_comment
                FROM dictionary_sys_view_tab
                WHERE view_type IN ('A', 'B', 'R', 'S')  -- Main view types
                  AND ROWNUM <= 5000  -- Limit for performance
                ORDER BY lu_name, view_name
            """,
            "navigator_entries": """
                SELECT nav.label, 
                       nav.projection,
                       pes.entity_name,
                       nav.page_type,
                       nav.entry_type,
                       nav.sort_order
                FROM fnd_navigator_all nav
                JOIN md_projection_entityset pes ON nav.projection = pes.projection_name
                WHERE nav.label IS NOT NULL
                  AND nav.entry_type IN ('PAGE', 'LIST')
                  AND nav.label NOT LIKE '%NavEntry%'
                  AND nav.label NOT LIKE '%Analysis%'
                  AND LENGTH(nav.label) BETWEEN 5 AND 50
                  AND pes.entity_name NOT LIKE '%Virtual%'
                  AND pes.entity_name NOT LIKE '%Lov%'
                  AND pes.entity_name NOT LIKE '%Query%'
                  AND pes.entity_name NOT LIKE '%Lookup%'
                  AND ROWNUM <= 5000  -- Limit for performance
                ORDER BY nav.label, pes.entity_name
            """,
        }

    def extract_from_database(self, ifs_version: str) -> MetadataExtract:
        """
        Extract metadata directly from database

        Args:
            ifs_version: IFS Cloud version identifier

        Returns:
            MetadataExtract object with all extracted data
        """
        if not self.db_connection:
            raise ValueError("Database connection required for extraction")

        logger.info(f"Starting metadata extraction for IFS {ifs_version}")

        # Extract logical units
        logical_units = self._extract_logical_units()
        logger.info(f"Extracted {len(logical_units)} logical units")

        # Extract modules
        modules = self._extract_modules()
        logger.info(f"Extracted {len(modules)} modules")

        # Extract domain mappings
        domain_mappings = self._extract_domain_mappings()
        logger.info(f"Extracted {len(domain_mappings)} domain mappings")

        # Extract views
        views = self._extract_views()
        logger.info(f"Extracted {len(views)} views")

        # Extract navigator entries
        navigator_entries = self._extract_navigator_entries()
        logger.info(f"Extracted {len(navigator_entries)} navigator entries")

        # Create extract
        extract_date = datetime.now()
        checksum = self._calculate_checksum(
            logical_units, modules, domain_mappings, views, navigator_entries
        )

        return MetadataExtract(
            ifs_version=ifs_version,
            extract_date=extract_date,
            logical_units=logical_units,
            modules=modules,
            domain_mappings=domain_mappings,
            views=views,
            navigator_entries=navigator_entries,
            checksum=checksum,
        )

    def extract_from_mcp_queries(
        self, ifs_version: str, query_results: Dict[str, List[Dict]]
    ) -> MetadataExtract:
        """
        Extract metadata from pre-executed MCP query results

        Args:
            ifs_version: IFS Cloud version identifier
            query_results: Dictionary with query results from MCP calls

        Returns:
            MetadataExtract object with processed data
        """
        logger.info(f"Processing MCP query results for IFS {ifs_version}")

        # Process logical units
        logical_units = []
        for row in query_results.get("logical_units", []):
            logical_units.append(
                LogicalUnit(
                    module=row.get("MODULE", ""),
                    lu_name=row.get("LU_NAME", ""),
                    lu_prompt=row.get("LU_PROMPT"),
                    base_table=row.get("BASE_TABLE"),
                    base_view=row.get("BASE_VIEW"),
                    logical_unit_type=row.get("LOGICAL_UNIT_TYPE"),
                    custom_fields=row.get("CUSTOM_FIELDS"),
                )
            )

        # Process modules
        modules = []
        for row in query_results.get("modules", []):
            modules.append(
                ModuleInfo(
                    name=row.get("MODULE", ""),
                    lu_count=int(row.get("LU_COUNT", 0)),
                    description=f"{row.get('LU_COUNT', 0)} logical units",
                )
            )

        # Process domain mappings
        domain_mappings = []
        for row in query_results.get("domain_mappings", []):
            domain_mappings.append(
                DomainMapping(
                    lu_name=row.get("LU_NAME", ""),
                    package_name=row.get("PACKAGE_NAME", ""),
                    db_value=row.get("DB_VALUE", ""),
                    client_value=row.get("CLIENT_VALUE", ""),
                )
            )

        # Process views
        views = []
        for row in query_results.get("views", []):
            views.append(
                ViewInfo(
                    lu_name=row.get("LU_NAME", ""),
                    view_name=row.get("VIEW_NAME", ""),
                    view_type=row.get("VIEW_TYPE", ""),
                    view_prompt=row.get("VIEW_PROMPT"),
                    view_comment=row.get("VIEW_COMMENT"),
                )
            )

        # Create extract
        extract_date = datetime.now()
        checksum = self._calculate_checksum(
            logical_units, modules, domain_mappings, views
        )

        return MetadataExtract(
            ifs_version=ifs_version,
            extract_date=extract_date,
            logical_units=logical_units,
            modules=modules,
            domain_mappings=domain_mappings,
            views=views,
            checksum=checksum,
        )

    def _extract_logical_units(self) -> List[LogicalUnit]:
        """Extract logical units from database"""
        from sqlalchemy import text

        with self.db_connection.connect() as conn:
            result = conn.execute(text(self.extraction_queries["logical_units"]))

            logical_units = []
            for row in result:
                logical_units.append(
                    LogicalUnit(
                        module=row[0] or "",
                        lu_name=row[1] or "",
                        lu_prompt=row[2],
                        base_table=row[3],
                        base_view=row[4],
                        logical_unit_type=row[5],
                        custom_fields=row[6],
                    )
                )

            return logical_units

    def _extract_modules(self) -> List[ModuleInfo]:
        """Extract module information from database"""
        from sqlalchemy import text

        with self.db_connection.connect() as conn:
            result = conn.execute(text(self.extraction_queries["modules"]))

            modules = []
            for row in result:
                modules.append(
                    ModuleInfo(
                        name=row[0] or "",
                        lu_count=int(row[1] or 0),
                        description=f"{row[1]} logical units, {row[2]} tables",
                    )
                )

            return modules

    def _extract_domain_mappings(self) -> List[DomainMapping]:
        """Extract domain mappings from database"""
        from sqlalchemy import text

        with self.db_connection.connect() as conn:
            result = conn.execute(text(self.extraction_queries["domain_mappings"]))

            mappings = []
            for row in result:
                mappings.append(
                    DomainMapping(
                        lu_name=row[0] or "",
                        package_name=row[1] or "",
                        db_value=row[2] or "",
                        client_value=row[3] or "",
                    )
                )

            return mappings

    def _extract_views(self) -> List[ViewInfo]:
        """Extract view information from database"""
        from sqlalchemy import text

        with self.db_connection.connect() as conn:
            result = conn.execute(text(self.extraction_queries["views"]))

            views = []
            for row in result:
                views.append(
                    ViewInfo(
                        lu_name=row[0] or "",
                        view_name=row[1] or "",
                        view_type=row[2] or "",
                        view_prompt=row[3],
                        view_comment=row[4],
                    )
                )

            return views

    def _extract_navigator_entries(self) -> List[NavigatorEntry]:
        """Extract navigator entries from database"""
        from sqlalchemy import text

        with self.db_connection.connect() as conn:
            result = conn.execute(text(self.extraction_queries["navigator_entries"]))

            navigator_entries = []
            for row in result:
                navigator_entries.append(
                    NavigatorEntry(
                        label=row[0] or "",
                        projection=row[1] or "",
                        entity_name=row[2] or "",
                        page_type=row[3],
                        entry_type=row[4],
                        sort_order=int(row[5]) if row[5] is not None else None,
                    )
                )

            return navigator_entries

    def _calculate_checksum(
        self,
        logical_units: List[LogicalUnit],
        modules: List[ModuleInfo],
        domain_mappings: List[DomainMapping],
        views: List[ViewInfo],
        navigator_entries: List[NavigatorEntry] = None,
    ) -> str:
        """Calculate checksum for metadata consistency verification"""
        nav_count = len(navigator_entries) if navigator_entries else 0
        data_str = f"{len(logical_units)}-{len(modules)}-{len(domain_mappings)}-{len(views)}-{nav_count}"
        if logical_units:
            data_str += f"-{logical_units[0].lu_name}-{logical_units[-1].lu_name}"

        return hashlib.md5(data_str.encode()).hexdigest()


class MetadataManager:
    """Manages metadata extraction, storage, and loading"""

    def __init__(self, metadata_dir: Path):
        """Initialize metadata manager with storage directory"""
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, MetadataExtract] = {}

    def get_metadata_path(self, ifs_version: str) -> Path:
        """Get path for metadata file for specific IFS version"""
        return self.metadata_dir / f"{ifs_version}" / "metadata_extract.json"

    def has_metadata(self, ifs_version: str) -> bool:
        """Check if metadata exists for specific IFS version"""
        return self.get_metadata_path(ifs_version).exists()

    def save_metadata(self, extract: MetadataExtract) -> Path:
        """Save metadata extract to file"""
        filepath = self.get_metadata_path(extract.ifs_version)
        extract.save_to_file(filepath)

        # Update cache
        self._cache[extract.ifs_version] = extract

        return filepath

    def load_metadata(
        self, ifs_version: str, use_cache: bool = True
    ) -> Optional[MetadataExtract]:
        """Load metadata for specific IFS version"""
        # Check cache first
        if use_cache and ifs_version in self._cache:
            return self._cache[ifs_version]

        filepath = self.get_metadata_path(ifs_version)
        if not filepath.exists():
            return None

        try:
            extract = MetadataExtract.load_from_file(filepath)
            self._cache[ifs_version] = extract
            return extract
        except Exception as e:
            logger.error(f"Failed to load metadata for {ifs_version}: {e}")
            return None

    def get_available_versions(self) -> List[str]:
        """Get list of available IFS versions with metadata"""
        versions = []
        for version_dir in self.metadata_dir.iterdir():
            if (
                version_dir.is_dir()
                and (version_dir / "metadata_extract.json").exists()
            ):
                versions.append(version_dir.name)

        return sorted(versions)

    def cleanup_old_versions(self, keep_latest: int = 3) -> None:
        """Remove old metadata versions, keeping only the latest N"""
        versions = self.get_available_versions()
        if len(versions) <= keep_latest:
            return

        # Remove oldest versions
        to_remove = versions[:-keep_latest]
        for version in to_remove:
            version_dir = self.metadata_dir / version
            if version_dir.exists():
                import shutil

                shutil.rmtree(version_dir)
                logger.info(f"Removed old metadata version: {version}")

                # Remove from cache
                self._cache.pop(version, None)
