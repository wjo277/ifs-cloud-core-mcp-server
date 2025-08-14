"""Specialized parsers for different IFS Cloud file types."""

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass


@dataclass
class ParsedFile:
    """Result of parsing an IFS Cloud file."""

    entities: List[str]
    dependencies: List[str]
    functions: List[str]
    imports: List[str]
    metadata: Dict[str, Any]
    complexity_indicators: int


class IFSFileParser:
    """Base parser for IFS Cloud files."""

    def parse(self, content: str, file_type: str) -> ParsedFile:
        """Parse file content based on its type."""
        if file_type == ".entity":
            return self._parse_entity(content)
        elif file_type == ".plsql":
            return self._parse_plsql(content)
        elif file_type == ".views":
            return self._parse_views(content)
        elif file_type == ".storage":
            return self._parse_storage(content)
        elif file_type == ".fragment":
            return self._parse_fragment(content)
        elif file_type == ".client":
            return self._parse_client(content)
        elif file_type == ".projection":
            return self._parse_projection(content)
        elif file_type == ".plsvc":
            return self._parse_plsvc(content)
        else:
            return self._parse_generic(content)

    def _parse_entity(self, content: str) -> ParsedFile:
        """Parse .entity XML files."""
        entities = []
        dependencies = []
        functions = []
        imports = []
        metadata = {}
        complexity_indicators = 0

        try:
            # Parse XML content
            root = ET.fromstring(content)

            # Extract entity name
            name_elem = root.find(".//{urn:ifsworld-com:schemas:entity_entity}NAME")
            if name_elem is not None:
                entities.append(name_elem.text)

            # Extract component
            component_elem = root.find(
                ".//{urn:ifsworld-com:schemas:entity_entity}COMPONENT"
            )
            if component_elem is not None:
                metadata["component"] = component_elem.text

            # Extract based_on (inheritance)
            based_on_elem = root.find(
                ".//{urn:ifsworld-com:schemas:entity_entity}BASED_ON"
            )
            if based_on_elem is not None:
                dependencies.append(based_on_elem.text)

            # Extract attributes and their properties
            attributes = root.findall(
                ".//{urn:ifsworld-com:schemas:entity_entity}ATTRIBUTE"
            )
            metadata["attribute_count"] = len(attributes)

            # Extract references (foreign keys)
            references = root.findall(
                ".//{urn:ifsworld-com:schemas:entity_entity}REFERENCE"
            )
            for ref in references:
                name_elem = ref.find(".//{urn:ifsworld-com:schemas:entity_entity}NAME")
                ref_entity_elem = ref.find(
                    ".//{urn:ifsworld-com:schemas:entity_entity}ENTITY_NAME"
                )
                if name_elem is not None and ref_entity_elem is not None:
                    dependencies.append(ref_entity_elem.text)

            # Calculate complexity based on structure
            complexity_indicators = len(attributes) + len(references) * 2

        except ET.ParseError:
            # Fallback to text parsing if XML parsing fails
            entities = self._extract_camelcase_words(content)
            complexity_indicators = content.count("<ATTRIBUTE>") + content.count(
                "<REFERENCE>"
            )

        return ParsedFile(
            entities, dependencies, functions, imports, metadata, complexity_indicators
        )

    def _parse_plsql(self, content: str) -> ParsedFile:
        """Parse .plsql files (PL/SQL with IFS extensions)."""
        entities = []
        dependencies = []
        functions = []
        imports = []
        metadata = {}
        complexity_indicators = 0

        lines = content.split("\n")

        # Extract procedures and functions
        proc_func_pattern = re.compile(
            r"^\s*(PROCEDURE|FUNCTION)\s+(\w+)", re.IGNORECASE
        )
        override_pattern = re.compile(r"@(Override|Overtake)", re.IGNORECASE)

        for line in lines:
            # Find procedures and functions
            match = proc_func_pattern.match(line)
            if match:
                func_type, func_name = match.groups()
                functions.append(f"{func_type.lower()}:{func_name}")

            # Find override/overtake annotations
            if override_pattern.search(line):
                metadata["has_overrides"] = True

            # Find table/entity references
            if (
                "FROM " in line.upper()
                or "UPDATE " in line.upper()
                or "INSERT INTO " in line.upper()
            ):
                # Extract table names
                table_matches = re.findall(
                    r"(?:FROM|UPDATE|INSERT\s+INTO)\s+(\w+)", line, re.IGNORECASE
                )
                dependencies.extend(table_matches)

        # Calculate complexity indicators
        complexity_keywords = [
            "IF",
            "WHILE",
            "FOR",
            "LOOP",
            "CASE",
            "EXCEPTION",
            "CURSOR",
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
        ]
        complexity_indicators = sum(
            content.upper().count(keyword) for keyword in complexity_keywords
        )

        metadata["function_count"] = len(functions)
        metadata["has_sql"] = any(
            keyword in content.upper()
            for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE"]
        )

        return ParsedFile(
            entities, dependencies, functions, imports, metadata, complexity_indicators
        )

    def _parse_views(self, content: str) -> ParsedFile:
        """Parse .views files (IFS DSL for Oracle views)."""
        entities = []
        dependencies = []
        functions = []
        imports = []
        metadata = {}
        complexity_indicators = 0

        lines = content.split("\n")

        # Extract view definitions
        view_pattern = re.compile(r"^\s*VIEW\s+(\w+)", re.IGNORECASE)
        table_pattern = re.compile(r"FROM\s+(\w+)", re.IGNORECASE)
        join_pattern = re.compile(r"JOIN\s+(\w+)", re.IGNORECASE)
        override_pattern = re.compile(r"@(Override|Overtake)", re.IGNORECASE)

        for line in lines:
            # Extract view names
            view_match = view_pattern.match(line)
            if view_match:
                entities.append(view_match.group(1))

            # Extract table dependencies
            table_matches = table_pattern.findall(line)
            dependencies.extend(table_matches)

            join_matches = join_pattern.findall(line)
            dependencies.extend(join_matches)

            # Check for overrides
            if override_pattern.search(line):
                metadata["has_overrides"] = True

        # Calculate complexity
        complexity_keywords = [
            "SELECT",
            "JOIN",
            "WHERE",
            "GROUP BY",
            "ORDER BY",
            "HAVING",
            "UNION",
        ]
        complexity_indicators = sum(
            content.upper().count(keyword) for keyword in complexity_keywords
        )

        metadata["view_count"] = len(entities)
        metadata["has_joins"] = "JOIN" in content.upper()

        return ParsedFile(
            entities, dependencies, functions, imports, metadata, complexity_indicators
        )

    def _parse_storage(self, content: str) -> ParsedFile:
        """Parse .storage files (IFS DSL for Oracle DDL)."""
        entities = []
        dependencies = []
        functions = []
        imports = []
        metadata = {}
        complexity_indicators = 0

        lines = content.split("\n")

        # Extract table and index definitions
        table_pattern = re.compile(r"^\s*TABLE\s+(\w+)", re.IGNORECASE)
        index_pattern = re.compile(r"^\s*INDEX\s+(\w+)", re.IGNORECASE)
        sequence_pattern = re.compile(r"^\s*SEQUENCE\s+(\w+)", re.IGNORECASE)

        table_count = 0
        index_count = 0
        sequence_count = 0

        for line in lines:
            # Extract table names
            table_match = table_pattern.match(line)
            if table_match:
                entities.append(table_match.group(1))
                table_count += 1

            # Extract index names
            index_match = index_pattern.match(line)
            if index_match:
                entities.append(index_match.group(1))
                index_count += 1

            # Extract sequence names
            sequence_match = sequence_pattern.match(line)
            if sequence_match:
                entities.append(sequence_match.group(1))
                sequence_count += 1

        complexity_indicators = table_count * 3 + index_count + sequence_count

        metadata["table_count"] = table_count
        metadata["index_count"] = index_count
        metadata["sequence_count"] = sequence_count

        return ParsedFile(
            entities, dependencies, functions, imports, metadata, complexity_indicators
        )

    def _parse_fragment(self, content: str) -> ParsedFile:
        """Parse .fragment files (reusable Marble code)."""
        entities = []
        dependencies = []
        functions = []
        imports = []
        metadata = {}
        complexity_indicators = 0

        lines = content.split("\n")

        # Extract fragment name (usually from filename, but check content)
        fragment_pattern = re.compile(r"^\s*fragment\s+(\w+)", re.IGNORECASE)

        for line in lines:
            fragment_match = fragment_pattern.match(line)
            if fragment_match:
                entities.append(fragment_match.group(1))

        # Parse as both projection and client content since fragments can contain both
        projection_result = self._parse_projection_content(content)
        client_result = self._parse_client_content(content)

        # Merge results
        entities.extend(projection_result.entities)
        entities.extend(client_result.entities)
        dependencies.extend(projection_result.dependencies)
        dependencies.extend(client_result.dependencies)
        functions.extend(projection_result.functions)
        functions.extend(client_result.functions)

        complexity_indicators = (
            projection_result.complexity_indicators
            + client_result.complexity_indicators
        )

        metadata["is_fragment"] = True
        metadata["has_projection_content"] = len(projection_result.entities) > 0
        metadata["has_client_content"] = len(client_result.entities) > 0

        return ParsedFile(
            entities, dependencies, functions, imports, metadata, complexity_indicators
        )

    def _parse_projection(self, content: str) -> ParsedFile:
        """Parse .projection files (Marble data access layer)."""
        return self._parse_projection_content(content)

    def _parse_projection_content(self, content: str) -> ParsedFile:
        """Parse projection content (used by both .projection and .fragment files)."""
        entities = []
        dependencies = []
        functions = []
        imports = []
        metadata = {}
        complexity_indicators = 0

        lines = content.split("\n")

        # Extract projection name
        projection_pattern = re.compile(r"^\s*projection\s+(\w+)", re.IGNORECASE)
        # Extract entity sets
        entityset_pattern = re.compile(
            r"^\s*entityset\s+(\w+)\s+for\s+(\w+)", re.IGNORECASE
        )
        # Extract entity references
        entity_pattern = re.compile(r"^\s*entity\s+(\w+)", re.IGNORECASE)
        # Extract includes (fragment dependencies)
        include_pattern = re.compile(r"^\s*include\s+fragment\s+(\w+)", re.IGNORECASE)

        entityset_count = 0
        entity_count = 0

        for line in lines:
            # Extract projection name
            proj_match = projection_pattern.match(line)
            if proj_match:
                entities.append(proj_match.group(1))

            # Extract entitysets
            entityset_match = entityset_pattern.match(line)
            if entityset_match:
                entityset_name, entity_name = entityset_match.groups()
                entities.append(entityset_name)
                dependencies.append(entity_name)
                entityset_count += 1

            # Extract entity overrides
            entity_match = entity_pattern.match(line)
            if entity_match:
                dependencies.append(entity_match.group(1))
                entity_count += 1

            # Extract fragment includes
            include_match = include_pattern.match(line)
            if include_match:
                dependencies.append(include_match.group(1))

        # Calculate complexity based on structure
        complexity_keywords = [
            "entityset",
            "entity",
            "action",
            "function",
            "reference",
            "attribute",
        ]
        complexity_indicators = sum(
            content.lower().count(keyword) for keyword in complexity_keywords
        )

        metadata["entityset_count"] = entityset_count
        metadata["entity_override_count"] = entity_count
        metadata["is_projection"] = True

        return ParsedFile(
            entities, dependencies, functions, imports, metadata, complexity_indicators
        )

    def _parse_client(self, content: str) -> ParsedFile:
        """Parse .client files (Marble frontend)."""
        return self._parse_client_content(content)

    def _parse_client_content(self, content: str) -> ParsedFile:
        """Parse client content (used by both .client and .fragment files)."""
        entities = []
        dependencies = []
        functions = []
        imports = []
        metadata = {}
        complexity_indicators = 0

        lines = content.split("\n")

        # Extract client name
        client_pattern = re.compile(r"^\s*client\s+(\w+)", re.IGNORECASE)
        # Extract navigation entries
        navigator_pattern = re.compile(r"^\s*navigator\s+(\w+)", re.IGNORECASE)
        # Extract page definitions
        page_pattern = re.compile(r"^\s*page\s+(\w+)", re.IGNORECASE)

        page_count = 0
        navigator_count = 0

        for line in lines:
            # Extract client name
            client_match = client_pattern.match(line)
            if client_match:
                entities.append(client_match.group(1))

            # Extract navigators
            nav_match = navigator_pattern.match(line)
            if nav_match:
                entities.append(nav_match.group(1))
                navigator_count += 1

            # Extract pages
            page_match = page_pattern.match(line)
            if page_match:
                entities.append(page_match.group(1))
                page_count += 1

        # Calculate complexity based on UI elements
        ui_keywords = ["page", "list", "form", "group", "field", "action", "command"]
        complexity_indicators = sum(
            content.lower().count(keyword) for keyword in ui_keywords
        )

        metadata["page_count"] = page_count
        metadata["navigator_count"] = navigator_count
        metadata["is_client"] = True

        return ParsedFile(
            entities, dependencies, functions, imports, metadata, complexity_indicators
        )

    def _parse_plsvc(self, content: str) -> ParsedFile:
        """Parse .plsvc files (PL/SQL service layer for projections)."""
        # Parse as PL/SQL but with additional projection-specific metadata
        result = self._parse_plsql(content)
        result.metadata["is_service_layer"] = True
        return result

    def _parse_generic(self, content: str) -> ParsedFile:
        """Generic parser for unknown file types."""
        entities = self._extract_camelcase_words(content)
        dependencies = []
        functions = []
        imports = []
        metadata = {}
        complexity_indicators = len(content.split("\n"))

        return ParsedFile(
            entities, dependencies, functions, imports, metadata, complexity_indicators
        )

    def _extract_camelcase_words(self, content: str) -> List[str]:
        """Extract CamelCase words which are likely entity names in IFS."""
        # Pattern for CamelCase words (starting with uppercase, containing lowercase)
        camelcase_pattern = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]*)*\b")
        matches = camelcase_pattern.findall(content)

        # Filter out common words that aren't likely to be entities
        common_words = {
            "And",
            "Or",
            "Not",
            "The",
            "For",
            "With",
            "From",
            "To",
            "In",
            "On",
            "At",
            "By",
        }
        return [
            word for word in set(matches) if word not in common_words and len(word) > 2
        ]
