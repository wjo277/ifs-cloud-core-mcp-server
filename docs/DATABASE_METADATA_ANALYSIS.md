# IFS Cloud Database Metadata Analysis Methodology

## Overview

This document provides a systematic approach to analyzing the IFS Cloud database for search-enhancing metadata. This methodology was developed through comprehensive exploration of the IFSCDEV Oracle database and should be used if IFS changes their core table structure or when analyzing new IFS Cloud environments.

## Prerequisites

- Oracle database connection to IFS Cloud environment (e.g., IFSCDEV)
- SQLcl or similar Oracle client tools
- MCP SQLcl server for structured queries (optional but recommended)
- Understanding of IFS Cloud architecture concepts

## Analysis Phases

### Phase 1: Discovery - Identify Metadata Tables

#### Step 1.1: Find Navigation and Component Tables

```sql
-- Discover all navigation, metadata, and component tables
SELECT table_name
FROM all_tables
WHERE owner = 'IFSAPP'
  AND (table_name LIKE '%NAVIGATION%'
       OR table_name LIKE '%COMPONENT%'
       OR table_name LIKE '%METADATA%'
       OR table_name LIKE '%DICTIONARY%'
       OR table_name LIKE '%GUI%'
       OR table_name LIKE '%CLIENT%')
ORDER BY table_name;
```

**Expected Result**: 300-400+ tables containing metadata and navigation information.

#### Step 1.2: Identify Dictionary System Tables

```sql
-- Focus on the core dictionary system
SELECT table_name
FROM all_tables
WHERE owner = 'IFSAPP'
  AND table_name LIKE 'DICTIONARY%'
ORDER BY table_name;
```

**Expected Tables**:

- `DICTIONARY_SYS_LU_ACTIVE` (core logical units)
- `DICTIONARY_SYS_METHOD_TAB` (business methods)
- `DICTIONARY_SYS_DOMAIN_TAB` (value mappings)
- `DICTIONARY_SYS_VIEW_TAB` (presentation views)
- `DICTIONARY_SYS_STATE_*` (workflow states)

### Phase 2: Structure Analysis - Understand Table Schemas

#### Step 2.1: Analyze Core Logical Units Table

```sql
-- Get structure of the main logical units table
SELECT column_name, data_type, data_length, nullable
FROM all_tab_columns
WHERE table_name = 'DICTIONARY_SYS_LU_ACTIVE'
  AND owner = 'IFSAPP'
ORDER BY column_id;
```

**Key Columns to Identify**:

- `MODULE`: Business domain classification
- `LU_NAME`: Technical logical unit identifier
- `LU_PROMPT`: User-friendly description
- `BASE_TABLE`: Underlying database table
- `BASE_VIEW`: Associated view
- `LOGICAL_UNIT_TYPE`: Entity classification
- `CUSTOM_FIELDS`: Extension capabilities

#### Step 2.2: Analyze Methods and APIs

```sql
-- Understand method structure
SELECT column_name, data_type
FROM all_tab_columns
WHERE table_name = 'DICTIONARY_SYS_METHOD_TAB'
  AND owner = 'IFSAPP'
ORDER BY column_id;
```

**Key Columns**:

- `LU_NAME`: Link to logical units
- `PACKAGE_NAME`: API package organization
- `METHOD_NAME`: Available functionality
- `METHOD_TYPE`: Type of operation

### Phase 3: Content Analysis - Explore Data Patterns

#### Step 3.1: Module Distribution Analysis

```sql
-- Analyze business module distribution
SELECT module,
       COUNT(*) as lu_count,
       COUNT(DISTINCT base_table) as table_count,
       LISTAGG(DISTINCT lu_type, ',') WITHIN GROUP (ORDER BY lu_type) as lu_types
FROM dictionary_sys_lu_active
GROUP BY module
ORDER BY lu_count DESC;
```

**Analysis Points**:

- Identify largest business domains (ORDER, PERSON, PURCH, etc.)
- Understand module complexity by LU count
- Note entity types distribution

#### Step 3.2: Sample Business Entity Analysis

```sql
-- Examine specific business entities for patterns
SELECT lu_name, lu_prompt, module, base_table, base_view, lu_type
FROM dictionary_sys_lu_active
WHERE lu_name LIKE '%Customer%'
   OR lu_name LIKE '%Order%'
   OR lu_name LIKE '%Employee%'
   OR lu_name LIKE '%Person%'
ORDER BY module, lu_name;
```

**Look For**:

- Naming conventions and patterns
- Business terminology in prompts
- Cross-module relationships
- Entity hierarchies

#### Step 3.3: Method Complexity Assessment

```sql
-- Assess API method distribution
SELECT COUNT(*) as method_count,
       COUNT(DISTINCT package_name) as package_count,
       COUNT(DISTINCT lu_name) as lu_count
FROM dictionary_sys_method_tab;
```

### Phase 4: Relationship Mapping

#### Step 4.1: Domain Value Mappings

```sql
-- Analyze domain value translations
SELECT lu_name, package_name, COUNT(*) as domain_count
FROM dictionary_sys_domain_tab
WHERE ROWNUM <= 50
GROUP BY lu_name, package_name
ORDER BY domain_count DESC;
```

#### Step 4.2: View Associations

```sql
-- Understand view relationships
SELECT COUNT(*) as view_count,
       COUNT(DISTINCT lu_name) as lu_count,
       COUNT(DISTINCT view_type) as view_types,
       LISTAGG(DISTINCT view_type, ',') WITHIN GROUP (ORDER BY view_type) as types
FROM dictionary_sys_view_tab;
```

### Phase 5: Search Enhancement Opportunities

#### Step 5.1: Business Context Extraction

Document how to extract:

1. **Technical Layer**: Table names, column names, technical identifiers
2. **Business Layer**: LU prompts, module classifications, business terms
3. **User Layer**: GUI forms, navigation paths, user-friendly names

#### Step 5.2: Cross-Reference Patterns

```sql
-- Example: Find all customer-related entities
SELECT DISTINCT lu_name, module, lu_prompt
FROM dictionary_sys_lu_active
WHERE lu_name LIKE '%Customer%'
   OR lu_prompt LIKE '%Customer%'
ORDER BY module, lu_name;
```

## Change Detection Strategy

### Monitoring Key Tables

If IFS changes their structure, monitor these critical tables:

1. **`DICTIONARY_SYS_LU_ACTIVE`**: Core business entity definitions
2. **`DICTIONARY_SYS_METHOD_TAB`**: API and functionality mappings
3. **`DICTIONARY_SYS_VIEW_TAB`**: Presentation layer information
4. **Navigation/GUI tables**: User interface metadata

### Version Comparison Approach

```sql
-- Compare table structures between versions
SELECT table_name, column_name, data_type,
       CASE WHEN column_name IN (SELECT column_name FROM old_version_columns)
            THEN 'EXISTING'
            ELSE 'NEW' END as status
FROM all_tab_columns
WHERE table_name = 'DICTIONARY_SYS_LU_ACTIVE'
  AND owner = 'IFSAPP'
ORDER BY column_id;
```

## Integration Guidelines

### Data Extraction for Search Enhancement

1. **Module-Based Classification**

   - Extract module → LU mappings for domain-specific search
   - Use prompts for user-friendly descriptions

2. **Hierarchical Relationships**

   - Map LUs → Methods → Views for complete context
   - Build cross-reference tables for related entities

3. **Business Vocabulary**
   - Extract prompts and descriptions for natural language search
   - Map technical terms to business terms via domain tables

### Indexing Strategy

```python
# Pseudo-code for enhanced indexer
class EnhancedIFSCloudIndexer:
    def build_metadata_index(self):
        # Extract logical units with business context
        logical_units = self.extract_logical_units()

        # Map methods to functionality
        methods = self.extract_methods()

        # Build domain vocabulary
        domain_mappings = self.extract_domain_mappings()

        # Create hierarchical search structure
        return self.build_hierarchical_index(
            logical_units, methods, domain_mappings
        )
```

## Troubleshooting Common Issues

### Access Permissions

- Ensure database user has SELECT access to `IFSAPP` schema
- Some tables may require elevated privileges

### Empty Results

- Check if tables exist in current IFS version
- Verify correct schema owner (usually `IFSAPP`)
- Some metadata may be version-specific

### Performance Considerations

- Dictionary tables can be large (300K+ methods)
- Use ROWNUM limits for exploratory queries
- Consider partitioning analysis by module

## Future Considerations

### Version Tracking

- Document IFS Cloud version when analysis is performed
- Track changes in core metadata structure
- Maintain compatibility matrix

### Extension Points

- Look for custom field extensions in logical units
- Identify customer-specific modifications
- Plan for multi-tenant variations

## Example Analysis Results

Based on analysis of IFS Cloud 25.1.0 (IFSCDEV):

- **13,739** logical units across **152** modules
- **393,081** methods in **14,023** packages
- **20,223** views across **9,310** logical units
- Major modules: ORDER (505 LUs), PERSON (361 LUs), PURCH (366 LUs)

This rich metadata provides multiple layers for enhanced search:

- Technical identifiers for developers
- Business terminology for functional users
- Cross-module relationships for comprehensive results
- UI context for user-oriented search

---

## Last Updated

**Date**: August 16, 2025  
**IFS Version**: 25.1.0  
**Database**: IFSCDEV  
**Analyst**: GitHub Copilot

---

_Note: This methodology should be re-applied whenever IFS releases major version updates or structural changes to their core metadata system._
