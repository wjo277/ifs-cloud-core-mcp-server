# Enhanced Search with IFS Cloud Metadata Integration

This document describes the comprehensive metadata-enhanced search system for IFS Cloud files, providing intelligent business context and cross-module relationship discovery.

## Overview

The enhanced search system extends the existing Tantivy-based file indexer with rich metadata extracted from the IFS Cloud database. This enables:

- **Business-context search**: Search using business terminology instead of technical file names
- **Cross-module discovery**: Find related entities across different IFS modules
- **Intelligent suggestions**: Get related search terms based on business relationships
- **Domain-aware results**: Results ranked by business importance and module relevance

## Architecture

### Core Components

#### 1. MetadataExtractor (`metadata_extractor.py`)

- **Purpose**: Extracts and manages IFS Cloud database metadata
- **Key Classes**:
  - `LogicalUnit`: Represents IFS business entities with metadata
  - `ModuleInfo`: Business module information and statistics
  - `DomainMapping`: Database-to-client value translations
  - `MetadataExtract`: Container for all extracted metadata
  - `DatabaseMetadataExtractor`: Handles database extraction
  - `MetadataManager`: Manages metadata storage and versioning

#### 2. EnhancedSearchEngine (`enhanced_search.py`)

- **Purpose**: Provides intelligent search with metadata integration
- **Key Classes**:
  - `SearchResult`: Enhanced results with business context
  - `SearchContext`: Search parameters and filters
  - `BusinessTermMatcher`: Maps business terms to technical entities
  - `MetadataEnhancedSearchEngine`: Main search orchestrator

#### 3. Enhanced Indexer (`indexer.py` - extended)

- **Purpose**: Integrates metadata capabilities with existing indexer
- **New Methods**:
  - `set_ifs_version()`: Configure metadata for specific IFS version
  - `enhanced_search()`: Perform metadata-aware search
  - `extract_metadata_from_mcp_results()`: Process MCP query results

#### 4. Extraction Utilities (`extract_metadata.py`)

- **Purpose**: Command-line utilities for metadata extraction
- **Features**:
  - Step-by-step extraction instructions
  - CSV/JSON result processing
  - Automated metadata saving

## Usage Guide

### 1. Initial Setup

```python
from ifs_cloud_mcp_server.indexer import IFSCloudIndexer

# Create indexer with metadata support
indexer = IFSCloudIndexer("path/to/index")

# Set IFS version for metadata enhancement
indexer.set_ifs_version("25.1.0")
```

### 2. Extract Metadata from Database

#### Option A: Using MCP SQLcl Server (Recommended)

1. **Connect to database**:

   ```python
   # Use MCP SQLcl tools to connect to IFSCDEV or your IFS database
   ```

2. **Get extraction queries**:

   ```python
   queries = indexer.get_metadata_extract_queries()
   ```

3. **Execute queries using MCP and save results**

4. **Process results**:

   ```python
   # Results from MCP queries
   query_results = {
       'logical_units': [...],  # Results from logical units query
       'modules': [...],        # Results from modules query
       'domain_mappings': [...],# Results from domain mappings query
       'views': [...]           # Results from views query
   }

   # Extract and save metadata
   success = indexer.extract_metadata_from_mcp_results("25.1.0", query_results)
   ```

#### Option B: Using Command Line Utility

```bash
# Get extraction instructions
python -m ifs_cloud_mcp_server.extract_metadata --ifs-version 25.1.0 --instructions

# Create results template
python -m ifs_cloud_mcp_server.extract_metadata --create-template results_template.json

# Process results from JSON file
python -m ifs_cloud_mcp_server.extract_metadata --ifs-version 25.1.0 --process-json results.json
```

### 3. Enhanced Search Operations

#### Basic Enhanced Search

```python
# Simple enhanced search
results = indexer.enhanced_search("customer order")

# With filters
results = indexer.enhanced_search(
    "customer order",
    modules_filter=['ORDER', 'INVOIC'],
    content_types_filter=['entity', 'projection'],
    fuzzy_threshold=85.0
)

for result in results:
    print(f"File: {result.file_path}")
    print(f"Module: {result.module}")
    print(f"Business Description: {result.business_description}")
    print(f"Confidence: {result.confidence}%")
    print(f"Related Entities: {result.related_entities}")
    print("---")
```

#### Get Related Search Suggestions

```python
suggestions = indexer.suggest_related_searches("customer order")
# Returns: ['CustomerOrderLine', 'OrderQuotation', 'CustomerInvoice', ...]
```

#### Module Statistics

```python
stats = indexer.get_module_statistics()
# Returns module info with LU counts and sample entities
```

### 4. Search Result Enhancement

Enhanced search results include:

- **file_path**: Original file path
- **content_type**: File type (entity, plsql, client, etc.)
- **confidence**: Search confidence score (0-100)
- **logical_unit**: Associated IFS logical unit
- **module**: IFS business module (ORDER, PERSON, PURCH, etc.)
- **business_description**: User-friendly description
- **related_entities**: Related logical units in same module
- **search_context**: Additional context information

## Metadata Structure

### Logical Units

Core business entities with:

- **Module classification** (ORDER, PERSON, PURCH, etc.)
- **Business-friendly prompts**
- **Database table mappings**
- **Entity type information**

### Domain Mappings

Database-to-client value translations:

- Technical database values → User-friendly display values
- Enables search by business terminology

### View Information

UI presentation metadata:

- View types and prompts
- User interface context
- Navigation relationships

### Module Information

Business domain statistics:

- Logical unit counts per module
- Module complexity metrics
- Cross-module relationships

## Advanced Features

### 1. Fuzzy Matching

- Uses rapidfuzz library for high-performance approximate string matching
- Configurable threshold for fuzzy search tolerance
- Handles typos and variations in business terms
- Significantly faster than traditional fuzzy matching libraries

### 2. Cross-Module Relationships

- Discovers related entities across different modules
- Maps business process flows
- Suggests complementary search terms

### 3. Business Context Ranking

- Ranks results by business importance
- Considers module relationships
- Weights by entity complexity and usage

### 4. Version Management

- Supports multiple IFS Cloud versions
- Automatic metadata versioning
- Backward compatibility handling

## Performance Considerations

### Metadata Storage

- Metadata stored as JSON files per IFS version
- Cached in memory for fast access
- Periodic cleanup of old versions

### Search Performance

- Base Tantivy search enhanced, not replaced
- Metadata enhancement adds ~10-20ms per search
- Results limited to prevent overwhelming UI

### Memory Usage

- Metadata cached: ~10-50MB per IFS version
- Business term mappings: ~5-20MB indexed
- Total overhead: <100MB for typical installations

## Configuration Options

### Search Context Parameters

```python
SearchContext(
    query="customer order",
    modules_filter=['ORDER', 'INVOIC'],      # Limit to specific modules
    content_types_filter=['entity'],         # Limit to file types
    logical_units_filter=['CustomerOrder'],  # Limit to specific LUs
    fuzzy_threshold=80.0,                    # Fuzzy matching threshold
    include_related=True                     # Include related entity suggestions
)
```

### Metadata Manager Settings

```python
MetadataManager(
    metadata_dir=Path("metadata"),          # Metadata storage directory
    keep_latest=3                           # Number of versions to retain
)
```

## Troubleshooting

### Common Issues

1. **No metadata available**:

   - Check if metadata was extracted for the IFS version
   - Verify metadata files exist in `metadata/{version}/` directory

2. **Poor search results**:

   - Verify IFS version is set correctly
   - Check fuzzy threshold (lower = more matches, higher = more precise)
   - Ensure base file indexing is working properly

3. **Memory issues**:

   - Reduce number of cached metadata versions
   - Consider extracting fewer domain mappings

4. **Performance issues**:
   - Check if base Tantivy index needs rebuilding
   - Verify metadata files aren't corrupted
   - Monitor search query complexity

### Debug Information

```python
# Check metadata status
print(f"Current IFS Version: {indexer.get_current_ifs_version()}")
print(f"Has Metadata: {indexer.has_metadata_enhancement()}")
print(f"Available Versions: {indexer.get_available_ifs_versions()}")

# Get module statistics
stats = indexer.get_module_statistics()
print(f"Loaded Modules: {list(stats.keys())}")
```

## Future Enhancements

### Planned Features

1. **Real-time metadata sync** with database changes
2. **Custom business vocabulary** import/export
3. **Search analytics** and usage tracking
4. **Multi-language support** for international deployments
5. **Machine learning** for search result optimization

### Extension Points

- Custom metadata extractors for specific IFS modules
- Business-specific term mapping rules
- Integration with IFS Cloud REST APIs
- Custom search result ranking algorithms

## API Reference

See individual class documentation in:

- `metadata_extractor.py` - Metadata extraction and management
- `enhanced_search.py` - Enhanced search engine
- `extract_metadata.py` - Command-line utilities
- `indexer.py` - Enhanced indexer integration

---

**Last Updated**: August 16, 2025  
**IFS Cloud Version**: 25.1.0  
**System Version**: 1.0.0
