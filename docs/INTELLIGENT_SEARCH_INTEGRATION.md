# Intelligent Search Integration with Metadata Enhancement

This document describes the complete intelligent search integration that combines the Tantivy-based index with IFS Cloud metadata to provide optimal search results and related file discovery.

## Overview

The enhanced search system now provides three levels of search capability:

1. **Basic Search** (`search_deduplicated`) - Traditional index-based search
2. **Enhanced Search** (`enhanced_search`) - Metadata-augmented search with business context
3. **Intelligent Search** (`intelligent_search`) - Combines both approaches with related file discovery

## Key Features

### 1. Metadata-Enhanced Ranking

The system uses IFS Cloud database metadata to understand:

- Business terminology and context
- Module relationships
- Logical unit hierarchies
- Cross-reference mappings

### 2. Related File Discovery

When searching for a logical unit (e.g., "Activity"), the system automatically includes related files:

- `Activity.entity` - Core entity definition
- `Activity.plsql` / `ActivityAPI.plsql` - Business logic
- `Activity.views` - Database views
- `Activity.projection` - API projections
- `Activity.client` - UI client definitions
- `ActivityHandling.fragment` - UI fragments

### 3. Business Logic Boosting

For authorization and business rule queries, the system:

- Prioritizes `.plsql` files containing business logic
- Boosts large files (>1000 lines) that likely contain main implementations
- Applies context-aware scoring based on query keywords

### 4. Cross-Module Intelligence

The system leverages metadata to:

- Map business terms to technical entities
- Suggest related search terms
- Understand module boundaries and relationships

## New MCP Tools

### `intelligent_search`

**Primary search tool** - Use this for most searches:

```python
# Basic intelligent search
results = await intelligent_search("customer order authorization")

# With options
results = await intelligent_search(
    query="expense approval workflow",
    limit=15,
    include_related=True,
    boost_business_logic=True
)
```

**Best for:**

- Entity-focused searches
- Business logic discovery
- Authorization and workflow queries
- Complete implementation understanding

### `find_related_files`

Find all files for a specific logical unit:

```python
# Find all Activity-related files
results = await find_related_files("Activity")

# Find CustomerOrder implementation files
results = await find_related_files("CustomerOrder", limit=20)
```

**Returns:** All core files for the logical unit, sorted by importance.

### `set_ifs_version` & `get_available_ifs_versions`

Configure metadata enhancement:

```python
# Set IFS version for enhanced search
await set_ifs_version("25.1.0")

# Check available versions
versions = await get_available_ifs_versions()
```

### `get_module_statistics`

Get metadata statistics:

```python
# View module information
stats = await get_module_statistics()
```

## Search Strategies

### 1. Entity Discovery

**Query:** "CustomerOrder"
**Results:**

- CustomerOrder.entity (highest priority)
- CustomerOrderAPI.plsql (business logic)
- CustomerOrder.views (data access)
- CustomerOrder.projection (API layer)
- CustomerOrder.client (UI layer)

### 2. Authorization Searches

**Query:** "expense authorization"
**Enhanced Results:**

- ExpenseHeader.plsql (boosted for business logic)
- ExpenseSheet authorization methods
- Related workflow files
- Approval chain implementations

### 3. Business Process Discovery

**Query:** "purchase requisition workflow"
**Intelligent Results:**

- Core PurchaseRequisition files
- Workflow-related logical units
- Authorization chain files
- Related approval entities

## Implementation Architecture

### IndexerIntegration

The `IFSCloudIndexer` now includes:

```python
class IFSCloudIndexer:
    # Enhanced search integration
    def set_ifs_version(self, version: str) -> bool
    def enhanced_search(self, query: str, **options) -> List[EnhancedSearchResult]
    def intelligent_search(self, query: str, **options) -> List[SearchResult]
    def find_related_files(self, logical_unit: str) -> List[SearchResult]
    def suggest_related_searches(self, query: str) -> List[str]

    # Metadata utilities
    def get_module_statistics(self) -> Dict[str, Any]
    def has_metadata_enhancement(self) -> bool
    def get_current_ifs_version(self) -> Optional[str]
```

### Search Flow

1. **User Query** → `intelligent_search()`
2. **Metadata Available?**
   - **Yes**: Use `enhanced_search()` with metadata ranking
   - **No**: Use `search_with_related_files()` with business logic boosting
3. **Include Related Files** (if enabled)
4. **Apply Business Logic Boosting** (if enabled)
5. **Return Ranked Results**

### Related File Discovery

For each logical unit found in search results:

1. **Extract LU Name** from file path or metadata
2. **Find Related Files** using patterns:
   - `{LU}.entity`
   - `{LU}.plsql` or `{LU}API.plsql`
   - `{LU}.views`
   - `{LU}.projection`
   - `{LU}.client`
   - `{LU}Handling.fragment`
3. **Rank by File Type Priority**:
   - `.entity` (core definition)
   - `.plsql` (business logic)
   - `.views` (data access)
   - `.projection` (API layer)
   - `.client` (UI layer)
   - `.fragment` (components)
   - `.storage` (database schema)

## Performance Considerations

### Search Performance

- **Enhanced Search**: +10-20ms overhead for metadata processing
- **Related File Discovery**: +5-10ms per logical unit
- **Business Logic Boosting**: +2-5ms for scoring adjustments

### Memory Usage

- **Metadata Cache**: ~10-50MB per IFS version
- **Term Matcher**: ~5-20MB for business term mappings
- **Total Overhead**: <100MB for typical installations

### Optimization Strategies

1. **Lazy Loading**: Metadata loaded only when needed
2. **Result Limiting**: Prevent overwhelming responses
3. **Caching**: Metadata and search results cached appropriately

## Usage Examples

### Authorization Query

```bash
# Traditional search - may miss key files
search_content("expense sheet authorization")

# Intelligent search - comprehensive results
intelligent_search("expense authorization")
# Returns: ExpenseHeader.plsql, ExpenseSheet.*, related workflow files
```

### Entity Implementation Discovery

```bash
# Find complete Activity implementation
intelligent_search("Activity", include_related=true)

# Or directly find related files
find_related_files("Activity")
```

### Business Process Analysis

```bash
# Find purchase workflow implementation
intelligent_search("purchase requisition approval workflow", boost_business_logic=true)
```

## Migration Guide

### For Existing MCP Clients

1. **Update Search Calls**: Replace `search_content` with `intelligent_search` for better results
2. **Set IFS Version**: Call `set_ifs_version()` to enable enhanced features
3. **Leverage Related Files**: Use `find_related_files()` for comprehensive implementation discovery

### For Agent Implementations

1. **Primary Search Tool**: Use `intelligent_search` as the default search method
2. **Fallback Strategy**: Keep `search_content` for simple keyword searches
3. **Context Building**: Use `find_related_files` to build comprehensive context

## Troubleshooting

### No Enhanced Results

**Problem**: Search results don't show metadata enhancement
**Solution**:

1. Check if IFS version is set: `get_available_ifs_versions()`
2. Set version: `set_ifs_version("25.1.0")`
3. Verify metadata exists for the version

### Missing Related Files

**Problem**: Related files not appearing in results  
**Solution**:

1. Ensure `include_related=True` in search call
2. Check if files exist in the indexed directory
3. Verify logical unit naming conventions

### Poor Business Logic Ranking

**Problem**: .plsql files not prioritized for business queries
**Solution**:

1. Ensure `boost_business_logic=True`
2. Use business terminology in queries (e.g., "authorization" vs "auth")
3. Check if files are properly indexed with business logic content

## Future Enhancements

### Planned Features

1. **Real-time Metadata Sync**: Automatic metadata updates from database changes
2. **Custom Business Vocabulary**: User-defined term mappings
3. **Search Analytics**: Usage tracking and result optimization
4. **Machine Learning**: Adaptive ranking based on user behavior

### Extension Points

- Custom metadata extractors for specific IFS modules
- Business-specific term mapping rules
- Integration with IFS Cloud REST APIs
- Custom search result ranking algorithms

---

**Last Updated**: August 16, 2025  
**IFS Cloud Version**: 25.1.0+  
**System Version**: 2.0.0
