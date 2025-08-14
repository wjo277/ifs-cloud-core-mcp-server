# IFS Cloud MCP Server - Quick Start Guide

## Installation

### Prerequisites
- Python 3.9 or higher
- Git

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/graknol/ifs-cloud-core-mcp-server.git
cd ifs-cloud-core-mcp-server

# Install the package and dependencies
pip install -e .

# For development (includes testing tools)
pip install -e ".[dev]"
```

## Quick Start

### 1. Test Core Functionality (No Dependencies Required)

```bash
# Test core algorithms without external dependencies
python test_standalone.py

# View capabilities demonstration
python demo.py --mock
```

### 2. Start the MCP Server

```bash
# Start server with default settings
ifs-cloud-mcp-server --index-path ./index

# Start with custom settings
ifs-cloud-mcp-server \
  --index-path /path/to/index \
  --log-level DEBUG \
  --name my-ifs-server
```

### 3. Index Your IFS Cloud Project

Use any MCP client to connect and send commands:

```json
{
  "method": "tools/call",
  "params": {
    "name": "index_directory",
    "arguments": {
      "path": "/path/to/your/ifs/project",
      "recursive": true
    }
  }
}
```

### 4. Search Your Codebase

#### Search by Content
```json
{
  "method": "tools/call", 
  "params": {
    "name": "search_content",
    "arguments": {
      "query": "CustomerOrder",
      "limit": 10,
      "file_type": ".entity"
    }
  }
}
```

#### Find Entities
```json
{
  "method": "tools/call",
  "params": {
    "name": "search_entities", 
    "arguments": {
      "entity": "CustomerOrder",
      "limit": 5
    }
  }
}
```

#### Find Similar Files
```json
{
  "method": "tools/call",
  "params": {
    "name": "find_similar_files",
    "arguments": {
      "file_path": "/path/to/CustomerOrder.entity",
      "limit": 5
    }
  }
}
```

#### Search by Complexity
```json
{
  "method": "tools/call",
  "params": {
    "name": "search_by_complexity",
    "arguments": {
      "min_complexity": 0.3,
      "max_complexity": 0.8,
      "file_type": ".plsql"
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `search_content` | Full-text search with ranking and filtering |
| `search_entities` | Find files containing specific IFS entities |
| `find_similar_files` | Similarity search based on entities/content |
| `search_by_complexity` | Filter files by complexity score |
| `index_directory` | Index all files in a directory |
| `index_file` | Index a single file |
| `get_index_statistics` | Get index metrics and statistics |
| `fuzzy_search` | Typo-tolerant fuzzy search |

## Supported File Types

- `.entity` - Entity definitions
- `.plsql` - PL/SQL code and packages
- `.views` - Database view definitions
- `.storage` - Storage and table definitions
- `.fragment` - UI fragment definitions
- `.client` - Client page definitions
- `.projection` - Data projection definitions

## Example Usage Scenarios

### Code Review
```bash
# Find all files related to CustomerOrder
search_entities: "CustomerOrder"
find_similar_files: "CustomerOrder.entity"
search_by_complexity: min=0.3  # Find complex files to review carefully
```

### Impact Analysis
```bash
# Analyze impact of changing OrderNo field
search_content: "OrderNo"
search_entities: "OrderNo"
find_similar_files: "CustomerOrder.entity"
```

### Refactoring
```bash
# Find complex PL/SQL files that need refactoring
search_by_complexity: min=0.4 type=".plsql"
search_content: "EXCEPTION"  # Find error handling patterns
fuzzy_search: "proceedure~"   # Find typos
```

## Performance Expectations

- **Indexing Speed**: ~1000 files/second on typical hardware
- **Search Response**: <100ms for most queries
- **Memory Usage**: ~200MB for 1GB codebase index
- **Processing Rate**: ~17 MB/s for content analysis

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Client    │◄──►│   MCP Server    │◄──►│ Tantivy Index   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ File Processors │
                       │  - Entity       │
                       │  - PL/SQL       │
                       │  - Views        │
                       │  - Storage      │
                       │  - Fragment     │
                       │  - Client       │
                       │  - Projection   │
                       └─────────────────┘
```

## Development

### Run Tests
```bash
# Core functionality tests (no dependencies)
python test_standalone.py

# Full test suite (requires dependencies)
pytest

# Test coverage
pytest --cov=ifs_cloud_mcp_server
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code  
ruff check src/ tests/

# Type checking
mypy src/
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
   ```bash
   pip install -e .
   ```

2. **Permission denied**: Ensure index directory is writable
   ```bash
   chmod 755 ./index
   ```

3. **Search returns no results**: Check if files are indexed
   ```bash
   # Use get_index_statistics tool
   ```

4. **Slow performance**: Check available memory and disk space
   ```bash
   # Monitor with: htop, df -h
   ```

### Getting Help

- Check the logs: Server logs to stderr by default
- Use `--log-level DEBUG` for detailed logging
- Review the test files for usage examples
- Check GitHub issues for known problems

## Future Enhancements

The server is designed to support future AI/ML integrations:

- **Semantic Search**: Vector embeddings for meaning-based search
- **Pattern Recognition**: ML-based code pattern detection
- **Smart Suggestions**: AI-powered refactoring recommendations
- **Documentation Generation**: Automated API documentation
- **Code Quality Analysis**: ML-based quality scoring