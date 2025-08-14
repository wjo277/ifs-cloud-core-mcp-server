# IFS Cloud MCP Server

A high-performance Model Context Protocol (MCP) server for IFS Cloud codebases, featuring enterprise-grade search capabilities powered by Tantivy search engine. **Available as both an MCP server for AI integration and a standalone web UI for interactive exploration.**

## 🚀 Two Powerful Interfaces

### 1. **MCP Server Mode** (AI Integration)

- Integrates with Claude, GitHub Copilot, and other MCP clients
- Provides structured search APIs for AI-powered development
- Perfect for automated code analysis and AI-assisted development

### 2. **Web UI Mode** (Interactive Exploration) ⭐ NEW!

- Modern web interface with type-ahead search
- Visual exploration of IFS Cloud codebases
- Real-time search with intelligent suggestions
- **[See Web UI Documentation](./WEB_UI_README.md)**

## Features

### 🚀 High-Performance Search

- **Tantivy Integration**: Rust-based search engine for lightning-fast queries
- **Large Codebase Support**: Efficiently handles 1GB+ IFS Cloud projects
- **Sub-second Response Times**: Optimized for enterprise-scale development
- **Intelligent Caching**: 17.4x performance improvement with disk-based caching

### 📁 IFS Cloud File Support

Complete support for all IFS Cloud file types with **specialized parsers**:

- `*.entity` - Entity definitions (XML parsing)
- `*.plsql` - PL/SQL code with function/procedure extraction
- `*.views` - Database views (IFS DSL parsing)
- `*.storage` - Storage configurations (table/index definitions)
- `*.fragment` - Code fragments (mixed Marble content)
- `*.client` - Client-side code (Marble UI parsing) **✨ Enhanced**
- `*.projection` - Data projections (Marble data access parsing)
- `*.plsvc` - PL/SQL service layer for projections

### 🎨 Frontend Element Discovery **✨ NEW!**

Advanced parsing of IFS Cloud UI components:

- **Pages** - Client pages and dialogs
- **Lists** - Data lists and grids
- **Groups** - UI grouping elements
- **Iconsets** - Icon definitions and mappings
- **Trees** - Tree navigators and hierarchies
- **Navigators** - Navigation structures
- **Contexts** - Navigation contexts and selectors

### 🔍 Advanced Search Capabilities

- **Full-text Search**: Content search with relevance ranking
- **Entity Search**: Find files containing specific IFS entities
- **Frontend Element Search**: Find UI components (iconsets, trees, navigators)
- **Module-aware Search**: Search within specific IFS modules
- **Type-based Search**: Filter by file type and extension
- **Multi-criteria Search**: Combine content, type, complexity, and size filters
- **Similarity Search**: Find related files based on entities and dependencies
- **Fuzzy Search**: Handle typos and partial matches

### 📊 Code Intelligence & Analysis

- **Entity Dependency Analysis**: Map relationships and dependencies between entities
- **Override/Overtake Detection**: Find all @Override and @Overtake annotations
- **Complexity Scoring**: Type-specific automated code complexity analysis
- **Function/Procedure Extraction**: Identify all procedures and functions
- **Fragment Dependency Tracking**: Map fragment includes and usage
- **Cross-module Relationship Mapping**: Understand component interconnections

### ⚙️ Configuration Management

- **Persistent Core Codes Path**: Configure and remember IFS Cloud Core Codes location
- **Automatic Indexing**: Index configured core codes with a single command
- **Index Statistics**: Track indexing history and performance metrics

## Quick Start 🚀

### 🌐 **Web UI Mode** (Recommended for exploration)

1. **Install and start the web interface:**

```bash
# Clone and install
git clone https://github.com/graknol/ifs-cloud-core-mcp-server.git
cd ifs-cloud-core-mcp-server
uv sync

# Start web UI
uv run python -m src.ifs_cloud_mcp_server.launcher web
```

2. **Index your IFS Cloud project:**

```bash
# Build search index
uv run python -m src.ifs_cloud_mcp_server.launcher index build --directory /path/to/your/ifs/project
```

3. **Open browser:** Navigate to `http://localhost:8000` and start exploring!

### 🔌 **MCP Server Mode** (For AI integration)

1. **Start MCP server:**

```bash
# For Claude Desktop or other MCP clients
uv run python -m src.ifs_cloud_mcp_server.main
```

2. **Configure in Claude Desktop:**

```json
{
  "mcpServers": {
    "ifs-cloud": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.ifs_cloud_mcp_server.main"],
      "cwd": "/path/to/ifs-cloud-core-mcp-server"
    }
  }
}
```

## Installation

```bash
# Clone the repository
git clone https://github.com/graknol/ifs-cloud-core-mcp-server.git
cd ifs-cloud-core-mcp-server

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

## Quick Start

```bash
# Start the MCP server
ifs-cloud-mcp-server --port 8000 --index-path ./index

# Index your IFS Cloud codebase
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/ifs/cloud/project"}'

# Search for entities
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "CustomerOrder", "type": "entity"}'
```

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

```bash
# Run tests
pytest

# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Performance

- **Indexing Speed**: ~1000 files/second on typical hardware
- **Search Response**: <100ms for most queries
- **Memory Usage**: ~200MB for 1GB codebase index
- **Incremental Updates**: Real-time file change tracking

## Future Roadmap

- 🤖 **AI Integration**: FastAI/PyTorch for semantic search
- 🧠 **Pattern Recognition**: ML-based code pattern detection
- 📈 **Analytics**: Advanced codebase insights and metrics
- 🔗 **IDE Integration**: VS Code and IntelliJ plugins

## License

Licensed under the terms specified in the LICENSE file.
