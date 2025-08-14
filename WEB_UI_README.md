# IFS Cloud Web UI

A modern, responsive web interface for exploring IFS Cloud codebases with intelligent type-ahead search.

## Features 🚀

### 🔍 **Intelligent Search**

- **Type-ahead suggestions** with context-aware results
- **Frontend element discovery** - find pages, iconsets, trees, navigators
- **Module-aware search** with IFS Cloud structure understanding
- **Real-time highlighting** and content preview

### 🎯 **Advanced Filtering**

- Filter by file type (`.entity`, `.client`, `.projection`, etc.)
- Filter by IFS module and logical unit
- Complexity-based filtering
- Customizable result limits

### 🎨 **Modern Interface**

- Responsive design that works on desktop and mobile
- Clean, intuitive UI with modern styling
- Color-coded tags and result categories
- Fast, smooth user experience

### 📊 **Comprehensive Results**

- **File Context**: Module, logical unit, component information
- **Frontend Elements**: Pages, lists, groups, iconsets, trees, navigators
- **Code Entities**: Functions, classes, dependencies
- **Metadata**: File size, complexity, modification time

## Quick Start 🚀

### 1. Start the Web UI

```bash
# Start with default settings
uv run python -m src.ifs_cloud_mcp_server.launcher web

# Custom port and index path
uv run python -m src.ifs_cloud_mcp_server.launcher web --port 8080 --index-path ./my_index
```

### 2. Index Your IFS Cloud Project

```bash
# Build index for your IFS Cloud codebase
uv run python -m src.ifs_cloud_mcp_server.launcher index build --directory /path/to/your/ifs/project
```

### 3. Open Your Browser

Navigate to `http://localhost:8000` (or your custom port) and start exploring!

## Usage Examples 🎯

### **Search for Frontend Elements**

- `iconset` - Find all iconset definitions
- `tree navigator` - Find tree navigators and their configurations
- `page overview` - Find overview pages
- `delivery structure` - Find delivery-related components

### **Search by Module/Component**

- Use the module filter: `PROJ`, `ORDER`, `ACCRUL`
- Combine with search: `customer order PROJ`

### **Find Specific Code Elements**

- `Activity` - Find Activity entities across modules
- `GetInfo` - Find GetInfo functions
- `projection CustomerOrder` - Find projections related to CustomerOrder

## API Endpoints 🔌

The web UI exposes several REST endpoints:

### Search

```
GET /api/search?q=query&limit=20&file_type=.client&module=PROJ
```

### Type-ahead Suggestions

```
GET /api/suggestions?q=partial_query&limit=10
```

### Index Statistics

```
GET /api/stats
```

### Rebuild Index

```
POST /api/index?directory=/path/to/project
```

### File Content

```
GET /api/file/{file_path}
```

## Configuration Options ⚙️

| Option         | Default   | Description                        |
| -------------- | --------- | ---------------------------------- |
| `--host`       | `0.0.0.0` | Host to bind to                    |
| `--port`       | `8000`    | Port to bind to                    |
| `--index-path` | `./index` | Path to store search index         |
| `--reload`     | `false`   | Enable auto-reload for development |

## Architecture 🏗️

The web UI is built with:

- **FastAPI** - High-performance web framework
- **Tantivy** - Fast full-text search engine
- **Alpine.js** - Lightweight frontend framework
- **Tailwind CSS** - Utility-first CSS framework
- **Font Awesome** - Icon library

## Comparison: MCP Server vs Web UI 📊

| Feature        | MCP Server                  | Web UI                      |
| -------------- | --------------------------- | --------------------------- |
| **Use Case**   | Claude/Copilot integration  | Interactive exploration     |
| **Interface**  | Programmatic (JSON-RPC)     | Visual web interface        |
| **Search**     | Precise, structured queries | Interactive with type-ahead |
| **Results**    | Structured data             | Rich visual presentation    |
| **Filtering**  | API parameters              | Interactive form controls   |
| **Deployment** | Integrated with AI tools    | Standalone web application  |

## Tips & Tricks 💡

### **Effective Search Strategies**

1. **Start broad, then filter** - Search "customer" then filter by module
2. **Use frontend terms** - Search "iconset", "navigator", "tree" for UI elements
3. **Combine concepts** - Search "delivery structure iconset" for specific UI elements
4. **Use file type filters** - Filter by `.client` for frontend, `.entity` for data models

### **Understanding Results**

- **Score** indicates relevance (higher = more relevant)
- **Tags** show file characteristics (module, logical unit, has-iconsets, etc.)
- **Frontend Elements** section shows UI components found in the file
- **Highlight** shows matching content with context

### **Performance Tips**

- Index is cached for fast subsequent searches
- Use result limits to avoid overwhelming results
- Type-ahead suggestions help refine queries quickly

## Development 🛠️

### Run in Development Mode

```bash
uv run python -m src.ifs_cloud_mcp_server.launcher web --reload --port 8080
```

### Custom Styling

Modify `templates/index.html` to customize the interface. The template uses:

- Tailwind CSS for styling
- Alpine.js for interactivity
- Font Awesome for icons

### Extending the API

Add new endpoints in `src/ifs_cloud_mcp_server/web_ui.py` by adding routes to the FastAPI app.

## Troubleshooting 🔧

### Common Issues

**No search results found:**

- Ensure index is built: `uv run python -m src.ifs_cloud_mcp_server.launcher index build --directory /path/to/project`
- Check index stats: `uv run python -m src.ifs_cloud_mcp_server.launcher index stats`

**Port already in use:**

- Use a different port: `--port 8080`
- Or stop the conflicting service

**Slow search performance:**

- Reduce result limit in search
- Consider rebuilding index if it's corrupted

### Debug Mode

Start with debug logging to troubleshoot issues:

```bash
export LOG_LEVEL=DEBUG
uv run python -m src.ifs_cloud_mcp_server.launcher web
```
