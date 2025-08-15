# 📦 IFS Cloud ZIP File Indexing Instructions

## Overview

The IFS Cloud MCP Server supports importing and indexing ZIP files containing IFS Cloud source code to create versioned catalogs for intelligent AI analysis. This guide provides step-by-step instructions for adding ZIP files to your versioned catalog.

## 🎯 Quick Start

### Basic Import Command

```bash
python -m src.ifs_cloud_mcp_server.main import <zip_file_path> --version <version_identifier>
```

### Example

```bash
python -m src.ifs_cloud_mcp_server.main import "C:\Downloads\IFS_Cloud_24.2.1.zip" --version "24.2.1"
```

## 📁 File System Structure

The system automatically creates the following structure:

```
%APPDATA%\ifs_cloud_mcp_server\          # Windows
~/.local/share/ifs_cloud_mcp_server\     # Linux
~/Library/Application Support/ifs_cloud_mcp_server\  # macOS
├── extracts\
│   ├── 24.2.1\                         # Version-specific extracted files
│   ├── latest\
│   └── custom_build\
└── indexes\
    ├── 24.2.1\                         # Version-specific search indexes
    ├── latest\
    └── custom_build\
```

## 🚀 Complete Import Process

### Step 1: Prepare Your ZIP File

Ensure your ZIP file contains IFS Cloud source files with supported extensions:

- `.entity` - Entity definitions
- `.plsql` - PL/SQL business logic
- `.client` - Client UI definitions
- `.projection` - Business projections
- `.fragment` - UI fragments
- `.views` - Database views
- `.storage` - Storage configurations
- `.plsvc` - Platform services
- `.enumeration` - Enumerations

### Step 2: Import the ZIP File

#### Basic Import

```bash
python -m src.ifs_cloud_mcp_server.main import "path/to/your/ifs_cloud.zip" --version "24.2.1"
```

#### Import with Custom Index Path

```bash
python -m src.ifs_cloud_mcp_server.main import "path/to/your/ifs_cloud.zip" --version "24.2.1" --index-path "C:\custom\index\location"
```

#### Import with Verbose Logging

```bash
python -m src.ifs_cloud_mcp_server.main import "path/to/your/ifs_cloud.zip" --version "24.2.1" --log-level DEBUG
```

### Step 3: Verify Import Success

The import process will:

1. ✅ **Extract** supported files from ZIP to versioned directory
2. ✅ **Filter** only IFS Cloud file types
3. ✅ **Index** files for fast searching with Tantivy
4. ✅ **Cache** metadata for efficient updates
5. ✅ **Log** detailed statistics

**Success Output:**

```
🚀 Starting IFS Cloud ZIP import...
📦 ZIP file: /path/to/ifs_cloud.zip
🏷️  Version: 24.2.1

✅ Import completed successfully!
📁 Extracted files: /data/ifs_cloud_mcp_server/extracts/24.2.1
🔍 Search index: /data/ifs_cloud_mcp_server/indexes/24.2.1
🏷️  Version: 24.2.1

Files indexed: 15,847
Files cached: 0
Files skipped: 23
Errors: 0

To use this version with the MCP server:
  python -m src.ifs_cloud_mcp_server.main server --version "24.2.1"
```

## 📋 Managing Versions

### List Available Versions

```bash
python -m src.ifs_cloud_mcp_server.main list
```

**Example Output:**

```
📦 Available IFS Cloud Versions:

Version: 24.2.1
  Files: 15,847
  Created: 2025-08-15 10:30:45
  Status: ✅ Indexed
  Path: C:\Users\...\ifs_cloud_mcp_server\extracts\24.2.1

Version: latest
  Files: 16,234
  Created: 2025-08-15 14:22:18
  Status: ✅ Indexed
  Path: C:\Users\...\ifs_cloud_mcp_server\extracts\latest
```

### List in JSON Format (for scripts)

```bash
python -m src.ifs_cloud_mcp_server.main list --json
```

## 🎯 Using Imported Versions

### Start Server with Specific Version

```bash
python -m src.ifs_cloud_mcp_server.main server --version "24.2.1"
```

### Start Server with Default Index

```bash
python -m src.ifs_cloud_mcp_server.main server --index-path "./index"
```

## 🔧 Advanced Usage

### Version Naming Best Practices

#### Semantic Versioning

```bash
--version "24.2.1"      # Official release
--version "24.2.1-RC1"  # Release candidate
--version "24.2.1-dev"  # Development build
```

#### Custom Identifiers

```bash
--version "latest"           # Latest build
--version "production"       # Production environment
--version "dev_branch"       # Development branch
--version "custom_build_001" # Custom build
```

### Handling Large ZIP Files

For very large ZIP files (>1GB), consider:

```bash
# Use DEBUG logging to monitor progress
python -m src.ifs_cloud_mcp_server.main import large_file.zip --version "24.2.1" --log-level DEBUG

# Process will show progress every 100 files:
# "Extracted 100 files..."
# "Extracted 200 files..."
```

### Re-importing a Version

If you need to re-import a version (e.g., updated build):

1. The system automatically **removes existing extracts**
2. **Re-extracts** all files from ZIP
3. **Rebuilds** the search index
4. **Updates** cache metadata

```bash
# This will overwrite existing "24.2.1" version
python -m src.ifs_cloud_mcp_server.main import updated_build.zip --version "24.2.1"
```

## ⚡ Performance & Optimization

### Extraction Performance

- **Parallel processing**: Files are processed in batches of 250
- **Memory efficient**: Batch processing prevents memory exhaustion
- **Progress tracking**: Every 100 files logged for large imports

### Indexing Performance

- **Intelligent caching**: Only changed files are re-indexed
- **Fast search**: Tantivy provides millisecond search times
- **Deduplication**: Duplicate files are handled efficiently

### Storage Optimization

- **Selective extraction**: Only supported file types are extracted
- **Compression**: Index uses efficient storage
- **Version isolation**: Each version is completely separate

## 🔍 Integration with AI Agent

Once indexed, the ZIP contents become available to the intelligent AI agent:

### Automatic Discovery

```python
# AI agent automatically finds relevant files from indexed ZIP
intelligent_context_analysis(
    "Create customer order validation",
    domain="ORDER"
)
```

### Version-Specific Analysis

```bash
# Start server with specific version for AI analysis
python -m src.ifs_cloud_mcp_server.main server --version "24.2.1"
```

## 🚨 Troubleshooting

### Common Issues

#### ZIP File Not Found

```
ERROR: ZIP file not found: /path/to/file.zip
```

**Solution**: Check file path and ensure ZIP file exists

#### Bad ZIP File

```
ERROR: Failed to import: zipfile.BadZipFile
```

**Solution**: Verify ZIP file is not corrupted

#### Permission Denied

```
ERROR: Permission denied writing to data directory
```

**Solution**: Ensure write permissions to data directory

#### Invalid Version Name

```
ERROR: Version must contain at least one alphanumeric character
```

**Solution**: Use valid version names (alphanumeric, dots, dashes, underscores)

### Debug Mode

For detailed troubleshooting:

```bash
python -m src.ifs_cloud_mcp_server.main import file.zip --version "test" --log-level DEBUG
```

This shows:

- Each file being extracted
- Indexing progress details
- Error details for failed files
- Performance statistics

## 💡 Tips & Best Practices

### 1. Version Organization

- Use semantic versioning for official releases
- Use descriptive names for development builds
- Keep production and development versions separate

### 2. Storage Management

- Large extracts consume disk space
- Consider removing old versions periodically
- Monitor data directory size

### 3. Performance Optimization

- Import during low-usage periods for large files
- Use SSD storage for better index performance
- Consider separate storage for different teams

### 4. Integration Workflow

```bash
# 1. Import new version
python -m src.ifs_cloud_mcp_server.main import new_build.zip --version "24.3.0"

# 2. Verify import
python -m src.ifs_cloud_mcp_server.main list

# 3. Start server with new version
python -m src.ifs_cloud_mcp_server.main server --version "24.3.0"

# 4. AI agent now has access to new version for analysis
```

## ✅ Summary

The IFS Cloud ZIP import system provides:

- **Automated extraction** of supported file types
- **Version management** for multiple IFS Cloud releases
- **High-performance indexing** with Tantivy search
- **Intelligent caching** for efficient updates
- **Seamless integration** with the AI agent

Your imported ZIP files become a **versioned catalog** that the intelligent AI agent can automatically search and analyze to understand existing patterns and implement new features consistently with IFS Cloud standards! 🚀
