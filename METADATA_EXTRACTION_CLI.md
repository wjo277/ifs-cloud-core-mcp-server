# 🚀 IFS Cloud Metadata Extraction

Extract metadata directly from your IFS Cloud database using the integrated MCP server CLI. Each developer can now have enhanced search capabilities that match their exact database schema and configuration.

## 🎯 **Why Use This?**

- ✅ **Environment-Specific**: Get metadata from YOUR database, not generic examples
- ✅ **Always Current**: Extracts live data that matches your actual system
- ✅ **Developer-Friendly**: Each developer gets metadata for their environment
- ✅ **Integrated**: Built right into the main MCP server CLI - no separate tools needed
- ✅ **Secure**: Multiple options for handling database credentials safely
- ✅ **Fast**: Direct database extraction optimized for production use

## 🚀 **Quick Start**

### Method 1: Environment Variables (Most Secure)

```bash
# Set environment variables for security
export IFS_DB_PASSWORD="your_secure_password"

# Extract metadata using environment password
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host your-db-host \
  --username ifsapp \
  --service IFSCDEV \
  25.1.0
```

### Method 2: Connection String

```bash
# Using complete connection string
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --connection "oracle://ifsapp:password@host:1521/IFSCDEV" \
  25.1.0
```

### Method 3: Individual Parameters

```bash
# Using individual parameters (password will be visible in command history)
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host your-db-host \
  --username ifsapp \
  --password your_password \
  --service IFSCDEV \
  25.1.0
```

## 📊 **What Gets Extracted**

The tool extracts comprehensive metadata from your IFS Cloud database:

| Data Type           | Description                     | Typical Count |
| ------------------- | ------------------------------- | ------------- |
| **Logical Units**   | Core business entities          | 10,000+       |
| **Modules**         | Functional areas                | 150+          |
| **Domain Mappings** | Business term translations      | 8,000+        |
| **Views**           | Database views and descriptions | 5,000+        |

## 🔧 **Command Reference**

### Basic Extraction

```bash
# Most common usage with environment password
export IFS_DB_PASSWORD="secure_password"
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host prod-db01 \
  --username ifsapp \
  --service IFSCPROD \
  25.1.0
```

### Advanced Options

```bash
# Custom output directory
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --connection "oracle://user:pass@host:1521/service" \
  --output ./my_metadata \
  25.1.0

# Quiet mode with JSON output (for automation)
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --connection "oracle://user:pass@host:1521/service" \
  --quiet --json \
  25.1.0 > extraction_results.json

# Verbose debug mode
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --connection "oracle://user:pass@host:1521/service" \
  --log-level DEBUG \
  25.1.0
```

## 📋 **Complete Options Reference**

### Connection Options

```
--connection, -c    Complete connection string (oracle://user:pass@host:port/service)
--host             Database host
--port             Database port (default: 1521)
--username, -u     Database username
--password, -p     Database password (or use IFS_DB_PASSWORD env var)
--service          Oracle service name
--sid              Oracle SID (alternative to service name)
--driver           SQLAlchemy driver (default: oracle+oracledb)
```

### Extraction Options

```
version            IFS Cloud version (required, positional argument)
--output, -o       Output directory (default: platform data directory)
```

### Output Options

```
--quiet, -q        Suppress non-error output
--json             Output results as JSON
--log-level        DEBUG, INFO, WARNING, ERROR (default: INFO)
```

## 🎯 **Example Workflows**

### Development Environment

```bash
# Developer extracting from development database
export IFS_DB_PASSWORD="dev_password"
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host dev-ifs-01 \
  --username ifsapp \
  --service IFSCDEV \
  25.1.0
```

### Production Environment

```bash
# Production extraction with security best practices
export IFS_DB_PASSWORD="$(read -s -p 'Database Password: ' pwd; echo $pwd)"
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host prod-ifs-01 \
  --username ifsapp \
  --service IFSCPROD \
  25.1.0
```

### Continuous Integration

```bash
# Automated extraction in CI/CD
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --connection "$IFS_CONNECTION_STRING" \
  --quiet --json \
  "$IFS_VERSION" > metadata_results.json
```

### Multiple Environments

```bash
# Development
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host dev-db --username ifsapp --service IFSCDEV \
  --output ./metadata/dev \
  25.1.0

# Testing
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host test-db --username ifsapp --service IFSCTEST \
  --output ./metadata/test \
  25.1.0

# Production
export IFS_DB_PASSWORD="prod_password"
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host prod-db --username ifsapp --service IFSCPROD \
  --output ./metadata/prod \
  25.1.0
```

## 🔒 **Security Best Practices**

### 1. Environment Variables (.env file)

```bash
# .env file (not committed to git)
IFS_DB_HOST=prod-db01.company.com
IFS_DB_USERNAME=ifsapp
IFS_DB_PASSWORD=secure_password_here
IFS_DB_SERVICE=IFSCPROD
IFS_VERSION=25.1.0
```

### 2. System Environment Variables

```bash
# Set in your shell profile
export IFS_DB_PASSWORD="secure_password"
# Use other parameters as command line args
```

### 3. Interactive Password Prompt (Future Enhancement)

```bash
# Password will be prompted securely (Future Enhancement)
# Currently use environment variables for security
export IFS_DB_PASSWORD="secure_password"
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host prod-db01 \
  --username ifsapp \
  --service IFSCPROD \
  25.1.0
```

## 📋 **Complete Options Reference**

### Connection Options

```
--connection, -c    Complete connection string
--host             Database host
--port             Database port (default: 1521)
--username, -u     Database username
--password, -p     Database password (or use IFS_DB_PASSWORD env var)
--service          Oracle service name
--sid              Oracle SID (alternative to service name)
--driver           SQLAlchemy driver (default: oracle+oracledb)
```

### Extraction Options

```
version            IFS Cloud version (required, positional argument)
--output, -o       Output directory (default: platform data directory)
```

### Output Options

```
--quiet, -q        Suppress non-error output
--json             Output results as JSON
--log-level        DEBUG, INFO, WARNING, ERROR (default: INFO)
```

## 🎯 **Example Workflows**

### Development Environment

```bash
# Developer extracting from development database
export IFS_DB_PASSWORD="dev_password"
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host dev-ifs-01 \
  --username ifsapp \
  --service IFSCDEV \
  25.1.0
```

### Production Environment

```bash
# Production extraction with security best practices
export IFS_DB_PASSWORD="$(read -s -p 'Database Password: ' pwd; echo $pwd)"
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host prod-ifs-01 \
  --username ifsapp \
  --service IFSCPROD \
  25.1.0
```

### Continuous Integration

```bash
# Automated extraction in CI/CD
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --connection "$IFS_CONNECTION_STRING" \
  --quiet --json \
  "$IFS_VERSION" > metadata_results.json
```

### Multiple Environments

```bash
# Development
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host dev-db --username ifsapp --service IFSCDEV \
  --output ./metadata/dev \
  25.1.0

# Testing
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host test-db --username ifsapp --service IFSCTEST \
  --output ./metadata/test \
  25.1.0

# Production
export IFS_DB_PASSWORD="prod_password"
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --host prod-db --username ifsapp --service IFSCPROD \
  --output ./metadata/prod \
  25.1.0
```

## 📁 **Output Structure**

After extraction, your metadata will be stored in the platform-appropriate data directory:

**Windows**: `%APPDATA%\ifs_cloud_mcp_server\metadata\25.1.0\`  
**macOS**: `~/Library/Application Support/ifs_cloud_mcp_server/metadata/25.1.0/`  
**Linux**: `~/.local/share/ifs_cloud_mcp_server/metadata/25.1.0/`

```
metadata/25.1.0/
├── metadata_extract.json      # Complete metadata extract
└── checksums.txt             # Verification checksums

# Or custom output with --output:
./your_output_dir/25.1.0/
├── metadata_extract.json
└── checksums.txt
```

## 🚨 **Troubleshooting**

### Common Issues

**1. Missing Dependencies**

```bash
# Install required packages
uv add sqlalchemy oracledb
```

**2. Connection Issues**

```bash
# Test connection first
uv run python -c "
import oracledb
conn = oracledb.connect(user='ifsapp', password='pass', host='host', port=1521, service_name='IFSCDEV')
print('✓ Connection successful')
conn.close()
"
```

**3. Permission Issues**

```bash
# Ensure user has read access to system tables
# Contact your DBA if needed
```

### Getting Help

```bash
# Show general help
uv run python -m src.ifs_cloud_mcp_server.main --help

# Show extract command help
uv run python -m src.ifs_cloud_mcp_server.main extract --help

# Verbose mode for debugging
uv run python -m src.ifs_cloud_mcp_server.main extract \
  --log-level DEBUG [other options] 25.1.0
```

## 🎉 **What's Next?**

After extraction, use your metadata with:

1. **Enhanced Search**: Use with the IFS Cloud MCP Server for intelligent search
2. **Development Tools**: Better code completion and context
3. **Documentation**: Generate up-to-date system documentation
4. **Analysis**: Understand your IFS Cloud customizations and extensions

---

## 💡 **Pro Tips**

1. **Regular Extractions**: Set up scheduled extractions to keep metadata current
2. **Version Control**: Keep different extractions for different environments
3. **Automation**: Use JSON output mode for integration with other tools
4. **Security**: Always use environment variables or .env files for credentials
5. **Performance**: Extraction typically takes 2-5 minutes for full IFS Cloud databases

**Ready to supercharge your IFS Cloud development with environment-specific metadata!** 🚀
