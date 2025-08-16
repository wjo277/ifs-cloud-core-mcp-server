# 🚀 GitHub Release Upload for Quantized Models

This directory contains scripts to automatically upload the quantized FastAI models to GitHub releases with device authentication.

## 📋 Prerequisites

### 1. GitHub CLI Installation

**Windows:**

```powershell
winget install GitHub.cli
```

**macOS:**

```bash
brew install gh
```

**Linux:**

```bash
# Ubuntu/Debian
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install gh

# Other distributions: https://cli.github.com/manual/installation
```

### 2. Model Preparation

Ensure models are prepared for release:

```bash
uv run python scripts/prepare_model_release.py
```

This creates:

- `dist/fastai_intent_classifier_quantized.pkl` (63.0 MB) - Primary model
- `dist/fastai_intent_classifier.pkl` (121.1 MB) - Fallback model

## 🚀 Upload Methods

### Method 1: Python Script (Recommended)

```bash
uv run python scripts/upload_to_github.py
```

**Features:**

- Interactive wizard interface
- Device authentication flow
- Progress tracking
- Automatic release notes generation
- Error handling and validation

### Method 2: PowerShell Script (Windows)

```powershell
.\scripts\upload_to_github.ps1
```

**Features:**

- Windows-optimized interface
- Automatic model preparation
- Colorized output
- Error handling

### Method 3: Batch Script (Windows)

```cmd
scripts\upload_to_github.bat
```

**Features:**

- Simple double-click execution
- Basic error handling
- Virtual environment activation

## 🔐 Authentication Process

The script will guide you through GitHub authentication:

1. **Check existing auth:** Verifies if you're already logged in
2. **Device flow:** Opens browser for GitHub login if needed
3. **Verification:** Confirms authentication success

**No tokens or passwords needed!** Uses secure device authentication.

## 📤 Upload Process

### Interactive Flow

1. **Pre-checks:** Validates GitHub CLI, authentication, and model files
2. **Release info:** Prompts for tag, title, and prerelease status
3. **Notes generation:** Creates comprehensive release notes automatically
4. **Confirmation:** Shows preview before upload
5. **Creation:** Creates GitHub release
6. **Upload:** Uploads both model files with progress tracking

### Example Session

```
🚀 GITHUB RELEASE UPLOAD WIZARD
============================================================

✅ GitHub CLI found: gh version 2.40.1
✅ Already authenticated with GitHub CLI

📁 Checking model files...
   ✅ fastai_intent_classifier_quantized.pkl (63.0 MB)
   ✅ fastai_intent_classifier.pkl (121.1 MB)

📊 Total upload size: 184.1 MB

📋 Existing releases:
   • v1.0.0

📝 Release Information:
   Last release: v1.0.0
   Enter release tag (e.g., v1.1.0): v1.1.0
   Enter release title (default: 'IFS Cloud MCP Server v1.1.0'):
   Is this a prerelease? (y/N): n

📄 Generated release notes preview:
----------------------------------------
# IFS Cloud MCP Server v1.1.0

## 🚀 Quantized Model Release

This release features the new **quantized FastAI intent classifier**...
----------------------------------------

Proceed with release creation and upload? (Y/n): y

🚀 Creating release v1.1.0...
✅ Release v1.1.0 created successfully!

📤 Uploading assets to release v1.1.0...
   Uploading fastai_intent_classifier_quantized.pkl...
   ✅ fastai_intent_classifier_quantized.pkl uploaded successfully
   Uploading fastai_intent_classifier.pkl...
   ✅ fastai_intent_classifier.pkl uploaded successfully

🎉 SUCCESS! Release v1.1.0 created and assets uploaded!
🔗 View release: https://github.com/graknol/ifs-cloud-core-mcp-server/releases/tag/v1.1.0
```

## 📋 Generated Release Notes

The script automatically generates comprehensive release notes including:

- **Performance metrics:** Size reduction, speed improvement
- **Usage instructions:** Download commands and configuration
- **Migration notes:** Upgrade process for existing users
- **Technical details:** Quantization benefits and compatibility

## 🔧 Configuration

### Repository Settings

Default: `graknol/ifs-cloud-core-mcp-server`

To change repository:

```python
uploader = GitHubReleaseUploader("your-username/your-repo")
```

### Model Files

Default paths:

- `dist/fastai_intent_classifier_quantized.pkl` - Primary model
- `dist/fastai_intent_classifier.pkl` - Fallback model

## ⚠️ Troubleshooting

### GitHub CLI Not Found

```bash
# Check installation
gh --version

# Reinstall if needed
winget install GitHub.cli  # Windows
brew install gh            # macOS
```

### Authentication Issues

```bash
# Check auth status
gh auth status

# Re-authenticate
gh auth login --web
```

### Model Files Missing

```bash
# Prepare models
uv run python scripts/prepare_model_release.py

# Check files exist
ls dist/
```

### Upload Failures

- Check internet connection
- Verify repository permissions
- Ensure sufficient GitHub storage quota
- Try smaller uploads if network issues

## 🎯 Post-Upload Steps

After successful upload:

1. **Update model downloader:**

   ```python
   # In src/ifs_cloud_mcp_server/model_downloader.py
   DEFAULT_TAG = "v1.1.0"  # Update to new release
   ```

2. **Test download:**

   ```bash
   # Test quantized model download
   uv run python -m ifs_cloud_mcp_server.model_downloader --tag v1.1.0
   ```

3. **Update documentation:**
   - README.md
   - Installation instructions
   - Version references

## 📊 Release Benefits

**For Users:**

- 48% smaller downloads
- 22% faster inference
- Same search accuracy
- Automatic fallback

**For Deployment:**

- Faster CI/CD pipelines
- Reduced bandwidth costs
- Better user experience
- Easier distribution

---

_The upload scripts handle all the complexity - just run and follow the interactive prompts!_ 🚀
