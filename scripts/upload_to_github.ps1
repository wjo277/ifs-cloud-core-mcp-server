# GitHub Release Upload Script for PowerShell
# This script uploads the quantized models to GitHub releases

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " GitHub Release Upload for Quantized Models" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "scripts\upload_to_github.py")) {
    Write-Host "❌ Error: Please run this script from the project root directory" -ForegroundColor Red
    Write-Host "   Expected to find: scripts\upload_to_github.py" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to continue"
    exit 1
}

# Check for UV
try {
    $uvVersion = & uv --version 2>$null
    Write-Host "✅ UV found: $uvVersion" -ForegroundColor Green
}
catch {
    Write-Host "❌ UV not found! Please install UV first." -ForegroundColor Red
    Write-Host "   Install: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to continue"
    exit 1
}

# Check if models are prepared
if (-not (Test-Path "dist\fastai_intent_classifier_quantized.pkl")) {
    Write-Host "⚠️ Model files not found in dist/ directory" -ForegroundColor Yellow
    Write-Host "   Preparing models first..." -ForegroundColor Yellow
    Write-Host ""

    try {
        & uv run python scripts\prepare_model_release.py
        if ($LASTEXITCODE -ne 0) {
            throw "Model preparation failed"
        }
        Write-Host "✅ Models prepared successfully!" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to prepare models: $_" -ForegroundColor Red
        Read-Host "Press Enter to continue"
        exit 1
    }
}

# Run the Python upload script
Write-Host "🚀 Starting upload process..." -ForegroundColor Green
Write-Host ""

try {
    & uv run python scripts\upload_to_github.py

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "🎉 Upload completed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "📋 Next steps:" -ForegroundColor Cyan
        Write-Host "1. Test download with new release" -ForegroundColor White
        Write-Host "2. Update DEFAULT_TAG in model_downloader.py" -ForegroundColor White
        Write-Host "3. Update documentation if needed" -ForegroundColor White
    }
    else {
        Write-Host ""
        Write-Host "❌ Upload failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
}
catch {
    Write-Host ""
    Write-Host "❌ Upload failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Read-Host "Press Enter to continue"
