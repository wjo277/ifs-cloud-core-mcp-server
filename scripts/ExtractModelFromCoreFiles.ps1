# Extract IFS Cloud Core Model Files
# This script copies specific file types from IFS Cloud Core Codes directory
# while preserving the folder structure

param(
  [string]$SourcePath = $null
)

# Define the target extensions (case-insensitive)
$TargetExtensions = @('.entity', '.plsql', '.views', '.storage', '.fragment', '.client', '.projection', '.plsvc')

# Get the script's directory and create the work directory path
$WorkDir = Join-Path $PSScriptRoot "../_work"

function Get-IFSCloudCoreLocation {
  param([string]$ProvidedPath)

  if ($ProvidedPath -and (Test-Path $ProvidedPath)) {
    return $ProvidedPath
  }

  do {
    Write-Host "Please enter the path to the IFS Cloud Core Codes directory:" -ForegroundColor Yellow
    $userPath = Read-Host

    if ([string]::IsNullOrWhiteSpace($userPath)) {
      Write-Host "Path cannot be empty. Please try again." -ForegroundColor Red
      continue
    }

    if (-not (Test-Path $userPath)) {
      Write-Host "Path '$userPath' does not exist. Please try again." -ForegroundColor Red
      continue
    }

    if (-not (Test-Path $userPath -PathType Container)) {
      Write-Host "Path '$userPath' is not a directory. Please try again." -ForegroundColor Red
      continue
    }

    return $userPath
  } while ($true)
}

function Copy-FilesWithStructure {
  param(
    [string]$SourceRoot,
    [string]$DestinationRoot,
    [string[]]$Extensions
  )

  Write-Host "Searching for files with extensions: $($Extensions -join ', ')" -ForegroundColor Green

  # Check if destination directory exists and is not empty
  if (Test-Path $DestinationRoot) {
    $existingFiles = Get-ChildItem -Path $DestinationRoot -Recurse -File
    if ($existingFiles.Count -gt 0) {
      Write-Host "Error: Destination directory '$DestinationRoot' already exists and is not empty." -ForegroundColor Red
      Write-Host "Please delete or rename the existing directory before running this script." -ForegroundColor Red
      throw "Destination directory is not empty. Operation aborted."
    }
  }

  # Create destination directory if it doesn't exist
  if (-not (Test-Path $DestinationRoot)) {
    New-Item -ItemType Directory -Path $DestinationRoot -Force | Out-Null
    Write-Host "Created destination directory: $DestinationRoot" -ForegroundColor Green
  }

  # Build filter for Get-ChildItem
  $IncludeFilter = $Extensions | ForEach-Object { "*$_" }

  # Find all matching files recursively
  $FilesToCopy = Get-ChildItem -Path $SourceRoot -Recurse -File -Include $IncludeFilter

  if ($FilesToCopy.Count -eq 0) {
    Write-Host "No files found with the specified extensions." -ForegroundColor Yellow
    return
  }

  Write-Host "Found $($FilesToCopy.Count) files to copy..." -ForegroundColor Green

  $CopiedCount = 0
  $ErrorCount = 0

  foreach ($File in $FilesToCopy) {
    try {
      # Calculate relative path from source root
      $RelativePath = $File.FullName.Substring($SourceRoot.Length).TrimStart('\', '/')

      # Skip files in source/<model>/replication directories
      if ($RelativePath -match '\\source\\[^\\]+\\replication\\' -or $RelativePath -match '/source/[^/]+/replication/') {
        continue
      }

      # Create destination path
      $DestinationPath = Join-Path $DestinationRoot $RelativePath
      $DestinationDir = Split-Path $DestinationPath -Parent

      # Create destination directory if it doesn't exist
      if (-not (Test-Path $DestinationDir)) {
        New-Item -ItemType Directory -Path $DestinationDir -Force | Out-Null
      }

      # Copy the file
      Copy-Item -Path $File.FullName -Destination $DestinationPath -Force
      $CopiedCount++

      # Show progress every 10 files
      if ($CopiedCount % 10 -eq 0) {
        Write-Host "`rCopied $CopiedCount files..." -ForegroundColor Cyan -NoNewline
      }
    }
    catch {
      Write-Host "`n`nError copying file '$($File.FullName)': $($_.Exception.Message)" -ForegroundColor Red
      $ErrorCount++
    }
  }

  Write-Host "`n`nCopy operation completed!" -ForegroundColor Green
  Write-Host "Successfully copied: $CopiedCount files" -ForegroundColor Green
  if ($ErrorCount -gt 0) {
    Write-Host "Errors encountered: $ErrorCount files" -ForegroundColor Red
  }
}

function Show-Summary {
  param(
    [string]$SourcePath,
    [string]$DestinationPath,
    [string[]]$Extensions
  )

  Write-Host "`n" + "="*60 -ForegroundColor Cyan
  Write-Host "EXTRACTION SUMMARY" -ForegroundColor Cyan
  Write-Host "="*60 -ForegroundColor Cyan
  Write-Host "Source Directory: $SourcePath" -ForegroundColor White
  Write-Host "Destination Directory: $DestinationPath" -ForegroundColor White
  Write-Host "File Extensions: $($Extensions -join ', ')" -ForegroundColor White
  Write-Host "="*60 -ForegroundColor Cyan
}

# Main execution
try {
  Write-Host "IFS Cloud Core Model Files Extractor" -ForegroundColor Cyan
  Write-Host "=====================================" -ForegroundColor Cyan

  # Get the source directory
  $SourceDirectory = Get-IFSCloudCoreLocation -ProvidedPath $SourcePath
  Write-Host "Source directory: $SourceDirectory" -ForegroundColor Green

  # Show what we're about to do
  Write-Host "`nTarget extensions: $($TargetExtensions -join ', ')" -ForegroundColor Yellow
  Write-Host "Destination directory: $WorkDir" -ForegroundColor Yellow

  # Ask for confirmation
  Write-Host "`nThis will copy all matching files to the destination directory." -ForegroundColor Yellow
  $Confirmation = Read-Host "Do you want to continue? (Y/N)"

  if ($Confirmation -notmatch '^[Yy]$') {
    Write-Host "Operation cancelled by user." -ForegroundColor Yellow
    exit 0
  }

  # Perform the copy operation
  Copy-FilesWithStructure -SourceRoot $SourceDirectory -DestinationRoot $WorkDir -Extensions $TargetExtensions

  # Show summary
  Show-Summary -SourcePath $SourceDirectory -DestinationPath $WorkDir -Extensions $TargetExtensions

  Write-Host "`nOperation completed successfully!" -ForegroundColor Green
}
catch {
  Write-Host "An error occurred: $($_.Exception.Message)" -ForegroundColor Red
  exit 1
}