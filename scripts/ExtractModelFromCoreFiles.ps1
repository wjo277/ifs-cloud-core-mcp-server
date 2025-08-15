# Extract IFS Cloud Core Model Files
# This script copies specific file types from IFS Cloud Core Codes directory
# while preserving the folder structure

param(
  [string]$SourcePath = $null
)

# Define the target extensions (case-insensitive)
$TargetExtensions = @('.entity', '.plsql', '.views', '.storage', '.fragment', '.client', '.projection', '.plsvc', '.enumeration')

# Get the script's directory and create the work directory path
$WorkDir = Join-Path $PSScriptRoot "../_work"

function Get-IFSCloudCoreLocation {
  param([string]$ProvidedPath)

  if ($ProvidedPath -and (Test-Path $ProvidedPath)) {
    # Validate that this looks like an IFS Cloud Core directory
    if (Test-IFSCloudCoreDirectory -Path $ProvidedPath) {
      return $ProvidedPath
    }
    else {
      Write-Host "Warning: The provided path '$ProvidedPath' does not appear to be a valid IFS Cloud Core Codes directory." -ForegroundColor Yellow
    }
  }

  # Try to find common IFS Cloud locations
  $CommonPaths = @(
    "C:\IFS\Cloud\Core",
    "C:\IFS\Cloud\Codes",
    "C:\Program Files\IFS\Cloud\Core",
    "D:\IFS\Cloud\Core",
    "E:\IFS\Cloud\Core"
  )

  Write-Host "`nLooking for IFS Cloud Core directories in common locations..." -ForegroundColor Yellow
  $FoundPaths = @()

  foreach ($path in $CommonPaths) {
    if (Test-Path $path) {
      if (Test-IFSCloudCoreDirectory -Path $path) {
        $FoundPaths += $path
        Write-Host "  ✓ Found: $path" -ForegroundColor Green
      }
    }
  }

  if ($FoundPaths.Count -gt 0) {
    Write-Host "`nFound $($FoundPaths.Count) potential IFS Cloud Core directories:" -ForegroundColor Green
    for ($i = 0; $i -lt $FoundPaths.Count; $i++) {
      Write-Host "  [$($i + 1)] $($FoundPaths[$i])" -ForegroundColor Cyan
    }
    Write-Host "  [0] None of these - I'll provide my own path" -ForegroundColor Cyan

    do {
      $choice = Read-Host "`nSelect a directory [0-$($FoundPaths.Count)]"
      if ($choice -match '^\d+$' -and [int]$choice -ge 0 -and [int]$choice -le $FoundPaths.Count) {
        if ([int]$choice -eq 0) {
          break
        }
        else {
          return $FoundPaths[[int]$choice - 1]
        }
      }
      Write-Host "Invalid choice. Please enter a number between 0 and $($FoundPaths.Count)." -ForegroundColor Red
    } while ($true)
  }

  do {
    Write-Host "`nPlease enter the path to the IFS Cloud Core Codes directory:" -ForegroundColor Yellow
    Write-Host "Example: C:\IFS\Cloud\Core" -ForegroundColor Gray
    $selectedPath = Read-Host "Path"

    if ([string]::IsNullOrWhiteSpace($selectedPath)) {
      Write-Host "Path cannot be empty. Please try again." -ForegroundColor Red
      continue
    }

    Write-Host "Entered path: $selectedPath" -ForegroundColor Cyan    if (-not (Test-Path $selectedPath)) {
      Write-Host "Selected path '$selectedPath' does not exist. Please try again." -ForegroundColor Red
      continue
    }

    if (-not (Test-Path $selectedPath -PathType Container)) {
      Write-Host "Selected path '$selectedPath' is not a directory. Please try again." -ForegroundColor Red
      continue
    }

    # Validate that this looks like an IFS Cloud Core directory
    if (-not (Test-IFSCloudCoreDirectory -Path $selectedPath)) {
      Write-Host "Selected path '$selectedPath' does not appear to be a valid IFS Cloud Core Codes directory." -ForegroundColor Red
      Write-Host "Expected to find subdirectories with IFS Cloud modules (like 'order', 'proj', 'invent', etc.)" -ForegroundColor Red
      Write-Host "Please select a different directory." -ForegroundColor Red
      continue
    }

    return $selectedPath
  } while ($true)
}

function Test-IFSCloudCoreDirectory {
  param([string]$Path)

  # Check for common IFS Cloud module directories
  $CommonModules = @('order', 'proj', 'invent', 'fndcob', 'mfgstd', 'purch', 'genled', 'appsrv')
  $FoundModules = 0

  foreach ($module in $CommonModules) {
    if (Test-Path (Join-Path $Path $module)) {
      $FoundModules++
    }
  }

  # If we find at least 3 common modules, consider it valid
  if ($FoundModules -ge 3) {
    return $true
  }

  # Alternative check: look for any directories containing our target file types
  $SubDirectories = Get-ChildItem -Path $Path -Directory -ErrorAction SilentlyContinue
  foreach ($dir in $SubDirectories) {
    $IncludeFilter = $TargetExtensions | ForEach-Object { "*$_" }
    $FilesFound = Get-ChildItem -Path $dir.FullName -Recurse -File -Include $IncludeFilter -ErrorAction SilentlyContinue
    if ($FilesFound.Count -gt 0) {
      return $true
    }
  }

  return $false
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
      Write-Host "Warning: Destination directory '$DestinationRoot' already exists and contains $($existingFiles.Count) files." -ForegroundColor Yellow
      Write-Host "Do you want to:" -ForegroundColor Yellow
      Write-Host "  [O] Overwrite (delete existing directory and create new)" -ForegroundColor Cyan
      Write-Host "  [M] Merge (add new files, overwrite existing files with same name)" -ForegroundColor Cyan
      Write-Host "  [C] Cancel operation" -ForegroundColor Cyan

      do {
        $choice = Read-Host "Enter your choice [O/M/C]"
        $choice = $choice.ToUpper()
      } while ($choice -notin @('O', 'M', 'C'))

      switch ($choice) {
        'O' {
          Write-Host "Removing existing directory..." -ForegroundColor Yellow
          Remove-Item -Path $DestinationRoot -Recurse -Force
          Write-Host "Existing directory removed." -ForegroundColor Green
        }
        'M' {
          Write-Host "Merging files into existing directory..." -ForegroundColor Green
        }
        'C' {
          Write-Host "Operation cancelled by user." -ForegroundColor Red
          return
        }
      }
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

  Write-Host ("`n" + ("=" * 60)) -ForegroundColor Cyan
  Write-Host "EXTRACTION SUMMARY" -ForegroundColor Cyan
  Write-Host ("=" * 60) -ForegroundColor Cyan
  Write-Host "Source Directory: $SourcePath" -ForegroundColor White
  Write-Host "Destination Directory: $DestinationPath" -ForegroundColor White
  Write-Host "File Extensions: $($Extensions -join ', ')" -ForegroundColor White
  Write-Host ("=" * 60) -ForegroundColor Cyan
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