# ============================================================================
# Sharadar SEP Daily Sync - PowerShell Script for Windows Task Scheduler
# ============================================================================
#
# SETUP:
#   1. Edit $BiotechScreenerPath below to your actual install path
#   2. Set your API key (choose one method):
#      - Environment variable: [Environment]::SetEnvironmentVariable("NASDAQ_DATA_LINK_API_KEY", "your_key", "User")
#      - Or create file: data\config\nasdaq_api_key.txt
#   3. Schedule in Task Scheduler (see SCHEDULING section)
#
# SCHEDULING (Windows Task Scheduler):
#   1. Open Task Scheduler (taskschd.msc)
#   2. Create Basic Task > Name: "Sharadar SEP Sync"
#   3. Trigger: Daily at 6:00 AM
#   4. Action: Start a program
#      - Program: powershell.exe
#      - Arguments: -ExecutionPolicy Bypass -File "C:\biotech_screener\scripts\sync_sharadar.ps1"
#      - Start in: C:\biotech_screener
#   5. In Properties > Settings:
#      - Check "Run task as soon as possible after scheduled start is missed"
#      - Check "If task fails, restart every: 1 hour" (up to 3 times)
#   6. Finish
#
# ============================================================================

# === CONFIGURATION (EDIT THIS) ===
$BiotechScreenerPath = "C:\biotech_screener"

# === LOG FILE ===
$LogDir = Join-Path $BiotechScreenerPath "logs"
$LogFile = Join-Path $LogDir "sync_$(Get-Date -Format 'yyyy-MM-dd').log"

# === FUNCTIONS ===
function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] $Message"
    Write-Host $LogMessage
    Add-Content -Path $LogFile -Value $LogMessage -ErrorAction SilentlyContinue
}

function Test-PythonAvailable {
    try {
        $null = python --version 2>&1
        return $true
    } catch {
        return $false
    }
}

# === MAIN ===
try {
    # Create log directory
    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }

    Write-Log "============================================================"
    Write-Log "Sharadar SEP Sync Starting"
    Write-Log "============================================================"

    # Change to project directory
    if (-not (Test-Path $BiotechScreenerPath)) {
        Write-Log "ERROR: Directory not found: $BiotechScreenerPath"
        exit 1
    }
    Set-Location $BiotechScreenerPath
    Write-Log "Working directory: $BiotechScreenerPath"

    # Check Python
    if (-not (Test-PythonAvailable)) {
        Write-Log "ERROR: Python not found in PATH"
        exit 1
    }
    $PythonVersion = python --version 2>&1
    Write-Log "Python: $PythonVersion"

    # Check API key
    $ApiKeyEnv = $env:NASDAQ_DATA_LINK_API_KEY
    $ApiKeyFile = Join-Path $BiotechScreenerPath "data\config\nasdaq_api_key.txt"
    
    if ($ApiKeyEnv) {
        Write-Log "API key: Environment variable set"
    } elseif (Test-Path $ApiKeyFile) {
        Write-Log "API key: File exists at $ApiKeyFile"
    } else {
        Write-Log "WARNING: No API key found - sync may fail"
    }

    # Run sync
    Write-Log "Running incremental sync..."
    $SyncOutput = python scripts/sync_sharadar_sep.py 2>&1
    $ExitCode = $LASTEXITCODE

    # Log output
    foreach ($line in $SyncOutput) {
        Write-Log "  $line"
    }

    # Check result
    if ($ExitCode -ne 0) {
        Write-Log "ERROR: Sync failed with exit code $ExitCode"
        exit $ExitCode
    }

    Write-Log "============================================================"
    Write-Log "Sync completed successfully"
    Write-Log "============================================================"

    # Check curated file
    $CuratedFile = Join-Path $BiotechScreenerPath "data\sharadar_sep.csv"
    if (Test-Path $CuratedFile) {
        $FileInfo = Get-Item $CuratedFile
        Write-Log "Curated file: $CuratedFile"
        Write-Log "  Size: $([math]::Round($FileInfo.Length / 1MB, 2)) MB"
        Write-Log "  Modified: $($FileInfo.LastWriteTime)"
    }

    exit 0

} catch {
    Write-Log "ERROR: Unhandled exception: $_"
    exit 1
}
