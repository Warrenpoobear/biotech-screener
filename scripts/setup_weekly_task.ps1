# Wake Robin Biotech Screener - Windows Task Scheduler Setup
# Run this script as Administrator to set up weekly screening automation

# Configuration
$TaskName = "WakeRobin_WeeklyScreen"
$TaskDescription = "Run Wake Robin Biotech Screener every Tuesday at 9:00 AM"
$ProjectPath = "C:\Projects\biotech_screener\biotech-screener"
$PythonPath = "python"  # Update if using specific Python installation

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

Write-Host "=" * 60
Write-Host "Wake Robin Biotech Screener - Task Setup"
Write-Host "=" * 60

# Check if task already exists
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "Task '$TaskName' already exists." -ForegroundColor Yellow
    $response = Read-Host "Do you want to replace it? (y/n)"
    if ($response -eq 'y') {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Existing task removed."
    } else {
        Write-Host "Setup cancelled."
        exit 0
    }
}

# Create the action
$scriptPath = Join-Path $ProjectPath "scripts\weekly_screen.py"
$action = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument "`"$scriptPath`"" `
    -WorkingDirectory $ProjectPath

# Create the trigger (Every Tuesday at 9:00 AM)
$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Tuesday `
    -At 9:00AM

# Create task settings
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -MultipleInstances IgnoreNew

# Create the principal (run whether user is logged in or not)
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

# Register the task
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Description $TaskDescription `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal

    Write-Host ""
    Write-Host "SUCCESS: Task '$TaskName' created!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Schedule:"
    Write-Host "  Day:  Tuesday"
    Write-Host "  Time: 9:00 AM"
    Write-Host ""
    Write-Host "To view task: Open Task Scheduler -> Task Scheduler Library"
    Write-Host "To run manually: Right-click task -> Run"
    Write-Host "To disable: Right-click task -> Disable"
    Write-Host ""
} catch {
    Write-Host "ERROR: Failed to create task" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

# Verify task was created
$task = Get-ScheduledTask -TaskName $TaskName
if ($task) {
    Write-Host "Task verified successfully!" -ForegroundColor Green

    # Show task details
    Write-Host ""
    Write-Host "Task Details:"
    Write-Host "  Name:   $($task.TaskName)"
    Write-Host "  State:  $($task.State)"
    Write-Host "  Path:   $($task.TaskPath)"
}

Write-Host ""
Write-Host "=" * 60
Write-Host "Setup Complete"
Write-Host "=" * 60
