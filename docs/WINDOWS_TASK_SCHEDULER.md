# Windows Task Scheduler Setup Guide

## Overview

This guide shows how to automate the Sharadar SEP sync and backtest pipeline using Windows Task Scheduler.

**Recommended Schedule:**
- **Daily sync**: 6:00 AM (after market data is finalized)
- **Monthly backtest**: 1st of each month at 7:00 AM

---

## Prerequisites

### 1. Set API Key (System Environment Variable)

1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Click **Advanced** tab → **Environment Variables**
3. Under **System variables**, click **New**
4. Add:
   - Variable name: `NASDAQ_DATA_LINK_API_KEY`
   - Variable value: `your_api_key_here`
5. Click OK to save

### 2. Verify Python Path

Open Command Prompt and run:
```cmd
python --version
```

If Python isn't found, add it to your PATH or update `PYTHON_PATH` in the batch files.

### 3. Test Scripts Manually First

```cmd
cd C:\path\to\biotech_screener

REM Test sync (with API key set)
scripts\run_sharadar_sync.bat --full-refresh

REM Test backtest
scripts\run_backtest.bat
```

---

## Task 1: Daily Sharadar Sync

### Create the Task

1. Press `Win + R`, type `taskschd.msc`, press Enter
2. Click **Create Task** (not Basic Task)

### General Tab
- **Name**: `Sharadar SEP Daily Sync`
- **Description**: `Sync Sharadar SEP prices from Nasdaq Data Link`
- **Security options**: 
  - Select "Run whether user is logged on or not"
  - Check "Run with highest privileges"

### Triggers Tab
1. Click **New**
2. **Begin the task**: On a schedule
3. **Settings**: Daily
4. **Start**: Today at `6:00:00 AM`
5. **Recur every**: 1 day
6. Check "Enabled"
7. Click OK

### Actions Tab
1. Click **New**
2. **Action**: Start a program
3. **Program/script**: `C:\path\to\biotech_screener\scripts\run_sharadar_sync.bat`
4. **Start in**: `C:\path\to\biotech_screener`
5. Click OK

### Conditions Tab
- Uncheck "Start the task only if the computer is on AC power" (optional)

### Settings Tab
- Check "Allow task to be run on demand"
- Check "Run task as soon as possible after a scheduled start is missed"
- Check "If the task fails, restart every": 1 hour, up to 3 times
- "Stop the task if it runs longer than": 2 hours

### Save
Click OK, enter your Windows password when prompted.

---

## Task 2: Monthly Backtest

### Create the Task

1. In Task Scheduler, click **Create Task**

### General Tab
- **Name**: `Biotech Screener Monthly Backtest`
- **Description**: `Run monthly backtest with latest Sharadar data`
- **Security options**: Same as above

### Triggers Tab
1. Click **New**
2. **Begin the task**: On a schedule
3. **Settings**: Monthly
4. **Start**: 1st of next month at `7:00:00 AM`
5. **Months**: Select all months
6. **Days**: 1
7. Click OK

### Actions Tab
1. Click **New**
2. **Program/script**: `C:\path\to\biotech_screener\scripts\run_backtest.bat`
3. **Start in**: `C:\path\to\biotech_screener`
4. Click OK

### Settings Tab
- Same as daily sync task

---

## Alternative: Single Pipeline Task (Sync + Backtest)

If you prefer one task that does everything:

### Actions Tab
- **Program/script**: `C:\path\to\biotech_screener\scripts\run_full_pipeline.bat`
- **Start in**: `C:\path\to\biotech_screener`

Schedule monthly on the 1st at 6:00 AM.

---

## Verify Tasks

### Check Task Status

1. In Task Scheduler, find your task
2. Right-click → **Run** to test
3. Check **Last Run Result**: `0x0` = success

### Check Logs

```cmd
cd C:\path\to\biotech_screener
type logs\pipeline_*.log
```

### Check Outputs

```cmd
dir data\sharadar_sep.csv
dir output\runs\sharadar_*
```

---

## Troubleshooting

### Task doesn't run

1. Check **History** tab in Task Scheduler
2. Verify the batch file path is correct
3. Ensure "Run whether user is logged on or not" is set
4. Check Windows Event Viewer for errors

### API errors

1. Verify `NASDAQ_DATA_LINK_API_KEY` is set as a **System** variable (not User)
2. Restart Task Scheduler after setting environment variables
3. Test manually: `echo %NASDAQ_DATA_LINK_API_KEY%`

### Python not found

1. Update `PYTHON_PATH` in batch files to full path:
   ```batch
   set PYTHON_PATH=C:\Python310\python.exe
   ```

### Permission errors

1. Run Task Scheduler as Administrator
2. Ensure batch file has execute permissions
3. Check the "Start in" directory is correct

---

## Quick Reference

| Task | Schedule | Batch File |
|------|----------|------------|
| Daily Sync | 6:00 AM daily | `scripts\run_sharadar_sync.bat` |
| Weekly Full Refresh | Sunday 5:00 AM | `scripts\run_sharadar_sync.bat --full-refresh` |
| Monthly Backtest | 1st @ 7:00 AM | `scripts\run_backtest.bat` |
| Full Pipeline | 1st @ 6:00 AM | `scripts\run_full_pipeline.bat` |

---

## Recommended Production Policy

### Sync Schedule
- **Daily incremental**: 6:00 AM (small delta, fast)
- **Weekly full refresh**: Sunday 5:00 AM (catch any corrections, eliminate drift)

### Task 3: Weekly Full Refresh (Recommended)

Create a third task for weekly full history refresh:

1. In Task Scheduler, click **Create Task**
2. **Name**: `Sharadar SEP Weekly Full Refresh`
3. **Trigger**: Weekly, Sunday at 5:00 AM
4. **Action**: 
   - Program: `C:\path\to\biotech_screener\scripts\run_sharadar_sync.bat`
   - Arguments: `--full-refresh`
   - Start in: `C:\path\to\biotech_screener`

This ensures any corrections from Nasdaq Data Link are captured.

---

| File | Purpose |
|------|---------|
| `data\sharadar_sep.csv` | Synced price data |
| `data\curated\sharadar\state.json` | Sync state |
| `output\runs\sharadar_*\` | Backtest results |
| `logs\pipeline_*.log` | Run logs |

---

## XML Export (Optional)

To export task for backup or sharing:

1. Right-click task → **Export**
2. Save as `sharadar_sync_task.xml`

To import on another machine:

1. Task Scheduler → **Import Task**
2. Select the XML file
3. Update paths as needed
