#!/usr/bin/env python3
"""
Setup dated outputs for weekly tracking.

Creates dated copies of all output files and archives them.
"""

import json
import shutil
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
import sys


def get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


def get_file_hash(filepath):
    """Calculate SHA256 hash of file."""
    if not filepath.exists():
        return None
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    # Get date from args or use today
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
        calc_date = datetime.strptime(date_str, '%Y-%m-%d')
    else:
        calc_date = datetime.now()

    date_suffix = calc_date.strftime('%Y%m%d')
    date_display = calc_date.strftime('%Y-%m-%d')

    print("=" * 70)
    print(f"SETTING UP DATED OUTPUTS FOR {date_display}")
    print("=" * 70)

    # Create archive directory
    archive_dir = Path('archives') / calc_date.strftime('%Y') / calc_date.strftime('%m')
    archive_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nArchive directory: {archive_dir}")

    # Files to version (source -> dated target)
    files_to_version = {
        'outputs/momentum_signals.json': f'outputs/momentum_signals_{date_suffix}.json',
        'outputs/ranked_full_308.json': f'outputs/ranked_{date_suffix}.json',
        'outputs/ranked_with_momentum.json': f'outputs/ranked_{date_suffix}_with_momentum.json',
        'outputs/sweet_spot_tickers.csv': f'outputs/sweet_spot_{date_suffix}.csv',
        'outputs/momentum_rankings_full.json': f'outputs/momentum_rankings_{date_suffix}.json',
    }

    manifest = {
        'run_date': date_display,
        'created_at': datetime.now().isoformat(),
        'git_commit': get_git_commit(),
        'files': {}
    }

    print("\nCreating dated outputs:")
    print("-" * 70)

    for source, target in files_to_version.items():
        source_path = Path(source)
        target_path = Path(target)

        if source_path.exists():
            # Calculate hash
            file_hash = get_file_hash(source_path)

            # Copy to dated version
            shutil.copy2(source_path, target_path)
            print(f"✅ {target_path.name}")

            # Also copy to archive
            archive_path = archive_dir / target_path.name
            shutil.copy2(source_path, archive_path)
            print(f"   → Archived: archives/{calc_date.strftime('%Y/%m')}/{target_path.name}")

            # Add to manifest
            manifest['files'][target_path.name] = {
                'source': source,
                'hash': file_hash,
                'size': source_path.stat().st_size,
                'archived': str(archive_path)
            }
        else:
            print(f"⚠️  Missing: {source}")
            manifest['files'][target] = {'error': 'source file missing'}

    # Load regime info if available
    momentum_file = Path('outputs/momentum_signals.json')
    if momentum_file.exists():
        with open(momentum_file, 'r') as f:
            mom_data = json.load(f)
        manifest['regime'] = mom_data.get('metadata', {}).get('regime', 'unknown')
        manifest['xbi_return_90d'] = mom_data.get('metadata', {}).get('xbi_return_90d', 0)
        manifest['tickers_processed'] = len(mom_data.get('signals', {}))

    # Save manifest
    manifest_path = Path(f'outputs/run_manifest_{date_suffix}.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✅ Manifest: {manifest_path.name}")

    # Also archive manifest
    shutil.copy2(manifest_path, archive_dir / manifest_path.name)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Date: {date_display}")
    print(f"Git commit: {manifest['git_commit']}")
    print(f"Files versioned: {len([f for f in manifest['files'].values() if 'error' not in f])}")
    print(f"Archive: {archive_dir}")

    if 'regime' in manifest:
        print(f"Regime: {manifest['regime']}")
        print(f"Tickers: {manifest['tickers_processed']}")

    print("\n✅ Dated outputs setup complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
