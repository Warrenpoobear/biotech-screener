#!/usr/bin/env python3
"""Fetch the scheduled alpha cohort table artifact from GitHub Actions.

Downloads the artifact produced by the ``rebuild-alpha-cohort-table``
workflow for a given date, extracts the table JSON, and writes it to
``--dest``.  Designed to run in CI before the production screen so that
the ``if_missing`` policy in ``_resolve_alpha_table_path`` finds the
pre-built table and skips the expensive in-process rebuild.

Exit codes:
    0  — artifact fetched and written successfully
    3  — artifact not found (non-fatal; caller should continue with fallback)
    2  — API / auth / IO error
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

# Exit codes
EXIT_OK = 0
EXIT_API_ERROR = 2
EXIT_NOT_FOUND = 3

DEFAULT_REPO = "Warrenpoobear/biotech-screener"
DEFAULT_WORKFLOW = "rebuild-alpha-cohort-table.yml"


def fetch_alpha_table(
    as_of_date: str,
    dest: Path,
    *,
    repo: str = DEFAULT_REPO,
    workflow: str = DEFAULT_WORKFLOW,
    token: str | None = None,
) -> int:
    """Download the alpha cohort table artifact and write to *dest*.

    Returns an exit code (0, 2, or 3).
    """
    if not token:
        print("ERROR: GITHUB_TOKEN not set", file=sys.stderr)
        return EXIT_API_ERROR

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    artifact_name = f"alpha-cohort-table-{as_of_date}"

    # Step 1: Find the artifact by name
    url = f"https://api.github.com/repos/{repo}/actions/artifacts"
    params = {"name": artifact_name, "per_page": 1}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
    except requests.RequestException as e:
        print(f"ERROR: API request failed: {e}", file=sys.stderr)
        return EXIT_API_ERROR

    if resp.status_code == 401:
        print("ERROR: GitHub token unauthorized (401)", file=sys.stderr)
        return EXIT_API_ERROR
    if resp.status_code == 403:
        print("ERROR: GitHub token forbidden (403)", file=sys.stderr)
        return EXIT_API_ERROR
    if resp.status_code != 200:
        print(
            f"ERROR: Artifact list failed (HTTP {resp.status_code}): {resp.text[:200]}",
            file=sys.stderr,
        )
        return EXIT_API_ERROR

    data = resp.json()
    artifacts = data.get("artifacts", [])
    if not artifacts:
        print(f"Artifact '{artifact_name}' not found", file=sys.stderr)
        return EXIT_NOT_FOUND

    # Use the first (most recent) matching artifact
    artifact = artifacts[0]
    if artifact.get("expired", False):
        print(f"Artifact '{artifact_name}' is expired", file=sys.stderr)
        return EXIT_NOT_FOUND

    download_url = artifact["archive_download_url"]
    print(f"Found artifact: id={artifact['id']} size={artifact.get('size_in_bytes', '?')}")

    # Step 2: Download the artifact zip
    try:
        dl_resp = requests.get(download_url, headers=headers, timeout=120)
    except requests.RequestException as e:
        print(f"ERROR: Download failed: {e}", file=sys.stderr)
        return EXIT_API_ERROR

    if dl_resp.status_code != 200:
        print(
            f"ERROR: Download failed (HTTP {dl_resp.status_code})",
            file=sys.stderr,
        )
        return EXIT_API_ERROR

    # Step 3: Extract the table JSON from the zip
    try:
        zf = zipfile.ZipFile(io.BytesIO(dl_resp.content))
    except zipfile.BadZipFile:
        print("ERROR: Downloaded artifact is not a valid zip", file=sys.stderr)
        return EXIT_API_ERROR

    # Find the v1_*.json file inside the zip
    json_names = [n for n in zf.namelist() if n.endswith(".json")]
    if not json_names:
        print("ERROR: No JSON file found in artifact zip", file=sys.stderr)
        return EXIT_API_ERROR

    table_name = json_names[0]
    table_bytes = zf.read(table_name)

    # Step 4: Atomic write to dest
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(dest.parent), suffix=".tmp",
        )
        os.write(fd, table_bytes)
        os.close(fd)
        os.replace(tmp_path, str(dest))
    except OSError as e:
        print(f"ERROR: Could not write to {dest}: {e}", file=sys.stderr)
        # Clean up temp file if it exists
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return EXIT_API_ERROR

    print(f"OK: wrote {dest} ({len(table_bytes)} bytes)")
    return EXIT_OK


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch alpha cohort table artifact from GitHub Actions",
    )
    parser.add_argument(
        "--as-of-date", required=True,
        help="Table date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--dest", required=True, type=Path,
        help="Destination path for the table JSON",
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO,
        help=f"GitHub repo (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--workflow", default=DEFAULT_WORKFLOW,
        help=f"Workflow file name (default: {DEFAULT_WORKFLOW})",
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    return fetch_alpha_table(
        as_of_date=args.as_of_date,
        dest=args.dest,
        repo=args.repo,
        workflow=args.workflow,
        token=token,
    )


if __name__ == "__main__":
    sys.exit(main())
