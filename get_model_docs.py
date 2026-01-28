#!/usr/bin/env python3
"""
Retrieve model documentation on demand.

Usage:
    python get_model_docs.py              # Print to stdout
    python get_model_docs.py --json       # Output as JSON with metadata
    python get_model_docs.py --path       # Print file path only
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DOCS_DIR = Path(__file__).parent / "docs"
MODEL_DOC_PATH = DOCS_DIR / "MODEL_DOCUMENTATION.md"
DOD_PATH = DOCS_DIR / "MODULE_5_DEFINITION_OF_DONE.md"


def get_file_hash(path: Path) -> str:
    """Compute MD5 hash of file contents."""
    if not path.exists():
        return "file_not_found"
    content = path.read_bytes()
    return hashlib.md5(content).hexdigest()[:12]


def get_model_documentation(as_json: bool = False) -> str:
    """
    Retrieve model documentation.

    Args:
        as_json: If True, return JSON with metadata. Otherwise return raw markdown.

    Returns:
        Documentation string (markdown or JSON)
    """
    if not MODEL_DOC_PATH.exists():
        raise FileNotFoundError(f"Model documentation not found at {MODEL_DOC_PATH}")

    content = MODEL_DOC_PATH.read_text()

    if as_json:
        return json.dumps({
            "document": "MODEL_DOCUMENTATION",
            "version": "1.0.0",
            "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "file_path": str(MODEL_DOC_PATH),
            "content_hash": get_file_hash(MODEL_DOC_PATH),
            "related_documents": {
                "definition_of_done": str(DOD_PATH),
                "dod_hash": get_file_hash(DOD_PATH),
            },
            "content": content,
        }, indent=2)

    return content


def main():
    parser = argparse.ArgumentParser(description="Retrieve model documentation")
    parser.add_argument("--json", action="store_true", help="Output as JSON with metadata")
    parser.add_argument("--path", action="store_true", help="Print file path only")
    parser.add_argument("--hash", action="store_true", help="Print content hash only")
    args = parser.parse_args()

    if args.path:
        print(MODEL_DOC_PATH)
        return

    if args.hash:
        print(get_file_hash(MODEL_DOC_PATH))
        return

    print(get_model_documentation(as_json=args.json))


if __name__ == "__main__":
    main()
