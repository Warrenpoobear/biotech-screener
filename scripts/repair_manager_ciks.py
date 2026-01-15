import argparse, json, re, shutil, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus
import urllib.request
import gzip

# ---- CONFIG ----
USER_AGENT = "Wake Robin Capital / biotech_screener (darren@wakerobincapital.com)"
REQUEST_DELAY_SECONDS = 0.20

def _sleep():
    time.sleep(REQUEST_DELAY_SECONDS)

def fetch_url(url: str, timeout: int = 20) -> Optional[bytes]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json,text/html,*/*",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "close",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            enc = (resp.headers.get("Content-Encoding") or "").lower()
            if enc == "gzip":
                raw = gzip.decompress(raw)
            return raw
    except Exception:
        return None
    finally:
        _sleep()

def pad10(cik: str) -> str:
    digits = re.sub(r"\D", "", str(cik))
    if not digits:
        return ""
    return digits.zfill(10)

def submissions_json(cik10: str) -> Optional[dict]:
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    raw = fetch_url(url)
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None

def has_13f(sub: dict) -> bool:
    recent = sub.get("filings", {}).get("recent", {})
    forms = recent.get("form", []) or []
    for f in forms:
        if isinstance(f, str) and f.startswith("13F-HR"):
            return True
    return False

def best_name_match_score(expected: str, actual: str) -> int:
    exp = re.sub(r"[^a-z0-9 ]+", " ", expected.lower()).split()
    act = re.sub(r"[^a-z0-9 ]+", " ", actual.lower()).split()
    exp_set, act_set = set(exp), set(act)
    if not exp_set or not act_set:
        return 0
    overlap = len(exp_set & act_set)
    return overlap * 10

@dataclass
class FixSuggestion:
    manager_name: str
    old_cik: str
    status: str
    new_cik: Optional[str]
    evidence: str

def analyze_manager(name: str, cik: str) -> FixSuggestion:
    cik10 = pad10(cik)
    if not cik10:
        return FixSuggestion(name, cik, "MISSING", None, "CIK not parseable")

    sub = submissions_json(cik10)
    if not sub:
        return FixSuggestion(name, cik10, "MISSING", None, "CIK returns 404")

    entity = (sub.get("name") or "").strip()
    match_score = best_name_match_score(name, entity)
    thirteenf = has_13f(sub)

    if not thirteenf:
        return FixSuggestion(name, cik10, "NO_13F", None, f"entity='{entity}' but no 13F-HR")

    if match_score < 15:
        return FixSuggestion(name, cik10, "MISMATCH", None, f"entity='{entity}', match_score={match_score}")
    
    return FixSuggestion(name, cik10, "OK", None, f"entity='{entity}', match_score={match_score}, has_13f=True")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    args = ap.parse_args()

    reg_path = Path(args.registry)
    reg = json.loads(reg_path.read_text(encoding="utf-8"))

    managers = reg.get("elite_core", [])
    suggestions: List[FixSuggestion] = []
    
    print("Checking manager CIKs...")
    for m in managers:
        name = m.get("name","").strip()
        cik = m.get("cik","")
        print(f"  Checking {name}...")
        suggestions.append(analyze_manager(name, cik))

    print("\n" + "="*80)
    print("MANAGER CIK HEALTH CHECK")
    print("="*80)
    ok = [s for s in suggestions if s.status == "OK"]
    bad = [s for s in suggestions if s.status != "OK"]
    print(f"OK: {len(ok)}   Needs Review: {len(bad)}")
    print()

    if ok:
        print("✅ WORKING MANAGERS:")
        for s in ok:
            print(f"  {s.manager_name:35} CIK {s.old_cik}")
        print()

    if bad:
        print("⚠️  NEEDS REVIEW:")
        for s in bad:
            print(f"  {s.manager_name:35} CIK {s.old_cik}  status={s.status}")
            print(f"     Evidence: {s.evidence}")
        print()

if __name__ == "__main__":
    main()
