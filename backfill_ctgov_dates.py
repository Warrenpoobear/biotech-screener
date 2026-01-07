import json
import requests
import time
from pathlib import Path

def fetch_ctgov_dates(nct_id):
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        protocol = data.get("protocolSection", {})
        status_module = protocol.get("statusModule", {})
        results_section = data.get("resultsSection", {})
        dates = {
            "last_update_posted": status_module.get("lastUpdatePostDateStruct", {}).get("date"),
            "primary_completion_date": status_module.get("primaryCompletionDateStruct", {}).get("date"),
            "primary_completion_type": status_module.get("primaryCompletionDateStruct", {}).get("type"),
            "completion_date": status_module.get("completionDateStruct", {}).get("date"),
            "completion_type": status_module.get("completionDateStruct", {}).get("type"),
            "results_first_posted": results_section.get("resultsFirstPostDateStruct", {}).get("date"),
        }
        return dates
    except Exception as e:
        print(f"  Failed {nct_id}: {e}")
        return None

input_path = Path("production_data/trial_records.json")
output_path = Path("production_data/trial_records_with_dates.json")

with open(input_path) as f:
    records = json.load(f)

print(f"Processing {len(records)} trials...")
enhanced_records = []
success = 0

for i, record in enumerate(records, 1):
    nct_id = record.get('nct_id', 'UNKNOWN')
    ticker = record.get('ticker', 'UNKNOWN')
    print(f"[{i}/{len(records)}] {ticker} - {nct_id}...", end=' ')
    if record.get('last_update_posted'):
        print("Already has dates")
        enhanced_records.append(record)
        continue
    dates = fetch_ctgov_dates(nct_id)
    if dates:
        record.update(dates)
        if dates.get('last_update_posted'):
            print(f"Added dates (last_update: {dates['last_update_posted']})")
            success += 1
        else:
            print("Added dates but missing last_update_posted")
        enhanced_records.append(record)
    else:
        print("Failed")
        enhanced_records.append(record)
    if i < len(records):
        time.sleep(1.0)

with open(output_path, 'w') as f:
    json.dump(enhanced_records, f, indent=2)

print(f"\nDone! Successfully enhanced {success}/{len(records)} trials")
print(f"Output: {output_path}")
