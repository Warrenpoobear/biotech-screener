#!/usr/bin/env python3
"""
collect_ctgov_data.py - Collect Clinical Trials Data from ClinicalTrials.gov

Fetches all clinical trial data for tickers in universe.

Usage:
    python collect_ctgov_data.py --output production_data/trial_records.json
"""

import json
import requests
import time
from pathlib import Path
from datetime import date
from typing import List, Dict
import argparse


def get_trials_for_ticker(ticker: str, max_retries: int = 3, max_results: int = 1000) -> List[Dict]:
    """Fetch clinical trials for a ticker from ClinicalTrials.gov API v2 with pagination"""

    base_url = "https://clinicaltrials.gov/api/v2/studies"

    all_trials = []
    next_page_token = None

    while len(all_trials) < max_results:
        params = {
            "query.term": ticker,
            "format": "json",
            "pageSize": 100,  # API max per page
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        success = False
        for attempt in range(max_retries):
            try:
                response = requests.get(base_url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    studies = data.get('studies', [])

                    if not studies:
                        return all_trials[:max_results]

                    for study in studies:
                        protocol = study.get('protocolSection', {})
                        id_module = protocol.get('identificationModule', {})
                        status_module = protocol.get('statusModule', {})
                        design_module = protocol.get('designModule', {})
                        sponsor_module = protocol.get('sponsorCollaboratorsModule', {})
                        conditions_module = protocol.get('conditionsModule', {})
                        arms_module = protocol.get('armsInterventionsModule', {})

                        trial = {
                            "ticker": ticker,
                            "nct_id": id_module.get('nctId'),
                            "title": id_module.get('briefTitle'),
                            "status": status_module.get('overallStatus'),
                            "phase": design_module.get('phases', ['N/A'])[0] if design_module.get('phases') else 'N/A',
                            "study_type": design_module.get('studyType'),
                            "conditions": conditions_module.get('conditions', []),
                            "interventions": [i.get('name') for i in arms_module.get('interventions', [])],
                            "primary_completion_date": status_module.get('primaryCompletionDateStruct', {}).get('date'),
                            "completion_date": status_module.get('completionDateStruct', {}).get('date'),
                            "results_first_posted": status_module.get('resultsFirstPostDateStruct', {}).get('date'),
                            "last_update_posted": status_module.get('lastUpdatePostDateStruct', {}).get('date'),
                            "enrollment": status_module.get('enrollmentInfo', {}).get('count'),
                            "sponsor": sponsor_module.get('leadSponsor', {}).get('name'),
                            "collected_at": date.today().isoformat()
                        }

                        all_trials.append(trial)

                    # Check for next page
                    next_page_token = data.get('nextPageToken')
                    success = True
                    break

                elif response.status_code == 429:
                    wait_time = 5 * (attempt + 1)
                    time.sleep(wait_time)
                    continue
                else:
                    return all_trials

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return all_trials

        if not success or not next_page_token:
            break

        # Rate limit between pages
        time.sleep(0.2)

    return all_trials[:max_results]


def collect_all_trials(universe_file: Path, output_file: Path):
    """Collect trials for all tickers"""
    
    print("="*80)
    print("CLINICAL TRIALS DATA COLLECTION (ClinicalTrials.gov)")
    print("="*80)
    print(f"Date: {date.today()}")
    
    # Load universe
    with open(universe_file) as f:
        universe = json.load(f)
    
    tickers = [s['ticker'] for s in universe if s.get('ticker') and s['ticker'] != '_XBI_BENCHMARK_']
    
    print(f"\nUniverse: {len(tickers)} tickers")
    print(f"Output: {output_file}")
    print(f"Estimated time: {len(tickers) * 0.6 / 60:.1f} minutes")
    
    # Collect
    all_trials = []
    stats = {'total': len(tickers), 'with_trials': 0, 'total_trials': 0}
    
    print(f"\n{'='*80}")
    print("COLLECTING TRIALS")
    print(f"{'='*80}\n")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:3d}/{len(tickers)}] {ticker:6s}", end=" ", flush=True)
        
        trials = get_trials_for_ticker(ticker)
        
        if trials:
            all_trials.extend(trials)
            stats['with_trials'] += 1
            stats['total_trials'] += len(trials)
            print(f"✅ {len(trials):2d} trials")
        else:
            print("   No trials")
        
        time.sleep(0.5)  # Be nice to API
        
        if i % 50 == 0:
            print(f"\n  Progress: {i}/{len(tickers)} ({i/len(tickers)*100:.1f}%)")
            print(f"  Trials found: {stats['total_trials']}\n")
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_trials, f, indent=2)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total tickers: {stats['total']}")
    print(f"Tickers with trials: {stats['with_trials']}")
    print(f"Total trials: {stats['total_trials']}")
    print(f"Avg trials/ticker: {stats['total_trials'] / stats['total']:.1f}")
    print(f"Coverage: {stats['with_trials'] / stats['total'] * 100:.1f}%")
    print(f"✅ Saved to: {output_file}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Collect clinical trials from ClinicalTrials.gov")
    parser.add_argument('--universe', type=Path, default=Path('production_data/universe.json'))
    parser.add_argument('--output', type=Path, default=Path('production_data/trial_records.json'))
    args = parser.parse_args()
    
    if not args.universe.exists():
        print(f"❌ Universe file not found: {args.universe}")
        return 1
    
    try:
        collect_all_trials(args.universe, args.output)
        return 0
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
