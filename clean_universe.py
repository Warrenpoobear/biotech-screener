import json
import shutil
import datetime

print('Cleaning universe.json...')

# Load universe
with open('production_data/universe.json', 'r') as f:
    universe = json.load(f)

print(f'Original entries: {len(universe)}')

# Filter out garbage tickers
def is_valid_ticker(ticker):
    if not ticker:
        return False
    if len(ticker) > 10:  # Tickers shouldn't be longer than 10 chars
        return False
    if ticker == '-':  # Just a dash
        return False
    # Allow only alphanumeric + underscore/dash
    cleaned = ticker.replace('_', '').replace('-', '')
    if not cleaned.isalnum():
        return False
    return True

# Clean the data
clean_universe = [s for s in universe if is_valid_ticker(s.get('ticker', ''))]

print(f'Clean entries: {len(clean_universe)}')
print(f'Removed: {len(universe) - len(clean_universe)}')

# Show what was removed
removed_tickers = [s.get('ticker', '') for s in universe if not is_valid_ticker(s.get('ticker', ''))]
print(f'\nRemoved tickers:')
for t in removed_tickers:
    display = t[:80] + '...' if len(t) > 80 else t
    print(f'  - {repr(display)}')

# Backup original
backup_name = f'production_data/universe_backup_{datetime.datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.json'
shutil.copy('production_data/universe.json', backup_name)
print(f'\n✅ Backup saved: {backup_name}')

# Save cleaned version
with open('production_data/universe.json', 'w') as f:
    json.dump(clean_universe, f, indent=2)

print(f'✅ Cleaned universe saved to: production_data/universe.json')
print(f'\nYou can now re-run: python test_morningstar_universe.py')
