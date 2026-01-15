import json

# Load current registry
registry = json.load(open('production_data/manager_registry.json'))

# CIK corrections (old -> new)
corrections = {
    '0001102854': '0001009258',  # Deerfield Management
    '0001555280': '0001822947',  # Ally Bridge Group
    '0001496147': '0001425738',  # Redmile Group
    '0001107208': '0001055951',  # OrbiMed Advisors
    '0001655352': '0001776382',  # Venbio Partners
    '0001067983': '0001703031',  # Bain Capital Life Sciences
    '0001713407': '0001493215',  # RTW Investments
    '0001556308': '0001232621',  # Tang Capital Partners
    '0001105977': '0000909661',  # Farallon Capital
}

# Update CIKs in elite_core
updated = 0
for manager in registry.get('elite_core', []):
    old_cik = manager.get('cik', '')
    if old_cik in corrections:
        new_cik = corrections[old_cik]
        print(f"Updating {manager['name']}: {old_cik} -> {new_cik}")
        manager['cik'] = new_cik
        updated += 1

# Save updated registry
with open('production_data/manager_registry.json', 'w') as f:
    json.dump(registry, f, indent=2)

print(f"\nUpdated {updated} manager CIKs")

# Show all managers
print("\nAll Managers in Registry:")
print("="*70)
for m in registry.get('elite_core', []):
    print(f"{m['name']:35} CIK: {m['cik']}")
