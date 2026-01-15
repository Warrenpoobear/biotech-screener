#!/usr/bin/env python3
"""
setup_environment.py

Secure setup script for Wake Robin Biotech Screening System.
Helps configure API keys and environment variables safely.

Author: Wake Robin Capital Management
Date: 2026-01-09
"""

import os
import sys
from pathlib import Path
import getpass

def check_gitignore():
    """Ensure sensitive files are in .gitignore"""
    gitignore_path = Path('.gitignore')
    
    sensitive_patterns = [
        '.env',
        'production_data/openfigi_api_key.txt',
        '*.pyc',
        '__pycache__/',
        '.DS_Store'
    ]
    
    if gitignore_path.exists():
        existing = gitignore_path.read_text()
        missing = [p for p in sensitive_patterns if p not in existing]
    else:
        missing = sensitive_patterns
    
    if missing:
        print("\n⚠️  Adding sensitive patterns to .gitignore...")
        with open(gitignore_path, 'a') as f:
            f.write('\n# Wake Robin - Sensitive files (auto-added)\n')
            for pattern in missing:
                f.write(f'{pattern}\n')
        print("✅ .gitignore updated")
    else:
        print("✅ .gitignore already configured")


def setup_env_file():
    """Create .env file with user's API key"""
    env_path = Path('.env')
    
    if env_path.exists():
        overwrite = input("\n.env file already exists. Overwrite? (y/N): ")
        if overwrite.lower() != 'y':
            print("Keeping existing .env file")
            return
    
    print("\n" + "="*60)
    print("OPENFIGI API KEY SETUP")
    print("="*60)
    print("\nYour API key will be stored securely in .env file")
    print("This file is gitignored and won't be committed")
    print("\nGet free API key at: https://www.openfigi.com/api")
    
    api_key = getpass.getpass("\nEnter your OpenFIGI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Skipping .env creation")
        return
    
    # Create .env file
    env_content = f"""# Wake Robin Biotech Screening - Environment Variables
# SECURITY: Never commit this file to git!

# OpenFIGI API Key
OPENFIGI_API_KEY={api_key}

# Optional: SEC EDGAR User Agent
SEC_USER_AGENT=Wake Robin Capital Management institutional.validation@wakerobincapital.com

# Data directory
DATA_DIR=production_data
"""
    
    env_path.write_text(env_content)
    
    # Set restrictive permissions (Linux/Mac)
    if os.name != 'nt':  # Not Windows
        os.chmod(env_path, 0o600)
        print(f"\n✅ .env file created with mode 600 (owner read/write only)")
    else:
        print(f"\n✅ .env file created")
    
    print(f"📁 Location: {env_path.absolute()}")


def verify_api_key():
    """Verify API key is accessible"""
    
    # Try loading from .env
    env_path = Path('.env')
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith('OPENFIGI_API_KEY='):
                api_key = line.split('=', 1)[1].strip()
                if api_key and api_key != 'your-api-key-here':
                    print(f"\n✅ API key loaded from .env: {api_key[:8]}...{api_key[-4:]}")
                    return True
    
    # Try environment variable
    api_key = os.getenv('OPENFIGI_API_KEY')
    if api_key:
        print(f"\n✅ API key loaded from environment: {api_key[:8]}...{api_key[-4:]}")
        return True
    
    print("\n❌ No API key found in .env or environment variables")
    return False


def create_data_directories():
    """Create necessary data directories"""
    dirs = [
        'production_data',
        'outputs'
    ]
    
    print("\n" + "="*60)
    print("DIRECTORY SETUP")
    print("="*60)
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            print(f"✅ Created: {dir_name}/")
        else:
            print(f"✅ Exists: {dir_name}/")


def initialize_cusip_mapper():
    """Initialize CUSIP mapper files"""
    print("\n" + "="*60)
    print("CUSIP MAPPER INITIALIZATION")
    print("="*60)
    
    data_dir = Path('production_data')
    
    static_map_path = data_dir / 'cusip_static_map.json'
    cache_path = data_dir / 'cusip_cache.json'
    
    # Create static map if doesn't exist
    if not static_map_path.exists():
        import json
        
        static_map = {
            "_README": {
                "description": "Static CUSIP→Ticker mapping for biotech universe",
                "instructions": "Populate with CUSIPs from SEC EDGAR or Yahoo Finance"
            }
        }
        
        with open(static_map_path, 'w') as f:
            json.dump(static_map, f, indent=2)
        
        print(f"✅ Created: {static_map_path}")
    else:
        print(f"✅ Exists: {static_map_path}")
    
    # Create empty cache
    if not cache_path.exists():
        with open(cache_path, 'w') as f:
            f.write('{}')
        
        print(f"✅ Created: {cache_path}")
    else:
        print(f"✅ Exists: {cache_path}")


def show_next_steps():
    """Show user what to do next"""
    print("\n" + "="*60)
    print("SETUP COMPLETE! 🎉")
    print("="*60)
    
    print("\nNext steps:")
    print("\n1. Populate static CUSIP map:")
    print("   python build_static_cusip_map.py generate-template \\")
    print("       --universe production_data/universe.json \\")
    print("       --output biotech_cusips_template.csv")
    
    print("\n2. Test CUSIP mapper:")
    print("   python cusip_mapper.py query 037833100 \\")
    print("       --data-dir production_data")
    
    print("\n3. Extract 13F holdings:")
    print("   python edgar_13f_extractor.py \\")
    print("       --quarter-end 2024-09-30 \\")
    print("       --manager-registry production_data/manager_registry.json \\")
    print("       --universe production_data/universe.json \\")
    print("       --cusip-map production_data/cusip_static_map.json \\")
    print("       --output production_data/holdings_snapshots.json")
    
    print("\n" + "="*60)


def main():
    """Run complete setup"""
    print("\n" + "="*60)
    print("WAKE ROBIN - INSTITUTIONAL VALIDATION SETUP")
    print("="*60)
    
    # 1. Check/update .gitignore
    check_gitignore()
    
    # 2. Create data directories
    create_data_directories()
    
    # 3. Setup .env file
    setup_env_file()
    
    # 4. Verify API key
    verify_api_key()
    
    # 5. Initialize CUSIP mapper
    initialize_cusip_mapper()
    
    # 6. Show next steps
    show_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
