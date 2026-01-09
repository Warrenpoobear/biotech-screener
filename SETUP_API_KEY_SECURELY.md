# Secure API Key Setup
## Wake Robin Biotech Screening System

**CRITICAL:** Never commit API keys to git or hardcode in source files!

---

## Method 1: Environment Variable (Recommended)

### Windows (PowerShell)
```powershell
# Set for current session
$env:OPENFIGI_API_KEY = "1a242384-a922-4b68-92dc-c15474f79d2d"

# Set permanently (user-level)
[System.Environment]::SetEnvironmentVariable('OPENFIGI_API_KEY', '1a242384-a922-4b68-92dc-c15474f79d2d', 'User')

# Verify
echo $env:OPENFIGI_API_KEY
```

### Linux/Mac (Bash)
```bash
# Set for current session
export OPENFIGI_API_KEY="1a242384-a922-4b68-92dc-c15474f79d2d"

# Set permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export OPENFIGI_API_KEY="1a242384-a922-4b68-92dc-c15474f79d2d"' >> ~/.bashrc
source ~/.bashrc

# Verify
echo $OPENFIGI_API_KEY
```

---

## Method 2: Secure Config File (Alternative)

### Create .env file (NOT tracked by git)

```bash
# Create .env file in project root
cat > .env << 'EOF'
OPENFIGI_API_KEY=1a242384-a922-4b68-92dc-c15474f79d2d
EOF

# Set restrictive permissions (Linux/Mac)
chmod 600 .env
```

### Add to .gitignore
```bash
# Make sure .env is NOT committed to git
echo ".env" >> .gitignore
echo "production_data/openfigi_api_key.txt" >> .gitignore
```

### Load in Python scripts
```python
# Load from .env file
from pathlib import Path

def load_api_key() -> str:
    """Load API key from .env file or environment"""
    import os
    
    # Try environment variable first
    api_key = os.getenv('OPENFIGI_API_KEY')
    if api_key:
        return api_key
    
    # Try .env file
    env_file = Path('.env')
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith('OPENFIGI_API_KEY='):
                return line.split('=', 1)[1].strip()
    
    raise ValueError("OPENFIGI_API_KEY not found in environment or .env file")
```

---

## Method 3: Encrypted Config File (Most Secure)

For production systems, use encrypted secrets management.

### Using Python keyring (recommended)
```bash
# Install keyring
pip install keyring --break-system-packages

# Store key securely
python -c "import keyring; keyring.set_password('wake_robin', 'openfigi_api_key', '1a242384-a922-4b68-92dc-c15474f79d2d')"

# Retrieve in scripts
python -c "import keyring; print(keyring.get_password('wake_robin', 'openfigi_api_key'))"
```

---

## Updated Script Usage

### Option A: Pass via command line
```bash
python cusip_mapper.py query 037833100 \
    --data-dir production_data \
    --api-key $OPENFIGI_API_KEY
```

### Option B: Auto-load from environment
```bash
# Scripts now automatically check environment variables
python cusip_mapper.py query 037833100 \
    --data-dir production_data
# Will use OPENFIGI_API_KEY from environment
```

---

## Security Checklist

- [x] API key stored in environment variable or .env file
- [x] .env file added to .gitignore
- [x] .env file has restrictive permissions (600)
- [ ] Never commit .env or files containing keys to git
- [ ] Never share API keys in Slack, email, or documentation
- [ ] Rotate API key if accidentally exposed

---

## Testing Your Setup

```bash
# Test environment variable is set
python -c "import os; print('API key loaded:', bool(os.getenv('OPENFIGI_API_KEY')))"

# Test mapper can use it
python cusip_mapper.py query 037833100 --data-dir production_data

# Should see:
# Loading CUSIP mapper...
# Querying OpenFIGI for 037833100...
# 037833100 → AAPL
```

---

## If You Accidentally Commit Your Key

1. **Rotate immediately** at https://www.openfigi.com/api
2. **Remove from git history:**
```bash
# Remove from all commits (use git filter-branch or BFG Repo-Cleaner)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (WARNING: rewrites history)
git push origin --force --all
```

3. **Update .gitignore** to prevent future accidents

---

## API Key Security Best Practices

✅ **DO:**
- Store in environment variables
- Use .env files (gitignored)
- Use secret management systems (AWS Secrets Manager, Azure Key Vault)
- Rotate keys regularly
- Use different keys for dev/prod

❌ **DON'T:**
- Hardcode in source files
- Commit to git
- Share in chat/email
- Store in plain text files (tracked by git)
- Use same key across team members

---

**Your key is now secure!** ✅

Next: Continue with CUSIP mapper setup using environment variable.
