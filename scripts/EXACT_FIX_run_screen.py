"""
EXACT FIX FOR run_screen.py - EventType JSON Serialization

Apply these changes to your run_screen.py file:
"""

# ============================================================================
# STEP 1: ADD THIS IMPORT AT THE TOP OF THE FILE
# ============================================================================

# Find the imports section at the top of run_screen.py (around line 1-20)
# Add this import with the other imports:

from enum import Enum

# ============================================================================
# STEP 2: UPDATE THE CustomJSONEncoder CLASS
# ============================================================================

# Find this class (around line 120-130 based on your error traceback):

# BEFORE (OLD CODE):
"""
class CustomJSONEncoder(json.JSONEncoder):
    '''JSON encoder that handles Decimal and date types'''
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)
"""

# AFTER (NEW CODE):
class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal, date, and enum types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Enum):  # ← ADD THIS LINE
            return obj.value        # ← ADD THIS LINE
        return super().default(obj)


# ============================================================================
# COMPLETE UPDATED IMPORTS SECTION EXAMPLE:
# ============================================================================

"""
import json
import sys
import argparse
from pathlib import Path
from datetime import date
from decimal import Decimal
from enum import Enum  # ← ADD THIS LINE

# ... rest of your imports ...
"""

# ============================================================================
# THAT'S IT! SAVE AND RE-RUN
# ============================================================================

"""
After making these changes:

1. Save run_screen.py
2. Run: python run_screen.py --as-of-date 2026-01-07 --data-dir production_data --output screening_complete.json
3. Should complete successfully now!
"""
