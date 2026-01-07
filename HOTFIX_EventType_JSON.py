"""
HOTFIX: EventType JSON Serialization in run_screen.py

The error is: "Object of type EventType is not JSON serializable"

This happens because Module 3A catalyst events contain EventType enums 
that need to be converted to strings for JSON output.

FIX: Update the CustomJSONEncoder class in run_screen.py
"""

# ============================================================================
# FIND THIS CODE in run_screen.py (around line 120-130):
# ============================================================================

class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal and date types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)

# ============================================================================
# REPLACE WITH THIS:
# ============================================================================

class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal, date, and enum types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, date):
            return obj.isoformat()
        # Handle enums (like EventType from Module 3A)
        if hasattr(obj, '__class__') and hasattr(obj.__class__, '__name__'):
            if hasattr(obj, 'value'):
                # This is an enum - return its value
                return obj.value
            if hasattr(obj, 'name'):
                # Alternative: return its name
                return obj.name
        return super().default(obj)

# ============================================================================
# ALTERNATIVE SIMPLER FIX (if you want to handle all enums):
# ============================================================================

# Add this import at the top of run_screen.py:
from enum import Enum

# Then update CustomJSONEncoder:
class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal, date, and enum types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value  # or obj.name if you prefer
        return super().default(obj)

# ============================================================================
# EXPLANATION:
# ============================================================================

# The Module 3A catalyst events contain EventType enums like:
# EventType.STATUS_CHANGE, EventType.PHASE_CHANGE, etc.

# These need to be converted to strings for JSON serialization.
# The fix checks if an object is an Enum and returns its value/name as a string.

# After this fix, re-run:
# python run_screen.py --as-of-date 2026-01-07 --data-dir production_data --output screening_complete.json
