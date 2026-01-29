"""
CUSIP to Ticker Resolver for Wake Robin Biotech Alpha System

13F filings report holdings by CUSIP (Committee on Uniform Securities 
Identification Procedures), but we need tickers for display and cross-referencing.

This module provides:
1. Local cache for determinism (same CUSIP always returns same ticker)
2. OpenFIGI API integration for resolving unknown CUSIPs
3. Manual override table for edge cases

Point-in-time safety note:
- CUSIP→ticker mappings are generally stable
- But corporate actions (mergers, ticker changes) can break mappings
- The cache preserves the mapping as of when we first resolved it
- For backtesting, you'd want a historical mapping service

Usage:
    from wake_robin.providers.sec_13f.cusip_resolver import CUSIPResolver
    
    resolver = CUSIPResolver(cache_path='data/cusip_cache.json')
    ticker = resolver.resolve('594918104')  # Returns 'MSFT'
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
# KNOWN MAPPINGS (biotech-focused, manually curated)
# =============================================================================
# These are common biotech CUSIPs we'll encounter frequently.
# Add to this as you discover new mappings.

KNOWN_CUSIP_MAPPINGS = {
    
    # Elite manager overlap (8-char format)
    '76243J10': 'RYTM',    # Rhythm Pharmaceuticals (4 managers)
    '71331710': 'PEPG',    # Pepgen (4 managers)
    '98985Y10': 'ZYME',    # Zymeworks
    'N9006410': 'QURE',    # uniQure
    '05370A10': 'RNA',     # Avidity Biosciences
    '00370M10': 'ABVX',    # Abivax
    '50157510': 'KYMR',    # Kymera
    '15102K10': 'CELC',    # Celcuity
    '67070310': 'NUVL',    # Nuvalent
    '17175720': 'CDTX',    # Cidara
    '03940C10': 'ACLX',    # Arcellx
    '21217B10': 'CTNM',    # Contineum
    '98401F10': 'XNCR',    # Xencor
    '28036F10': 'EWTX',    # Edgewise
    '86366E10': 'GPCR',    # Structure Therapeutics
    '23284F10': 'CTMX',    # CytomX
    '82940110': 'SION',    # Sionna
    '04351P10': 'ASND',    # Ascendis
    '90240B10': 'TYRA',    # Tyra Biosciences
    '81734D10': 'SEPN',    # Septerna
    '05338F30': 'AVLO',    # Avalo
    '86889P11': 'SRZN',    # Surrozen
    '76155X10': 'RVMD',    # Revolution Medicines
    '55886810': 'MDGL',    # Madrigal
    '45766930': 'INSM',    # Insmed
    '22663K10': 'CRNX',    # Crinetics
    '55287L10': 'MBX',     # MBX Biosciences
    '50180M10': 'LBPH',    # Longboard Pharma
    '74366E10': 'PTGX',    # Protagonist
    '15230910': 'CNTA',    # Centessa

    # Full 9-char CUSIPs (some 13F filers use 9-char format)
    '04351P101': 'ASND',   # Ascendis Pharma
    '05338F306': 'AVLO',   # Avalo Therapeutics
    '22663K107': 'CRNX',   # Crinetics Pharmaceuticals
    '23284F105': 'CTMX',   # CytomX Therapeutics
    '457669307': 'INSM',   # Insmed
    '55287L101': 'MBX',    # MBX Biosciences
    '558868105': 'MDGL',   # Madrigal Pharmaceuticals
    '76155X100': 'RVMD',   # Revolution Medicines
    '81734D104': 'SEPN',   # Septerna
    '829401108': 'SION',   # Sionna Therapeutics
    '86889P117': 'SRZN',   # Surrozen
    '90240B106': 'TYRA',   # Tyra Biosciences
    'N90064101': 'QURE',   # uniQure
    '38341P102': 'GOSS',   # Gossamer Bio

    # AI/diagnostics biotech
    '88023B103': 'TEM',    # Tempus AI

    # Large-cap biotech
    '031162100': 'AMGN',   # Amgen
    '92532F100': 'VRTX',   # Vertex Pharmaceuticals
    '60770K107': 'MRNA',   # Moderna
    '09062X103': 'BIIB',   # Biogen
    '456132106': 'INCY',   # Incyte
    '69331C108': 'PCVX',   # Vaxcyte
    '00287Y109': 'ABBV',   # AbbVie
    '58933Y105': 'MRK',    # Merck
    '478160104': 'JNJ',    # Johnson & Johnson
    '717081103': 'PFE',    # Pfizer
    '91324P102': 'UNH',    # UnitedHealth
    '552953101': 'BMY',    # Bristol-Myers Squibb
    
    # Mid-cap biotech (commonly held by elite managers)
    '82489T104': 'SIGA',   # SIGA Technologies
    '871765106': 'SWTX',   # SpringWorks Therapeutics
    '00773T109': 'ADMA',   # ADMA Biologics
    '92556V106': 'VKTX',   # Viking Therapeutics
    '45826H109': 'INSM',   # Insmed
    '98956P102': 'ZNTL',   # Zentalis
    '29260X109': 'ENTA',   # Enanta Pharmaceuticals
    '68622V106': 'ORIC',   # ORIC Pharmaceuticals
    '75886F107': 'REGN',   # Regeneron
    '05329W102': 'AUTL',   # Autolus Therapeutics
    '032511107': 'ANAB',   # AnaptysBio
    '896878104': 'TXG',    # 10x Genomics
    
    # Cell/gene therapy
    '07400F101': 'BEAM',   # Beam Therapeutics
    '24703L102': 'BLUE',   # bluebird bio
    '17323P108': 'CRSP',   # CRISPR Therapeutics
    '33767L109': 'FATE',   # Fate Therapeutics
    '454140100': 'IMTX',   # Immatics
    '45826H109': 'INSM',   # Insmed
    '49327M109': 'KITE',   # Kite Pharma (acquired)
    
    # RNA therapeutics
    '043168106': 'ARWR',   # Arrowhead Pharmaceuticals
    '48666K109': 'KRTX',   # Karuna Therapeutics
    '00751Y106': 'AKTA',   # Akita
    
    # Common large-cap (for position sizing context)
    '594918104': 'MSFT',   # Microsoft
    '037833100': 'AAPL',   # Apple
    '02079K107': 'GOOG',   # Alphabet Class C
    '02079K305': 'GOOGL',  # Alphabet Class A
    '023135106': 'AMZN',   # Amazon
    '88160R101': 'TSLA',   # Tesla
    '67066G104': 'NVDA',   # NVIDIA
    '30303M102': 'META',   # Meta
}


# =============================================================================
# CUSIP RESOLVER CLASS
# =============================================================================

class CUSIPResolver:
    """
    Resolves CUSIP identifiers to stock tickers.
    
    Resolution order:
    1. Known mappings (hardcoded biotech CUSIPs)
    2. Local cache (persisted to disk)
    3. OpenFIGI API (if available and not cached)
    4. None (if unresolvable)
    """
    
    def __init__(
        self,
        cache_path: Optional[str] = None,
        use_openfigi: bool = True,
        openfigi_api_key: Optional[str] = None,
    ):
        """
        Initialize resolver.
        
        Args:
            cache_path: Path to JSON cache file. If None, uses in-memory only.
            use_openfigi: Whether to query OpenFIGI for unknown CUSIPs.
            openfigi_api_key: Optional API key for higher rate limits.
        """
        self.cache_path = Path(cache_path) if cache_path else None
        self.use_openfigi = use_openfigi and HAS_REQUESTS
        self.openfigi_api_key = openfigi_api_key
        
        # In-memory cache (populated from known mappings + disk cache)
        self._cache: dict[str, Optional[str]] = {}
        
        # Load known mappings
        self._cache.update(KNOWN_CUSIP_MAPPINGS)
        
        # Load disk cache
        if self.cache_path and self.cache_path.exists():
            self._load_cache()
        
        # Track stats
        self._stats = {
            'hits_known': 0,
            'hits_cache': 0,
            'hits_openfigi': 0,
            'misses': 0,
        }
    
    def _load_cache(self):
        """Load cache from disk."""
        try:
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
                # Only load the mappings, not metadata
                if 'mappings' in data:
                    self._cache.update(data['mappings'])
                else:
                    self._cache.update(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load CUSIP cache: {e}")
    
    def _save_cache(self):
        """Persist cache to disk."""
        if not self.cache_path:
            return
        
        # Ensure directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'version': '1.0',
            'updated_at': datetime.utcnow().isoformat(),
            'count': len(self._cache),
            'mappings': self._cache,
        }
        
        with open(self.cache_path, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
    
    def resolve(self, cusip: str) -> Optional[str]:
        """
        Resolve a CUSIP to its ticker symbol.
        
        Args:
            cusip: 9-character CUSIP identifier
            
        Returns:
            Ticker symbol or None if unresolvable
        """
        if not cusip:
            return None
        
        # Normalize: uppercase, strip whitespace
        cusip = cusip.strip().upper()
        
        # Check 9-char vs 6-char (some 13Fs use 6-char issuer ID)
        # We'll try both
        cusip_9 = cusip[:9] if len(cusip) >= 9 else cusip
        cusip_6 = cusip[:6]
        
        # 1. Check known mappings (exact 9-char)
        if cusip_9 in KNOWN_CUSIP_MAPPINGS:
            self._stats['hits_known'] += 1
            return KNOWN_CUSIP_MAPPINGS[cusip_9]
        
        # 2. Check cache
        if cusip_9 in self._cache:
            self._stats['hits_cache'] += 1
            return self._cache[cusip_9]
        
        # 3. Query OpenFIGI
        if self.use_openfigi:
            ticker = self._query_openfigi(cusip_9)
            if ticker:
                self._stats['hits_openfigi'] += 1
                self._cache[cusip_9] = ticker
                self._save_cache()
                return ticker
        
        # 4. Unresolvable
        self._stats['misses'] += 1
        # Cache the miss to avoid repeated lookups
        self._cache[cusip_9] = None
        self._save_cache()
        return None
    
    def _query_openfigi(self, cusip: str) -> Optional[str]:
        """
        Query OpenFIGI API for CUSIP resolution.
        
        Rate limit: 25 requests/minute without API key.
        """
        if not HAS_REQUESTS:
            return None
        
        url = 'https://api.openfigi.com/v3/mapping'
        
        headers = {'Content-Type': 'application/json'}
        if self.openfigi_api_key:
            headers['X-OPENFIGI-APIKEY'] = self.openfigi_api_key
        
        payload = [{'idType': 'ID_CUSIP', 'idValue': cusip}]
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 429:
                # Rate limited - wait and retry once
                time.sleep(2)
                response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            if data and len(data) > 0 and 'data' in data[0]:
                # Find US equity ticker
                for item in data[0]['data']:
                    if item.get('exchCode') in ('US', 'UN', 'UQ', 'UA', 'UW'):
                        return item.get('ticker')
                # Fallback: return first ticker found
                if data[0]['data']:
                    return data[0]['data'][0].get('ticker')
            
            return None
            
        except Exception as e:
            print(f"OpenFIGI query failed for {cusip}: {e}")
            return None
    
    def resolve_batch(self, cusips: list[str]) -> dict[str, Optional[str]]:
        """
        Resolve multiple CUSIPs efficiently.
        
        Uses batched OpenFIGI queries for unknowns.
        """
        results = {}
        unknowns = []
        
        for cusip in cusips:
            cusip_norm = cusip.strip().upper()[:9]
            
            # Check cache first
            if cusip_norm in self._cache:
                results[cusip] = self._cache[cusip_norm]
            else:
                unknowns.append(cusip_norm)
        
        # Batch query unknowns (OpenFIGI supports up to 100 per request)
        if unknowns and self.use_openfigi and HAS_REQUESTS:
            batch_results = self._query_openfigi_batch(unknowns)
            for cusip, ticker in batch_results.items():
                self._cache[cusip] = ticker
                results[cusip] = ticker
            self._save_cache()
        
        # Mark remaining unknowns as None
        for cusip in unknowns:
            if cusip not in results:
                results[cusip] = None
                self._cache[cusip] = None
        
        return results
    
    def _query_openfigi_batch(self, cusips: list[str]) -> dict[str, Optional[str]]:
        """Batch query OpenFIGI."""
        if not cusips:
            return {}
        
        results = {}
        
        # OpenFIGI allows up to 100 items per request
        for i in range(0, len(cusips), 100):
            batch = cusips[i:i+100]
            payload = [{'idType': 'ID_CUSIP', 'idValue': c} for c in batch]
            
            url = 'https://api.openfigi.com/v3/mapping'
            headers = {'Content-Type': 'application/json'}
            if self.openfigi_api_key:
                headers['X-OPENFIGI-APIKEY'] = self.openfigi_api_key
            
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    for j, item in enumerate(data):
                        cusip = batch[j]
                        if 'data' in item and item['data']:
                            # Prefer US exchange
                            ticker = None
                            for d in item['data']:
                                if d.get('exchCode') in ('US', 'UN', 'UQ', 'UA', 'UW'):
                                    ticker = d.get('ticker')
                                    break
                            if not ticker and item['data']:
                                ticker = item['data'][0].get('ticker')
                            results[cusip] = ticker
                        else:
                            results[cusip] = None
                
                # Rate limit courtesy
                time.sleep(0.5)
                
            except Exception as e:
                print(f"OpenFIGI batch query failed: {e}")
        
        return results
    
    def add_mapping(self, cusip: str, ticker: str):
        """Manually add a CUSIP→ticker mapping."""
        cusip = cusip.strip().upper()[:9]
        self._cache[cusip] = ticker
        self._save_cache()
    
    def get_stats(self) -> dict:
        """Return resolver statistics."""
        return {
            **self._stats,
            'cache_size': len(self._cache),
            'known_mappings': len(KNOWN_CUSIP_MAPPINGS),
        }


# =============================================================================
# DETERMINISTIC HASH FOR POINT-IN-TIME SAFETY
# =============================================================================

def cusip_mapping_hash(mappings: dict[str, str]) -> str:
    """
    Generate deterministic hash of CUSIP mappings.
    
    Use this to verify mapping consistency across runs.
    """
    # Sort for determinism
    sorted_items = sorted(mappings.items())
    content = json.dumps(sorted_items, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# CONVENIENCE SINGLETON
# =============================================================================

_default_resolver: Optional[CUSIPResolver] = None


def get_resolver(cache_path: str = 'data/cusip_cache.json') -> CUSIPResolver:
    """Get or create default resolver instance."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = CUSIPResolver(cache_path=cache_path)
    return _default_resolver


def resolve_cusip(cusip: str) -> Optional[str]:
    """Convenience function to resolve a single CUSIP."""
    return get_resolver().resolve(cusip)


if __name__ == '__main__':
    # Demo
    resolver = CUSIPResolver(cache_path=None)  # In-memory only for demo
    
    test_cusips = [
        '031162100',  # AMGN
        '92532F100',  # VRTX
        '594918104',  # MSFT
        '67066G104',  # NVDA
        '000000000',  # Unknown
    ]
    
    print("CUSIP Resolver Demo")
    print("=" * 50)
    
    for cusip in test_cusips:
        ticker = resolver.resolve(cusip)
        print(f"  {cusip} → {ticker or 'UNKNOWN'}")
    
    print()
    print("Stats:", resolver.get_stats())
