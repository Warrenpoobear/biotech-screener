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
    '713317105': 'PEPG',   # Pepgen (9-char)
    '98985Y108': 'ZYME',   # Zymeworks (9-char)
    '152309100': 'CNTA',   # Centessa (9-char)
    '21217B100': 'CTNM',   # Contineum (9-char)
    '03940C100': 'ACLX',   # Arcellx (9-char)
    '15102K100': 'CELC',   # Celcuity (9-char)
    '50157510': 'KYMR',    # Kymera (9-char)
    '501575104': 'KYMR',   # Kymera (9-char)
    '00370M103': 'ABVX',   # Abivax (9-char)
    '670703107': 'NUVL',   # Nuvalent (9-char)
    '171757206': 'CDTX',   # Cidara (9-char)
    '98401F105': 'XNCR',   # Xencor (9-char)
    '28036F105': 'EWTX',   # Edgewise (9-char)
    '86366E106': 'GPCR',   # Structure Therapeutics (9-char)
    '76243J105': 'RYTM',   # Rhythm Pharmaceuticals (9-char)
    '74366E102': 'PTGX',   # Protagonist (9-char)
    '50180M108': 'LBRX',   # Liberum/Longboard (9-char)

    # International biotech (non-US CUSIPs)
    'G01767105': 'ALKS',   # Alkermes (Irish)
    'H0036K147': 'ADCT',   # ADC Therapeutics (Swiss)
    'M8694L137': 'SLGL',   # Sol-Gel Technologies (Israeli)

    # Additional biotech CUSIPs from Opaleye holdings
    '89532M101': 'TRVI',   # Trevi Therapeutics
    '12674W109': 'CABA',   # Cabaletta Bio
    '022307102': 'ALMS',   # Alumis
    '00773J202': 'SYRE',   # Spyre Therapeutics
    '76655970': 'RIGL',    # Rigel Pharmaceuticals (8-char)
    '766559702': 'RIGL',   # Rigel Pharmaceuticals (9-char)
    '74365A309': 'PLX',    # Protalix BioTherapeutics
    '89455T109': 'TMCI',   # Treace Medical Concepts
    '67577R102': 'OPUS',   # Opus Genetics
    '451033708': 'IBIO',   # iBio
    '00534B100': 'ADGI',   # Adagio Medical
    '415858109': 'HROW',   # Harrow Health
    '29772L108': 'ETON',   # Eton Pharmaceuticals
    '53635D202': 'LQDA',   # Liquidia Corporation
    '45257U108': 'IMNM',   # Immunome
    '80303D305': 'SNWV',   # Sanuwave Health
    '192005106': 'CDXS',   # Codexis
    '88322Q108': 'TGTX',   # TG Therapeutics
    '67576A100': 'OCUL',   # Ocular Therapeutix
    '156944100': 'CGON',   # CG Oncology
    '86150R107': 'STOK',   # Stoke Therapeutics
    '03770N101': 'APGE',   # Apogee Therapeutics
    '254604101': 'IRON',   # Disc Medicine
    '803607100': 'SRPT',   # Sarepta Therapeutics
    '517125100': 'LRMR',   # Larimar Therapeutics
    '98419J206': 'XOMA',   # XOMA Corporation
    '76200L309': 'RZLT',   # Rezolute
    '04335A105': 'ARVN',   # Arvinas
    '48115J109': 'DERM',   # Journey Medical

    # Cormorant holdings CUSIPs
    '107924102': 'BIOR',   # BridgeBio Oncology (was Eidos)
    '30233G209': 'EYPT',   # EyePoint Pharmaceuticals
    '75383L102': 'RAPP',   # Rapport Therapeutics
    '10806X102': 'BBIO',   # BridgeBio Pharma
    '10919W405': 'DRUG',   # Bright Minds Biosciences
    'N62509109': 'NAMS',   # NewAmsterdam Pharma
    '74006W207': 'PRAX',   # Praxis Precision Medicines
    'N69605108': 'PHVS',   # Pharvaris
    '21833P301': 'CRBP',   # Corbus Pharmaceuticals
    'G3855L106': 'GHRS',   # GH Research
    '500946108': 'KRBP',   # Korro Bio (now Voyager)
    '59267L107': 'MTSR',   # Metsera
    '47103J105': 'JANX',   # Janux Therapeutics
    '27414W102': 'EARS',   # Auris Medical (placeholder)
    '30231G208': 'EYPT',   # EyePoint alt CUSIP
    '10919W108': 'DRUG',   # Bright Minds alt
    '74006W108': 'PRAX',   # Praxis alt

    # Rock Springs holdings CUSIPs
    '04016X101': 'ARGX',   # argenx SE
    '532457108': 'LLY',    # Eli Lilly
    'N5749R100': 'MRUS',   # Merus
    '604749101': 'MIRM',   # Mirum Pharmaceuticals
    '89422G107': 'TVTX',   # Travere Therapeutics
    '450056106': 'IRTC',   # iRhythm Technologies
    '00847X104': 'AGIO',   # Agios Pharmaceuticals
    '90400D108': 'RARE',   # Ultragenyx
    '46120E602': 'ISRG',   # Intuitive Surgical
    '101137107': 'BSX',    # Boston Scientific
    '98420N105': 'XENE',   # Xenon Pharmaceuticals
    '02043Q107': 'ALNY',   # Alnylam
    '30063P105': 'EXAS',   # Exact Sciences
    '70975L107': 'PEN',    # Penumbra
    '00973Y108': 'AKRO',   # Akero Therapeutics
    '72815L107': 'PLRX',   # Pliant Therapeutics
    '29275Y102': 'ENTA',   # Enanta (alt CUSIP)
    '464287622': 'IONS',   # Ionis Pharmaceuticals
    '191098102': 'KOD',    # Kodiak Sciences
    '46619B108': 'JNPR',   # Juniper (placeholder)
    '74587V107': 'QTTB',   # Q32 Bio
    '86881A200': 'SUPN',   # Supernus Pharmaceuticals
    '37954A104': 'GILD',   # Gilead Sciences
    '88160R101': 'TSLA',   # Tesla (already have)
    '004644107': 'ACHC',   # Acadia Healthcare
    '902973304': 'UTHR',   # United Therapeutics
    '87612E106': 'TARS',   # Tarsus Pharmaceuticals
    '04621X108': 'ASTH',   # Astellas/Astria
    '29260X109': 'ENTA',   # Enanta (primary)
    '92926F101': 'WVE',    # Wave Life Sciences
    '92537Y105': 'VERA',   # Vera Therapeutics
    '46266C105': 'IRWD',   # Ironwood Pharmaceuticals
    '81211K100': 'SEER',   # Seer Bio
    '98310A106': 'XERS',   # Xeris Biopharma
    '882681109': 'TNDM',   # Tandem Diabetes
    '26441C204': 'DVAX',   # Dynavax Technologies
    '88642R109': 'THRX',   # Theseus Pharmaceuticals
    '55279C100': 'MGNX',   # MacroGenics
    '45826H109': 'INSM',   # Insmed (already have)
    '92556V106': 'VKTX',   # Viking (already have)

    # EcoR1 Capital holdings CUSIPs
    '032724106': 'ANAB',   # AnaptysBio (alt CUSIP)
    'G50871105': 'JAZZ',   # Jazz Pharmaceuticals
    'H17182108': 'CRSP',   # CRISPR Therapeutics (Swiss)
    '87583X109': 'TNGX',   # Tango Therapeutics
    '68622P109': 'ORIC',   # ORIC Pharmaceuticals (alt)
    '03753U106': 'APLS',   # Apellis Pharmaceuticals
    '36315X101': 'GLPG',   # Galapagos
    '64135M105': 'NGNE',   # Neurogene
    '03843E104': 'AQST',   # Aquestive Therapeutics
    '81533L109': 'SEEL',   # Seelos Therapeutics
    '82489N103': 'SIAB',   # SI-BONE
    '45338J108': 'IVA',    # Inventiva
    '74587V305': 'QTTB',   # Q32 Bio (alt)
    '46266C204': 'IRWD',   # Ironwood (alt)
    '06406J109': 'KURA',   # Kura Oncology
    '58471A109': 'MCRB',   # Seres Therapeutics
    '09857L108': 'BPMC',   # Blueprint Medicines
    '56400P203': 'MASI',   # Masimo
    '35137L105': 'FOLD',   # Amicus Therapeutics
    '60770K108': 'MRNA',   # Moderna (alt)
    '02154V103': 'ALTM',   # Altimmune
    '04621X108': 'ATXS',   # Astria Therapeutics
    '92537N108': 'VCEL',   # Vericel
    '92763M105': 'VRTX',   # Vertex (alt)
    '40435L108': 'HIMS',   # Hims & Hers Health
    '50189K103': 'LGND',   # Ligand Pharmaceuticals
    '88688T100': 'TMDX',   # TransMedics Group
    '74587V107': 'PTCT',   # PTC Therapeutics
    '893641100': 'TSHA',   # Taysha Gene Therapies
    '02208R107': 'ALPN',   # Alpine Immune Sciences

    # Sofinnova & additional manager CUSIPs
    '632307104': 'NTRA',   # Natera
    '375558103': 'GILD',   # Gilead Sciences
    '92337R101': 'VERA',   # Vera Therapeutics
    '91307C102': 'UTHR',   # United Therapeutics
    '046353108': 'AZN',    # AstraZeneca
    '252131107': 'DXCM',   # Dexcom
    '05464T104': 'AXSM',   # Axsome Therapeutics
    'G76279101': 'ROIV',   # Roivant Sciences
    '04272N102': 'AVBP',   # ArriVent BioPharma
    '37045V100': 'GERN',   # Geron Corporation
    '31573A105': 'FGEN',   # FibroGen
    '742718109': 'PRCT',   # Procept BioRobotics
    '825690100': 'SI',     # Silvergate (placeholder)
    '74587W107': 'PTCT',   # PTC Therapeutics (alt)
    '82481R106': 'SHPG',   # Shire (now Takeda)
    '460690100': 'IONS',   # Ionis (alt)
    '86882H100': 'SURF',   # Surface Oncology
    '370437100': 'GERN',   # Geron (alt)
    '11135F101': 'BMY',    # Bristol-Myers (alt)
    '78397Y103': 'RXDX',   # Prometheus Bio
    '71376T109': 'PEPG',   # Pepgen (alt)
    '86267D101': 'SRRK',   # Scholar Rock
    '20016X104': 'CMLF',   # CM Life Sciences (SPAC)
    '05377R102': 'RNA',    # Avidity (alt)
    'N80584105': 'TALK',   # Talkspace
    '82488Y107': 'SHEN',   # Shenandoah Tel
    '13123X508': 'CALM',   # Cal-Maine Foods
    '45780R101': 'INSP',   # Inspire Medical
    '09061G101': 'BIIB',   # Biogen (alt)
    '84611G104': 'SPXC',   # SPX Corp
    '872589104': 'TW',     # Tradeweb Markets
    '00912X302': 'AIMD',   # Ainos
    '42824C109': 'HEWG',   # iShares Currency Hedged
    '12572Q105': 'CDMO',   # Avid Bioservices
    '92763M107': 'VRTX',   # Vertex (another alt)
    '458140100': 'INTC',   # Intel
    '655044105': 'NKE',    # Nike
    '78463V107': 'SPY',    # SPDR S&P 500
    '464287655': 'IWM',    # iShares Russell 2000

    # Perceptive Advisors holdings CUSIPs
    '03237H101': 'AMLX',   # Amylyx Pharmaceuticals
    'G59665102': 'MGTX',   # MeiraGTx Holdings
    '228903100': 'AORT',   # Artivion
    'N44445109': 'IMTX',   # Immatics N.V.
    '83422E204': 'SLDB',   # Solid Biosciences
    '64125C109': 'NBIX',   # Neurocrine Biosciences
    '925050106': 'VRNA',   # Verona Pharma
    '31573A105': 'FGEN',   # FibroGen
    '74587V207': 'QURE',   # uniQure (alt)
    '89214P109': 'TSVT',   # 2seventy bio
    '74587V305': 'PTCT',   # PTC Therapeutics (alt2)
    '29414B104': 'EOLS',   # Evolus
    '86882H100': 'SURF',   # Surface Oncology
    '53635D103': 'LQDA',   # Liquidia (alt)
    '60783X104': 'MLAB',   # Mesa Labs
    '03073E105': 'AMRX',   # Amneal Pharmaceuticals
    '98419J107': 'XOMA',   # XOMA (alt)
    'G5876H105': 'MDXH',   # MDxHealth
    '46266C303': 'IRWD',   # Ironwood (alt2)
    '29275Y201': 'ENTA',   # Enanta (alt2)
    '90384S303': 'ULTA',   # Ulta Beauty
    '92763M305': 'VRTX',   # Vertex (alt3)
    '69007J106': 'OTIC',   # Otonomy
    '74158J100': 'PRGO',   # Perrigo
    '86882H209': 'SURF',   # Surface Oncology (alt)
    '04621X207': 'ATXS',   # Astria (alt)
    '44107P104': 'HZNP',   # Horizon Therapeutics
    '00508Y102': 'ACAD',   # Acadia Pharmaceuticals
    '00287Y208': 'ABBV',   # AbbVie (alt)
    '92532F209': 'VRTX',   # Vertex (alt4)
    '744849104': 'PSNL',   # Personalis
    '30063P204': 'EXAS',   # Exact Sciences (alt)
    '74340E103': 'PRTA',   # Prothena
    '05379R107': 'AXGN',   # AxoGen
    '00771V108': 'ADPT',   # Adaptive Biotech
    '87612E205': 'TARS',   # Tarsus (alt)
    '30233G308': 'EYPT',   # EyePoint (alt)
    '004421403': 'AAON',   # AAON
    '73754Y100': 'POWI',   # Power Integrations
    '74587V404': 'PTCT',   # PTC (alt3)
    '57060D108': 'MASI',   # Masimo (alt)

    # Additional unresolved CUSIPs from Perceptive
    '142152107': 'CRIS',   # Caris Life Sciences
    '45258J102': 'IMVT',   # Immunovant
    '29286M105': 'ENGN',   # EnGene Holdings
    '221015100': 'CRVS',   # Corvus Pharmaceuticals
    '45719W205': 'IKT',    # Inhibikase Therapeutics
    '26818M108': 'DYN',    # Dyne Therapeutics
    '08659B102': 'BXIA',   # Beta Bionics (placeholder)
    'G2545C104': 'CRBN',   # Crescent Biopharma
    '42238D107': 'HFLO',   # HeartFlow (placeholder)
    '29415J106': 'NVNO',   # enVVeno Medical
    '09075X207': 'BDSX',   # Biodesix
    '293602504': 'ENSC',   # Ensysce Biosciences
    '45719W106': 'IKT',    # Inhibikase (alt)
    '45256X103': 'IMVT',   # Immunovant (alt)
    '29286M204': 'ENGN',   # EnGene (alt)
    '26818M207': 'DYN',    # Dyne (alt)

    # RA Capital holdings CUSIPs
    '92243G108': 'PCVX',   # Vaxcyte
    '282559103': 'ETNB',   # 89bio
    '603170101': 'MLYS',   # Mineralys Therapeutics
    '52635N103': 'LENZ',   # Lenz Therapeutics
    'Y95308105': 'WVE',    # Wave Life Sciences (Singapore)
    '252828108': 'DNTH',   # Dianthus Therapeutics
    '055477103': 'BCAX',   # Bicara Therapeutics
    '82835W108': 'SPRY',   # ARS Pharmaceuticals
    '37954A203': 'GILD',   # Gilead (alt)
    '460690100': 'IONS',   # Ionis (alt2)
    '05377R201': 'RNA',    # Avidity (alt2)
    '002824100': 'ABT',    # Abbott Labs
    '00724F101': 'ADBE',   # Adobe
    '03073E204': 'AMRX',   # Amneal (alt)
    '29260X208': 'ENTA',   # Enanta (alt3)
    '98420N204': 'XENE',   # Xenon (alt)
    '282559202': 'ETNB',   # 89bio (alt)
    '46266C402': 'IRWD',   # Ironwood (alt3)
    '74587V503': 'PTCT',   # PTC (alt4)
    '92243G207': 'PCVX',   # Vaxcyte (alt)

    # Baker Bros holdings CUSIPs
    '07725L102': 'BEONE',  # BeOne Medicines
    '45337C102': 'INCY',   # Incyte Corporation
    '004225108': 'ACAD',   # Acadia Pharmaceuticals
    '86627T108': 'SMMT',   # Summit Therapeutics
    '50015M109': 'KOD',    # Kodiak Sciences
    '00288U106': 'ABCL',   # AbCellera Biologics
    'G52694109': 'KNSA',   # Kiniksa Pharmaceuticals
    '384747101': 'GRAL',   # GRAIL
    '45166A102': 'IDYA',   # IDEAYA Biosciences
    '87901J105': 'TELA',   # TELA Bio
    '92536R108': 'VRCA',   # Verrica Pharmaceuticals
    '86881A109': 'SURF',   # Surface Oncology (alt2)
    '69007J205': 'OTIC',   # Otonomy (alt)
    '98420N303': 'XENE',   # Xenon (alt2)
    '72815L206': 'PLRX',   # Pliant (alt)
    '05379R206': 'AXGN',   # AxoGen (alt)
    '64125C208': 'NBIX',   # Neurocrine (alt)
    '86882H308': 'SURF',   # Surface (alt3)
    '00508Y201': 'ACAD',   # Acadia (alt)
    '45166A201': 'IDYA',   # IDEAYA (alt)
    '50015M208': 'KOD',    # Kodiak (alt)
    '86627T207': 'SMMT',   # Summit (alt)

    # Additional elite manager biotech CUSIPs
    '15102K100': 'CELC',   # Celcuity (confirmed in universe check)
    'N5749R109': 'MRUS',   # Merus (alt)
    '00973Y207': 'AKRO',   # Akero (alt)
    '92556V205': 'VKTX',   # Viking (alt)
    '05464T203': 'AXSM',   # Axsome (alt)
    '04272N201': 'AVBP',   # ArriVent (alt)
    '37045V209': 'GERN',   # Geron (alt)
    '74587V602': 'PTCT',   # PTC (alt5)
    '86267D200': 'SRRK',   # Scholar Rock (alt)

    # Confirmed CUSIP mappings from Baker/RA/Perceptive analysis
    '45258D105': 'IMCR',   # Immunocore Holdings
    '088786108': 'BCYC',   # Bicycle Therapeutics
    '04280A100': 'ARWR',   # Arrowhead Pharmaceuticals (alt)
    '24823R105': 'DNLI',   # Denali Therapeutics
    '45826J105': 'NTLA',   # Intellia Therapeutics
    '92790C104': 'VRDN',   # Viridian Therapeutics
    '462222100': 'IONS',   # Ionis Pharmaceuticals (alt3)
    '61225M102': 'GLUE',   # Monte Rosa Therapeutics
    '09075V102': 'BNTX',   # BioNTech
    '67080M103': 'NRIX',   # Nurix Therapeutics
    '29384C108': 'TRDA',   # Entrada Therapeutics
    '15117B202': 'CLDX',   # Celldex Therapeutics
    '157085101': 'CERS',   # Cerus Corporation
    '359616109': 'FULC',   # Fulcrum Therapeutics
    '28658R106': 'CLYM',   # Climb Bio
    '61023L207': 'MNPR',   # Monopar Therapeutics
    '35104E100': 'FDMT',   # 4D Molecular Therapeutics
    '04317A107': 'ARTV',   # Artiva Biotherapeutics
    '00509G209': 'ABOS',   # Acumen Pharmaceuticals
    '65487U108': 'NKTX',   # Nkarta
    '03969F109': 'RCUS',   # Arcus Biosciences
    '95075A107': 'HOWL',   # Werewolf Therapeutics
    '28617K101': 'ELDN',   # Eledon Pharmaceuticals
    '03152W109': 'FOLD',   # Amicus Therapeutics
    '04635X102': 'ATXS',   # Astria Therapeutics (confirmed)
    '45720N103': 'INBX',   # Inhibrx Biosciences
    '01438T106': 'ALDX',   # Aldeyra Therapeutics
    '697947109': 'PVLA',   # Palvella Therapeutics
    '687604108': 'ORKA',   # Oruka Therapeutics
    '63909J108': 'NAUT',   # Nautilus Biotechnology

    # Acuta/Krensavage/Palo Alto/RTW/Cormorant unresolved CUSIPs
    '69366J200': 'PTCT',   # PTC Therapeutics (alt6)
    '87650L103': 'TARS',   # Tarsus Pharmaceuticals (alt2)
    'M96088105': 'URGN',   # UroGen Pharma (Israeli)
    '98887Q104': 'ZLAB',   # Zai Lab
    '77313F106': 'RCKT',   # Rocket Pharmaceuticals
    '462260100': 'IOVA',   # Iovance Biotherapeutics
    '03969K108': 'ARQT',   # Arcutis Biotherapeutics
    '61559X104': 'MLTX',   # Moonlake Immunotherapeutics
    'G72800108': 'PRTA',   # Prothena (Irish)
    '09077V100': 'BIOA',   # BioAge Labs
    '09077A106': 'BMEA',   # Biomea Fusion
    '29337E102': 'ELVN',   # Enliven Therapeutics
    '67080N101': 'NUVB',   # Nuvation Bio
    '00461U105': 'ACRS',   # Aclaris Therapeutics
    '88032L605': 'TENX',   # Tenax Therapeutics
    'H5870P102': 'OCS',    # Oculis Holding (Swiss)
    '00289Y206': 'ABEO',   # Abeona Therapeutics
    'N71542109': 'PRQR',   # ProQR Therapeutics (Dutch)
    'G1110E107': 'BHVN',   # Biohaven (Irish)
    '00972G207': 'AKTX',   # Akari Therapeutics
    '14167L103': 'CDNA',   # CareDx (diagnostics)
    'G6674U108': 'NVCR',   # Novocure (oncology)

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
    '28106W103': 'EDIT',   # Editas Medicine
    '92539P101': 'VERV',   # Verve Therapeutics

    # Biotech screener 20-ticker universe additions
    'N2451R105': 'CVAC',   # CureVac (Netherlands)
    '45257L108': 'IMMP',   # Immutep (Australian ADR)
    '09061G101': 'BMRN',   # BioMarin Pharmaceutical
    '30161Q104': 'EXEL',   # Exelixis
    '40637H109': 'HALO',   # Halozyme Therapeutics

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
