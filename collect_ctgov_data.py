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


# =============================================================================
# SPONSOR NAME MAPPING
# =============================================================================
# Maps ticker symbols to official company names used in ClinicalTrials.gov
# sponsor fields. This avoids noise from generic ticker-term searches
# (e.g., "BEAM" matching "Ion Beam Applications" trials).

TICKER_TO_SPONSORS = {
    "AARD": ["Aardvark Therapeutics", "Aardvark Therapeutics, Inc."],
    "ABBV": ["AbbVie"],
    "ABCL": ["AbCellera Biologics", "AbCellera Biologics Inc."],
    "ABEO": ["Abeona Therapeutics", "Abeona Therapeutics Inc."],
    "ABOS": ["Acumen Pharmaceuticals", "Acumen Pharmaceuticals, Inc."],
    "ABUS": ["Arbutus Biopharma", "Arbutus Biopharma Corporation"],
    "ABVX": ["Abivax", "Abivax SA"],
    "ACAD": ["ACADIA Pharmaceuticals", "ACADIA Pharmaceuticals Inc."],
    "ACIU": ["AC Immune", "AC Immune SA"],
    "ACLX": ["Arcellx", "Arcellx, Inc."],
    "ACRS": ["Aclaris Therapeutics", "Aclaris Therapeutics, Inc."],
    "ACRV": ["Acrivon Therapeutics", "Acrivon Therapeutics, Inc."],
    "ADCT": ["ADC Therapeutics", "ADC Therapeutics SA"],
    "ADMA": ["ADMA Biologics", "ADMA Biologics Inc"],
    "AGIO": ["Agios Pharmaceuticals", "Agios Pharmaceuticals, Inc."],
    "AKRO": ["Akero Therapeutics"],
    "AKTX": ["Akari Therapeutics, Plc"],
    "ALDX": ["Aldeyra Therapeutics", "Aldeyra Therapeutics, Inc."],
    "ALGS": ["Aligos Therapeutics", "Aligos Therapeutics, Inc."],
    "ALKS": ["Alkermes, Inc.", "Alkermes"],
    "ALLO": ["Allogene Therapeutics", "Allogene Therapeutics, Inc."],
    "ALMS": ["Alumis", "Alumis Inc."],
    "ALNY": ["Alnylam Pharmaceuticals"],
    "ALT": ["Altimmune", "Altimmune, Inc."],
    "ALVO": ["Alvotech"],
    "AMGN": ["Amgen"],
    "AMLX": ["Amylyx Pharmaceuticals", "Amylyx Pharmaceuticals, Inc."],
    "AMPH": ["Amphastar Pharmaceuticals", "Amphastar Pharmaceuticals, Inc."],
    "AMRX": ["Amneal Pharmaceuticals", "Amneal Pharmaceuticals, Inc."],
    "ANAB": ["AnaptysBio", "AnaptysBio, Inc."],
    "ANIP": ["ANI Pharmaceuticals", "ANI Pharmaceuticals, Inc."],
    "ANNX": ["Annexon", "Annexon, Inc."],
    "ANRO": ["Alto Neuroscience", "Alto Neuroscience, Inc."],
    "APGE": ["Apogee Therapeutics"],
    "APLS": ["Apellis Pharmaceuticals"],
    "AQST": ["Aquestive Therapeutics", "Aquestive Therapeutics, Inc."],
    "ARCT": ["Arcturus Therapeutics Holdings", "Arcturus Therapeutics Holdings Inc."],
    "ARDX": ["Ardelyx", "Ardelyx, Inc."],
    "ARGX": ["argenx"],
    "ARQT": ["Arcutis Biotherapeutics"],
    "ARTV": ["Artiva Biotherapeutics", "Artiva Biotherapeutics, Inc."],
    "ARVN": ["Arvinas"],
    "ARWR": ["Arrowhead Pharmaceuticals", "Arrowhead Pharmaceuticals, Inc."],
    "ASMB": ["Assembly Biosciences", "Assembly Biosciences, Inc."],
    "ASND": ["Ascendis Pharma A/S", "Ascendis Pharma"],
    "ATAI": ["atai Life Sciences"],
    "ATXS": ["Astria Therapeutics", "Astria Therapeutics, Inc."],
    "AUPH": ["Aurinia Pharmaceuticals", "Aurinia Pharmaceuticals Inc"],
    "AURA": ["Aura Biosciences", "Aura Biosciences, Inc."],
    "AUTL": ["Autolus Therapeutics", "Autolus Therapeutics plc"],
    "AVBP": ["ArriVent BioPharma", "ArriVent BioPharma, Inc."],
    "AVIR": ["Atea Pharmaceuticals", "Atea Pharmaceuticals, Inc."],
    "AVXL": ["Anavex Life Sciences", "Anavex Life Sciences Corp."],
    "AXSM": ["Axsome Therapeutics"],
    "AZN": ["Astrazeneca", "Astrazeneca PLC"],
    "BBIO": ["BridgeBio Pharma", "BridgeBio"],
    "BBOT": ["BridgeBio Oncology Therapeutics"],
    "BCAX": ["Bicara Therapeutics", "Bicara Therapeutics Inc."],
    "BCRX": ["BioCryst Pharmaceuticals"],
    "BCYC": ["Bicycle Therapeutics", "Bicycle Therapeutics plc"],
    "BDTX": ["Black Diamond Therapeutics", "Black Diamond Therapeutics, Inc."],
    "BEAM": ["Beam Therapeutics"],
    "BHVN": ["Biohaven", "Biohaven Ltd."],
    "BIIB": ["Biogen"],
    "BIOA": ["BioAge Labs", "BioAge Labs, Inc."],
    "BLTE": ["Belite Bio", "Belite Bio, Inc"],
    "BLUE": ["bluebird bio"],
    "BMEA": ["Biomea Fusion", "Biomea Fusion, Inc."],
    "BMRN": ["BioMarin Pharmaceutical", "BioMarin"],
    "BNTX": ["BioNTech SE", "BioNTech"],
    "CABA": ["Cabaletta Bio", "Cabaletta Bio, Inc."],
    "CADL": ["Candel Therapeutics", "Candel Therapeutics, Inc."],
    "CATX": ["Perspective Therapeutics", "Perspective Therapeutics, Inc."],
    "CCCC": ["C4 Therapeutics", "C4 Therapeutics, Inc."],
    "CDTX": ["Cidara Therapeutics"],
    "CELC": ["Celcuity", "Celcuity Inc."],
    "CERS": ["Cerus", "Cerus Corporation"],
    "CGEM": ["Cullinan Therapeutics", "Cullinan Therapeutics, Inc."],
    "CGON": ["CG Oncology", "CG Oncology, Inc."],
    "CHRS": ["Coherus Oncology", "Coherus Oncology, Inc."],
    "CLDX": ["Celldex Therapeutics"],
    "CLLS": ["Cellectis S.A."],
    "CLYM": ["Climb Bio", "Climb Bio, Inc."],
    "CMPS": ["COMPASS Pathways"],
    "CNTA": ["Centessa Pharmaceuticals", "Centessa Pharmaceuticals plc"],
    "CNTX": ["Context Therapeutics"],
    "COGT": ["Cogent Biosciences", "Cogent Biosciences, Inc."],
    "COLL": ["Collegium Pharmaceutical", "Collegium Pharmaceutical, Inc."],
    "CRBU": ["Caribou Biosciences", "Caribou Biosciences, Inc."],
    "CRMD": ["CorMedix", "CorMedix Inc."],
    "CRNX": ["Crinetics Pharmaceuticals", "Crinetics Pharmaceuticals, Inc."],
    "CRSP": ["CRISPR Therapeutics"],
    "CRVS": ["Corvus Pharmaceuticals", "Corvus Pharmaceuticals, Inc."],
    "CTMX": ["CytomX Therapeutics", "CytomX Therapeutics, Inc."],
    "CTNM": ["Contineum Therapeutics", "Contineum Therapeutics, Inc."],
    "CVAC": ["CureVac"],
    "CYTK": ["Cytokinetics", "Cytokinetics, Incorporated"],
    "DAWN": ["Day One Biopharmaceuticals", "Day One Biopharmaceuticals, Inc."],
    "DNLI": ["Denali Therapeutics"],
    "DNTH": ["Dianthus Therapeutics", "Dianthus Therapeutics, Inc."],
    "DRMA": ["Dermata Therapeutics", "Dermata Therapeutics, Inc."],
    "DRUG": ["Bright Minds Biosciences", "Bright Minds Biosciences Inc."],
    "DSGN": ["Design Therapeutics", "Design Therapeutics, Inc."],
    "DVAX": ["Dynavax Technologies", "Dynavax Technologies Corporation"],
    "DYN": ["Dyne Therapeutics"],
    "EBS": ["Emergent BioSolutions", "Emergent BioSolutions Inc."],
    "EDIT": ["Editas Medicine"],
    "ELDN": ["Eledon Pharmaceuticals", "Eledon Pharmaceuticals, Inc."],
    "ELVN": ["Enliven Therapeutics", "Enliven Therapeutics, Inc."],
    "ENGN": ["enGene Holdings", "enGene Holdings Inc."],
    "ENTA": ["Enanta Pharmaceuticals", "Enanta Pharmaceuticals, Inc."],
    "EOLS": ["Evolus"],
    "EPRX": ["Eupraxia Pharmaceuticals", "Eupraxia Pharmaceuticals Inc."],
    "ERAS": ["Erasca", "Erasca, Inc."],
    "ESPR": ["Esperion Therapeutics", "Esperion Therapeutics, Inc."],
    "ETON": ["Eton Pharmaceuticals", "Eton Pharmaceuticals, Inc."],
    "EWTX": ["Edgewise Therapeutics"],
    "EXEL": ["Exelixis"],
    "EYPT": ["EyePoint Pharmaceuticals"],
    "FATE": ["Fate Therapeutics"],
    "FDMT": ["4D Molecular Therapeutics", "4D Molecular Therapeutics, Inc."],
    "FGEN": ["FibroGen", "FibroGen, Inc."],
    "FHTX": ["Foghorn Therapeutics", "Foghorn Therapeutics Inc."],
    "FOLD": ["Amicus Therapeutics"],
    "FULC": ["Fulcrum Therapeutics", "Fulcrum Therapeutics, Inc."],
    "GBIO": ["Generation Bio", "Generation Bio Co."],
    "GERN": ["Geron", "Geron Corporation"],
    "GHRS": ["GH Research", "GH Research PLC"],
    "GILD": ["Gilead Sciences"],
    "GLPG": ["Galapagos NV"],
    "GLUE": ["Monte Rosa Therapeutics", "Monte Rosa Therapeutics, Inc."],
    "GMAB": ["Genmab"],
    "GOSS": ["Gossamer Bio", "Gossamer Bio, Inc."],
    "GPCR": ["Structure Therapeutics", "Structure Therapeutics Inc."],
    "GRAL": ["GRAIL", "GRAIL, Inc."],
    "HALO": ["Halozyme Therapeutics"],
    "HOWL": ["Werewolf Therapeutics", "Werewolf Therapeutics, Inc."],
    "HRMY": ["Harmony Biosciences"],
    "HROW": ["Harrow", "Harrow, Inc."],
    "HRTX": ["Heron Therapeutics", "Heron Therapeutics, Inc."],
    "HUMA": ["Humacyte", "Humacyte, Inc."],
    "IBRX": ["ImmunityBio", "ImmunityBio, Inc."],
    "IDYA": ["IDEAYA Biosciences", "IDEAYA Biosciences, Inc."],
    "IKT": ["Inhibikase Therapeutics", "Inhibikase Therapeutics, Inc."],
    "IMCR": ["Immunocore"],
    "IMMP": ["Immutep"],
    "IMRX": ["Immuneering", "Immuneering Corporation"],
    "IMTX": ["Immatics", "Immatics N.V."],
    "IMVT": ["Immunovant", "Immunovant, Inc."],
    "INBX": ["Inhibrx Biosciences", "Inhibrx Biosciences, Inc."],
    "INCY": ["Incyte Corporation", "Incyte"],
    "INDV": ["Indivior", "Indivior PLC"],
    "INSM": ["Insmed Incorporated", "Insmed"],
    "IONS": ["Ionis Pharmaceuticals"],
    "IOVA": ["Iovance Biotherapeutics"],
    "IRON": ["Disc Medicine", "Disc Medicine, Inc."],
    "IRWD": ["Ironwood Pharmaceuticals", "Ironwood Pharmaceuticals, Inc."],
    "IVVD": ["Invivyd", "Invivyd, Inc."],
    "JANX": ["Janux Therapeutics"],
    "JAZZ": ["Jazz Pharmaceuticals"],
    "JBIO": ["Jade Biosciences", "Jade Biosciences, Inc."],
    "KALA": ["KALA BIO", "KALA BIO, Inc."],
    "KALV": ["KalVista Pharmaceuticals", "KalVista Pharmaceuticals, Inc."],
    "KMDA": ["Kamada", "Kamada Ltd."],
    "KNSA": ["Kiniksa Pharmaceuticals", "Kiniksa Pharmaceuticals, Ltd."],
    "KOD": ["Kodiak Sciences", "Kodiak Sciences Inc"],
    "KROS": ["Keros Therapeutics", "Keros Therapeutics, Inc."],
    "KRYS": ["Krystal Biotech"],
    "KURA": ["Kura Oncology", "Kura Oncology, Inc."],
    "KYMR": ["Kymera Therapeutics"],
    "KYTX": ["Kyverna Therapeutics", "Kyverna Therapeutics, Inc."],
    "LBRX": ["LB Pharmaceuticals", "LB Pharmaceuticals Inc"],
    "LCTX": ["Lineage Cell Therapeutics", "Lineage Cell Therapeutics, Inc."],
    "LEGN": ["Legend Biotech", "Legend Biotech Corporation"],
    "LENZ": ["LENZ Therapeutics", "LENZ Therapeutics, Inc."],
    "LFCR": ["Lifecore Biomedical", "Lifecore Biomedical, Inc."],
    "LQDA": ["Liquidia Technologies", "Liquidia"],
    "LRMR": ["Larimar Therapeutics", "Larimar Therapeutics, Inc."],
    "LXEO": ["Lexeo Therapeutics", "Lexeo Therapeutics, Inc."],
    "LYEL": ["Lyell Immunopharma", "Lyell Immunopharma, Inc."],
    "LYRA": ["Lyra Therapeutics", "Lyra Therapeutics, Inc."],
    "MAZE": ["Maze Therapeutics", "Maze Therapeutics, Inc."],
    "MBX": ["MBX Biosciences", "MBX Biosciences, Inc."],
    "MCRB": ["Seres Therapeutics", "Seres Therapeutics, Inc."],
    "MDGL": ["Madrigal Pharmaceuticals"],
    "MENS": ["Jyong Biotech", "Jyong Biotech Ltd."],
    "MESO": ["Mesoblast", "Mesoblast Limited"],
    "MGTX": ["MeiraGTx Holdings", "MeiraGTx Holdings plc"],
    "MIRM": ["Mirum Pharmaceuticals"],
    "MLTX": ["MoonLake Immunotherapeutics"],
    "MLYS": ["Mineralys Therapeutics", "Mineralys Therapeutics, Inc."],
    "MNKD": ["MannKind", "MannKind Corporation"],
    "MNMD": ["Mind Medicine (MindMed)", "Mind Medicine (MindMed) Inc."],
    "MRK": ["Merck Sharp & Dohme LLC", "Merck Sharp & Dohme"],
    "MRNA": ["ModernaTX, Inc.", "Moderna"],
    "MRSN": ["Mersana Therapeutics", "Mersana Therapeutics, Inc."],
    "MRUS": ["Merus", "Merus N.V."],
    "NAMS": ["NewAmsterdam Pharma Company", "NewAmsterdam Pharma"],
    "NBIX": ["Neurocrine Biosciences"],
    "NBP": ["NovaBridge Biosciences"],
    "NGNE": ["Neurogene"],
    "NKTX": ["Nkarta", "Nkarta, Inc."],
    "NMRA": ["Neumora Therapeutics", "Neumora Therapeutics, Inc."],
    "NRIX": ["Nurix Therapeutics", "Nurix Therapeutics, Inc."],
    "NTLA": ["Intellia Therapeutics"],
    "NUVB": ["Nuvation Bio", "Nuvation Bio Inc."],
    "NUVL": ["Nuvalent"],
    "NVAX": ["Novavax", "Novavax, Inc."],
    "NVCR": ["NovoCure", "NovoCure Limited"],
    "OBIO": ["Orchestra BioMed Holdings", "Orchestra BioMed Holdings, Inc."],
    "OCGN": ["Ocugen", "Ocugen, Inc."],
    "OCS": ["Oculis Holding", "Oculis Holding AG"],
    "OCUL": ["Ocular Therapeutix"],
    "OLMA": ["Olema Pharmaceuticals", "Olema Pharmaceuticals, Inc."],
    "OMER": ["Omeros", "Omeros Corporation"],
    "ONC": ["BeOne Medicines", "BeOne Medicines Ltd."],
    "ORIC": ["Oric Pharmaceuticals", "Oric Pharmaceuticals, Inc."],
    "ORKA": ["Oruka Therapeutics", "Oruka Therapeutics, Inc."],
    "PACB": ["Pacific Biosciences of California"],
    "PAHC": ["Phibro Animal Health Corporation"],
    "PBYI": ["Puma Biotechnology", "Puma Biotechnology Inc"],
    "PCRX": ["Pacira BioSciences", "Pacira BioSciences, Inc."],
    "PCVX": ["Vaxcyte"],
    "PEPG": ["PepGen"],
    "PFE": ["Pfizer"],
    "PGEN": ["Precigen", "Precigen, Inc."],
    "PHAT": ["Phathom Pharmaceuticals", "Phathom Pharmaceuticals, Inc."],
    "PHVS": ["Pharvaris", "Pharvaris N.V."],
    "PRAX": ["Praxis Precision Medicines", "Praxis Precision Medicine"],
    "PRLD": ["Prelude Therapeutics", "Prelude Therapeutics Incorporated"],
    "PRME": ["Prime Medicine", "Prime Medicine, Inc."],
    "PRQR": ["ProQR Therapeutics", "ProQR Therapeutics N.V."],
    "PRTA": ["Prothena Corporation", "Prothena"],
    "PTCT": ["PTC Therapeutics"],
    "PTGX": ["Protagonist Therapeutics", "Protagonist Therapeutics, Inc."],
    "PVLA": ["Palvella Therapeutics", "Palvella Therapeutics, Inc."],
    "PYXS": ["Pyxis Oncology", "Pyxis Oncology, Inc."],
    "QURE": ["uniQure"],
    "RANI": ["Rani Therapeutics Holdings", "Rani Therapeutics"],
    "RAPP": ["Rapport Therapeutics", "Rapport Therapeutics, Inc."],
    "RAPT": ["RAPT Therapeutics", "RAPT Therapeutics, Inc."],
    "RARE": ["Ultragenyx Pharmaceutical", "Ultragenyx"],
    "RCKT": ["Rocket Pharmaceuticals"],
    "RCUS": ["Arcus Biosciences"],
    "REGN": ["Regeneron Pharmaceuticals"],
    "REPL": ["Replimune"],
    "RGNX": ["REGENXBIO", "REGENXBIO Inc."],
    "RIGL": ["Rigel Pharmaceuticals", "Rigel Pharmaceuticals, Inc."],
    "RLAY": ["Relay Therapeutics", "Relay Therapeutics, Inc."],
    "RNA": ["Avidity Biosciences", "Avidity Biosciences, Inc."],
    "ROIV": ["Roivant Sciences"],
    "RVMD": ["Revolution Medicines"],
    "RYTM": ["Rhythm Pharmaceuticals"],
    "SABS": ["SAB Biotherapeutics", "SAB Biotherapeutics, Inc."],
    "SANA": ["Sana Biotechnology", "Sana Biotechnology, Inc."],
    "SEPN": ["Septerna", "Septerna, Inc."],
    "SERA": ["Sera Prognostics", "Sera Prognostics, Inc."],
    "SGMO": ["Sangamo Therapeutics", "Sangamo Therapeutics, Inc."],
    "SGMT": ["Sagimet Biosciences"],
    "SIGA": ["SIGA Technologies", "SIGA Technologies Inc."],
    "SION": ["Sionna Therapeutics", "Sionna Therapeutics, Inc."],
    "SKYE": ["Skye Bioscience", "Skye Bioscience, Inc."],
    "SLDB": ["Solid Biosciences", "Solid Biosciences Inc."],
    "SLN": ["Silence Therapeutics"],
    "SMMT": ["Summit Therapeutics"],
    "SNDX": ["Syndax Pharmaceuticals"],
    "SNY": ["Sanofi"],
    "SPRY": ["ARS Pharmaceuticals", "ARS Pharmaceuticals, Inc."],
    "SRPT": ["Sarepta Therapeutics"],
    "SRRK": ["Scholar Rock"],
    "SRZN": ["Surrozen"],
    "STOK": ["Stoke Therapeutics", "Stoke Therapeutics, Inc."],
    "STRO": ["Sutro Biopharma", "Sutro Biopharma, Inc."],
    "SUPN": ["Supernus Pharmaceuticals", "Supernus Pharmaceuticals, Inc."],
    "SVRA": ["Savara", "Savara, Inc."],
    "SYRE": ["Spyre Therapeutics", "Spyre Therapeutics, Inc."],
    "TARA": ["Protara Therapeutics", "Protara Therapeutics, Inc."],
    "TARS": ["Tarsus Pharmaceuticals", "Tarsus Pharmaceuticals, Inc."],
    "TBPH": ["Theravance Biopharma", "Theravance Biopharma, Inc."],
    "TCRX": ["TScan Therapeutics", "TScan Therapeutics, Inc."],
    "TECX": ["Tectonic Therapeutic", "Tectonic Therapeutic, Inc."],
    "TENX": ["Tenax Therapeutics", "Tenax Therapeutics, Inc."],
    "TERN": ["Terns Pharmaceuticals", "Terns Pharmaceuticals, Inc."],
    "TGTX": ["TG Therapeutics", "TG Therapeutics, Inc."],
    "TNGX": ["Tango Therapeutics", "Tango Therapeutics, Inc."],
    "TNYA": ["Tenaya Therapeutics", "Tenaya Therapeutics, Inc."],
    "TRDA": ["Entrada Therapeutics", "Entrada Therapeutics, Inc."],
    "TRVI": ["Trevi Therapeutics", "Trevi Therapeutics, Inc."],
    "TSHA": ["Taysha Gene Therapies"],
    "TVTX": ["Travere Therapeutics"],
    "TYRA": ["Tyra Biosciences", "Tyra Biosciences, Inc."],
    "UPB": ["Upstream Bio", "Upstream Bio, Inc."],
    "URGN": ["UroGen Pharma", "UroGen Pharma Ltd."],
    "UTHR": ["United Therapeutics"],
    "VCEL": ["Vericel", "Vericel Corporation"],
    "VERA": ["Vera Therapeutics", "Vera Therapeutics, Inc."],
    "VERV": ["Verve Therapeutics"],
    "VIR": ["Vir Biotechnology", "Vir Biotechnology, Inc."],
    "VKTX": ["Viking Therapeutics"],
    "VNDA": ["Vanda Pharmaceuticals", "Vanda Pharmaceuticals Inc."],
    "VOR": ["Vor Biopharma", "Vor Biopharma Inc."],
    "VRDN": ["Viridian Therapeutics", "Viridian Therapeutics, Inc."],
    "VRTX": ["Vertex Pharmaceuticals"],
    "VTYX": ["Ventyx Biosciences", "Ventyx Biosciences, Inc."],
    "VYGR": ["Voyager Therapeutics", "Voyager Therapeutics, Inc."],
    "WVE": ["Wave Life Sciences"],
    "XENE": ["Xenon Pharmaceuticals"],
    "XERS": ["Xeris Biopharma Holdings", "Xeris Biopharma Holdings, Inc."],
    "XNCR": ["Xencor", "Xencor, Inc."],
    "ZBIO": ["Zenas BioPharma", "Zenas BioPharma, Inc."],
    "ZLAB": ["Zai Lab"],
    "ZVRA": ["Zevra Therapeutics", "Zevra Therapeutics, Inc."],
    "ZYME": ["Zymeworks", "Zymeworks Inc."],
}


def _fetch_trials_page(base_url: str, params: dict, max_retries: int = 3) -> tuple:
    """Fetch a single page of trials. Returns (studies, next_page_token, success)."""
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                studies = data.get('studies', [])
                next_token = data.get('nextPageToken')
                return studies, next_token, True

            elif response.status_code == 429:
                wait_time = 5 * (attempt + 1)
                time.sleep(wait_time)
                continue
            else:
                return [], None, False

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return [], None, False

    return [], None, False


def _parse_study(study: dict, ticker: str) -> dict:
    """Parse a CT.gov study into our trial record format."""
    protocol = study.get('protocolSection', {})
    id_module = protocol.get('identificationModule', {})
    status_module = protocol.get('statusModule', {})
    design_module = protocol.get('designModule', {})
    sponsor_module = protocol.get('sponsorCollaboratorsModule', {})
    conditions_module = protocol.get('conditionsModule', {})
    arms_module = protocol.get('armsInterventionsModule', {})

    return {
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


def _fetch_all_trials(query_params: dict, ticker: str, max_retries: int = 3, max_results: int = 1000) -> List[Dict]:
    """Fetch all pages of trials for a given set of query params."""
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    all_trials = []
    next_page_token = None

    while len(all_trials) < max_results:
        params = {**query_params, "format": "json", "pageSize": 100}
        if next_page_token:
            params["pageToken"] = next_page_token

        studies, next_page_token, success = _fetch_trials_page(base_url, params, max_retries)

        if not studies:
            break

        for study in studies:
            all_trials.append(_parse_study(study, ticker))

        if not success or not next_page_token:
            break

        time.sleep(0.2)

    return all_trials[:max_results]


def get_trials_for_ticker(ticker: str, max_retries: int = 3, max_results: int = 1000) -> List[Dict]:
    """Fetch clinical trials for a ticker from ClinicalTrials.gov API v2.

    Uses sponsor name mapping (TICKER_TO_SPONSORS) when available for precise
    results via query.spons. Falls back to query.term (ticker symbol) when no
    sponsor mapping exists. De-duplicates results by NCT ID.
    """
    seen_nct_ids = set()
    all_trials = []

    sponsors = TICKER_TO_SPONSORS.get(ticker)

    if sponsors:
        # Primary: search by sponsor name(s)
        for sponsor in sponsors:
            trials = _fetch_all_trials({"query.spons": sponsor}, ticker, max_retries, max_results)
            for trial in trials:
                nct_id = trial.get("nct_id")
                if nct_id and nct_id not in seen_nct_ids:
                    seen_nct_ids.add(nct_id)
                    all_trials.append(trial)
    else:
        # Fallback: search by ticker symbol
        trials = _fetch_all_trials({"query.term": ticker}, ticker, max_retries, max_results)
        for trial in trials:
            nct_id = trial.get("nct_id")
            if nct_id and nct_id not in seen_nct_ids:
                seen_nct_ids.add(nct_id)
                all_trials.append(trial)

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
