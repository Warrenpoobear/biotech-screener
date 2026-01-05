"""
Scoring modules for Wake Robin Biotech Alpha System.

Module 3: Catalyst scoring (trial event proximity, sponsor reliability)
Module 4: Clinical development scoring (phase weighting, pipeline depth)

These modules accept TrialRow data from providers and produce deterministic scores.
The provider owns the PIT boundary; modules just score and emit flags.

Example signatures (to be implemented):

def module3_catalyst(
    as_of_date: date,
    pit_cutoff: date,
    trials_by_ticker: dict[str, list[TrialRow]],
    sponsor_lag_model: LagModel,
) -> Module3Output:
    ...

def module4_clinical(
    as_of_date: date,
    pit_cutoff: date,
    trials_by_ticker: dict[str, list[TrialRow]],
) -> Module4Output:
    ...
"""
