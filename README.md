# Pharmacophore Generator

Builds pharmacophore MODELS (not just feature tables) and validates them.

## Files
- `app.py`          Streamlit interface
- `pharmacore.py`   Pipelines: ligand-based, target-based, screening/validation, export
- `geometry.py`     Kabsch fit, consensus clustering, 3D matching, EF (numpy/scipy only)
- `rcsb.py`         RCSB download, ligand detection, ligand pose preparation
- `test_geometry.py` Synthetic-data tests for geometry.py (`python test_geometry.py`)

## Ligand-based pipeline
CSV (smiles, activity) -> scaffold split -> ETKDGv3 conformers -> Crippen-O3A
alignment of actives to a rigid reference -> common-feature clustering
-> model -> validation on held-out new-scaffold actives vs inactives.

## Target-based pipeline
PDB IDs or UniProt -> download -> auto-detect ligand -> keep only features that
interact with the protein -> superimpose complexes on the pocket -> consensus
features + exclusion volumes -> model -> optional validation.

## Run
    pip install -r requirements.txt
    streamlit run app.py
Internet is required for the target-based workflow.

## Known limitations
- Features are points with a tolerance radius. No directionality for H-bonds
  or aromatic ring normals.
- Ligand-based alignment uses Crippen-O3A (atom-type/shape based), not a
  feature-based alignment, and assumes the actives share one binding mode.
- Charge/protonation states come from the input SMILES.
- Target-based interaction rules are distance-based; waters, metals and
  cofactors are ignored.
- Validation numbers from small datasets are noisy. Use property-matched
  decoys (e.g. DUD-E style) or confirmed inactives, never random molecules
  of different size.
