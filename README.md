# General Pharmacophore Modelling Platform

A research prototype for target-based and ligand-based
pharmacophore feature analysis.

## Features

### Target-based
- Input by PDB ID, UniProt accession, or an uploaded complex file
  (PDB or mmCIF). No separate ligand file is needed.
- Automatic download from RCSB PDB and automatic detection of bound
  ligands (water, buffers, ions and glycans filtered out; filter is optional)
- Ligand bond orders restored from the RCSB chemical-component template
- Pharmacophore feature detection on the crystallographic pose
- Candidate interaction labels, contacting residues and pocket residue list
- Interactive 3D view, CSV and SDF export, PDF report

### Ligand-based
- Upload a CSV with `smiles` and `activity` columns
- Duplicate and conflicting structures removed before modelling
- User-selected positive class
- 2D feature-count Random Forest with held-out test evaluation
- CSV export and PDF report

## Run locally

Install dependencies:

    pip install -r requirements.txt

Launch:

    streamlit run app.py

An internet connection is required for the target-based workflow
(RCSB PDB is queried live).

## Scientific limitations
- Target-based results come from a single complex. Interaction labels are
  distance-based candidates only; waters, metals and cofactors are not
  considered, and only the first model of NMR entries is used.
- Apo structures (no bound ligand) cannot be analysed.
- If ligand template matching fails (for example, missing atoms in the
  crystal structure), connectivity is distance-based and aromatic or
  ionisable features may be incomplete. The app shows a warning.
- The ligand-based workflow is a 2D feature-based baseline classifier,
  not a full 3D common-feature pharmacophore engine. A random split can
  overestimate performance for congeneric series.

Generated hypotheses require independent validation (for example decoy
sets with ROC and enrichment analysis) before any scientific claim.
