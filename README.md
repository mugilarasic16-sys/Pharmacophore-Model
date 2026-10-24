
# General Pharmacophore Modelling Platform

A research prototype for target-based and ligand-based
pharmacophore feature analysis.

## Features
- Protein PDB and bound-ligand SDF input
- Ligand pharmacophore feature detection
- Ligand-based activity classification
- Training and held-out test evaluation
- CSV export and downloadable PDF reports

## Run locally
Install dependencies:

pip install -r requirements.txt

Launch:

streamlit run app.py

## Scientific limitations
The target-based workflow analyses feature positions and
protein proximity. The ligand-based workflow is a 2D
feature-based baseline classifier, not a full 3D
common-feature pharmacophore engine.

Model performance requires suitable data, independent
testing and further scientific validation.
