"""RCSB PDB access, ligand detection and ligand-pose preparation."""
import io
from functools import lru_cache

import numpy as np
import requests
from Bio.PDB import MMCIFParser, PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem

FILES_URL = "https://files.rcsb.org/download/{pid}.{ext}"
CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{cid}"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
HEADERS = {"User-Agent": "pharmacophore-platform/3.0"}

WATER = {"HOH", "WAT", "DOD", "H2O"}
ADDITIVES = {
    "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "PG4", "P6G", "1PE", "DMS",
    "ACT", "FMT", "ACY", "NO3", "TRS", "MPD", "BME", "EPE", "MES", "IMD",
    "CIT", "TAR", "SCN", "PEO", "BOG", "CL", "NA", "K", "MG", "CA", "BR",
    "IOD", "ZN", "MN", "CD", "NI", "CO", "CU", "FE", "FE2", "NAG", "BMA",
    "MAN", "FUC", "GAL", "NDG", "SIA", "FLC", "DTT", "TLA", "OXY",
}


@lru_cache(maxsize=256)
def fetch_structure(pdb_id):
    pdb_id = pdb_id.strip().upper()
    for ext in ("pdb", "cif"):
        r = requests.get(FILES_URL.format(pid=pdb_id, ext=ext), headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r.text, ext
    raise ValueError(f"Could not download {pdb_id} from RCSB (check the ID).")


@lru_cache(maxsize=256)
def fetch_entry_info(pdb_id):
    query = """
    query ($id: String!) {
      entry(entry_id: $id) {
        struct { title }
        exptl { method }
        rcsb_entry_info { resolution_combined }
        nonpolymer_entities { nonpolymer_comp { chem_comp { id name } } }
      }
    }"""
    try:
        r = requests.post(
            GRAPHQL_URL,
            json={"query": query, "variables": {"id": pdb_id.upper()}},
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        entry = (r.json().get("data") or {}).get("entry") or {}
    except Exception:
        return {}
    comps = {}
    for ent in entry.get("nonpolymer_entities") or []:
        chem = ((ent or {}).get("nonpolymer_comp") or {}).get("chem_comp") or {}
        if chem.get("id"):
            comps[chem["id"]] = chem.get("name", "")
    res = (entry.get("rcsb_entry_info") or {}).get("resolution_combined")
    methods = [e.get("method") for e in (entry.get("exptl") or []) if e]
    return {
        "title": (entry.get("struct") or {}).get("title", ""),
        "method": ", ".join(m for m in methods if m),
        "resolution": res[0] if res else None,
        "comps": comps,
    }


@lru_cache(maxsize=512)
def fetch_template_smiles(comp_id):
    r = requests.get(CHEMCOMP_URL.format(cid=comp_id.strip().upper()), headers=HEADERS, timeout=20)
    r.raise_for_status()
    descriptors = r.json().get("pdbx_chem_comp_descriptor") or []
    for want_type, program in [
        ("SMILES_CANONICAL", "OpenEye"),
        ("SMILES_CANONICAL", "CACTVS"),
        ("SMILES", "OpenEye"),
        ("SMILES", "CACTVS"),
        ("SMILES", ""),
    ]:
        for d in descriptors:
            if d.get("type") == want_type and program in (d.get("program") or ""):
                return d.get("descriptor")
    return None


@lru_cache(maxsize=64)
def search_pdb_by_uniprot(accession, limit=50):
    ident = "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers"
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": ident + ".database_accession",
                    "operator": "exact_match", "value": accession.strip().upper()}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": ident + ".database_name",
                    "operator": "exact_match", "value": "UniProt"}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                    "operator": "greater", "value": 0}},
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": limit},
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }
    r = requests.post(SEARCH_URL, json=query, headers=HEADERS, timeout=30)
    if r.status_code == 204:
        return []
    r.raise_for_status()
    return [x["identifier"] for x in r.json().get("result_set", [])]


def parse_structure(text, ext):
    parser = PDBParser(QUIET=True) if ext == "pdb" else MMCIFParser(QUIET=True)
    return parser.get_structure("complex", io.StringIO(text))


def heavy_atoms(residue):
    return [a for a in residue if (a.element or "").strip().upper() not in ("H", "D")]


def collect_ligands(model, allowed_ids=None, min_heavy=6, drop_additives=True, force_comp=None):
    rows = []
    for chain in model:
        for res in chain:
            if res.id[0] in (" ", "W"):
                continue
            name = res.get_resname().strip()
            if name in WATER:
                continue
            if force_comp:
                if name != force_comp.strip().upper():
                    continue
            else:
                if drop_additives and name in ADDITIVES:
                    continue
                if allowed_ids and name not in allowed_ids:
                    continue
            n_heavy = len(heavy_atoms(res))
            if n_heavy < min_heavy:
                continue
            rows.append({"comp": name, "chain": chain.id, "resseq": res.id[1],
                         "n_heavy": n_heavy, "residue": res})
    rows.sort(key=lambda r: r["n_heavy"], reverse=True)
    return rows


def protein_atom_table(model):
    coords, meta = [], []
    for chain in model:
        for res in chain:
            if res.id[0] != " ":
                continue
            for atom in res:
                el = (atom.element or "").strip().upper()
                if el in ("H", "D"):
                    continue
                coords.append(atom.coord)
                meta.append((chain.id, res.get_resname(), res.id[1], atom.get_name(), el, res.id))
    return np.array(coords), meta


def pdb_line(record, serial, name, resname, chain, resseq, coord, element):
    name_field = f" {name:<3}" if len(element) == 1 and len(name) < 4 else f"{name:<4}"
    return (
        f"{record:<6}{serial % 100000:5d} {name_field[:4]} {resname:>3} "
        f"{chain[:1] or 'A'}{resseq % 10000:4d}    "
        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{1.0:6.2f}{0.0:6.2f}"
        f"          {element.rjust(2)}"
    )


def ligand_to_mol(residue, comp_id):
    """RDKit molecule at the crystallographic pose, bond orders from the RCSB template."""
    lines = [
        pdb_line("HETATM", i, a.get_name(), residue.get_resname(), "A", 1, a.coord,
                 (a.element or "").strip().upper())
        for i, a in enumerate(heavy_atoms(residue), 1)
    ]
    lines.append("END")
    raw = Chem.MolFromPDBBlock("\n".join(lines), sanitize=False, removeHs=False)
    if raw is None or raw.GetNumAtoms() == 0:
        return None, "RDKit could not read the ligand coordinates."
    try:
        smiles = fetch_template_smiles(comp_id)
    except Exception:
        smiles = None
    if smiles:
        template = Chem.MolFromSmiles(smiles)
        if template is not None:
            try:
                mol = AllChem.AssignBondOrdersFromTemplate(template, raw)
                Chem.SanitizeMol(mol)
                return mol, "template"
            except Exception:
                pass
    try:
        raw.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            raw,
            Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES ^ Chem.SANITIZE_KEKULIZE
            ^ Chem.SANITIZE_SETAROMATICITY,
            catchErrors=True,
        )
    except Exception:
        pass
    return raw, "approximate"
