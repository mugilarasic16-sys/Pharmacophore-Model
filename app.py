import io
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from Bio.PDB import MMCIFParser, PDBParser
from rdkit import Chem, RDConfig
from rdkit.Chem import (
    AllChem,
    ChemicalFeatures,
    Descriptors,
    rdMolDescriptors,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    import py3Dmol
except ImportError:  # 3D view is optional
    py3Dmol = None

st.set_page_config(page_title="General Pharmacophore Platform", layout="wide")

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
FILES_URL = "https://files.rcsb.org/download/{pid}.{ext}"
CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{cid}"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
HEADERS = {"User-Agent": "pharmacophore-platform/2.0"}

FEATURES = [
    "Donor",
    "Acceptor",
    "Hydrophobe",
    "Aromatic",
    "PosIonizable",
    "NegIonizable",
]
FEATURE_COLORS = {
    "Donor": "blue",
    "Acceptor": "red",
    "Hydrophobe": "green",
    "Aromatic": "orange",
    "PosIonizable": "cyan",
    "NegIonizable": "magenta",
}

WATER = {"HOH", "WAT", "DOD", "H2O"}
# Crystallisation additives, buffers, ions and glycans that are rarely the
# ligand of interest. Can be switched off in the UI.
ADDITIVES = {
    "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "PG4", "P6G", "1PE", "DMS",
    "ACT", "FMT", "ACY", "NO3", "TRS", "MPD", "BME", "EPE", "MES", "IMD",
    "CIT", "TAR", "SCN", "PEO", "BOG", "CL", "NA", "K", "MG", "CA", "BR",
    "IOD", "ZN", "MN", "CD", "NI", "CO", "CU", "FE", "FE2", "NAG", "BMA",
    "MAN", "FUC", "GAL", "NDG", "SIA", "FLC", "DTT", "TLA", "OXY",
}

ACIDIC = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
BASIC = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2")}
AROMATIC_RES = {"PHE", "TYR", "TRP", "HIS"}
BACKBONE = {"N", "CA", "C", "O", "CB"}

factory = ChemicalFeatures.BuildFeatureFactory(
    str(RDConfig.RDDataDir) + "/BaseFeatures.fdef"
)


# ----------------------------------------------------------------------
# RCSB access
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_structure(pdb_id):
    """Download coordinates. Large entries have no .pdb, so fall back to .cif."""
    pdb_id = pdb_id.strip().upper()
    for ext in ("pdb", "cif"):
        r = requests.get(
            FILES_URL.format(pid=pdb_id, ext=ext), headers=HEADERS, timeout=60
        )
        if r.status_code == 200:
            return r.text, ext
    raise ValueError(f"Could not download {pdb_id} from RCSB (check the ID).")


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_entry_info(pdb_id):
    """Title, method, resolution and the real (non-polymer) ligand IDs."""
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


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_template_smiles(comp_id):
    """SMILES of the chemical component, used to restore bond orders."""
    r = requests.get(
        CHEMCOMP_URL.format(cid=comp_id.strip().upper()),
        headers=HEADERS,
        timeout=20,
    )
    r.raise_for_status()
    descriptors = r.json().get("pdbx_chem_comp_descriptor") or []
    preference = [
        ("SMILES_CANONICAL", "OpenEye"),
        ("SMILES_CANONICAL", "CACTVS"),
        ("SMILES", "OpenEye"),
        ("SMILES", "CACTVS"),
        ("SMILES", ""),
    ]
    for want_type, program in preference:
        for d in descriptors:
            if d.get("type") == want_type and program in (d.get("program") or ""):
                return d.get("descriptor")
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def search_pdb_by_uniprot(accession, limit=50):
    """PDB entries for a UniProt accession that contain a non-polymer entity."""
    ident = "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers"
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": ident + ".database_accession",
                        "operator": "exact_match",
                        "value": accession.strip().upper(),
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": ident + ".database_name",
                        "operator": "exact_match",
                        "value": "UniProt",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                        "operator": "greater",
                        "value": 0,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": limit},
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc",
                }
            ],
        },
    }
    r = requests.post(SEARCH_URL, json=query, headers=HEADERS, timeout=30)
    if r.status_code == 204:
        return []
    r.raise_for_status()
    return [x["identifier"] for x in r.json().get("result_set", [])]


# ----------------------------------------------------------------------
# Structure handling
# ----------------------------------------------------------------------
def parse_structure(text, ext):
    parser = PDBParser(QUIET=True) if ext == "pdb" else MMCIFParser(QUIET=True)
    return parser.get_structure("complex", io.StringIO(text))


def heavy_atoms(residue):
    return [a for a in residue if (a.element or "").strip().upper() not in ("H", "D")]


def collect_ligands(model, allowed_ids, min_heavy, drop_additives):
    rows = []
    for chain in model:
        for res in chain:
            flag = res.id[0]
            if flag in (" ", "W"):
                continue
            name = res.get_resname().strip()
            if name in WATER:
                continue
            if drop_additives and name in ADDITIVES:
                continue
            if allowed_ids and name not in allowed_ids:
                continue
            n_heavy = len(heavy_atoms(res))
            if n_heavy < min_heavy:
                continue
            rows.append(
                {
                    "comp": name,
                    "chain": chain.id,
                    "resseq": res.id[1],
                    "n_heavy": n_heavy,
                    "residue": res,
                }
            )
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
                meta.append(
                    (chain.id, res.get_resname(), res.id[1],
                     atom.get_name(), el, res.id)
                )
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
    """Build an RDKit molecule from the bound-pose coordinates and restore
    bond orders using the RCSB chemical-component template."""
    lines = []
    for i, atom in enumerate(heavy_atoms(residue), 1):
        lines.append(
            pdb_line(
                "HETATM", i, atom.get_name(), residue.get_resname(), "A", 1,
                atom.coord, (atom.element or "").strip().upper(),
            )
        )
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
                return mol, "Bond orders assigned from the RCSB chemical-component template."
            except Exception:
                pass

    # Fallback: distance-based connectivity only (single bonds). Aromatic and
    # charged groups may be missed, so the user must be warned.
    try:
        raw.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            raw,
            Chem.SANITIZE_ALL
            ^ Chem.SANITIZE_PROPERTIES
            ^ Chem.SANITIZE_KEKULIZE
            ^ Chem.SANITIZE_SETAROMATICITY,
            catchErrors=True,
        )
    except Exception:
        pass
    return raw, (
        "WARNING: template matching failed (missing atoms or no template). "
        "Connectivity is distance-based, so aromatic/ionisable features may "
        "be incomplete."
    )


# ----------------------------------------------------------------------
# Pharmacophore feature analysis
# ----------------------------------------------------------------------
def classify(family, point, tree, coords, meta):
    def hits_within(radius, keep):
        out = []
        for idx in tree.query_ball_point(point, radius):
            if keep(meta[idx]):
                out.append((float(np.linalg.norm(coords[idx] - point)), idx))
        return sorted(out)

    if family in ("Donor", "Acceptor"):
        hits = hits_within(3.5, lambda m: m[4] in ("N", "O"))
        label = "H-bond candidate" if hits else "No polar partner <=3.5 A"
    elif family == "Hydrophobe":
        hits = hits_within(4.5, lambda m: m[4] in ("C", "S"))
        label = "Hydrophobic contact" if len(hits) >= 3 else "Weak (<3 C within 4.5 A)"
    elif family == "Aromatic":
        hits = hits_within(
            5.5, lambda m: m[1] in AROMATIC_RES and m[3] not in BACKBONE
        )
        label = "Aromatic contact candidate" if hits else "No aromatic residue <=5.5 A"
    elif family == "PosIonizable":
        hits = hits_within(4.5, lambda m: (m[1], m[3]) in ACIDIC)
        label = "Salt-bridge candidate" if hits else "No acidic partner <=4.5 A"
    else:  # NegIonizable
        hits = hits_within(4.5, lambda m: (m[1], m[3]) in BASIC)
        label = "Salt-bridge candidate" if hits else "No basic partner <=4.5 A"

    contacts = "; ".join(
        f"{meta[i][1]}{meta[i][2]}{meta[i][0]}:{meta[i][3]}({d:.1f})"
        for d, i in hits[:4]
    )
    return label, contacts


def analyse_ligand(model, entry, pocket_cutoff):
    mol, note = ligand_to_mol(entry["residue"], entry["comp"])
    if mol is None:
        raise ValueError(note)

    coords, meta = protein_atom_table(model)
    if len(coords) == 0:
        raise ValueError("No protein atoms found in model 1.")
    tree = cKDTree(coords)

    rows = []
    for feat in factory.GetFeaturesForMol(mol):
        family = feat.GetFamily()
        if family not in FEATURES:
            continue
        p = feat.GetPos()
        point = np.array([p.x, p.y, p.z])
        dist, idx = tree.query(point)
        label, contacts = classify(family, point, tree, coords, meta)
        rows.append(
            {
                "Feature": family,
                "X": round(float(point[0]), 3),
                "Y": round(float(point[1]), 3),
                "Z": round(float(point[2]), 3),
                "Nearest protein atom": f"{meta[idx][1]}{meta[idx][2]}{meta[idx][0]}:{meta[idx][3]}",
                "Nearest distance (A)": round(float(dist), 3),
                "Candidate interaction": label,
                "Contacts": contacts,
            }
        )
    features = pd.DataFrame(rows)

    lig_xyz = mol.GetConformer().GetPositions()
    pocket_idx = set()
    for hit in tree.query_ball_point(lig_xyz, pocket_cutoff):
        pocket_idx.update(hit)
    pocket_keys = sorted(
        {(meta[i][0], meta[i][5], meta[i][1], meta[i][2]) for i in pocket_idx},
        key=lambda k: (k[0], k[3]),
    )
    pocket_labels = [f"{k[2]}{k[3]}{k[0]}" for k in pocket_keys]

    chain_lookup = {c.id: c for c in model}
    pocket_residues = [(k[0], chain_lookup[k[0]][k[1]]) for k in pocket_keys]
    pocket_lines, serial = [], 1
    for chain_id, res in pocket_residues:
        for atom in heavy_atoms(res):
            pocket_lines.append(
                pdb_line(
                    "ATOM", serial, atom.get_name(), res.get_resname(),
                    chain_id, res.id[1], atom.coord,
                    (atom.element or "").strip().upper(),
                )
            )
            serial += 1
    pocket_pdb = "\n".join(pocket_lines) + "\nEND\n"

    return {
        "mol": mol,
        "note": note,
        "features": features,
        "pocket_labels": pocket_labels,
        "pocket_pdb": pocket_pdb,
        "ligand_block": Chem.MolToMolBlock(mol),
    }


def show_3d(result):
    if py3Dmol is None:
        st.info("Add py3Dmol to requirements.txt to enable the 3D view.")
        return
    view = py3Dmol.view(width=900, height=600)
    view.addModel(result["pocket_pdb"], "pdb")
    view.setStyle({"model": 0}, {"stick": {"radius": 0.08, "colorscheme": "whiteCarbon"}})
    view.addModel(result["ligand_block"], "sdf")
    view.setStyle({"model": 1}, {"stick": {"radius": 0.22, "colorscheme": "greenCarbon"}})
    for _, row in result["features"].iterrows():
        view.addSphere(
            {
                "center": {"x": row["X"], "y": row["Y"], "z": row["Z"]},
                "radius": 0.45,
                "color": FEATURE_COLORS[row["Feature"]],
                "alpha": 0.7,
            }
        )
    view.zoomTo({"model": 1})
    components.html(view._make_html(), height=620)
    st.caption(
        "Spheres: " + ", ".join(f"{k} = {v}" for k, v in FEATURE_COLORS.items())
    )


# ----------------------------------------------------------------------
# Ligand-descriptor helper and PDF report
# ----------------------------------------------------------------------
def get_features(mol):
    result = {name: 0 for name in FEATURES}
    for feature in factory.GetFeaturesForMol(mol):
        family = feature.GetFamily()
        if family in result:
            result[family] += 1
    result["MolWt"] = Descriptors.MolWt(mol)
    result["LogP"] = Descriptors.MolLogP(mol)
    result["TPSA"] = rdMolDescriptors.CalcTPSA(mol)
    result["HBD"] = rdMolDescriptors.CalcNumHBD(mol)
    result["HBA"] = rdMolDescriptors.CalcNumHBA(mol)
    return result


def make_pdf(title, sections, tables=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(title, styles["Title"]),
        Paragraph("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M"), styles["Normal"]),
        Spacer(1, 15),
    ]
    for heading, body in sections:
        story.append(Paragraph(heading, styles["Heading2"]))
        safe = str(body).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe, styles["BodyText"]))
        story.append(Spacer(1, 10))
    for heading, dataframe in tables or []:
        story.append(Paragraph(heading, styles["Heading2"]))
        if dataframe is not None and not dataframe.empty:
            view = dataframe.iloc[:35, :7].fillna("")
            data = [list(map(str, view.columns))]
            data.extend(view.astype(str).values.tolist())
            table = Table(data, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                        ("FONTSIZE", (0, 0), (-1, -1), 7),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 10))
    doc.build(story)
    return buffer.getvalue()


# ----------------------------------------------------------------------
# Target-based workflow (automatic ligand detection)
# ----------------------------------------------------------------------
def load_complex(text, ext, label, pdb_id=None):
    structure = parse_structure(text, ext)
    info = fetch_entry_info(pdb_id) if pdb_id else {}
    st.session_state["cx"] = {
        "label": label,
        "pdb_id": pdb_id,
        "structure": structure,
        "info": info,
    }
    st.session_state.pop("analysis", None)


def target_mode():
    st.header("Target-based pharmacophore feature analysis")
    st.caption(
        "Give a PDB ID (or UniProt accession) and the app downloads the "
        "complex, finds the bound ligand(s) and extracts the pharmacophore "
        "features automatically."
    )

    source = st.radio(
        "Input source",
        ["PDB ID", "UniProt accession", "Upload complex file"],
        horizontal=True,
    )

    try:
        if source == "PDB ID":
            pid = st.text_input("PDB ID", placeholder="e.g. 1M17").strip().upper()
            if st.button("Fetch structure"):
                if len(pid) != 4 or not pid.isalnum():
                    st.error("Enter a valid 4-character PDB ID.")
                else:
                    with st.spinner(f"Downloading {pid} from RCSB..."):
                        text, ext = fetch_structure(pid)
                        load_complex(text, ext, pid, pid)

        elif source == "UniProt accession":
            acc = st.text_input("UniProt accession", placeholder="e.g. P00533").strip()
            if st.button("Find PDB entries with ligands"):
                with st.spinner("Searching RCSB..."):
                    st.session_state["uniprot_hits"] = search_pdb_by_uniprot(acc)
                if not st.session_state["uniprot_hits"]:
                    st.warning("No ligand-bound entries found for that accession.")
            hits = st.session_state.get("uniprot_hits")
            if hits:
                choice = st.selectbox("Entries (best resolution first)", hits)
                if st.button("Load selected entry"):
                    with st.spinner(f"Downloading {choice}..."):
                        text, ext = fetch_structure(choice)
                        load_complex(text, ext, choice, choice)

        else:
            upload = st.file_uploader(
                "Upload protein-ligand complex (PDB or mmCIF)",
                type=["pdb", "ent", "cif", "mmcif"],
            )
            if upload and st.button("Load file"):
                ext = "pdb" if upload.name.lower().endswith((".pdb", ".ent")) else "cif"
                load_complex(upload.getvalue().decode("utf-8", "ignore"), ext, upload.name)
    except requests.RequestException as error:
        st.error(f"Network problem while contacting RCSB: {error}")
        return
    except Exception as error:
        st.error(f"Could not load the structure: {error}")
        return

    cx = st.session_state.get("cx")
    if not cx:
        return

    info = cx["info"]
    st.subheader(f"Loaded: {cx['label']}")
    if info.get("title"):
        st.write(info["title"])
    meta_bits = [b for b in (info.get("method"),
                             f"{info['resolution']} A" if info.get("resolution") else None) if b]
    if meta_bits:
        st.caption(" | ".join(meta_bits))

    col1, col2, col3 = st.columns(3)
    min_heavy = col1.slider("Min. ligand heavy atoms", 1, 30, 6)
    drop_additives = col2.checkbox("Ignore common additives, ions, glycans", value=True)
    pocket_cutoff = col3.slider("Pocket radius (A)", 3.0, 8.0, 5.0, 0.5)

    model = cx["structure"][0]
    allowed = set(info.get("comps", {}).keys()) or None
    ligands = collect_ligands(model, allowed, min_heavy, drop_additives)

    if not ligands:
        st.warning(
            "No ligand passed the filters. This may be an apo structure. "
            "Lower the heavy-atom filter, untick the additive filter, or pick "
            "another entry."
        )
        return

    labels = [
        f"{l['comp']} | chain {l['chain']} | res {l['resseq']} | {l['n_heavy']} heavy atoms"
        for l in ligands
    ]
    chosen = st.selectbox("Detected ligands (largest first)", labels)
    entry = ligands[labels.index(chosen)]

    if st.button("Analyse selected ligand"):
        try:
            with st.spinner("Extracting pharmacophore features..."):
                result = analyse_ligand(model, entry, pocket_cutoff)
            result["meta"] = {
                "ligand": chosen,
                "structure": cx["label"],
                "title": info.get("title", ""),
            }
            st.session_state["analysis"] = result
        except Exception as error:
            st.session_state.pop("analysis", None)
            st.error(f"Analysis failed: {error}")

    result = st.session_state.get("analysis")
    if not result:
        return

    st.info(result["note"])
    features = result["features"]
    if features.empty:
        st.warning("No pharmacophore features were detected on this ligand.")
        return

    st.subheader("Detected pharmacophore feature points")
    st.dataframe(features, use_container_width=True)
    st.write("Feature counts:", features["Feature"].value_counts().to_dict())
    st.write(
        f"Pocket residues within {pocket_cutoff} A ({len(result['pocket_labels'])}): "
        + ", ".join(result["pocket_labels"])
    )

    st.subheader("3D view")
    show_3d(result)

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download feature table CSV",
        features.to_csv(index=False).encode("utf-8"),
        "target_features.csv",
        "text/csv",
    )
    c2.download_button(
        "Download ligand SDF (bound pose)",
        result["ligand_block"].encode("utf-8"),
        "bound_ligand.sdf",
        "chemical/x-mdl-sdfile",
    )
    pdf = make_pdf(
        "Target-based Feature Analysis Report",
        [
            ("Structure", f"{result['meta']['structure']} - {result['meta']['title']}"),
            ("Ligand", result["meta"]["ligand"]),
            ("Ligand preparation", result["note"]),
            ("Method", "Automatic RCSB retrieval, ligand detection, RDKit feature "
                       "perception on the crystallographic pose, KD-tree distance "
                       "analysis against protein heavy atoms."),
            ("Pocket residues", ", ".join(result["pocket_labels"])),
            ("Limitation", "Distance-based interaction labels are candidates only. "
                           "Waters, metals and other hetero groups are not "
                           "considered. Proximity alone does not validate a "
                           "pharmacophore."),
        ],
        [("Feature results", features)],
    )
    c3.download_button("Download PDF report", pdf, "target_report.pdf", "application/pdf")


# ----------------------------------------------------------------------
# Ligand-based workflow
# ----------------------------------------------------------------------
def ligand_based_mode():
    st.header("Ligand-based activity modelling")

    csv_file = st.file_uploader(
        "Upload CSV with smiles and activity columns", type=["csv"]
    )
    test_fraction = st.slider("Held-out test fraction", 0.10, 0.40, 0.20, 0.05)

    positive = None
    raw = None
    if csv_file is not None:
        raw = pd.read_csv(io.BytesIO(csv_file.getvalue()))
        if not {"smiles", "activity"}.issubset(raw.columns):
            st.error("Your CSV needs columns named smiles and activity.")
            return
        classes = sorted(raw["activity"].dropna().astype(str).str.strip().unique())
        if len(classes) == 2:
            positive = st.selectbox("Which class is the positive (active) class?", classes)

    if st.button("Train and evaluate"):
        if raw is None:
            st.error("Upload a CSV dataset.")
            return
        if positive is None:
            st.error("The activity column must contain exactly two classes.")
            return

        records = []
        for _, row in raw.iterrows():
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is None or pd.isna(row["activity"]):
                continue
            records.append(
                {
                    "smiles": str(row["smiles"]),
                    "canon": Chem.MolToSmiles(mol),
                    "activity": str(row["activity"]).strip(),
                    **get_features(mol),
                }
            )
        data = pd.DataFrame(records)
        if data.empty:
            st.error("No valid SMILES found.")
            return

        n_before = len(data)
        conflict = data.groupby("canon")["activity"].nunique()
        data = data[~data["canon"].isin(conflict[conflict > 1].index)]
        data = data.drop_duplicates("canon").reset_index(drop=True)
        st.write(
            f"Removed {n_before - len(data)} duplicate/conflicting structures "
            f"({len(data)} remain)."
        )

        if data["activity"].nunique() != 2 or data["activity"].value_counts().min() < 3:
            st.error("Need two classes with at least 3 unique compounds each.")
            return

        columns = FEATURES + ["MolWt", "LogP", "TPSA", "HBD", "HBA"]
        X, y = data[columns], data["activity"]
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_fraction, random_state=2026, stratify=y
            )
        except ValueError as error:
            st.error(f"Unable to split the dataset: {error}")
            return

        model = RandomForestClassifier(
            n_estimators=300, random_state=2026, class_weight="balanced"
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "Training compounds": len(X_train),
            "Test compounds": len(X_test),
            "Accuracy": round(accuracy_score(y_test, predictions), 4),
            "Precision": round(precision_score(y_test, predictions, pos_label=positive, zero_division=0), 4),
            "Recall": round(recall_score(y_test, predictions, pos_label=positive, zero_division=0), 4),
            "F1 score": round(f1_score(y_test, predictions, pos_label=positive, zero_division=0), 4),
        }
        if y_test.nunique() == 2:
            proba = model.predict_proba(X_test)[:, list(model.classes_).index(positive)]
            metrics["ROC-AUC"] = round(
                roc_auc_score((y_test == positive).astype(int), proba), 4
            )

        metric_table = pd.DataFrame(list(metrics.items()), columns=["Metric", "Result"])
        predictions_table = pd.DataFrame(
            {
                "SMILES": data.loc[X_test.index, "smiles"].values,
                "Observed": y_test.values,
                "Predicted": predictions,
            }
        )
        pdf = make_pdf(
            "Ligand-based Modelling Report",
            [
                ("Dataset", csv_file.name),
                ("Method", "RDKit 2D pharmacophore feature counts and physicochemical "
                           "descriptors with Random Forest."),
                ("Split", f"Training: {len(X_train)}; random held-out test: {len(X_test)}."),
                ("Metrics", "; ".join(f"{k}: {v}" for k, v in metrics.items())),
                ("Limitation", "Baseline 2D feature classifier, not a 3D common-feature "
                               "pharmacophore. Random split can overestimate performance "
                               "for congeneric series; external validation is needed."),
            ],
            [("Test metrics", metric_table), ("Held-out predictions", predictions_table)],
        )
        st.session_state["lb"] = {
            "metrics": metric_table,
            "predictions": predictions_table,
            "pdf": pdf,
        }

    lb = st.session_state.get("lb")
    if lb:
        st.subheader("Held-out test results")
        st.dataframe(lb["metrics"], use_container_width=True)
        st.download_button(
            "Download test predictions CSV",
            lb["predictions"].to_csv(index=False).encode("utf-8"),
            "test_predictions.csv",
            "text/csv",
        )
        st.download_button(
            "Download complete PDF report", lb["pdf"],
            "ligand_based_report.pdf", "application/pdf",
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.title("General Pharmacophore Modelling Platform")
st.write("A research prototype for target-based and ligand-based pharmacophore analysis.")
st.warning(
    "A generated hypothesis is not automatically validated. Reliable "
    "structures, activity data and independent testing are needed."
)

mode = st.sidebar.radio("Choose modelling workflow", ["Target-based", "Ligand-based"])
if mode == "Target-based":
    target_mode()
else:
    ligand_based_mode()
