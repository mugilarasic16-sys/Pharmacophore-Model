"""Pharmacophore GENERATION pipelines.

Ligand-based : conformers -> align actives -> common-feature consensus -> model
Target-based : ligand-bound complexes -> keep interacting features ->
               (superimpose + consensus across complexes) -> model + exclusion volumes
Validation   : screen actives/inactives against the model (ROC-AUC, EF)

A model is a plain dict:
  {"type", "features": [{family,x,y,z,radius,support,n_sources,...}],
   "exclusion": [{x,y,z,radius}], "meta": {...}}
"""
import json
import os

import numpy as np
import pandas as pd
from Bio import Align
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, rdMolAlign, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Geometry import Point3D
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, roc_curve

import geometry as geo
from rcsb import heavy_atoms, ligand_to_mol, pdb_line, protein_atom_table

FEATURE_COLORS = {
    "Donor": "blue", "Acceptor": "red", "Hydrophobe": "green",
    "Aromatic": "orange", "PosIonizable": "cyan", "NegIonizable": "magenta",
}
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
ACIDIC = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
BASIC = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2")}
AROMATIC_RES = {"PHE", "TYR", "TRP", "HIS"}
BACKBONE = {"N", "CA", "C", "O", "CB"}

factory = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(str(RDConfig.RDDataDir), "BaseFeatures.fdef")
)


# ======================================================================
# Shared helpers
# ======================================================================
def mol_features(mol, conf_id=-1):
    """[(family, xyz)] for one conformer. Position = centroid of the feature atoms."""
    xyz = mol.GetConformer(conf_id).GetPositions()
    out = []
    for f in factory.GetFeaturesForMol(mol, confId=conf_id):
        fam = f.GetFamily()
        if fam in geo.SPECIFICITY:
            out.append((fam, xyz[list(f.GetAtomIds())].mean(0)))
    return out


def embed(smiles, n_confs=50, seed=42):
    """Largest fragment -> ETKDGv3 conformers -> MMFF/UFF minimisation.
    Returns (heavy-atom mol with conformers, conformer ids, energies) or None."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    except Exception:
        pass
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.5
    params.numThreads = 0
    cids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
    if not cids:
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
    if not cids:
        return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=500)
        else:
            res = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=500)
        energies = [e for _, e in res]
    except Exception:
        energies = [0.0] * len(cids)
    return Chem.RemoveHs(mol), cids, energies


def set_positions(mol, X):
    conf = mol.GetConformer()
    for i, (x, y, z) in enumerate(X):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))


def _finalise(cons, max_features, tol):
    return [
        {"family": c["family"], "x": round(float(c["xyz"][0]), 3),
         "y": round(float(c["xyz"][1]), 3), "z": round(float(c["xyz"][2]), 3),
         "radius": tol, "support": c["support"], "n_sources": c["n_sources"],
         "spread": c["spread"]}
        for c in cons[:max_features]
    ]


def scaffold_split(smiles, test_fraction=0.25):
    """Group by Bemis-Murcko scaffold; whole scaffold groups go to train OR test."""
    groups = {}
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""
        except Exception:
            scaf = ""
        groups.setdefault(scaf, []).append(i)
    n_train = int(round((1 - test_fraction) * len(smiles)))
    train, test = [], []
    for g in sorted(groups.values(), key=len, reverse=True):
        (train if len(train) + len(g) <= n_train else test).extend(g)
    return sorted(train), sorted(test)


# ======================================================================
# 1. LIGAND-BASED pharmacophore generation
# ======================================================================
def choose_reference(items, name=None):
    if name:
        for it in items:
            if it["name"] == name:
                return it
    return min(
        items,
        key=lambda it: (rdMolDescriptors.CalcNumRotatableBonds(it["mol"]),
                        -len(mol_features(it["mol"], it["cids"][0]))),
    )


def build_ligand_based(smiles, names, n_confs=50, tol=1.0, min_support=0.7,
                       max_features=5, ref_name=None, progress=None):
    """Common-feature pharmacophore from a set of ACTIVE ligands."""
    items, skipped = [], []
    for k, (s, n) in enumerate(zip(smiles, names)):
        if progress:
            progress(k / max(len(smiles), 1) * 0.5, f"Conformers: {n}")
        r = embed(s, n_confs)
        if r is None:
            skipped.append(f"{n}: conformer generation failed")
            continue
        mol, cids, energies = r
        mol.SetProp("_Name", str(n))
        items.append({"name": n, "smiles": s, "mol": mol, "cids": cids, "energies": energies})
    if len(items) < 3:
        raise ValueError("Need at least 3 actives with valid 3D conformers.")

    ref = choose_reference(items, ref_name)
    ref_cid = ref["cids"][int(np.argmin(ref["energies"]))]
    ref_feats = mol_features(ref["mol"], ref_cid)
    points = [{"family": f, "xyz": x, "source": ref["name"]} for f, x in ref_feats]
    aligned = [(ref, ref_cid)]

    for k, it in enumerate(items):
        if it is ref:
            continue
        if progress:
            progress(0.5 + k / len(items) * 0.45, f"Aligning: {it['name']}")
        best = None
        for cid in it["cids"]:
            try:
                o3a = rdMolAlign.GetCrippenO3A(it["mol"], ref["mol"], cid, ref_cid)
                o3a.Align()
                score = o3a.Score()
            except Exception:
                continue
            feats = mol_features(it["mol"], cid)
            key = (geo.feature_overlap(ref_feats, feats, 2.0), score)
            if best is None or key > best[0]:
                best = (key, cid, feats)
        if best is None:
            skipped.append(f"{it['name']}: alignment to reference failed")
            continue
        aligned.append((it, best[1]))
        points.extend({"family": f, "xyz": x, "source": it["name"]} for f, x in best[2])

    n_used = len(aligned)
    cons = geo.consensus_features(points, tol, min_support, n_used)
    features = _finalise(cons, max_features, tol)
    sdf = "".join(Chem.MolToMolBlock(it["mol"], confId=cid) + "$$$$\n" for it, cid in aligned)

    return {
        "type": "ligand_based",
        "features": features,
        "exclusion": [],
        "meta": {
            "reference": ref["name"], "n_actives_used": n_used, "tolerance": tol,
            "min_support": min_support, "n_confs": n_confs, "skipped": skipped,
            "method": "ETKDGv3 conformers, Crippen-O3A alignment to a rigid reference, "
                      "common-feature clustering (complete linkage)",
        },
        "aligned_sdf": sdf,
        "n_aligned": n_used,
    }


# ======================================================================
# 2. TARGET-BASED pharmacophore generation
# ======================================================================
def interaction_check(family, point, tree, coords, meta):
    """Is this ligand feature actually engaged by the protein? (distance-based)"""
    def hits_within(radius, keep):
        out = [(float(np.linalg.norm(coords[i] - point)), i)
               for i in tree.query_ball_point(point, radius) if keep(meta[i])]
        return sorted(out)

    if family in ("Donor", "Acceptor"):
        hits = hits_within(3.5, lambda m: m[4] in ("N", "O"))
        ok, label = len(hits) > 0, "H-bond"
    elif family == "Hydrophobe":
        hits = hits_within(4.5, lambda m: m[4] in ("C", "S"))
        ok, label = len(hits) >= 3, "Hydrophobic"
    elif family == "Aromatic":
        hits = hits_within(5.5, lambda m: m[1] in AROMATIC_RES and m[3] not in BACKBONE)
        ok, label = len(hits) > 0, "Aromatic"
    elif family == "PosIonizable":
        hits = hits_within(4.5, lambda m: (m[1], m[3]) in ACIDIC)
        ok, label = len(hits) > 0, "Salt bridge"
    else:
        hits = hits_within(4.5, lambda m: (m[1], m[3]) in BASIC)
        ok, label = len(hits) > 0, "Salt bridge"
    contacts = "; ".join(f"{meta[i][1]}{meta[i][2]}{meta[i][0]}:{meta[i][3]}({d:.1f})"
                         for d, i in hits[:3])
    return ok, label, contacts


def analyse_complex(label, struct_model, entry):
    mol, note = ligand_to_mol(entry["residue"], entry["comp"])
    if mol is None:
        raise ValueError(note)
    coords, meta = protein_atom_table(struct_model)
    if len(coords) == 0:
        raise ValueError("no protein atoms")
    tree = cKDTree(coords)
    feats, dropped = [], 0
    for fam, xyz in mol_features(mol):
        ok, kind, contacts = interaction_check(fam, xyz, tree, coords, meta)
        if ok:
            feats.append({"family": fam, "xyz": xyz, "kind": kind, "contacts": contacts})
        else:
            dropped += 1
    return {"label": label, "model": struct_model, "entry": entry, "mol": mol,
            "ligand_note": note, "feats": feats, "n_dropped": dropped,
            "coords": coords, "meta": meta}


def _chain_near_ligand(struct_model, residue):
    centre = np.mean([a.coord for a in heavy_atoms(residue)], axis=0)
    best, best_n = None, -1
    for chain in struct_model:
        cas = np.array([r["CA"].coord for r in chain if r.id[0] == " " and "CA" in r])
        if len(cas) == 0:
            continue
        n = int((np.linalg.norm(cas - centre, axis=1) < 10).sum())
        if n > best_n:
            best, best_n = chain, n
    return best


def superposition(ref, mob):
    """Superimpose the 'mob' complex onto 'ref' via sequence-aligned CA atoms,
    refined on the binding pocket. Returns (R, Pc, Qc, pocket_rmsd, n_pairs, n_pocket)."""
    rc = _chain_near_ligand(ref["model"], ref["entry"]["residue"])
    mc = _chain_near_ligand(mob["model"], mob["entry"]["residue"])
    if rc is None or mc is None:
        raise ValueError("no protein chain near ligand")
    rres = [r for r in rc if r.id[0] == " " and "CA" in r]
    mres = [r for r in mc if r.id[0] == " " and "CA" in r]
    sr = "".join(AA3.get(r.get_resname(), "X") for r in rres)
    sm = "".join(AA3.get(r.get_resname(), "X") for r in mres)

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score, aligner.extend_gap_score = -10, -0.5
    aln = aligner.align(sr, sm)[0]
    pairs = []
    for (a0, a1), (b0, b1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(a1 - a0):
            i, j = a0 + k, b0 + k
            if sr[i] == sm[j] and sr[i] != "X":
                pairs.append((i, j))
    if len(pairs) < 20:
        raise ValueError(f"only {len(pairs)} identical aligned residues - proteins too different")

    ref_ca = np.array([rres[i]["CA"].coord for i, _ in pairs])
    mob_ca = np.array([mres[j]["CA"].coord for _, j in pairs])
    rmsd, R, Pc, Qc = geo.kabsch(mob_ca, ref_ca)
    centre = np.mean([a.coord for a in heavy_atoms(ref["entry"]["residue"])], axis=0)
    sel = np.linalg.norm(ref_ca - centre, axis=1) < 12.0
    if sel.sum() >= 8:
        rmsd, R, Pc, Qc = geo.kabsch(mob_ca[sel], ref_ca[sel])
    return R, Pc, Qc, rmsd, len(pairs), int(sel.sum())


def exclusion_spheres(coords, lig_xyz, shell, radius):
    idx = set()
    for hit in cKDTree(coords).query_ball_point(lig_xyz, shell):
        idx.update(hit)
    return [{"x": round(float(coords[i][0]), 3), "y": round(float(coords[i][1]), 3),
             "z": round(float(coords[i][2]), 3), "radius": radius} for i in sorted(idx)]


def pocket_pdb(ref, lig_xyz, cutoff=6.0):
    tree = cKDTree(ref["coords"])
    idx = set()
    for hit in tree.query_ball_point(lig_xyz, cutoff):
        idx.update(hit)
    keys = sorted({(ref["meta"][i][0], ref["meta"][i][5]) for i in idx},
                  key=lambda k: (k[0], k[1][1]))
    chains = {c.id: c for c in ref["model"]}
    lines, n = [], 1
    for cid, rid in keys:
        res = chains[cid][rid]
        for a in heavy_atoms(res):
            lines.append(pdb_line("ATOM", n, a.get_name(), res.get_resname(), cid, res.id[1],
                                  a.coord, (a.element or "").strip().upper()))
            n += 1
    return "\n".join(lines) + "\nEND\n"


def build_target_based(items, tol=1.0, min_support=0.6, max_features=8,
                       excl_shell=5.0, excl_radius=1.0, progress=None):
    """items: [{label, model, entry}] - first item is the reference frame.
    One complex -> interacting features of that ligand.
    Several complexes -> superimpose on the reference pocket, pool the interacting
    features and keep clusters supported by >= min_support of the complexes."""
    notes, analysed = [], []
    for k, it in enumerate(items):
        if progress:
            progress(k / len(items) * 0.5, f"Analysing {it['label']}")
        try:
            analysed.append(analyse_complex(it["label"], it["model"], it["entry"]))
        except Exception as e:
            notes.append(f"{it['label']}: skipped ({e})")
    if not analysed:
        raise ValueError("No complex could be analysed. " + " | ".join(notes))

    ref = analysed[0]
    ref_lig = ref["mol"].GetConformer().GetPositions()
    ref_centre = ref_lig.mean(0)
    kept, points, align_info = [ref], [], [{"complex": ref["label"], "pocket_ca_rmsd": 0.0}]
    points += [{"family": f["family"], "xyz": f["xyz"], "source": ref["label"]} for f in ref["feats"]]

    for k, a in enumerate(analysed[1:], 1):
        if progress:
            progress(0.5 + k / len(analysed) * 0.4, f"Superimposing {a['label']}")
        try:
            R, Pc, Qc, rmsd, n_pairs, n_pocket = superposition(ref, a)
        except Exception as e:
            notes.append(f"{a['label']}: skipped, cannot superimpose ({e})")
            continue
        lig = geo.apply_transform(a["mol"].GetConformer().GetPositions(), R, Pc, Qc)
        if np.linalg.norm(lig.mean(0) - ref_centre) > 10.0:
            notes.append(f"{a['label']}: skipped, ligand sits >10 A from the reference site")
            continue
        if rmsd > 2.5:
            notes.append(f"{a['label']}: kept but pocket CA RMSD is {rmsd:.2f} A (flexible site?)")
        set_positions(a["mol"], lig)
        for f in a["feats"]:
            f["xyz"] = geo.apply_transform(f["xyz"], R, Pc, Qc)
            points.append({"family": f["family"], "xyz": f["xyz"], "source": a["label"]})
        kept.append(a)
        align_info.append({"complex": a["label"], "pocket_ca_rmsd": round(rmsd, 2),
                           "aligned_pairs": n_pairs, "pocket_pairs": n_pocket})

    cons = geo.consensus_features(points, tol, min_support, len(kept))
    features = _finalise(cons, max_features, tol)
    for f in features:  # attach example interaction from the reference complex
        f["example_interaction"] = next(
            (f"{r['kind']}: {r['contacts']}" for r in ref["feats"]
             if r["family"] == f["family"]
             and np.linalg.norm(r["xyz"] - np.array([f["x"], f["y"], f["z"]])) <= 2 * tol), "")

    exclusion = exclusion_spheres(ref["coords"], ref_lig, excl_shell, excl_radius)
    sdf = "".join(Chem.MolToMolBlock(a["mol"]) + "$$$$\n" for a in kept)
    return {
        "type": "target_based",
        "features": features,
        "exclusion": exclusion,
        "meta": {
            "reference": ref["label"], "complexes_used": [a["label"] for a in kept],
            "alignment": align_info, "tolerance": tol, "min_support": min_support,
            "notes": notes,
            "ligand_prep": {a["label"]: a["ligand_note"] for a in kept},
            "method": "interacting ligand features (distance rules), CA superposition on "
                      "the pocket, consensus clustering, exclusion spheres from reference pocket",
        },
        "aligned_sdf": sdf,
        "n_aligned": len(kept),
        "pocket_pdb": pocket_pdb(ref, ref_lig),
    }


# ======================================================================
# 3. Screening / validation (works for both model types)
# ======================================================================
def screen(model, smiles, names, tol=None, n_confs=30, allow_missing=0,
           use_exclusion=True, progress=None):
    feats = [(f["family"], np.array([f["x"], f["y"], f["z"]])) for f in model["features"]]
    n_model = len(feats)
    tol = tol or model["meta"]["tolerance"]
    excl = None
    if use_exclusion and model.get("exclusion"):
        excl = np.array([[e["x"], e["y"], e["z"]] for e in model["exclusion"]])
    excl_r = model["exclusion"][0]["radius"] if model.get("exclusion") else 1.0

    rows = []
    for k, (s, n) in enumerate(zip(smiles, names)):
        if progress:
            progress(k / max(len(smiles), 1), f"Screening {n}")
        emb = embed(s, n_confs)
        best = None
        if emb is not None:
            mol, cids, _ = emb
            for cid in cids:
                xyz = mol.GetConformer(cid).GetPositions()
                m = geo.match_model(feats, mol_features(mol, cid), tol, allow_missing,
                                    excl, excl_r, xyz)
                if m and (best is None or (m["n_matched"], -m["rmsd"]) > (best["n_matched"], -best["rmsd"])):
                    best = m
        n_matched = best["n_matched"] if best else 0
        rmsd = best["rmsd"] if best else np.nan
        score = n_matched - min(rmsd, 9.99) / 10 if best else -1.0
        rows.append({"name": n, "smiles": s, "n_matched": n_matched,
                     "fit_rmsd": None if best is None else round(rmsd, 3),
                     "score": round(score, 4), "full_match": n_matched >= n_model})
    return pd.DataFrame(rows)


def evaluate(df, y_true):
    y = np.asarray(y_true)
    out = {"n_active": int(y.sum()), "n_inactive": int((1 - y).sum())}
    if out["n_active"] and out["n_inactive"]:
        out["ROC-AUC"] = round(float(roc_auc_score(y, df["score"])), 3)
        fpr, tpr, _ = roc_curve(y, df["score"])
        out["roc"] = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        out["EF 10%"] = round(geo.enrichment_factor(y, df["score"], 0.10), 2)
        out["EF 5%"] = round(geo.enrichment_factor(y, df["score"], 0.05), 2)
        full = df["full_match"].values
        out["Actives fully matched"] = round(float(full[y == 1].mean()), 3)
        out["Inactives fully matched"] = round(float(full[y == 0].mean()), 3)
    return out


# ======================================================================
# 4. Export
# ======================================================================
def model_to_json(model):
    keep = {k: model[k] for k in ("type", "features", "exclusion", "meta")}
    return json.dumps(keep, indent=2, default=str)


def model_to_pymol(model):
    """PyMOL script: spheres for features and exclusion volumes (not tested in PyMOL)."""
    pymol_colors = {"blue": "blue", "red": "red", "green": "green",
                    "orange": "orange", "cyan": "cyan", "magenta": "magenta"}
    lines = ["# Pharmacophore model - PyMOL script", "bg_color white"]
    for i, f in enumerate(model["features"], 1):
        obj = f"ph_{i}_{f['family']}"
        lines += [
            f"pseudoatom {obj}, pos=[{f['x']}, {f['y']}, {f['z']}], vdw={f['radius']}",
            f"show spheres, {obj}",
            f"color {pymol_colors[FEATURE_COLORS[f['family']]]}, {obj}",
            f"set sphere_transparency, 0.5, {obj}",
        ]
    for e in model.get("exclusion", []):
        lines.append(f"pseudoatom excl, pos=[{e['x']}, {e['y']}, {e['z']}], vdw={e['radius']}")
    if model.get("exclusion"):
        lines += ["show spheres, excl", "color gray60, excl", "set sphere_transparency, 0.7, excl"]
    return "\n".join(lines) + "\n"
