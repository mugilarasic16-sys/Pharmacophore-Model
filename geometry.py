"""Pure-geometry core of the pharmacophore pipeline (numpy/scipy only).

Features are passed around as (family, xyz) tuples, xyz = np.ndarray(3,).
"""
import itertools

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree

# lower number = more specific, preferred when ranking equally-supported features
SPECIFICITY = {
    "PosIonizable": 0,
    "NegIonizable": 0,
    "Donor": 1,
    "Acceptor": 1,
    "Aromatic": 2,
    "Hydrophobe": 3,
}


# ----------------------------------------------------------------------
# Rigid-body fitting
# ----------------------------------------------------------------------
def kabsch(P, Q):
    """Best rigid fit of P onto Q (proper rotation only).

    Returns rmsd, R, Pc, Qc with  fitted = (X - Pc) @ R.T + Qc
    """
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    fitted = (P - Pc) @ R.T + Qc
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - Q) ** 2, axis=1))))
    return rmsd, R, Pc, Qc


def apply_transform(X, R, Pc, Qc):
    return (np.asarray(X, float) - Pc) @ R.T + Qc


# ----------------------------------------------------------------------
# Consensus (common-feature) clustering
# ----------------------------------------------------------------------
def consensus_features(points, tol, min_support, n_sources):
    """Cluster same-family points from many ligands/complexes.

    points: list of dicts {family, xyz, source}
    A cluster is kept when features from >= min_support * n_sources distinct
    sources fall inside it. Complete linkage with cut-off 2*tol guarantees the
    cluster diameter <= 2*tol, i.e. every member lies within tol of the centroid
    (approximately).
    """
    kept = []
    for family in sorted({p["family"] for p in points}):
        pts = [p for p in points if p["family"] == family]
        X = np.array([p["xyz"] for p in pts], float)
        if len(pts) == 1:
            labels = np.array([1])
        else:
            labels = fcluster(linkage(X, method="complete"), t=2 * tol, criterion="distance")
        for lab in np.unique(labels):
            idx = np.where(labels == lab)[0]
            sources = {pts[i]["source"] for i in idx}
            support = len(sources) / n_sources
            if support + 1e-9 >= min_support:
                centre = X[idx].mean(0)
                spread = float(np.max(np.linalg.norm(X[idx] - centre, axis=1)))
                kept.append(
                    {
                        "family": family,
                        "xyz": centre,
                        "support": round(support, 3),
                        "n_sources": len(sources),
                        "spread": round(spread, 3),
                    }
                )
    kept.sort(key=lambda f: (-f["support"], SPECIFICITY.get(f["family"], 9), f["spread"]))
    return kept


def feature_overlap(ref_feats, feats, tol):
    """How many reference features have a same-family feature within tol."""
    n = 0
    for fam, xyz in ref_feats:
        for fam2, xyz2 in feats:
            if fam == fam2 and np.linalg.norm(xyz - xyz2) <= tol:
                n += 1
                break
    return n


# ----------------------------------------------------------------------
# Matching a molecule conformer against a pharmacophore model
# ----------------------------------------------------------------------
def _assignments(model_feats, mol_feats, dtol):
    """Backtracking enumeration of feature assignments whose inter-feature
    distances agree with the model within dtol."""
    n = len(model_feats)
    mxyz = np.array([f[1] for f in model_feats])
    Dm = np.linalg.norm(mxyz[:, None] - mxyz[None], axis=2)
    lxyz = np.array([f[1] for f in mol_feats]) if mol_feats else np.zeros((0, 3))
    Dl = np.linalg.norm(lxyz[:, None] - lxyz[None], axis=2) if len(lxyz) else None
    cands = [
        [j for j, f in enumerate(mol_feats) if f[0] == fam] for fam, _ in model_feats
    ]
    if any(not c for c in cands):
        return

    def rec(i, chosen):
        if i == n:
            yield tuple(chosen)
            return
        for j in cands[i]:
            if j in chosen:
                continue
            if all(abs(Dl[j, chosen[k]] - Dm[i, k]) <= dtol for k in range(i)):
                chosen.append(j)
                yield from rec(i + 1, chosen)
                chosen.pop()

    yield from rec(0, [])


def match_model(model_feats, mol_feats, tol, allow_missing=0,
                excl_centres=None, excl_radius=1.0, mol_xyz=None):
    """Best match of a conformer's features to the model.

    Returns dict(n_matched, rmsd, R, Pc, Qc) or None when nothing fits.
    Exclusion spheres are checked only when mol_xyz (heavy-atom coordinates of
    the conformer) is given: the fitted pose must not place a heavy atom within
    excl_radius of an exclusion centre.
    """
    n = len(model_feats)
    if n == 0:
        return None
    tree = cKDTree(excl_centres) if excl_centres is not None and len(excl_centres) else None

    for size in range(n, max(n - allow_missing, 1) - 1, -1):
        best = None
        for subset in itertools.combinations(range(n), size):
            sub_model = [model_feats[i] for i in subset]
            Q = np.array([f[1] for f in sub_model])
            hits = []
            for combo in _assignments(sub_model, mol_feats, dtol=2 * tol):
                P = np.array([mol_feats[j][1] for j in combo])
                rmsd, R, Pc, Qc = kabsch(P, Q)
                if rmsd <= tol:
                    hits.append((rmsd, R, Pc, Qc))
            hits.sort(key=lambda h: h[0])
            for rmsd, R, Pc, Qc in hits:
                if tree is not None and mol_xyz is not None:
                    fitted = apply_transform(mol_xyz, R, Pc, Qc)
                    if tree.query(fitted)[0].min() < excl_radius:
                        continue
                if best is None or rmsd < best["rmsd"]:
                    best = {"n_matched": size, "rmsd": rmsd, "R": R, "Pc": Pc, "Qc": Qc}
                break
        if best is not None:
            return best
    return None


# ----------------------------------------------------------------------
# Screening metrics
# ----------------------------------------------------------------------
def enrichment_factor(y_true, scores, fraction=0.1):
    y = np.asarray(y_true)
    s = np.asarray(scores, float)
    n_top = max(1, int(round(len(y) * fraction)))
    order = np.argsort(-s, kind="stable")[:n_top]
    return float((y[order].sum() / n_top) / (y.sum() / len(y))) if y.sum() else float("nan")
