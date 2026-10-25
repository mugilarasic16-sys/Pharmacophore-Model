import numpy as np
from scipy.spatial.transform import Rotation
from geometry import *

rng = np.random.default_rng(0)

# 1. Kabsch recovers a random rigid transform (and never returns a reflection)
P = rng.normal(size=(6, 3)) * 3
Rt = Rotation.random(random_state=1).as_matrix()
Q = P @ Rt.T + np.array([5.0, -2.0, 1.0])
rmsd, R, Pc, Qc = kabsch(P, Q)
assert rmsd < 1e-8 and np.linalg.det(R) > 0, (rmsd, np.linalg.det(R))
assert np.allclose(apply_transform(P, R, Pc, Qc), Q)
# mirror image must NOT fit perfectly
Pm = P * np.array([1, 1, -1])
assert kabsch(Pm, Q)[0] > 0.5
print("kabsch OK")

# 2. consensus clustering: 10 ligands, a donor at A in 9/10, an acceptor at B in 4/10, noise
A, B = np.array([0., 0, 0]), np.array([5., 0, 0])
pts = []
for i in range(10):
    if i < 9: pts.append({"family": "Donor", "xyz": A + rng.normal(scale=0.25, size=3), "source": f"L{i}"})
    if i < 4: pts.append({"family": "Acceptor", "xyz": B + rng.normal(scale=0.25, size=3), "source": f"L{i}"})
    pts.append({"family": "Hydrophobe", "xyz": rng.uniform(-30, 30, 3), "source": f"L{i}"})
cons = consensus_features(pts, tol=1.0, min_support=0.7, n_sources=10)
fams = [c["family"] for c in cons]
assert fams == ["Donor"], fams
assert abs(cons[0]["support"] - 0.9) < 1e-9 and np.linalg.norm(cons[0]["xyz"] - A) < 0.4
cons2 = consensus_features(pts, tol=1.0, min_support=0.4, n_sources=10)
assert sorted(c["family"] for c in cons2) == ["Acceptor", "Donor"]
print("consensus OK")

# 3. matching: model of 4 features, molecule = rotated/translated copy + decoy features
model = [("Donor", np.array([0, 0, 0.])), ("Acceptor", np.array([4, 0, 0.])),
         ("Aromatic", np.array([2, 3, 0.])), ("Hydrophobe", np.array([2, -3, 1.]))]
Rr = Rotation.random(random_state=7).as_matrix()
mol = [(f, x @ Rr.T + np.array([10., 10, 10])) for f, x in model]
mol += [("Donor", rng.uniform(-8, 8, 3) + 10) for _ in range(5)]
rng.shuffle(mol)
m = match_model(model, mol, tol=0.5)
assert m and m["n_matched"] == 4 and m["rmsd"] < 1e-6, m
# distorted molecule (acceptor moved 3 A) -> no full match, but 3/4 with allow_missing
bad = [(f, x.copy()) for f, x in mol]
for k, (f, x) in enumerate(bad):
    if f == "Acceptor": bad[k] = (f, x + 3.0)
assert match_model(model, bad, tol=0.5) is None
m2 = match_model(model, bad, tol=0.5, allow_missing=1)
assert m2 and m2["n_matched"] == 3
print("matching OK")

# 4. exclusion volume rejects a clashing pose
mol_xyz = rng.normal(size=(8, 3)) + 10
fitted = apply_transform(mol_xyz, m["R"], m["Pc"], m["Qc"])
excl_clash = fitted[:1] + 0.1          # sits on top of a ligand atom in the fitted pose
excl_free = np.array([[100., 100, 100]])
assert match_model(model, mol, 0.5, excl_centres=excl_free, mol_xyz=mol_xyz) is not None
assert match_model(model, mol, 0.5, excl_centres=excl_clash, excl_radius=1.0, mol_xyz=mol_xyz) is None
print("exclusion OK")

# 5. EF sanity: perfect ranking of 10 actives among 100 -> EF10% = 10
y = np.array([1] * 10 + [0] * 90); s = np.linspace(1, 0, 100)
assert abs(enrichment_factor(y, s, 0.1) - 10.0) < 1e-9
print("EF OK")
