import io

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

import pharmacore as pc
import rcsb

try:
    import py3Dmol
except ImportError:
    py3Dmol = None

st.set_page_config(page_title="Pharmacophore Generator", layout="wide")
st.title("Pharmacophore Generator")
st.caption(
    "Builds ligand-based (common-feature) and target-based (structure-based, "
    "multi-complex consensus) pharmacophore MODELS, then validates them."
)
st.warning(
    "A model is a hypothesis. Only the validation section tells you whether it "
    "discriminates actives from inactives, and small datasets give noisy numbers."
)


# ----------------------------------------------------------------------
# shared rendering
# ----------------------------------------------------------------------
def show_3d(model):
    if py3Dmol is None:
        st.info("Add py3Dmol to requirements.txt for the 3D view.")
        return
    view = py3Dmol.view(width=900, height=600)
    n_models = 0
    if model.get("pocket_pdb"):
        view.addModel(model["pocket_pdb"], "pdb")
        view.setStyle({"model": 0}, {"stick": {"radius": 0.07, "colorscheme": "whiteCarbon"}})
        n_models = 1
    view.addModels(model["aligned_sdf"], "sdf")
    for i in range(n_models, n_models + model["n_aligned"]):
        view.setStyle({"model": i}, {"stick": {"radius": 0.15}})
    for f in model["features"]:
        view.addSphere({"center": {"x": f["x"], "y": f["y"], "z": f["z"]},
                        "radius": f["radius"], "color": pc.FEATURE_COLORS[f["family"]],
                        "alpha": 0.6})
    for e in model.get("exclusion", []):
        view.addSphere({"center": {"x": e["x"], "y": e["y"], "z": e["z"]},
                        "radius": e["radius"] * 0.6, "color": "gray", "alpha": 0.25})
    view.zoomTo()
    components.html(view._make_html(), height=620)
    st.caption("Spheres: " + ", ".join(f"{k} = {v}" for k, v in pc.FEATURE_COLORS.items())
               + ("; grey = exclusion volumes" if model.get("exclusion") else ""))


def show_model(model, key):
    feats = pd.DataFrame(model["features"])
    if feats.empty:
        st.error("No feature reached the support threshold. Lower the minimum "
                 "support, raise the tolerance, or check that the ligands share a binding mode.")
        return False
    st.subheader("Pharmacophore model")
    st.dataframe(feats, use_container_width=True)
    st.write(f"{len(feats)} features" +
             (f", {len(model['exclusion'])} exclusion volumes" if model["exclusion"] else "")
             + f"; built from {model['n_aligned']} ligand(s).")
    for note in model["meta"].get("skipped", []) + model["meta"].get("notes", []):
        st.caption("Note: " + note)
    if model["type"] == "target_based":
        approx = [k for k, v in model["meta"]["ligand_prep"].items() if v == "approximate"]
        if approx:
            st.warning("Bond orders could not be restored from the RCSB template for: "
                       + ", ".join(approx) + ". Their aromatic/ionisable features may be incomplete.")
    show_3d(model)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Model JSON", pc.model_to_json(model), f"{key}_model.json", "application/json")
    c2.download_button("PyMOL script", pc.model_to_pymol(model), f"{key}_model.pml", "text/plain")
    c3.download_button("Aligned ligands SDF", model["aligned_sdf"], f"{key}_aligned.sdf",
                       "chemical/x-mdl-sdfile")
    return True


def show_validation(result):
    m = result["metrics"]
    if "ROC-AUC" not in m:
        st.warning("Need both actives and inactives to compute metrics.")
    else:
        cols = st.columns(5)
        for col, name in zip(cols, ["ROC-AUC", "EF 5%", "EF 10%",
                                    "Actives fully matched", "Inactives fully matched"]):
            col.metric(name, m[name])
        st.line_chart(m["roc"].set_index("FPR")["TPR"])
        st.caption(f"{m['n_active']} actives vs {m['n_inactive']} inactives. "
                   "AUC 0.5 = random. Be sceptical of high values from tiny sets.")
    st.dataframe(result["table"], use_container_width=True)
    st.download_button("Validation table CSV", result["table"].to_csv(index=False).encode(),
                       "validation.csv", "text/csv")


def read_activity_csv(file):
    raw = pd.read_csv(io.BytesIO(file.getvalue()))
    if not {"smiles", "activity"}.issubset(raw.columns):
        raise ValueError("CSV needs columns named smiles and activity (optional: name).")
    raw = raw.dropna(subset=["smiles", "activity"]).copy()
    raw["activity"] = raw["activity"].astype(str).str.strip()
    if "name" not in raw.columns:
        raw["name"] = [f"cmpd_{i}" for i in range(len(raw))]
    raw["name"] = raw["name"].astype(str)
    return raw.drop_duplicates("smiles").reset_index(drop=True)


def validation_panel(model, key):
    st.subheader("Validate the model")
    st.caption("Upload actives AND inactives that were NOT used to build the model "
               "(CSV: smiles, activity, optional name).")
    vfile = st.file_uploader("Validation CSV", type=["csv"], key=f"v_{key}")
    c1, c2, c3 = st.columns(3)
    n_confs = c1.slider("Conformers per compound", 10, 100, 30, key=f"vc_{key}")
    allow_missing = c2.selectbox("Allowed missing features", [0, 1], key=f"vm_{key}")
    use_excl = c3.checkbox("Apply exclusion volumes", value=bool(model["exclusion"]),
                           disabled=not model["exclusion"], key=f"ve_{key}")
    if vfile is None:
        return
    try:
        df = read_activity_csv(vfile)
    except Exception as e:
        st.error(str(e))
        return
    classes = sorted(df["activity"].unique())
    if len(classes) != 2:
        st.error("The activity column must contain exactly two classes.")
        return
    active = st.selectbox("Active class", classes, key=f"va_{key}")
    if st.button("Run validation", key=f"vb_{key}"):
        bar = st.progress(0.0)
        table = pc.screen(model, df["smiles"].tolist(), df["name"].tolist(),
                          n_confs=n_confs, allow_missing=allow_missing,
                          use_exclusion=use_excl,
                          progress=lambda f, t: bar.progress(min(f, 1.0), t))
        y = (df["activity"] == active).astype(int).values
        table["observed"] = np.where(y == 1, "active", "inactive")
        st.session_state[f"val_{key}"] = {"table": table, "metrics": pc.evaluate(table, y)}
        bar.empty()
    if st.session_state.get(f"val_{key}"):
        show_validation(st.session_state[f"val_{key}"])


# ----------------------------------------------------------------------
# Ligand-based
# ----------------------------------------------------------------------
def ligand_based_mode():
    st.header("Ligand-based pharmacophore")
    st.caption("Common-feature model from aligned ACTIVE ligands. Held-out validation "
               "uses a scaffold split so the model is tested on new chemotypes.")
    file = st.file_uploader("CSV: smiles, activity (+ optional name)", type=["csv"], key="lb_csv")
    c1, c2, c3, c4 = st.columns(4)
    n_confs = c1.slider("Conformers per active", 10, 200, 50)
    tol = c2.slider("Feature tolerance (A)", 0.5, 2.0, 1.0, 0.1)
    min_support = c3.slider("Min. fraction of actives sharing a feature", 0.3, 1.0, 0.7, 0.05)
    max_features = c4.slider("Max. features", 3, 8, 5)
    c5, c6 = st.columns(2)
    max_train = c5.slider("Max. actives used to build model", 5, 40, 20)
    test_frac = c6.slider("Held-out fraction of actives (scaffold split)", 0.1, 0.5, 0.25, 0.05)

    if file is None:
        return
    try:
        df = read_activity_csv(file)
    except Exception as e:
        st.error(str(e))
        return
    classes = sorted(df["activity"].unique())
    if len(classes) != 2:
        st.error("The activity column must contain exactly two classes.")
        return
    active = st.selectbox("Which class is ACTIVE?", classes)
    actives = df[df["activity"] == active].reset_index(drop=True)
    inactives = df[df["activity"] != active].reset_index(drop=True)
    st.write(f"{len(actives)} actives, {len(inactives)} inactives.")

    if st.button("Build ligand-based pharmacophore"):
        if len(actives) < 6:
            st.error("Need at least 6 actives (3+ to build, a few held out to test).")
            return
        tr, te = pc.scaffold_split(actives["smiles"].tolist(), test_frac)
        train = actives.iloc[tr].head(max_train)
        test = actives.iloc[te]
        bar = st.progress(0.0)
        try:
            model = pc.build_ligand_based(
                train["smiles"].tolist(), train["name"].tolist(), n_confs, tol,
                min_support, max_features,
                progress=lambda f, t: bar.progress(min(f, 1.0), t))
        except Exception as e:
            bar.empty()
            st.error(f"Model building failed: {e}")
            return
        bar.empty()
        st.session_state["lb_model"] = model
        st.session_state["lb_holdout"] = (test, inactives)
        st.session_state.pop("val_lb", None)

    model = st.session_state.get("lb_model")
    if not model:
        return
    st.info(f"Reference ligand: {model['meta']['reference']}. "
            "Built from the training split only.")
    if not show_model(model, "lb"):
        return

    st.subheader("Held-out validation (automatic)")
    test, inactives = st.session_state["lb_holdout"]
    st.caption(f"{len(test)} held-out actives (new scaffolds) vs up to 100 inactives.")
    if len(test) >= 2 and len(inactives) >= 5 and st.button("Validate on held-out actives + inactives"):
        inact = inactives.head(100)
        smi = test["smiles"].tolist() + inact["smiles"].tolist()
        nam = test["name"].tolist() + inact["name"].tolist()
        y = np.array([1] * len(test) + [0] * len(inact))
        bar = st.progress(0.0)
        table = pc.screen(model, smi, nam, n_confs=30,
                          progress=lambda f, t: bar.progress(min(f, 1.0), t))
        table["observed"] = np.where(y == 1, "active", "inactive")
        bar.empty()
        st.session_state["val_lb"] = {"table": table, "metrics": pc.evaluate(table, y)}
    elif len(test) < 2 or len(inactives) < 5:
        st.warning("Not enough held-out actives or inactives for automatic validation.")
    if st.session_state.get("val_lb"):
        show_validation(st.session_state["val_lb"])


# ----------------------------------------------------------------------
# Target-based
# ----------------------------------------------------------------------
def load_entries(pdb_ids, min_heavy, drop_additives, force_comp):
    items, log = [], []
    for pid in pdb_ids:
        try:
            text, ext = rcsb.fetch_structure(pid)
            struct = rcsb.parse_structure(text, ext)
            info = rcsb.fetch_entry_info(pid)
            model0 = struct[0]
            ligs = rcsb.collect_ligands(model0, set(info.get("comps", {}).keys()) or None,
                                        min_heavy, drop_additives, force_comp or None)
            if not ligs:
                log.append(f"{pid}: no ligand passed the filters (apo or only additives)")
                continue
            items.append({"label": pid, "structure": struct, "model": model0, "entry": ligs[0],
                          "info": info})
        except requests.RequestException as e:
            log.append(f"{pid}: network error ({e})")
        except Exception as e:
            log.append(f"{pid}: {e}")
    return items, log


def target_based_mode():
    st.header("Target-based pharmacophore")
    st.caption("Uses ligand-bound crystal structures. With several complexes of the same "
               "target, they are superimposed and only features shared by enough "
               "complexes are kept.")
    source = st.radio("Input", ["PDB IDs", "UniProt accession"], horizontal=True)
    ids = []
    if source == "PDB IDs":
        raw = st.text_area("PDB IDs (comma/space separated; the FIRST is the reference frame)",
                           placeholder="1M17, 4HJO, 3W2S")
        ids = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
    else:
        acc = st.text_input("UniProt accession", placeholder="P00533")
        n_top = st.slider("Number of best-resolution entries to use", 1, 12, 5)
        if acc and st.button("Search RCSB"):
            try:
                st.session_state["uni_ids"] = rcsb.search_pdb_by_uniprot(acc)[:n_top]
                if not st.session_state["uni_ids"]:
                    st.warning("No ligand-bound entries found.")
            except Exception as e:
                st.error(f"Search failed: {e}")
        ids = st.session_state.get("uni_ids", [])
        if ids:
            st.write("Entries:", ", ".join(ids))

    c1, c2, c3 = st.columns(3)
    min_heavy = c1.slider("Min. ligand heavy atoms", 1, 30, 10)
    drop_add = c2.checkbox("Ignore additives, ions, glycans", value=True)
    force_comp = c3.text_input("Force ligand ID (optional, e.g. AQ4)").strip()

    if ids and st.button("Load complexes"):
        with st.spinner("Downloading and detecting ligands..."):
            items, log = load_entries(ids, min_heavy, drop_add, force_comp)
        st.session_state["tb_items"], st.session_state["tb_log"] = items, log
        st.session_state.pop("tb_model", None)

    items = st.session_state.get("tb_items")
    for line in st.session_state.get("tb_log", []):
        st.caption("Note: " + line)
    if not items:
        return

    table = pd.DataFrame([{
        "PDB": it["label"], "Ligand": it["entry"]["comp"], "Chain": it["entry"]["chain"],
        "Heavy atoms": it["entry"]["n_heavy"],
        "Resolution (A)": it["info"].get("resolution"),
        "Title": (it["info"].get("title") or "")[:70]} for it in items])
    st.dataframe(table, use_container_width=True)
    chosen = st.multiselect("Complexes to use (first = reference)", [i["label"] for i in items],
                            default=[i["label"] for i in items])

    p1, p2, p3, p4, p5 = st.columns(5)
    tol = p1.slider("Feature tolerance (A)", 0.5, 2.0, 1.0, 0.1, key="tb_tol")
    min_support = p2.slider("Min. fraction of complexes", 0.3, 1.0, 0.6, 0.05, key="tb_sup")
    max_features = p3.slider("Max. features", 3, 10, 8, key="tb_max")
    shell = p4.slider("Exclusion shell (A)", 3.0, 7.0, 5.0, 0.5)
    excl_r = p5.slider("Exclusion radius (A)", 0.5, 2.0, 1.0, 0.1)

    if st.button("Build target-based pharmacophore"):
        sel = [it for it in items if it["label"] in chosen]
        if not sel:
            st.error("Select at least one complex.")
            return
        bar = st.progress(0.0)
        try:
            model = pc.build_target_based(
                sel, tol, min_support if len(sel) > 1 else 1.0, max_features, shell, excl_r,
                progress=lambda f, t: bar.progress(min(f, 1.0), t))
        except Exception as e:
            bar.empty()
            st.error(f"Model building failed: {e}")
            return
        bar.empty()
        st.session_state["tb_model"] = model
        st.session_state.pop("val_tb", None)

    model = st.session_state.get("tb_model")
    if not model:
        return
    if len(model["meta"]["complexes_used"]) == 1:
        st.info("Single complex: this is that ligand's interaction pharmacophore, not a "
                "consensus. Use several complexes of the same target for a robust model.")
    st.write("Alignment quality:", pd.DataFrame(model["meta"]["alignment"]))
    if show_model(model, "tb"):
        validation_panel(model, "tb")


mode = st.sidebar.radio("Workflow", ["Target-based pharmacophore", "Ligand-based pharmacophore"])
if mode == "Target-based pharmacophore":
    target_based_mode()
else:
    ligand_based_mode()
