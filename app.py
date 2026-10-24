
import io
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem, RDConfig
from rdkit.Chem import (
    ChemicalFeatures,
    Descriptors,
    rdMolDescriptors,
)
from Bio.PDB import PDBParser
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(
    page_title="General Pharmacophore Platform",
    layout="wide",
)

st.title("General Pharmacophore Modelling Platform")
st.write(
    "A research prototype for target-based and "
    "ligand-based pharmacophore analysis."
)

st.warning(
    "A generated hypothesis is not automatically validated. "
    "Reliable structures, activity data and independent "
    "testing are needed."
)

FEATURES = [
    "Donor",
    "Acceptor",
    "Hydrophobe",
    "Aromatic",
    "PosIonizable",
    "NegIonizable",
]

factory = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)


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
        Paragraph(
            "Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M"),
            styles["Normal"],
        ),
        Spacer(1, 15),
    ]

    for heading, body in sections:
        story.append(Paragraph(heading, styles["Heading2"]))
        safe_body = (
            str(body)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        story.append(Paragraph(safe_body, styles["BodyText"]))
        story.append(Spacer(1, 10))

    for heading, dataframe in (tables or []):
        story.append(Paragraph(heading, styles["Heading2"]))
        if dataframe is not None and not dataframe.empty:
            view = dataframe.iloc[:35, :7].fillna("")
            data = [list(map(str, view.columns))]
            data.extend(view.astype(str).values.tolist())

            table = Table(data, repeatRows=1)
            table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ])
            )
            story.append(table)
            story.append(Spacer(1, 10))

    doc.build(story)
    return buffer.getvalue()


mode = st.sidebar.radio(
    "Choose modelling workflow",
    ["Target-based", "Ligand-based"],
)

if mode == "Target-based":
    st.header("Target-based feature analysis")

    protein_file = st.file_uploader(
        "Upload protein PDB",
        type=["pdb", "ent"],
    )
    ligand_file = st.file_uploader(
        "Upload bound ligand SDF",
        type=["sdf", "sd"],
    )

    st.info(
        "The ligand must have 3D coordinates from the same "
        "protein-ligand complex as the protein PDB."
    )

    if st.button("Analyse target"):
        if not protein_file or not ligand_file:
            st.error("Upload both files.")
            st.stop()

        supplier = Chem.ForwardSDMolSupplier(
            io.BytesIO(ligand_file.getvalue()),
            removeHs=False,
        )
        ligand = next(
            (mol for mol in supplier if mol is not None),
            None,
        )

        if ligand is None or ligand.GetNumConformers() == 0:
            st.error(
                "Could not read a ligand with 3D coordinates. "
                "Please upload a valid complex-derived SDF."
            )
            st.stop()

        protein_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".pdb", delete=False
            ) as tmp:
                tmp.write(protein_file.getvalue())
                protein_path = tmp.name

            structure = PDBParser(QUIET=True).get_structure(
                "protein", protein_path
            )

            atoms = [
                atom
                for atom in structure.get_atoms()
                if atom.get_parent().id[0] == " "
                and atom.element.upper() not in ["H", "D"]
            ]
        finally:
            if protein_path and os.path.exists(protein_path):
                os.unlink(protein_path)

        if not atoms:
            st.error("No protein atoms were found in the PDB.")
            st.stop()

        rows = []
        for feature in factory.GetFeaturesForMol(ligand):
            family = feature.GetFamily()
            if family not in FEATURES:
                continue

            point = np.array(feature.GetPos())
            distances = [
                np.linalg.norm(atom.coord - point)
                for atom in atoms
            ]
            nearest_index = int(np.argmin(distances))

            rows.append({
                "Feature": family,
                "X": round(float(point[0]), 3),
                "Y": round(float(point[1]), 3),
                "Z": round(float(point[2]), 3),
                "Nearest protein atom": atoms[nearest_index].get_name(),
                "Nearest distance (A)": round(
                    float(distances[nearest_index]), 3
                ),
            })

        results = pd.DataFrame(rows)
        st.subheader("Detected pharmacophore feature points")
        st.dataframe(results, use_container_width=True)

        st.download_button(
            "Download feature table CSV",
            results.to_csv(index=False).encode("utf-8"),
            "target_features.csv",
            "text/csv",
        )

        pdf = make_pdf(
            "Target-based Feature Analysis Report",
            [
                ("Protein", protein_file.name),
                ("Ligand", ligand_file.name),
                ("Method", "RDKit ligand feature detection and "
                           "nearest protein heavy-atom distance analysis."),
                ("Limitation", "Proximity alone does not prove a specific "
                               "interaction or validate a pharmacophore."),
            ],
            [("Feature results", results)],
        )

        st.download_button(
            "Download PDF report",
            pdf,
            "target_report.pdf",
            "application/pdf",
        )

else:
    st.header("Ligand-based activity modelling")

    csv_file = st.file_uploader(
        "Upload CSV with smiles and activity columns",
        type=["csv"],
    )

    test_fraction = st.slider(
        "Held-out test fraction",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
    )

    if st.button("Train and evaluate"):
        if csv_file is None:
            st.error("Upload a CSV dataset.")
            st.stop()

        raw = pd.read_csv(csv_file)

        if not {"smiles", "activity"}.issubset(raw.columns):
            st.error(
                "Your CSV needs columns named smiles and activity."
            )
            st.stop()

        records = []
        for _, row in raw.iterrows():
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is None or pd.isna(row["activity"]):
                continue

            records.append({
                "smiles": str(row["smiles"]),
                "activity": str(row["activity"]).strip(),
                **get_features(mol),
            })

        data = pd.DataFrame(records)

        if (
            data.empty
            or data["activity"].nunique() != 2
            or data["activity"].value_counts().min() < 3
        ):
            st.error(
                "Provide valid SMILES and exactly two activity classes, "
                "with at least 3 compounds per class. More data is better."
            )
            st.stop()

        columns = FEATURES + [
            "MolWt", "LogP", "TPSA", "HBD", "HBA"
        ]
        X = data[columns]
        y = data["activity"]

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_fraction,
                random_state=2026,
                stratify=y,
            )
        except ValueError as error:
            st.error(f"Unable to split the dataset: {error}")
            st.stop()

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=2026,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        positive = sorted(y.unique())[-1]
        metrics = {
            "Training compounds": len(X_train),
            "Test compounds": len(X_test),
            "Accuracy": round(accuracy_score(y_test, predictions), 4),
            "Precision": round(
                precision_score(
                    y_test, predictions,
                    pos_label=positive, zero_division=0
                ), 4
            ),
            "Recall": round(
                recall_score(
                    y_test, predictions,
                    pos_label=positive, zero_division=0
                ), 4
            ),
            "F1 score": round(
                f1_score(
                    y_test, predictions,
                    pos_label=positive, zero_division=0
                ), 4
            ),
        }

        if y_test.nunique() == 2:
            probabilities = model.predict_proba(X_test)[
                :, list(model.classes_).index(positive)
            ]
            metrics["ROC-AUC"] = round(
                roc_auc_score(
                    (y_test == positive).astype(int),
                    probabilities,
                ), 4
            )

        metric_table = pd.DataFrame(
            list(metrics.items()),
            columns=["Metric", "Result"],
        )
        st.subheader("Held-out test results")
        st.dataframe(metric_table, use_container_width=True)

        predictions_table = pd.DataFrame({
            "SMILES": data.loc[X_test.index, "smiles"].values,
            "Observed": y_test.values,
            "Predicted": predictions,
        })

        st.download_button(
            "Download test predictions CSV",
            predictions_table.to_csv(index=False).encode("utf-8"),
            "test_predictions.csv",
            "text/csv",
        )

        pdf = make_pdf(
            "Ligand-based Modelling Report",
            [
                ("Dataset", csv_file.name),
                ("Method", "RDKit 2D pharmacophore feature counts and "
                           "physicochemical descriptors with Random Forest."),
                ("Split", f"Training: {len(X_train)}; "
                          f"independent held-out test: {len(X_test)}."),
                ("Metrics", "; ".join(
                    f"{key}: {value}" for key, value in metrics.items()
                )),
                ("Limitation", "This is a baseline 2D feature classifier, "
                               "not a full 3D common-feature pharmacophore. "
                               "External validation and careful dataset "
                               "curation are still needed."),
            ],
            [
                ("Test metrics", metric_table),
                ("Held-out predictions", predictions_table),
            ],
        )

        st.download_button(
            "Download complete PDF report",
            pdf,
            "ligand_based_report.pdf",
            "application/pdf",
        )
