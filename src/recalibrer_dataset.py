"""Recale le dataset synthétique sur les coûts réels de l'utilisateur (règles de l'estimation rapide d'app.py).

Pour chaque ligne et chaque lot :  nouveau montant = montant des règles Quotora × écart propre à la ligne,
où « l'écart » est le rapport entre la ligne et la médiane de son groupe (même standing, même tranche d'étages)
dans le dataset d'origine. On garde ainsi les effets du sol, de la saison, de l'accès, du type de fondation… tout
en recentrant les niveaux de prix sur les projets réels. Ajoute aussi des lignes « Moyen standing » et « Luxe »,
absentes du dataset d'origine.

Usage :  python recalibrer_dataset.py dataset_quotora_final.csv dataset_quotora_recalibre.csv
(app.py doit être dans le même dossier : les règles et les prix en sont lus directement.)
"""
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LOTS = ["lot1_terrassement", "lot2_fondations", "lot3_elevation", "lot4_charpente_couverture", "lot5_enduits_chape_etancheite",
        "lot6_menuiseries", "lot7_plomberie", "lot8_electricite", "lot9_carrelage_faux_plafond_peinture", "lot10_vrd_exterieurs"]
BASE = ["emprise", "emprise", "construit", "emprise", "construit", "construit", "construit", "construit", "construit", "construit"]


def load_rules():
    """Charge les fonctions de calcul de l'estimation rapide depuis app.py, sans lancer l'interface Streamlit."""
    ns = {"math": math, "np": np, "pd": pd}
    for node in ast.parse((HERE / "app.py").read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.Assign, ast.FunctionDef)):
            try: exec(compile(ast.Module([node], []), "app.py", "exec"), ns)
            except Exception: pass
    return ns


def level_bucket(n):
    return 0 if n == 0 else 1 if n == 1 else 2 if n == 2 else 3 if n <= 5 else 4 if n <= 10 else 5


def main(src, dst, seed=42):
    R = load_rules()
    d = pd.read_csv(src)
    rng = np.random.default_rng(seed)
    # Lignes Moyen standing et Luxe (absentes du dataset) : copies de lignes Standard / Haut standing
    moy = d[d.standing.isin(["Standard", "Haut standing"])].sample(4000, random_state=seed).assign(standing="Moyen standing")
    lux = d[d.standing == "Haut standing"].sample(min(2000, int((d.standing == "Haut standing").sum())), random_state=seed + 1).assign(standing="Luxe")
    d = pd.concat([d, moy, lux], ignore_index=True)
    d["id"] = np.arange(1, len(d) + 1)

    surf = [json.loads(s) if isinstance(s, str) else None for s in d.surfaces_niveaux_m2_json]
    emprise = d.surface_rdc_m2.to_numpy(float); built = d.surface_totale_construite_m2.to_numpy(float)
    nlev = d.niveaux_au_dessus_rdc.to_numpy(int); bucket = np.array([level_bucket(n) for n in nlev])

    # Écart de chaque ligne à la médiane de son groupe (standing d'origine × tranche d'étages), par lot
    dev = np.ones((len(d), len(LOTS)))
    orig_std = d.standing.replace({"Moyen standing": "Standard", "Luxe": "Haut standing"})
    for j, (lot, base) in enumerate(zip(LOTS, BASE)):
        per = d[lot].to_numpy(float) / (emprise if base == "emprise" else built)
        g = pd.Series(per).groupby([orig_std.to_numpy(), bucket]).transform("median").to_numpy()
        dev[:, j] = np.clip(np.where(g > 0, per / g, 1.0), 0.6, 1.6)

    new = np.zeros((len(d), len(LOTS)))
    for i, r in enumerate(d.itertuples(index=False)):
        s = surf[i] if surf[i] and len(surf[i]) == r.niveaux_au_dessus_rdc + 1 else None
        tg = R["quick_lot_targets"](r.region, float(r.surface_rdc_m2), int(r.niveaux_au_dessus_rdc), r.standing, r.type_mur, r.type_toiture, s)
        for k in range(9):
            new[i, k] = tg[k][0] * dev[i, k]
    new[:, 9] = R["QUICK_VRD_RATE"] * new[:, :9].sum(axis=1) * dev[:, 9]     # VRD : 2,5 % des autres lots
    new *= rng.lognormal(0.0, 0.05, new.shape)                              # léger bruit (chantiers réels jamais identiques)
    for j, lot in enumerate(LOTS): d[lot] = np.round(new[:, j])
    # Colonnes de synthèse recalculées (le lot 11 planchers et les finitions de plancher restent ceux d'origine)
    d["cout_total"] = d[LOTS].sum(axis=1) + d["lot11_planchers_dalles"]
    d["cout_m2"] = d["cout_total"] / d["surface_totale_construite_m2"]
    d.to_csv(dst, index=False)
    print(f"✅ {dst} : {len(d)} lignes (dont {len(moy)} Moyen standing et {len(lux)} Luxe ajoutées).")
    print(d.groupby("standing")["cout_m2"].median().round(0).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset_quotora_final.csv",
         sys.argv[2] if len(sys.argv) > 2 else "dataset_quotora_recalibre.csv")
