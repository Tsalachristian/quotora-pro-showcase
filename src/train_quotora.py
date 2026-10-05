"""Réentraîne le modèle de Quotora à partir de vos projets réels (et, si besoin, du dataset synthétique d'origine).

Exemples (PowerShell, dans le dossier où se trouvent app.py et ce script) :
  python train_quotora.py --reel quotora_projets_reels.xlsx --synthetique dataset_synthetique.csv
  python train_quotora.py --reel quotora_projets_reels.xlsx            # vos projets seuls (10 minimum)

Le modèle produit : quotora_model_final.joblib (l'ancien est sauvegardé en quotora_model_final_ancien.joblib).
Il prédit les lots 2 à 11 (terrassement … VRD), comme le modèle actuel ; l'installation (lot 1) et les planchers
(lot 12) sont calculés par l'application elle-même.
"""
import argparse
import ast
import datetime as dt
import math
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
FEATURES = ["region", "surface_rdc_m2", "niveaux_au_dessus_rdc", "surface_totale_construite_m2", "surface_planchers_intermediaires_m2",
            "surface_dallage_rdc_m2", "surface_dalle_pleine_m2", "surface_dalle_hourdis_m2", "type_plancher_rdc", "type_mur",
            "prix_mur_region_fcfa_m2", "surface_mur_exterieur_m2", "part_facade_vitree_pct", "type_ouverture", "type_toiture",
            "type_plafond", "prix_plafond_reference_fcfa_m2", "type_fondation", "type_sol", "standing", "saison", "acces_chantier",
            "distance_transport_km"]
CATEGORICAL = ["region", "type_plancher_rdc", "type_mur", "type_ouverture", "type_toiture", "type_plafond", "type_fondation",
               "type_sol", "standing", "saison", "acces_chantier"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
DEFAULT_LOT_COLS = ["lot1_terrassement", "lot2_fondations", "lot3_elevation", "lot4_charpente_couverture",
                    "lot5_enduits_chape_etancheite", "lot6_menuiseries", "lot7_plomberie", "lot8_electricite",
                    "lot9_carrelage_faux_plafond_peinture", "lot10_vrd_exterieurs"]   # noms du dataset_quotora_final.csv
# En-têtes de l'onglet « Projets » de quotora_projets_reels.xlsx
H = {"region": "Région", "rdc": "Surface RDC (m²)", "n": "Niveaux au-dessus du RDC",
     "etages": "Surface totale des étages (m²) — vide = RDC × niveaux", "dallage": "Type de dallage RDC",
     "dalle": "Type de dalle des étages", "mur": "Type de mur principal", "toiture": "Type de toiture", "fondation": "Type de fondation",
     "sol": "Type de sol", "standing": "Standing", "ouverture": "Ouverture principale", "plafond": "Plafond",
     "saison": "Saison des travaux", "acces": "Accès chantier", "distance": "Distance transport (km)",
     "murs_ext": "Surface murs extérieurs (m²) — vide = estimée", "vitre": "Part façade vitrée (%)", "nom": "Nom du projet"}
LOT_HEADERS = ["Lot 2 Terrassement (FCFA)", "Lot 3 Fondations (FCFA)", "Lot 4 Élévation — murs + béton armé (FCFA)",
               "Lot 5 Charpente & couverture (FCFA)", "Lot 6 Enduits, chape & étanchéité (FCFA)", "Lot 7 Menuiseries (FCFA)",
               "Lot 8 Plomberie & sanitaires (FCFA)", "Lot 9 Électricité (FCFA)", "Lot 10 Carrelage, plafond & peinture (FCFA)",
               "Lot 11 VRD & extérieurs (FCFA)"]


def catalogs():
    """Prix catalogue repris de app.py (même dossier) pour rester cohérent avec l'application."""
    out = {"WALL_CATALOG": {}, "REGION_MULT": {}, "CEIL_PRICE": {}}
    app = HERE / "app.py"
    if app.exists():
        for node in ast.parse(app.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in out:
                try: out[node.targets[0].id] = ast.literal_eval(node.value)
                except Exception: pass
    else:
        print("⚠️  app.py introuvable à côté du script : prix des murs/plafonds par défaut.")
    wall = {k: v.get("default", 10000) for k, v in out["WALL_CATALOG"].items()}
    return wall, out["REGION_MULT"], out["CEIL_PRICE"]


def txt(v, default):
    return default if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "" else str(v).strip()


def num(v, default=np.nan):
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def load_real(path):
    df = pd.read_excel(path, sheet_name="Projets", header=4)
    missing = [h for h in list(H.values()) + LOT_HEADERS if h not in df.columns]
    if missing:
        sys.exit(f"❌ Colonnes absentes dans {path} : {missing}. Utilisez le modèle quotora_projets_reels.xlsx sans renommer les en-têtes.")
    df = df[pd.to_numeric(df[H["rdc"]], errors="coerce") > 0]
    wall_price, region_mult, ceil_price = catalogs()
    rows, targets, names = [], [], []
    for _, r in df.iterrows():
        rdc = num(r[H["rdc"]]); n = int(num(r[H["n"]], 0))
        etages = num(r[H["etages"]], rdc * n) if n > 0 else 0.0
        levels = [rdc] + ([etages / n] * n if n > 0 else [])
        dalle = txt(r[H["dalle"]], "Dalle à hourdis / corps creux")
        region, mur = txt(r[H["region"]], "Littoral"), txt(r[H["mur"]], "Parpaing 15")
        plafond = txt(r[H["plafond"]], "Staff standard")
        vitre = num(r[H["vitre"]], 0.2); vitre = vitre * 100 if vitre <= 1 else vitre
        murs_ext = num(r[H["murs_ext"]], sum(4.2 * math.sqrt(max(s, 1)) * 3.0 * .82 for s in levels))
        rows.append({
            "region": region, "surface_rdc_m2": rdc, "niveaux_au_dessus_rdc": n, "surface_totale_construite_m2": rdc + etages,
            "surface_planchers_intermediaires_m2": etages, "surface_dallage_rdc_m2": rdc,
            "surface_dalle_pleine_m2": etages if dalle.startswith("Dalle pleine") else 0.0,
            "surface_dalle_hourdis_m2": 0.0 if dalle.startswith("Dalle pleine") else etages,
            "type_plancher_rdc": txt(r[H["dallage"]], "Dallage sur terre-plein"), "type_mur": mur,
            "prix_mur_region_fcfa_m2": wall_price.get(mur, 10000) * region_mult.get(region, 1.0),
            "surface_mur_exterieur_m2": murs_ext, "part_facade_vitree_pct": vitre,
            "type_ouverture": txt(r[H["ouverture"]], "Fenêtre aluminium vitrée"), "type_toiture": txt(r[H["toiture"]], "Charpente métallique + tôle"),
            "type_plafond": plafond, "prix_plafond_reference_fcfa_m2": ceil_price.get(plafond, 0),
            "type_fondation": txt(r[H["fondation"]], "Semelles isolées + longrines"), "type_sol": txt(r[H["sol"]], "Latéritique"),
            "standing": txt(r[H["standing"]], "Standard"), "saison": txt(r[H["saison"]], "Saison sèche"),
            "acces_chantier": txt(r[H["acces"]], "Accès normal"), "distance_transport_km": num(r[H["distance"]], 15.0)})
        targets.append([num(r[h]) for h in LOT_HEADERS])
        names.append(txt(r[H["nom"]], "projet"))
    return pd.DataFrame(rows, columns=FEATURES), np.array(targets, dtype=float), names


def load_old_model(path):
    try:
        art = joblib.load(path)
        return art if isinstance(art, dict) and "pipeline" in art else None
    except Exception as e:
        print(f"⚠️  Ancien modèle illisible ({e}) : les lots manquants des projets réels ne pourront pas être complétés.")
        return None


def build_pipeline(n_rows):
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL), ("num", "passthrough", NUMERIC)])
    gbr = HistGradientBoostingRegressor(max_iter=500 if n_rows >= 1000 else 150, learning_rate=0.06, max_leaf_nodes=31 if n_rows >= 1000 else 7,
                                        min_samples_leaf=20 if n_rows >= 1000 else 2, l2_regularization=0.1, random_state=42)
    model = TransformedTargetRegressor(regressor=MultiOutputRegressor(gbr), func=np.log1p, inverse_func=np.expm1, check_inverse=False)
    return Pipeline([("prep", prep), ("model", model)])


def metrics(y_true, y_pred):
    t, p = y_true.sum(axis=1), y_pred.sum(axis=1)
    mae = float(np.mean(np.abs(t - p))); mape = float(np.mean(np.abs(t - p) / np.maximum(t, 1)) * 100)
    r2 = float(1 - np.sum((t - p) ** 2) / max(np.sum((t - t.mean()) ** 2), 1e-9)) if len(t) > 1 else float("nan")
    return {"MAE_FCFA": round(mae), "MAPE_%": round(mape, 2), "R2": round(r2, 4), "n": int(len(t))}


def main():
    ap = argparse.ArgumentParser(description="Réentraîne le modèle Quotora sur vos projets réels.")
    ap.add_argument("--reel", default="quotora_projets_reels.xlsx", help="Fichier Excel des projets réels")
    ap.add_argument("--synthetique", default=None, help="Dataset synthétique d'origine (.csv ou .xlsx), optionnel")
    ap.add_argument("--ancien", default="quotora_model_final.joblib", help="Modèle actuel (pour noms des lots et lots manquants)")
    ap.add_argument("--poids-reel", type=float, default=None, help="Poids d'un projet réel (défaut : projets réels = 5 %% du poids total)")
    ap.add_argument("--sortie", default="quotora_model_final.joblib")
    ap.add_argument("--forcer", action="store_true", help="Entraîner même avec moins de 10 projets réels et sans dataset synthétique")
    a = ap.parse_args()

    old = load_old_model(HERE / a.ancien) if (HERE / a.ancien).exists() else None
    lot_cols = list(old["lot_cols"]) if old and len(old.get("lot_cols", [])) == 10 else DEFAULT_LOT_COLS

    Xr, yr, names = load_real(HERE / a.reel if not Path(a.reel).is_absolute() else a.reel)
    print(f"📋 {len(Xr)} projet(s) réel(s) lu(s) : {', '.join(names)}")
    Xs = ys = None
    if a.synthetique:
        p = HERE / a.synthetique if not Path(a.synthetique).is_absolute() else Path(a.synthetique)
        ds = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
        lacks = [c for c in FEATURES + lot_cols if c not in ds.columns]
        if lacks:
            sys.exit(f"❌ Colonnes absentes dans {p.name} : {lacks}\n   Colonnes attendues pour les lots : {lot_cols}")
        ds = ds.dropna(subset=FEATURES + lot_cols)
        Xs, ys = ds[FEATURES].reset_index(drop=True), ds[lot_cols].to_numpy(dtype=float)
        print(f"🧪 {len(Xs)} ligne(s) synthétique(s) lue(s).")
    elif len(Xr) < 10 and not a.forcer:
        sys.exit(f"⛔ Seulement {len(Xr)} projet(s) réel(s). Un modèle entraîné sur si peu de chantiers apprendrait vos projets par cœur "
                 "et donnerait n'importe quoi sur les autres.\n   → Ajoutez le dataset synthétique d'origine avec --synthetique, "
                 "ou complétez la base jusqu'à 10 projets au moins (30 à 50 idéalement).")

    missing = np.isnan(yr)
    if missing.any():
        filler = old["pipeline"] if old is not None else None
        if filler is None and Xs is not None:   # pas d'ancien modèle : modèle provisoire entraîné sur le synthétique
            filler = build_pipeline(len(Xs)); filler.fit(Xs, ys)
        if filler is not None:
            fill = np.maximum(filler.predict(Xr), 0)
            yr = np.where(missing, fill, yr)
            print(f"   {int(missing.sum())} montant(s) de lot vide(s) complété(s) par le modèle (à remplacer par vos montants réels dès que possible).")
        else:
            keep = ~missing.any(axis=1)
            print(f"   Lignes incomplètes ignorées (pas d'ancien modèle pour compléter) : {int((~keep).sum())}")
            Xr, yr = Xr[keep].reset_index(drop=True), yr[keep]
            names = [nm for nm, k in zip(names, keep) if k]

    X = pd.concat([d for d in (Xs, Xr) if d is not None and len(d)], ignore_index=True)
    y = np.vstack([d for d in (ys, yr) if d is not None and len(d)])
    if a.poids_reel is None:   # projets réels = 5 % du poids total (au moins 25 lignes synthétiques chacun)
        a.poids_reel = max(25.0, 0.05 * (0 if Xs is None else len(Xs)) / max(len(Xr), 1)) if Xs is not None else 1.0
    print(f"⚖️  Poids de chaque projet réel : {a.poids_reel:,.0f}")
    w = np.concatenate([np.ones(0 if Xs is None else len(Xs)), np.full(len(Xr), a.poids_reel)])

    met = {}
    if Xs is not None and len(Xs) >= 50:   # contrôle : 20 % du synthétique mis de côté
        rng = np.random.default_rng(42); idx = rng.permutation(len(Xs)); cut = int(len(Xs) * .8)
        tr, te = idx[:cut], idx[cut:]
        Xt = pd.concat([Xs.iloc[tr], Xr], ignore_index=True); yt = np.vstack([ys[tr], yr])
        wt = np.concatenate([np.ones(len(tr)), np.full(len(Xr), a.poids_reel)])
        pc = build_pipeline(len(Xt)); pc.fit(Xt, yt, model__sample_weight=wt)
        met["10_lots_total"] = metrics(ys[te], np.maximum(pc.predict(Xs.iloc[te]), 0))
    pipe = build_pipeline(len(X)); pipe.fit(X, y, model__sample_weight=w)
    pr = np.maximum(pipe.predict(Xr), 0) if len(Xr) else np.zeros((0, 10))
    if len(Xr): met["projets_reels_ajustement"] = metrics(yr, pr)
    met.setdefault("10_lots_total", met.get("projets_reels_ajustement", {}))

    out = HERE / a.sortie if not Path(a.sortie).is_absolute() else Path(a.sortie)
    if out.exists():
        shutil.copy2(out, out.with_name(out.stem + "_ancien" + out.suffix)); print(f"💾 Ancien modèle sauvegardé : {out.stem}_ancien{out.suffix}")
    joblib.dump({"pipeline": pipe, "features": FEATURES, "lot_cols": lot_cols, "metrics": met,
                 "trained_at": dt.datetime.now().isoformat(timespec="seconds"), "n_reel": int(len(Xr)),
                 "n_synthetique": 0 if Xs is None else int(len(Xs)), "sklearn_version": sklearn.__version__}, out)
    print(f"✅ Modèle enregistré : {out.name}")
    for k, v in met.items(): print(f"   {k} : {v}")
    if len(Xr):
        print("\n   Projet réel                                   Total réel        Total prédit (lots 2-11)")
        for nm, t, p in zip(names, yr.sum(axis=1), pr.sum(axis=1)):
            print(f"   {nm[:44]:44s} {t:>15,.0f}  {p:>15,.0f}  ({p / max(t, 1) - 1:+.0%})")
    print(f"\n⚠️  IMPORTANT : ce modèle a été créé avec scikit-learn {sklearn.__version__}. Dans requirements.txt (GitHub), "
          f"la ligne doit être exactement :  scikit-learn=={sklearn.__version__}\n   Sinon Streamlit Cloud risque de ne pas pouvoir l'ouvrir.")


if __name__ == "__main__":
    main()
