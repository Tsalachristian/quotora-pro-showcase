"""Quotora Pro - core cost and geomatics formulas (excerpt of app.py, no user interface, no secrets).

IMPORTANT: all prices, cost grids and coefficients in this file are ILLUSTRATIVE EXAMPLE VALUES.
The production application uses a private price grid calibrated on real construction quotes.
The methods (formulas, interpolation, guard rule, geomatics) are identical to production.

Author: Tsala Christian. All rights reserved. Shared for academic review only."""
import math
import numpy as np

REGIONS = ["Adamaoua", "Centre", "Est", "Extrême-Nord", "Littoral", "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest"]

STANDINGS = ["Économique", "Standard", "Moyen standing", "Haut standing", "Luxe"]

WALL_CATALOG = {   # EXAMPLE values, FCFA per m2 of wall
    "Parpaing 10": {"thickness": "10 cm", "default": 8000},
    "Parpaing 15": {"thickness": "15 cm", "default": 10000},
    "Parpaing 20": {"thickness": "20 cm", "default": 12000},
    "Brique cuite": {"thickness": "10-15 cm", "default": 18000},
    "BTC": {"thickness": "15 cm", "default": 16000},
    "Béton armé coulé": {"thickness": "15 cm", "default": 50000},
    "Carabot": {"thickness": "-", "default": 5000},
}

WALL_PRICE = {k: v["default"] for k, v in WALL_CATALOG.items()}

ROOF_PRICE = {"Charpente métallique + tôle": 28000, "Toiture-terrasse accessible": 45000}   # EXAMPLE values

def distance_m(a,b):
    R=6371000.0; p1=math.radians(a[0]); dp=math.radians(b[0]-a[0]); dl=math.radians(b[1]-a[1])
    h=math.sin(dp/2)**2+math.cos(p1)*math.cos(math.radians(b[0]))*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(h))

def bearing(a,b):
    y=math.sin(math.radians(b[1]-a[1]))*math.cos(math.radians(b[0]))
    x=math.cos(math.radians(a[0]))*math.sin(math.radians(b[0]))-math.sin(math.radians(a[0]))*math.cos(math.radians(b[0]))*math.cos(math.radians(b[1]-a[1]))
    return (math.degrees(math.atan2(y,x))+360)%360

def local_xy(lat,lon,lat0,lon0):
    return (lon-lon0)*111320*math.cos(math.radians(lat0)), (lat-lat0)*110540

def polygon_area_m2(poly):
    if len(poly)<3: return 0
    lat0=sum(p[0] for p in poly)/len(poly); lon0=sum(p[1] for p in poly)/len(poly)
    xy=[local_xy(a,b,lat0,lon0) for a,b in poly]
    return abs(sum(xy[i][0]*xy[(i+1)%len(xy)][1]-xy[(i+1)%len(xy)][0]*xy[i][1] for i in range(len(xy)))/2)

def interpolate_elev(point,sample_points,sample_z):
    if not sample_points: return 0.0
    d=np.array([distance_m(point,p) for p in sample_points]); idx=np.argsort(d)[:min(4,len(d))]; w=1/(d[idx]+0.5)
    return float(np.sum(np.array(sample_z)[idx]*w)/np.sum(w))

def calculate_elevation_rows(rows):
    """Build the physical elevation inputs and the user's reference cost.
    The resulting area/reference cost are then supplied to the ML estimator;
    they are not themselves the final Lot 3 prediction.
    """
    clean=[]
    total_area=0.0
    total_cost=0.0
    for i,r in enumerate(rows or [],1):
        length=float(r.get("length_m",0) or 0)
        height=float(r.get("height_m",0) or 0)
        openings=float(r.get("openings_m2",0) or 0)
        price=float(r.get("unit_price",0) or 0)
        gross_area=max(length*height,0.0)
        net_area=max(gross_area-openings,0.0)
        amount=net_area*price
        item=dict(r)
        item.update({"gross_area_m2":gross_area,"net_area_m2":net_area,"amount_fcfa":amount})
        clean.append(item)
        total_area+=net_area
        total_cost+=amount
    return clean,total_area,total_cost

def calculate_installation_chantier_cost(surface_m2):
    """Site set-up cost by footprint band (EXAMPLE values)."""
    s = max(float(surface_m2 or 0), 0.0)
    for limit, amount in ((100, 600000.0), (300, 1200000.0), (1000, 2500000.0)):
        if s <= limit:
            return amount
    return 2500000.0 + (s - 1000.0) * 2500.0

REF_ROOF_SLOPE = 1.20                          # surface de couverture ÷ emprise (devis : 75 m² pour 62 m²)

REF_ROOF_BASE = {"Charpente bois": 25000, "Charpente métallique": 28000, "Charpente artisanale": 20000}   # EXAMPLE values

REF_ROOF_BAC_ACIER_ADD = 4000   # EXAMPLE value

NO_CONCRETE_WALLS = {"Carabot", "Mur bois", "Poto-poto (terre + bois)", "Paille / matériaux végétaux"}

QUICK_LOT_GRID = {   # EXAMPLE values - index -> (min, max, base) in FCFA/m2
    0: (4000, 8000, "emprise"),      # Earthworks
    1: (20000, 30000, "emprise"),    # Foundations (replaced by the per-storey table below)
    3: (30000, 60000, "emprise"),    # Roof structure & covering
    4: (20000, 40000, "construit"),  # Plastering, screed, waterproofing
    5: (20000, 50000, "construit"),  # Joinery
    6: (10000, 25000, "construit"),  # Plumbing
    7: (15000, 35000, "construit"),  # Electrical
    8: (40000, 100000, "construit"), # Tiling, ceilings, painting
}

QUICK_WALLS_GRID = (12000, 20000)      # EXAMPLE values, FCFA per built m2

QUICK_BA_GRID = (13000, 23000)         # EXAMPLE values, FCFA per built m2

QUICK_ML_BAND = (0.65, 1.35)          # sortie du modèle conservée si elle reste à ±35 % de la grille ; sinon la grille s'applique (garde-fou)

QUICK_VRD_RATE = 0.025                # Lot 11 VRD (optionnel) : 2,5 % du montant des autres lots

QUICK_DALLAGE_GRID = (10000, 18000)   # EXAMPLE values

QUICK_PLANCHER_GRID = (30000, 50000)  # EXAMPLE values

QUICK_FOUNDATION_M2 = {0: 30000, 1: 40000, 3: 50000, 5: 90000}   # EXAMPLE values, FCFA per m2 of footprint by storeys

def _interp_levels(table, n):
    n = max(int(n or 0), 0); ks = sorted(table)
    if n in table: return float(table[n])
    if n > ks[-1]:
        a, b = ks[-2], ks[-1]
        return table[b] + (table[b] - table[a]) / (b - a) * (n - b)
    return float(np.interp(n, ks, [table[k] for k in ks]))

def quick_foundation_m2(n):
    return _interp_levels(QUICK_FOUNDATION_M2, n)

QUICK_BA_LEVEL_COEF = {0: 1.0, 1: 1.0, 3: 2.0, 5: 3.5}   # EXAMPLE values

QUICK_TERRASSEMENT_LEVEL_COEF = {0: 1.0, 3: 1.0, 5: 6.0}   # EXAMPLE values

QUICK_ECO_FINISH_K = 0.75   # EXAMPLE value

QUICK_FINISH_LOTS = {4, 5, 6, 7, 8}

QUICK_STANDING_POS = {"Économique": (0.0, 0.85), "Standard": (0.0, 1.0), "Moyen standing": (0.25, 1.0),
                      "Haut standing": (0.6, 1.0), "Luxe": (1.0, 1.0)}

def _grid_point(lo, hi, standing):
    """(valeur retenue, borne basse, borne haute) en FCFA/m² pour le standing."""
    pos, k = QUICK_STANDING_POS.get(standing, (0.0, 1.0))
    return (lo + pos * (hi - lo)) * k, lo * min(k, 1.0), hi

def quick_region_factor(region):
    m, mo, tr = REF_REGION_COEF.get(region, (1, 1, 1))
    return .60 * m + .25 * mo + .15 * tr

QUICK_STRUCT_STANDING_CAP = "Moyen standing"

QUICK_STRUCT_LOTS = {2, 3, 4}   # élévation, charpente & couverture, enduits & chape (fondations : déjà indépendantes du standing)

def _struct_standing(standing):
    order = STANDINGS
    if standing in order and order.index(standing) > order.index(QUICK_STRUCT_STANDING_CAP):
        return QUICK_STRUCT_STANDING_CAP
    return standing

def quick_lot_targets(region, rdc, n, standing, wall, roof, surfaces=None):
    """Montants cibles par lot (cible, min, max) en FCFA, d'après la grille par lot de l'utilisateur.
    surfaces : [RDC, R+1, …] ; par défaut la surface du RDC est reprise à chaque étage."""
    rf = quick_region_factor(region)
    emprise = float(rdc)
    surfaces = list(surfaces) if surfaces else [emprise] * (int(n) + 1)
    built = float(sum(surfaces))
    s_std = _struct_standing(standing)
    out = {}
    for k, (lo, hi, base) in QUICK_LOT_GRID.items():
        q = (emprise if base == "emprise" else built) * rf
        t, a, b = _grid_point(lo, hi, s_std if k in QUICK_STRUCT_LOTS else standing)
        if standing == "Économique" and k in QUICK_FINISH_LOTS:
            f = QUICK_ECO_FINISH_K / QUICK_STANDING_POS["Économique"][1]; t, a = t * f, a * f
        out[k] = (t * q, a * q, b * q)
    eco_k = min(QUICK_STANDING_POS.get(standing, (0, 1.0))[1], 1.0)   # Économique : −15 % ; sinon indépendant du standing
    fnd = quick_foundation_m2(n) * emprise * rf * eco_k   # fondations selon le nombre d'étages
    out[1] = (fnd, fnd, fnd * 1.25)
    tc = _interp_levels(QUICK_TERRASSEMENT_LEVEL_COEF, n)
    out[0] = tuple(v * tc for v in out[0])
    price, sloped = roof_reference_unit_price(roof)   # charpente : plus ou moins cher selon la toiture choisie
    if price:
        t, a, b = out[3]; r = price / 28000.0 / (1.0 if sloped else REF_ROOF_SLOPE)
        out[3] = (min(max(t * r, a), b), a, b)
    ratio = min(max(WALL_PRICE.get(wall, WALL_PRICE["Parpaing 15"]) / WALL_PRICE["Parpaing 15"], 0.3), 2.5)  # matériau du mur
    tw, aw, bw = _grid_point(*QUICK_WALLS_GRID, s_std)
    tb, ab, bb = (0.0, 0.0, 0.0) if wall in NO_CONCRETE_WALLS else _grid_point(*QUICK_BA_GRID, s_std)
    bc = _interp_levels(QUICK_BA_LEVEL_COEF, n); tb, ab, bb = tb * bc, ab * bc, bb * bc
    q = built * rf
    out[2] = ((tw * ratio + tb) * q, (aw * ratio + ab) * q, (bw * ratio + bb) * q)
    if wall in NO_CONCRETE_WALLS: out[4] = (0.0, 0.0, 0.0)   # pas d'enduit ciment
    out["dallage_m2"] = _grid_point(*QUICK_DALLAGE_GRID, s_std)[0] * rf
    out["plancher_m2"] = _grid_point(*QUICK_PLANCHER_GRID, s_std)[0] * rf
    return out

def roof_reference_unit_price(roof):
    """(prix FCFA/m², pente ?) d'une toiture décrite par « structure + couverture » ou par une entrée de ROOF_PRICE."""
    r = str(roof or "")
    if r in ROOF_PRICE: return float(ROOF_PRICE[r]), ("errasse" not in r)
    structure, _, sheet = r.partition(" + ")
    base = REF_ROOF_BASE.get(structure)
    if base is None: return None, True
    return float(base + (REF_ROOF_BAC_ACIER_ADD if "bac acier" in sheet.lower() else 0)), True

REF_REGION_COEF = {  # EXAMPLE values: (materials, labour, transport), base = 1
    "Littoral": (1, 1, 1), "Centre": (1.02, 1.0, 1.05), "Ouest": (1.02, 1.0, 1.10), "Nord-Ouest": (1.05, 1.0, 1.15),
    "Sud-Ouest": (1.02, 1.0, 1.10), "Sud": (1.05, 1.0, 1.15), "Est": (1.05, 1.0, 1.20), "Adamaoua": (1.08, 1.0, 1.30),
    "Nord": (1.10, 1.0, 1.40), "Extrême-Nord": (1.12, 1.0, 1.50)}
