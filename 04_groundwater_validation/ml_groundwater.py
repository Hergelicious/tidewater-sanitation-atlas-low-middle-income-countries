"""Does a machine-learning model predict shallow groundwater (within 3 m) better than the tidal rule at geographically
independent wells? Gradient-boosted trees and logistic regression, with leave-country-out and 2-degree spatial-block
cross-validation; performance assessed on low- and middle-income wells only.
Features: elevation above HAT (mean-sea-level referenced), distance to the coastline, distance to the generalised coastline
(estuary indicator), Fan et al. modelled water table depth, HAT, 100-year surge above HAT, absolute latitude.
Writes supplementary/Table_S23_ml_groundwater.csv"""
import glob, numpy as np, pandas as pd, xarray as xr
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

w = pd.read_csv("/home/claude/data/gw/wells_all_msl_dist.csv")
# Fan model depth at wells
w["fan"] = np.nan
for f in sorted(glob.glob("/home/claude/data/fanwtd/*_WTD_annualmean.nc")):
    d = xr.open_dataset(f); la, lo = d.lat.values.astype(float), d.lon.values.astype(float); ch = d.WTD.encoding["chunksizes"]
    sel = np.where(w.fan.isna() & w.lat.between(la[0], la[-1]) & w.lon.between(lo[0], lo[-1]))[0]
    if sel.size == 0:
        continue
    i = np.clip(np.round((w.lat.values[sel] - la[0]) / (la[1] - la[0])).astype(int), 0, la.size - 1)
    j = np.clip(np.round((w.lon.values[sel] - lo[0]) / (lo[1] - lo[0])).astype(int), 0, lo.size - 1)
    k = pd.DataFrame({"k": sel, "i": i, "j": j, "ci": i // ch[1], "cj": j // ch[2]})
    for (ci, cj), g in k.groupby(["ci", "cj"]):
        r0, c0 = ci * ch[1], cj * ch[2]
        blk = d.WTD.isel(time=0, lat=slice(r0, r0 + ch[1]), lon=slice(c0, c0 + ch[2])).values
        msk = d["mask"].isel(lat=slice(r0, r0 + ch[1]), lon=slice(c0, c0 + ch[2])).values
        w.loc[g.k.values, "fan"] = np.where(msk[g.i - r0, g.j - c0] == 1, -blk[g.i - r0, g.j - c0], np.nan)


def xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


E = "/home/claude/data/esl"
hat = pd.read_csv(f"{E}/max_Astronomical_Tide_Baseline.csv").dropna(subset=["valueMedian"])
_, i = cKDTree(xyz(hat.latitude.values, hat.longitude.values)).query(xyz(w.lat.values, w.lon.values), k=3)
w["hat"] = hat.valueMedian.values[i].mean(1)
eslf = [f for f in glob.glob(f"{E}/*.csv") if "ESL" in f.upper() or "Total" in f or "TWL" in f.upper()]
print("ESL files:", eslf[:3])
w["abslat"] = w.lat.abs()
w["y"] = (w.dtw <= 3).astype(int)
w["lmic"] = w.set.str.startswith("Low")
w["country"] = np.where(w.iso3.isin(["IND"]), "India", np.where(w.iso3.isin(["ZAF"]), "South Africa", np.where(w.lmic, "Other LMIC", "HIC")))
w["block"] = (np.floor(w.lat / 2).astype(int)).astype(str) + "_" + (np.floor(w.lon / 2).astype(int)).astype(str)
F = ["e_msl", "dist_km", "dist110_km", "fan", "hat", "abslat"]
w = w.dropna(subset=["e_msl", "dist_km"]).reset_index(drop=True)


def gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, random_state=0)


def logit():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def fill(X):
    X = X.copy(); X["fan"] = X.fan.fillna(X.fan.median()); return X


def evaluate(test, p_ml, name, nboot=500):
    y = test.y.values; rule = (test.e_msl <= 3).values; score = -test.e_msl.values
    thr = np.quantile(p_ml, 1 - rule.mean()) if rule.mean() > 0 else 1
    pm = p_ml >= thr
    def m(pred):
        tp, fp, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum()
        return tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    ppv_r, sen_r = m(rule); ppv_m, sen_m = m(pm)
    # paired cluster bootstrap over 2-degree blocks: difference in AUC (model minus continuous rule score)
    rng = np.random.default_rng(11); blk = test.block.values; ub = np.unique(blk)
    idx = {b: np.where(blk == b)[0] for b in ub}; diffs, ar, am = [], [], []
    for _ in range(nboot):
        ii = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
        if y[ii].min() == y[ii].max():
            continue
        a1, a2 = roc_auc_score(y[ii], score[ii]), roc_auc_score(y[ii], p_ml[ii])
        ar.append(a1); am.append(a2); diffs.append(a2 - a1)
    q = lambda v: np.percentile(v, [2.5, 97.5])
    return dict(test=name, wells=len(test), shallow_pct=100 * y.mean(),
                rule_auc=roc_auc_score(y, score), rule_auc_lo=q(ar)[0], rule_auc_hi=q(ar)[1],
                rule_binary_auc=roc_auc_score(y, rule), rule_ppv=ppv_r, rule_sens=sen_r,
                ml_auc=roc_auc_score(y, p_ml), ml_auc_lo=q(am)[0], ml_auc_hi=q(am)[1],
                auc_diff=roc_auc_score(y, p_ml) - roc_auc_score(y, score), auc_diff_lo=q(diffs)[0], auc_diff_hi=q(diffs)[1],
                ml_prauc=average_precision_score(y, p_ml), ml_brier=brier_score_loss(y, p_ml),
                ml_ppv_same_flag_rate=ppv_m, ml_sens_same_flag_rate=sen_m)


rows = []
for model_name, mk in (("Gradient-boosted trees", gbm), ("Logistic regression", logit)):
    # leave-country-out on LMIC groups (training on everything else, including HIC wells)
    for grp in ["India", "South Africa", "Other LMIC"]:
        tr, te = w[w.country != grp], w[w.country == grp]
        clf = mk().fit(fill(tr[F]), tr.y)
        rows.append(dict(model=model_name, scheme="leave-country-out", **evaluate(te, clf.predict_proba(fill(te[F]))[:, 1], grp)))
    # spatial block cross-validation (2-degree blocks, 5 folds), scored on LMIC wells
    blocks = np.array(w.block.unique(), dtype=object); rng = np.random.default_rng(1); rng.shuffle(blocks)
    fold = {b: i % 5 for i, b in enumerate(blocks)}
    w["fold"] = w.block.map(fold)
    p = np.full(len(w), np.nan)
    for k in range(5):
        tr, te = w[w.fold != k], w[w.fold == k]
        p[te.index] = mk().fit(fill(tr[F]), tr.y).predict_proba(fill(te[F]))[:, 1]
    te = w[w.lmic]
    rows.append(dict(model=model_name, scheme="2-degree spatial blocks", **evaluate(te, p[te.index], "All LMIC wells")))
    te = w[~w.lmic]
    rows.append(dict(model=model_name, scheme="2-degree spatial blocks", **evaluate(te, p[te.index], "High-income wells")))
t = pd.DataFrame(rows).round(3)
pd.set_option("display.width", 260)
print(t.to_string(index=False))
t.to_csv("supplementary/Table_S23_ml_groundwater.csv", index=False)
