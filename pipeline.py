"""FixFirst pipeline.  Run:  python pipeline.py
Reads locations.csv, incidents.csv, public_complaints.csv from this folder.
Writes scores.csv (the queue) and report.json (validation + data issues).
Only edit the CONFIG block if your column names differ.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# ============================ CONFIG ============================
L = "location_id"                      # join key present in all 3 files
INC_TIME, INC_SEV, INC_RES = "timestamp", "severity", "__none__"     # incidents.csv has no resolved_at column
CMP_TIME, CMP_RES = "created_at", "__none__"
LAT, LON = "latitude", "longitude"     # in locations.csv
CTX = ["zone", "road_type", "road_condition", "population_density_per_km2",
       "near_school", "near_hospital", "pedestrian_activity"]       # location context used as features
SERIOUS_MIN = 3                        # severity >= 3 (High, Critical) counts as "serious"; fixed BEFORE evaluation
OPEN_CMP_STATUS = ["Open", "Assigned", "In progress"]               # complaint statuses treated as not yet resolved
W = 30                                 # window length in days
TOP_FRAC = 0.05                        # queue size used for evaluation (top 5% of locations)
SEED = 42
# ================================================================

SEV_MAP = {"low": 1, "minor": 1, "medium": 2, "moderate": 2, "high": 3,
           "major": 3, "critical": 4, "severe": 4}
rng = np.random.default_rng(SEED)


class PoissonGLM:
    """Poisson regression trained from scratch with gradient descent (numpy only)."""

    def __init__(self, lr=0.05, iters=2000, l2=1e-3):
        self.lr, self.iters, self.l2 = lr, iters, l2

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.w = np.zeros(Z.shape[1])
        self.b = float(np.log(np.mean(y) + 1e-6))
        for _ in range(self.iters):
            m = np.exp(np.clip(Z @ self.w + self.b, -10, 10))
            self.w -= self.lr * (Z.T @ (m - y) / len(y) + self.l2 * self.w)
            self.b -= self.lr * float(np.mean(m - y))
        return self

    def z(self, X):
        return (np.asarray(X, float) - self.mu) / self.sd

    def predict(self, X):
        return np.exp(np.clip(self.z(X) @ self.w + self.b, -10, 10))


def sev_num(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return s.astype(str).str.lower().str.strip().map(SEV_MAP).fillna(2).astype(float)


def clean(locs, inc, cmp_):
    issues = {}
    out = []
    for name, df, t, r in [("incidents", inc, INC_TIME, INC_RES),
                           ("complaints", cmp_, CMP_TIME, CMP_RES)]:
        df = df.copy()
        df[t] = pd.to_datetime(df[t], errors="coerce")
        has_r = r in df.columns
        if has_r:
            df[r] = pd.to_datetime(df[r], errors="coerce")
        issues[f"{name}: duplicate rows"] = int(df.duplicated().sum())
        issues[f"{name}: missing timestamp"] = int(df[t].isna().sum())
        issues[f"{name}: location_id not in locations.csv"] = int((~df[L].isin(locs[L])).sum())
        df = df.drop_duplicates()
        df = df[df[t].notna() & df[L].isin(locs[L])].copy()
        if has_r:
            bad = df[r] < df[t]
            issues[f"{name}: resolved before created (set to open)"] = int(bad.sum())
            df.loc[bad, r] = pd.NaT
            issues[f"{name}: still open / no resolve time"] = int(df[r].isna().sum())
            df["res_days"] = (df[r] - df[t]).dt.days
        elif "res_days" not in df.columns:
            df["res_days"] = np.nan
        out.append(df)
    miss = locs.isna().mean()
    issues["locations: columns with missing values"] = {k: round(float(v), 3) for k, v in miss.items() if v > 0}
    return locs, out[0], out[1], issues


def build_panel(locs, inc, cmp_):
    t0 = min(inc[INC_TIME].min(), cmp_[CMP_TIME].min())
    inc["w"] = ((inc[INC_TIME] - t0).dt.days // W).astype(int)
    cmp_["w"] = ((cmp_[CMP_TIME] - t0).dt.days // W).astype(int)
    inc["sev"] = sev_num(inc[INC_SEV])
    inc["serious"] = (inc["sev"] >= SERIOUS_MIN).astype(int)
    span = (max(inc[INC_TIME].max(), cmp_[CMP_TIME].max()) - t0).days + 1
    nw = span // W                                        # full windows only; a short last window would mislead
    inc, cmp_ = inc[inc["w"] < nw], cmp_[cmp_["w"] < nw]

    g = inc.groupby([L, "w"]).agg(n=("sev", "size"), sev=("sev", "sum"),
                                  serious=("serious", "sum"), res=("res_days", "median")).reset_index()
    c = cmp_.groupby([L, "w"]).agg(cmp=("res_days", "size"), cres=("res_days", "median")).reset_index()
    grid = pd.MultiIndex.from_product([locs[L], range(nw)], names=[L, "w"]).to_frame(index=False)
    P = grid.merge(g, on=[L, "w"], how="left").merge(c, on=[L, "w"], how="left")
    for col in ["n", "sev", "serious", "cmp"]:
        P[col] = P[col].fillna(0)
    P[["res", "cres"]] = P[["res", "cres"]].fillna(P[["res", "cres"]].median()).fillna(0)
    P = P.sort_values([L, "w"]).reset_index(drop=True)

    lag_cols = []
    for col in ["n", "sev", "serious", "cmp"]:
        P[f"{col}_l1"] = P.groupby(L)[col].shift(1).fillna(0)
        P[f"{col}_r3"] = P.groupby(L)[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
        lag_cols += [f"{col}_l1", f"{col}_r3"]
    P["cum_serious"] = P.groupby(L)["serious"].cumsum()
    P["y"] = P.groupby(L)["serious"].shift(-1)          # target: serious incidents next window

    ctx = locs[[L] + CTX].copy()
    cat = [c_ for c_ in CTX if not pd.api.types.is_numeric_dtype(ctx[c_])]
    ctx = pd.get_dummies(ctx, columns=cat, dtype=float)
    ctx = ctx.fillna(ctx.median(numeric_only=True))
    ctx_cols = [c_ for c_ in ctx.columns if c_ != L]
    P = P.merge(ctx, on=L, how="left")
    feats = ["n", "sev", "serious", "cmp", "res", "cres"] + lag_cols + ctx_cols
    return P, feats, ctx_cols


def capture_at_k(df, score, k):
    """Share of next-window serious incidents that fall in the top-k of the queue (avg over windows)."""
    vals = []
    for _, d in df.groupby("w"):
        tot = d["y"].sum()
        if tot == 0:
            continue
        s = d[score].values + rng.random(len(d)) * 1e-6      # random tie-break
        top = np.argsort(-s)[:k]
        vals.append(d["y"].values[top].sum() / tot)
    return float(np.mean(vals)) if vals else float("nan")


PHRASE = {"n": "many recent incidents", "sev": "high recent severity", "serious": "serious incidents this period",
          "cmp": "many complaints", "res": "slow incident resolution", "cres": "slow complaint resolution",
          "n_l1": "incidents last period", "sev_l1": "severity last period", "serious_l1": "serious incidents last period",
          "cmp_l1": "complaints last period", "n_r3": "sustained incident volume", "sev_r3": "sustained high severity",
          "serious_r3": "repeated serious incidents", "cmp_r3": "sustained complaints"}


def main():
    locs = pd.read_csv("locations.csv")
    inc = pd.read_csv("incidents.csv")
    cmp_ = pd.read_csv("public_complaints.csv")
    # map this dataset's resolution columns to a common form
    inc["res_days"] = inc["resolution_hours"] / 24.0
    inc["open"] = (inc["resolved"] == 0).astype(int)
    cmp_["res_days"] = cmp_["days_to_resolution"]
    cmp_["open"] = cmp_["status"].isin(OPEN_CMP_STATUS).astype(int)
    locs, inc, cmp_, issues = clean(locs, inc, cmp_)
    issues["incidents: flagged unresolved"] = int(inc["open"].sum())
    issues["complaints: not yet resolved (open/assigned/in progress)"] = int(cmp_["open"].sum())
    P, FEATS, ctx_cols = build_panel(locs, inc, cmp_)

    # ---------- time-based split ----------
    last = int(P["w"].max())
    cut = int(last * 0.7)
    lab = P[P["y"].notna()]
    train, test = lab[lab["w"] < cut - 1].copy(), lab[lab["w"] >= cut].copy()
    K = max(5, int(TOP_FRAC * len(locs)))

    glm = PoissonGLM().fit(train[FEATS].values, train["y"].values)
    gbm = HistGradientBoostingRegressor(loss="poisson", max_iter=200, learning_rate=0.05,
                                        random_state=SEED).fit(train[FEATS].values, train["y"].values)
    test["glm"] = glm.predict(test[FEATS].values)
    test["gbm"] = gbm.predict(test[FEATS].values)
    test["base_last"] = test["serious"]
    test["base_cum"] = test["cum_serious"]
    cap = {m: capture_at_k(test, m, K) for m in ["base_cum", "base_last", "glm", "gbm"]}
    cap["random (expected)"] = K / len(locs)
    best_base = max(cap["base_cum"], cap["base_last"])
    forecast_validated = cap["glm"] > best_base + 0.02 and cap["glm"] > 1.2 * cap["random (expected)"]
    mae = {m: float(np.mean(np.abs(test[m] - test["y"]))) for m in ["glm", "gbm"]}
    mae["always predict train mean"] = float(np.mean(np.abs(train["y"].mean() - test["y"])))

    # ---------- final scoring: refit on all labelled windows ----------
    Xall, yall = lab[FEATS].values, lab["y"].values
    final = PoissonGLM().fit(Xall, yall)
    now = P[P["w"] == last].reset_index(drop=True)
    Xnow = now[FEATS].values
    point = final.predict(Xnow)

    B = 30
    preds = np.zeros((B, len(now)))
    for b in range(B):
        i = rng.integers(0, len(Xall), len(Xall))
        preds[b] = PoissonGLM(iters=600).fit(Xall[i], yall[i]).predict(Xnow)
    ranks = (-preds).argsort(axis=1).argsort(axis=1) + 1
    lo, hi = np.percentile(ranks, [5, 95], axis=0)
    width = hi - lo
    q1, q2 = np.quantile(width, [0.33, 0.66])
    conf = np.where(width <= q1, "High", np.where(width <= q2, "Medium", "Low"))
    if not forecast_validated:
        conf = np.array(["Not validated"] * len(conf))     # the forecast did not beat baselines/random on held-out data

    # reasons = top positive contributions (coefficient x standardized value)
    contrib = final.z(Xnow) * final.w
    reasons = []
    for r in contrib:
        top = [FEATS[j] for j in np.argsort(-r)[:3] if r[j] > 0]
        txt = ", ".join(PHRASE.get(t, t) for t in top) or "no single dominant driver"
        reasons.append(f"Ranked here mainly because of: {txt}.")

    # ---------- model B: complaints expected from incidents + context ----------
    sf = ["n", "sev", "serious"] + ctx_cols
    sm = PoissonGLM().fit(P[sf].values, P["cmp"].values)
    P["exp_cmp"] = sm.predict(P[sf].values)
    rec = P[P["w"] > last - 3].groupby(L).agg(act=("cmp", "sum"), exp=("exp_cmp", "sum"), inc=("n", "sum"))
    rec["silence_ratio"] = (rec["act"] + 1) / (rec["exp"] + 1)
    thr = rec.loc[rec["inc"] > 0, "silence_ratio"].quantile(0.10)
    rec["silent_flag"] = (rec["silence_ratio"] <= thr) & (rec["inc"] > 0)

    # ---------- current-backlog queue (state of the city now, not a forecast) ----------
    inc["sev_n"] = sev_num(inc[INC_SEV])
    oi = inc[inc["open"] == 1]
    bl = pd.DataFrame(index=locs[L])
    bl["open_incidents"] = oi.groupby(L).size()
    bl["open_serious"] = oi[oi["sev_n"] >= SERIOUS_MIN].groupby(L).size()
    bl["open_sev_sum"] = oi.groupby(L)["sev_n"].sum()
    bl["open_injuries"] = oi.groupby(L)["injuries"].sum() if "injuries" in oi.columns else 0
    oc = cmp_[cmp_["open"] == 1]
    bl["open_complaints"] = oc.groupby(L).size()
    hi_col = "priority_reported"
    bl["open_complaints_high"] = oc[oc[hi_col] == "High"].groupby(L).size() if hi_col in oc.columns else 0
    bl = bl.fillna(0)
    # transparent weights: severity points + 2 per injury + 1 per open complaint + 1 extra per high-priority complaint
    bl["backlog_raw"] = bl["open_sev_sum"] + 2 * bl["open_injuries"] + bl["open_complaints"] + bl["open_complaints_high"]
    bl["backlog_score"] = 100 * bl["backlog_raw"] / bl["backlog_raw"].max()
    zb = (bl["backlog_raw"] - bl["backlog_raw"].mean()) / (bl["backlog_raw"].std() + 1e-9)
    bl["signal"] = np.where(zb >= 2, "Well above city typical", np.where(zb >= 1, "Above typical", "Typical"))
    bl["backlog_reason"] = (bl["open_serious"].astype(int).astype(str) + " open High/Critical incidents, "
                            + bl["open_incidents"].astype(int).astype(str) + " open incidents, "
                            + bl["open_injuries"].astype(int).astype(str) + " injuries on open incidents, "
                            + bl["open_complaints"].astype(int).astype(str) + " unresolved complaints ("
                            + bl["open_complaints_high"].astype(int).astype(str) + " high priority).")
    bl = bl.reset_index()

    # ---------- do locations really differ beyond chance? ----------
    cnt_i = inc.groupby(L).size().reindex(locs[L], fill_value=0)
    cnt_c = cmp_.groupby(L).size().reindex(locs[L], fill_value=0)
    mid = inc[INC_TIME].median()
    h1 = inc[inc[INC_TIME] <= mid].groupby(L).size().reindex(locs[L], fill_value=0)
    h2 = inc[inc[INC_TIME] > mid].groupby(L).size().reindex(locs[L], fill_value=0)
    chance = {"incident_count_variance_to_mean": float(cnt_i.var() / cnt_i.mean()),
              "complaint_count_variance_to_mean": float(cnt_c.var() / cnt_c.mean()),
              "corr_first_half_vs_second_half_incidents": float(np.corrcoef(h1, h2)[0, 1]),
              "corr_incidents_vs_complaints_per_location": float(np.corrcoef(cnt_i, cnt_c)[0, 1])}

    total_inc = inc.groupby(L).size().rename("incidents_total")
    silence_supported = abs(chance["corr_incidents_vs_complaints_per_location"]) >= 0.1
    if not silence_supported:
        rec["silent_flag"] = False                        # complaints are unrelated to incidents, so "below expected" means nothing
    out = pd.DataFrame({L: now[L], "pred_serious_next": point, "rank_lo": lo.astype(int),
                        "rank_hi": hi.astype(int), "confidence": conf, "reason": reasons})
    out = (out.merge(total_inc, on=L, how="left").merge(rec[["silence_ratio", "silent_flag"]], on=L, how="left")
              .merge(locs[[c_ for c_ in [L, LAT, LON] if c_ in locs.columns]], on=L, how="left"))
    out = out.rename(columns={"reason": "forecast_drivers", "rank_lo": "forecast_rank_lo",
                              "rank_hi": "forecast_rank_hi", "confidence": "forecast_confidence"})
    out = out.merge(bl, on=L, how="left").rename(columns={LAT: "lat", LON: "lon"})
    out = out.sort_values(["backlog_raw", "pred_serious_next"], ascending=False)
    out.insert(1, "rank", range(1, len(out) + 1))
    out.to_csv("scores.csv", index=False)

    report = {"capture_at_k": cap, "k": K, "n_locations": int(len(locs)), "windows": last + 1,
              "window_days": W, "train_windows": f"0-{cut - 2}", "test_windows": f"{cut}-{last - 1}",
              "mean_abs_error": mae, "serious_threshold": SERIOUS_MIN, "data_issues": issues,
              "silent_locations_flagged": int(rec["silent_flag"].sum()),
              "chance_tests": chance, "locations_well_above_typical": int((bl["signal"] == "Well above city typical").sum()),
              "forecast_validated": bool(forecast_validated), "silence_flag_supported": bool(silence_supported)}
    json.dump(report, open("report.json", "w"), indent=2, default=str)
    print(json.dumps({k_: report[k_] for k_ in ["capture_at_k", "mean_abs_error", "k", "chance_tests",
                                              "locations_well_above_typical", "forecast_validated", "silence_flag_supported"]}, indent=2))


if __name__ == "__main__":
    main()
