"""Diagnóstico de la especificación logística. Ejecutar desde la raíz con python -m notebooks.phase4_logistic."""

from io import BytesIO
from itertools import combinations, product
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, xlogy
from scipy.stats import chi2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from src.evaluation import probability_metrics, threshold_metrics, youden_threshold
from src.models import (
    AGE_CURVE, FINAL_CURVES, FINAL_INTERACTIONS, INTERACTIONS, LIMIT_CURVES,
    Phase4Logistic, add_logistic_terms, fit_binomial, fit_phase4_logistic,
    make_logistic_design,
)
from src.preprocessing import BILLS, BILL_CHANGE, load_and_split, make_design_pipeline

ALL_CURVES = (*LIMIT_CURVES, AGE_CURVE)
ALL_INTERACTIONS = tuple(INTERACTIONS)
PHASE3_SPEC = ((), ())
FINAL_SPEC = (FINAL_CURVES, FINAL_INTERACTIONS)
MONETARY_PROFILE = ["LIMIT_BAL", "PAY_0", "BILL_AMT1", "PAY_AMT1"]


def load_analysis_data(data_path):
    splits = load_and_split(data_path)
    with ZipFile(data_path) as archive:
        archived = archive.read("data/phase1/split_indices.npz")
    with np.load(BytesIO(archived)) as indices:
        for split, key in [("train", "train"), ("val", "validation")]:
            np.testing.assert_array_equal(splits[split][1].index.to_numpy(), indices[key])
    return splits["train"], splits["val"]


def likelihood_ratio(restricted, expanded):
    df = int(expanded.df_model - restricted.df_model)
    statistic = 2 * (expanded.llf - restricted.llf)
    return {"LR": statistic, "gl": df, "p LRT": chi2.sf(statistic, df)}


def model_stats(result):
    return {"k": len(result.params), "logLik": result.llf, "AIC": result.aic,
            "BIC": result.bic_llf}


def inference(result):
    ci = np.exp(result.conf_int())
    return pd.DataFrame({
        "beta": result.params, "EE": result.bse, "p Wald": result.pvalues,
        "OR": np.exp(result.params), "OR IC95 inf": ci.iloc[:, 0], "OR IC95 sup": ci.iloc[:, 1],
    })


def vif(design, columns):
    correlation = design.drop(columns="const").corr()
    values = pd.Series(np.diag(np.linalg.inv(correlation)), index=correlation.index)
    return values.loc[columns].to_dict()


def save_diagnostic_grid(diagnostics, most_influential, destination, measures,
                         title, reference_lines):
    background, foreground, highlight = "#f7f9fc", "#233853", "#a54122"
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 7.2), layout="constrained")
    figure.set_facecolor(background)
    figure.suptitle(title, fontsize=17, weight="bold", color=foreground)
    x_specs = [("p", "Probabilidad estimada de incumplimiento"),
               ("fila_csv", "Índice de fila en el CSV limpio (base 0)")]
    for row, (measure, ylabel, scale) in enumerate(measures):
        for column, (x, xlabel) in enumerate(x_specs):
            axis = axes[row, column]
            axis.set_facecolor("white")
            axis.set_axisbelow(True)
            axis.grid(color="#e8edf4", linewidth=0.65)
            axis.scatter(diagnostics[x], diagnostics[measure], s=4.5, alpha=0.24,
                         color="#276295", linewidths=0)
            axis.scatter(most_influential[x], most_influential[measure], s=24,
                         color=highlight, edgecolors="white", linewidths=0.55,
                         label="10 mayores distancias de Cook", zorder=4)
            for line, label in reference_lines[measure]:
                axis.axhline(line, color="#6c7687", lw=0.9, ls="--", label=label)
            axis.set(xlabel=xlabel, ylabel=ylabel)
            if scale == "symlog":
                linear_region = {"Pearson": 0.5, "leverage": 0.0005, "Cook": 0.00001}
                axis.set_yscale(scale, linthresh=linear_region[measure])
                lower = diagnostics[measure].min() * 1.3 if measure == "Pearson" else 0
                axis.set_ylim(lower, diagnostics[measure].max() * 1.3)
            for edge in ("top", "right"):
                axis.spines[edge].set_visible(False)
            if column == 0:
                axis.legend(loc="best", fontsize=8, framealpha=0.94)
    figure.savefig(destination, dpi=300)
    plt.close(figure)


def diagnose(base, design, X, y, figures):
    influence = base.get_influence(observed=False)
    diagnostics = pd.DataFrame({
        "fila_csv": X.index, "p": base.fittedvalues, "devianza": base.resid_deviance,
        "Pearson": base.resid_pearson, "leverage": influence.hat_matrix_diag,
        "Cook": influence.cooks_distance[0],
    }, index=X.index)
    k, n = design.shape[1], len(X)
    h = diagnostics["leverage"]
    pregibon = diagnostics["Pearson"].pow(2) * h / (k * (1 - h).pow(2))
    np.testing.assert_allclose(diagnostics["Cook"], pregibon, rtol=1e-10)
    np.testing.assert_allclose(h.sum(), k)
    top = diagnostics.nlargest(10, "Cook")
    profile = top.join(X[MONETARY_PROFILE]).join(y.rename("default"))
    keep = ~X.index.isin(top.index)
    # Mantiene las unidades (DE del train completo) al comparar coeficientes y OR.
    sensitivity = fit_binomial(design.loc[keep], y.loc[keep])
    change = pd.DataFrame({"beta completo": base.params, "beta sin 10": sensitivity.params,
                           "OR completo": np.exp(base.params),
                           "OR sin 10": np.exp(sensitivity.params),
                           "p completo": base.pvalues, "p sin 10": sensitivity.pvalues})
    delta = sensitivity.params - base.params
    change["delta beta"] = delta
    change["cambio relativo OR (%)"] = 100 * np.expm1(delta)
    change["|delta beta| / EE original"] = abs(delta) / base.bse
    for name, fit in [("completo", base), ("sin 10", sensitivity)]:
        intervals = np.exp(fit.conf_int())
        change[f"OR IC95 inf {name}"] = intervals.iloc[:, 0]
        change[f"OR IC95 sup {name}"] = intervals.iloc[:, 1]

    save_diagnostic_grid(
        diagnostics, top, figures / "05_logistic_diagnostics_residuals.png",
        [("devianza", "Residuo de devianza", "linear"), ("Pearson", "Residuo de Pearson", "symlog")],
        "Diagnóstico de residuos del modelo logístico (muestra de entrenamiento)",
        {"devianza": [(2, "|d| = 2"), (-2, "_"), (3, "|d| = 3"), (-3, "_")],
         "Pearson": [(0, "Residuo cero")]},
    )
    save_diagnostic_grid(
        diagnostics, top, figures / "06_logistic_influence_cooks.png",
        [("leverage", "Apalancamiento", "symlog"), ("Cook", "Distancia de Cook", "symlog")],
        "Apalancamiento e influencia del modelo logístico (muestra de entrenamiento)",
        {"leverage": [(2 * k / n, "Referencia 2k/n")], "Cook": [(4 / n, "Referencia 4/n")]},
    )
    residue_rates = []
    for threshold in (2, 3):
        count = int(diagnostics["devianza"].abs().gt(threshold).sum())
        residue_rates.append({"|d| >": threshold, "n": count, "%": 100 * count / n})
    return {
        "devianza": residue_rates, "top10": profile.to_dict("records"),
        "sensibilidad": change.rename_axis("término").reset_index().to_dict("records"),
        "referencias_influencia": {"2k/n": 2 * k / n, "4/n": 4 / n,
                                   "leverage máximo": h.max(), "fila leverage máximo": int(h.idxmax()),
                                   "Cook máximo": diagnostics["Cook"].max(),
                                   "n leverage > 2k/n": int(h.gt(2 * k / n).sum()),
                                   "n Cook > 4/n": int(diagnostics["Cook"].gt(4 / n).sum())},
    }


RATIO_STATES = {
    "utilizacion": ("cero", "negativa"),
    "pago": ("cero", "indefinida"),
}
BT_MAIN = ["LIMIT_BAL", "AGE", *RATIO_STATES]
BT_TERMS = [f"{name}_BT" for name in BT_MAIN]
RATIO_FLAGS = [f"{name}_{state}" for name, states in RATIO_STATES.items() for state in states]


def ratios(X):
    return {"utilizacion": X["BILL_AMT1"] / X["LIMIT_BAL"],
            "pago": X["PAY_AMT1"] / X["BILL_AMT2"].where(X["BILL_AMT2"].gt(0))}


class BoxTidwellFeatures(TransformerMixin, BaseEstimator):
    """Representa las ratios por tramos y estima topes en la rama positiva."""

    def __init__(self, quantile=None):
        self.quantile = quantile

    def fit(self, X, y=None):
        self.caps_ = {name: ratio[ratio.gt(0)].quantile(self.quantile)
                      for name, ratio in ratios(X).items()} if self.quantile is not None else {}
        return self

    def transform(self, X):
        terms = {}
        for name, ratio in ratios(X).items():
            positive = ratio.where(ratio.gt(0), 0).clip(upper=self.caps_.get(name))
            terms[name] = positive
            terms[f"{name}_BT"] = xlogy(positive, positive)
            states = {"cero": ratio.eq(0), "negativa": ratio.lt(0), "indefinida": ratio.isna()}
            for state in RATIO_STATES[name]:
                terms[f"{name}_{state}"] = states[state].astype(float)
        for name, divisor in [("LIMIT_BAL", 100_000), ("AGE", 10)]:
            positive = X[name] / divisor
            terms[f"{name}_BT"] = xlogy(positive, positive)
        return pd.DataFrame(terms, index=X.index)


class StableLogit(sm.Logit):
    """Evalúa el auxiliar de ratios crudas con aritmética estable en la cola."""

    def cdf(self, X):
        return expit(X)

    def loglikeobs(self, params):
        signed_logit = (2 * self.endog - 1) * self.predict(params, which="linear")
        return -np.logaddexp(0, -signed_logit)

    def loglike(self, params):
        return self.loglikeobs(params).sum()


def box_tidwell(X, y, base_design):
    results = []
    for quantile in (None, 0.99, 0.975, 0.995):
        aux = Pipeline([
            ("features", BoxTidwellFeatures(quantile)),
            ("design", ColumnTransformer([
                ("scaled", StandardScaler(), [*RATIO_STATES, *BT_TERMS]),
                ("states", "passthrough", RATIO_FLAGS),
            ], verbose_feature_names_out=False).set_output(transform="pandas")),
        ])
        design = base_design.join(aux.fit_transform(X))
        model = StableLogit(y, design, missing="raise")
        n = len(y)
        # Amortigua los primeros pasos en la cola cruda antes del ajuste de Wald.
        start = minimize(
            lambda beta: -model.loglike(beta) / n, np.zeros(design.shape[1]),
            jac=lambda beta: -model.score(beta) / n, hess=lambda beta: -model.hessian(beta) / n,
            method="Newton-CG", options={"maxiter": 1000},
        )
        result = model.fit(start_params=start.x, method="newton", disp=False, maxiter=60)
        if not result.mle_retvals["converged"] or not np.isfinite(result.llf):
            raise RuntimeError(f"Box–Tidwell no convergió para tope {quantile}.")
        np.testing.assert_allclose(model.score(result.params) / n, 0, atol=1e-10)
        p = result.pvalues.loc[BT_TERMS]
        adjusted = multipletests(p, method="holm")[1]
        results.append({"cuantil_tope": quantile, "topes": aux["features"].caps_,
                        "logLik": result.llf, "términos": [
                            {"término": name, "beta": result.params[name], "EE": result.bse[name],
                             "p Wald": p[name], "p Holm": p_holm}
                            for name, p_holm in zip(BT_TERMS, adjusted)
                        ]})
    domains = {}
    for name, ratio in ratios(X).items():
        domains[name] = {"positiva": int(ratio.gt(0).sum()), "cero": int(ratio.eq(0).sum()),
                         "negativa": int(ratio.lt(0).sum()), "indefinida": int(ratio.isna().sum()),
                         "máximo": ratio.max()}
    return {"dominios": domains, "ajustes": results}


def select_specification(design, y):
    # Se conservan los efectos principales al contrastar interacciones.
    interaction_sets = [subset for size in range(4) for subset in combinations(ALL_INTERACTIONS, size)]
    curve_sets = [(*limit, *age) for limit, age in product(
        [(), *((curve,) for curve in LIMIT_CURVES)], [(), (AGE_CURVE,)],
    )]
    main = design.drop(columns=[*ALL_CURVES, *ALL_INTERACTIONS])
    fits = {(curves, terms): fit_binomial(design.loc[:, [*main, *curves, *terms]], y)
            for curves, terms in product(curve_sets, interaction_sets)}
    # El BIC ordena modelos elegibles. Cada término nuevo debe mejorar frente a su omisión.
    for spec in sorted(fits, key=lambda candidate: (fits[candidate].bic_llf, len(fits[candidate].params))):
        curves, terms = spec
        parents = [(tuple(c for c in curves if c != term), terms) for term in curves]
        parents += [(curves, tuple(t for t in terms if t != term)) for term in terms]
        if all(likelihood_ratio(fits[parent], fits[spec])["p LRT"] < 0.05 for parent in parents):
            return spec, fits


def comparison_table(fits):
    limit, persistence, bill = ALL_INTERACTIONS
    sequence = [PHASE3_SPEC, ((), (limit,)), ((), (limit, persistence)),
                ((), (limit, persistence, bill))]
    sequence += [((), FINAL_INTERACTIONS), FINAL_SPEC]
    comparisons = []
    for position, spec in enumerate(sequence):
        row = {"modelo": f"M{position}", "curvaturas": list(spec[0]),
               "interacciones": list(spec[1]), **model_stats(fits[spec])}
        if position:
            parent = [0, 0, 1, 2, 1, 4][position]
            row.update({"referencia LRT": f"M{parent}",
                        **likelihood_ratio(fits[sequence[parent]], fits[spec])})
        comparisons.append(row)
    drops = []
    curves, terms = FINAL_SPEC
    for term in (*curves, *terms):
        parent = (tuple(c for c in curves if c != term), tuple(t for t in terms if t != term))
        drops.append({"término": term, **likelihood_ratio(fits[parent], fits[FINAL_SPEC])})
    candidates = [{"curvaturas": list(spec[0]), "interacciones": list(spec[1]), **model_stats(fit)}
                  for spec, fit in sorted(fits.items(), key=lambda item: item[1].bic_llf)]
    return {"modelos": comparisons, "omisión individual del final": drops,
            "candidatos ordenados por BIC": candidates}


def cross_validate(X, y):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof = pd.Series(index=X.index, dtype=float)
    records = []
    fixed_records = []
    for fold, (fit_idx, eval_idx) in enumerate(cv.split(X, y), 1):
        X_fit, X_eval = X.iloc[fit_idx], X.iloc[eval_idx]
        preprocessing = make_logistic_design(ALL_CURVES)
        design = add_logistic_terms(preprocessing.fit_transform(X_fit), ALL_INTERACTIONS)
        spec, fits = select_specification(design, y.iloc[fit_idx])
        fitted = Phase4Logistic(preprocessing, fits[spec], spec[1])
        pred = fitted.predict_proba(X_eval)
        oof.iloc[eval_idx] = pred
        records.append({"pliegue": fold, "curvaturas": list(spec[0]), "interacciones": list(spec[1]),
                        **probability_metrics(y.iloc[eval_idx], pred)})
        fixed = Phase4Logistic(preprocessing, fits[FINAL_SPEC], FINAL_INTERACTIONS)
        fixed_records.append({"pliegue": fold, **probability_metrics(y.iloc[eval_idx], fixed.predict_proba(X_eval))})
    metric_names = list(probability_metrics(y, oof))
    def summarize(rows):
        metrics = pd.DataFrame(rows).loc[:, metric_names]
        return pd.DataFrame({"media": metrics.mean(), "DE muestral": metrics.std(ddof=1)}).to_dict("index")

    return {"pliegues": records, "resumen": summarize(records),
            "pliegues con especificación fija": fixed_records,
            "resumen con especificación fija": summarize(fixed_records)}, oof


def run_phase4(root):
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    (X, y), (X_val, y_val) = load_analysis_data(root / "sources" / "phase1_deliverables.zip")
    preprocessing = make_logistic_design(ALL_CURVES)
    design = add_logistic_terms(preprocessing.fit_transform(X), ALL_INTERACTIONS)
    selected, fits = select_specification(design, y)
    if selected != FINAL_SPEC:
        raise RuntimeError(f"La especificación seleccionada difiere del modelo M5: {selected}.")
    base = fits[PHASE3_SPEC]
    base_design = design.loc[:, base.params.index]
    np.testing.assert_allclose(base.llf, -9709.120435505902, rtol=0, atol=1e-8)
    full = add_logistic_terms(make_design_pipeline().fit_transform(X), ())
    diagnostics = diagnose(base, base_design, X, y, figures)
    bt = box_tidwell(X, y, base_design)
    cv, oof = cross_validate(X, y)
    oof_cut = youden_threshold(y, oof)
    final = fit_phase4_logistic(X, y)
    reference_params = fits[FINAL_SPEC].params.reindex(final.result.params.index)
    np.testing.assert_allclose(final.result.params, reference_params, rtol=1e-9, atol=1e-10)
    p_val = final.predict_proba(X_val)
    val_cut = youden_threshold(y_val, p_val)
    scale = final.preprocessing["design"].named_transformers_["continuous"]
    return {
        "partición": {"train": {"n": len(y), "defaults": int(y.sum())},
                      "val": {"n": len(y_val), "defaults": int(y_val.sum())}},
        "fase3": model_stats(base), "inferencia fase3": inference(base).to_dict("index"),
        "VIF saldos": {"Seis saldos": vif(full, BILLS),
                       "M0": vif(base_design, [BILLS[0], BILL_CHANGE]),
                       "M5": vif(design.loc[:, final.result.params.index], [BILLS[0], BILL_CHANGE])},
        "diagnósticos": diagnostics,
        "Box–Tidwell": bt,
        "comparación": comparison_table(fits),
        "inferencia final": inference(final.result).to_dict("index"),
        "escala continua final": dict(zip(scale.feature_names_in_, map(float, scale.scale_))),
        "media continua final": dict(zip(scale.feature_names_in_, map(float, scale.mean_))),
        "CV": cv,
        "val": {"probabilísticas": probability_metrics(y_val, p_val),
                "prevalencia": y_val.mean(), "PD media": p_val.mean(),
                "umbrales": {"estándar": threshold_metrics(y_val, p_val, 0.5),
                            "Youden OOF train": threshold_metrics(y_val, p_val, oof_cut),
                            "Youden descriptivo val": threshold_metrics(y_val, p_val, val_cut)},
                "J OOF train": threshold_metrics(y, oof, oof_cut)["J"]},
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    report = root / "reports" / "phase4_results.json"
    report.parent.mkdir(exist_ok=True)
    results = run_phase4(root)
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(report)
