"""Métricas probabilísticas y de clasificación para incumplimiento binario."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, log_loss,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .preprocessing import CATEGORICAL


def prediction_frame(models, X, y) -> pd.DataFrame:
    """Evalúa una vez cada estimador y conserva el orden de la partición."""
    if not X.index.equals(y.index):
        raise ValueError("Los predictores y el objetivo deben estar alineados.")
    return pd.DataFrame({
        "y_true": y,
        **{
            name: np.asarray(model.predict_proba(X), dtype=float)[:, 1]
            for name, model in models.items()
        },
    }, index=y.index)


def stratified_auc_bootstrap(predictions: pd.DataFrame, *, n_replicates=1000, seed=42):
    """Devuelve AUC replicados, con las mismas filas remuestreadas para cada modelo."""
    y = predictions["y_true"].to_numpy()
    scores = predictions.drop(columns="y_true")
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    rng = np.random.default_rng(seed)
    p = scores.to_numpy()
    replicas = np.empty((n_replicates, p.shape[1]))
    for replicate in replicas:
        rows = np.concatenate([rng.choice(group, size=len(group)) for group in groups])
        for column in range(p.shape[1]):
            replicate[column] = roc_auc_score(y[rows], p[rows, column])
    return pd.DataFrame(replicas, columns=scores.columns)


def paired_auc_difference(replicas, first, second, observed):
    """IC percentil y prueba bilateral con la distribución bootstrap centrada."""
    differences = replicas[first] - replicas[second]
    lower, upper = differences.quantile([0.025, 0.975])
    count = np.count_nonzero(np.abs(differences - observed) >= abs(observed))
    return {
        "primero": first, "segundo": second, "delta": observed,
        "ic95_inferior": lower, "ic95_superior": upper,
        "p_bilateral": (count + 1) / (len(differences) + 1),
    }


def compare_test_predictions(predictions, thresholds, replicas):
    """Resume las métricas probabilísticas y las dos reglas de clasificación."""
    y = predictions["y_true"]
    probability_rows, classification_rows = [], []
    matrices = {}
    intervals = replicas.quantile([0.025, 0.975])
    for name, cutoff in thresholds.items():
        p = predictions[name]
        probability_rows.append({
            "modelo": name, **probability_metrics(y, p),
            "ROC IC95 inferior": intervals.at[0.025, name],
            "ROC IC95 superior": intervals.at[0.975, name],
            "AP − prevalencia": average_precision_score(y, p) - y.mean(),
        })
        for rule, threshold in [("0.50", 0.5), ("Youden val", cutoff)]:
            metrics = threshold_metrics(y, p, threshold)
            classification_rows.append({"modelo": name, "regla": rule, **metrics})
            absolute = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
            matrices[name, rule] = (absolute, absolute / absolute.sum(axis=1, keepdims=True))
    return pd.DataFrame(probability_rows).set_index("modelo"), pd.DataFrame(classification_rows), matrices


def score_ablation(y, predictions):
    """Resume el desempeño de cada variante al corte fijo de 0.50."""
    rows = []
    for (model, variant), p in predictions.items():
        metrics = probability_metrics(y, p)
        rows.append({
            "modelo": model, "variante": variant,
            **{name: metrics[name] for name in ("ROC-AUC", "PR-AUC (AP)", "Log Loss")},
            "F1": threshold_metrics(y, p, 0.5)["F1"],
        })
    return pd.DataFrame(rows).set_index(["modelo", "variante"])


def ablation_deltas(scores, blocks):
    """Calcula D menos cada bloque aislado y D menos cada variante sin el bloque."""
    metrics = ["ROC-AUC", "PR-AUC (AP)"]
    rows = {}
    for model in scores.index.get_level_values("modelo").unique():
        table = scores.loc[model, metrics]
        for block in blocks:
            rows[model, block, "D − bloque aislado"] = table.loc["D"] - table.loc[block]
            rows[model, block, "D − modelo sin bloque"] = table.loc["D"] - table.loc[f"sin {block}"]
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(
        index=["modelo", "bloque", "contraste"],
    ).add_prefix("Δ ")


def plot_test_curves(predictions, scores, labels):
    """Devuelve una figura con ROC y precisión–recall en el mismo conjunto."""
    import matplotlib.pyplot as plt

    fig, (roc, pr) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    y = predictions["y_true"]
    for name, label in labels.items():
        p = predictions[name]
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        roc.plot(fpr, tpr, label=f"{label} ({scores.at[name, 'ROC-AUC']:.4f})")
        pr.step(recall, precision, where="post", label=f"{label} ({scores.at[name, 'PR-AUC (AP)']:.4f})")
    roc.plot([0, 1], [0, 1], "--", color="0.45", label="Azar (0.5000)")
    pr.axhline(y.mean(), linestyle="--", color="0.45", label=f"Prevalencia ({y.mean():.4f})")
    roc.set(title="A. Curvas ROC", xlabel="Tasa de falsos positivos", ylabel="Sensibilidad")
    pr.set(title="B. Curvas precisión–recall", xlabel="Sensibilidad", ylabel="Precisión")
    for ax, legend_position in [(roc, "lower right"), (pr, "upper right")]:
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=0.2)
        ax.legend(loc=legend_position)
    return fig


def plot_test_confusions(matrices, labels):
    """Muestra conteos y porcentajes de fila para cada modelo y corte."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(labels), 2, figsize=(8, 12), layout="constrained")
    for row, (name, label) in zip(axes, labels.items()):
        for ax, rule in zip(row, ("0.50", "Youden val")):
            absolute, normalized = matrices[name, rule]
            ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            for (i, j), rate in np.ndenumerate(normalized):
                ax.text(j, i, f"{absolute[i, j]:,}\n{rate:.1%}", ha="center", va="center",
                        color="white" if rate > 0.55 else "#173454")
            ax.set(title=f"{label} · {rule}", xticks=[0, 1], yticks=[0, 1],
                   xticklabels=["Cumple", "Incumple"], yticklabels=["Cumple", "Incumple"],
                   xlabel="Predicción", ylabel="Observado")
    return fig


def plot_ablation(scores, labels):
    """Compara los cuatro subconjuntos principales en las cuatro métricas."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, metric in zip(axes.flat, ["ROC-AUC", "PR-AUC (AP)", "F1", "Log Loss"]):
        values = scores[metric].unstack("modelo").loc[list("ABCD")].rename(columns=labels)
        values.plot.bar(ax=ax, rot=0, legend=False, color=["#266a91", "#cc6639"][:len(values.columns)])
        ax.set(title=metric, xlabel="", xticklabels=["Demografía", "Historial", "Finanzas", "Completo"])
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=7)
        ax.set_ylim(0, values.to_numpy().max() * 1.15)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncols=len(names), frameon=False)
    return fig


def probability_metrics(y, p) -> dict[str, float]:
    return {
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC (AP)": average_precision_score(y, p),
        "Log Loss": log_loss(y, p, labels=[0, 1]),
        "Brier Score": brier_score_loss(y, p),
    }


def _probability_vector(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not p.size or not np.all((0 <= p) & (p <= 1)):
        raise ValueError("Se requiere un vector no vacío de probabilidades en [0, 1].")
    return p


def _binary_observations(y, p):
    y = np.asarray(y)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Las observaciones binarias deben estar alineadas con las PD.")
    return y


def fit_probability_calibrator(y_val, p_val, *, method):
    """Ajusta una función de PD usando únicamente pares (y, PD) de validación."""
    p = _probability_vector(p_val)
    y = _binary_observations(y_val, p)
    if np.unique(y).size != 2:
        raise ValueError("La calibración requiere ambas clases en validación.")
    if method == "sigmoid":
        sigmoid = LogisticRegression(penalty=None, solver="lbfgs", tol=1e-10)
        sigmoid.fit(p[:, None], y)

        def predict(p_new):
            return sigmoid.predict_proba(_probability_vector(p_new)[:, None])[:, 1]

        return predict
    if method == "isotonic":
        isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(p, y)

        def predict(p_new):
            return isotonic.predict(_probability_vector(p_new))

        return predict
    raise ValueError(f"Método de calibración desconocido: {method}")


def calibration_bins(y, p, *, strategy="uniform", n_bins=10):
    """Resume intervalos no vacíos. Los cuantiles repetidos conservan los empates."""
    p = _probability_vector(p)
    y = _binary_observations(y, p)
    if n_bins < 1:
        raise ValueError("Se requiere al menos un intervalo.")
    quantiles = np.linspace(0, 1, n_bins + 1)
    if strategy == "uniform":
        cuts = quantiles[1:-1]
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, quantiles))
        cuts = edges[1:-1]
    else:
        raise ValueError(f"Estrategia de intervalos desconocida: {strategy}")
    bins = np.searchsorted(cuts, p, side="right")
    return (
        pd.DataFrame({"intervalo": bins + 1, "pd_media": p, "tasa_observada": y})
        .groupby("intervalo")
        .agg(n=("pd_media", "size"), pd_media=("pd_media", "mean"),
             tasa_observada=("tasa_observada", "mean"))
    )


def calibration_metrics(y, p, *, n_bins=10):
    """Agrega ECE y la partición de Murphy para las PD promediadas por intervalo."""
    uniform = calibration_bins(y, p, n_bins=n_bins)
    quantile = calibration_bins(y, p, strategy="quantile", n_bins=n_bins)
    weight = uniform["n"] / uniform["n"].sum()
    gap = uniform["pd_media"] - uniform["tasa_observada"]
    prevalence = np.mean(y)
    reliability = float(weight @ gap.pow(2))
    resolution = float(weight @ (uniform["tasa_observada"] - prevalence).pow(2))
    uncertainty = float(prevalence * (1 - prevalence))
    binned_brier = reliability - resolution + uncertainty
    metrics = probability_metrics(y, p)
    return {
        **metrics,
        "ECE uniforme": float(weight @ gap.abs()),
        "ECE cuantiles": float(quantile["n"] @
                               (quantile["pd_media"] - quantile["tasa_observada"]).abs() / len(p)),
        "Bins uniformes": len(uniform), "Bins cuantiles": len(quantile),
        "Reliability": reliability, "Resolution": resolution, "Uncertainty": uncertainty,
        "Brier agrupado": binned_brier,
        "Residuo intra-bin": metrics["Brier Score"] - binned_brier,
    }


def audit_subgroups(groups, y, p, *, threshold):
    """Mide desempeño por categoría aplicando un mismo corte a todas las PD."""
    if any(isinstance(values, pd.Series) and not groups.index.equals(values.index)
           for values in (y, p)):
        raise ValueError("Los grupos, observaciones y PD deben estar alineados.")
    p = _probability_vector(p)
    y = _binary_observations(y, p)
    if len(groups) != len(y):
        raise ValueError("Se requiere una fila demográfica por observación.")
    predicted = p >= threshold
    rows = []
    for variable in groups:
        partitions = groups.groupby(variable, observed=True, sort=False, dropna=False).indices
        for group, idx in partitions.items():
            observed, prob = y[idx], p[idx]
            tn, fp, fn, tp = confusion_matrix(observed, predicted[idx], labels=[0, 1]).ravel()
            prevalence, average_pd = observed.mean(), prob.mean()
            rows.append({
                "variable": variable, "grupo": group, "n": len(idx), "defaults": tp + fn,
                "tasa_observada": prevalence, "pd_media": average_pd,
                "sesgo_pd": average_pd - prevalence,
                "ROC-AUC": roc_auc_score(observed, prob) if tn + fp and tp + fn else np.nan,
                "Brier Score": brier_score_loss(observed, prob),
                "FPR": fp / (tn + fp) if tn + fp else np.nan,
                "FNR": fn / (tp + fn) if tp + fn else np.nan,
                "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            })
    return pd.DataFrame(rows).set_index(["variable", "grupo"])


def simulate_expected_losses(exposures, p_uncalibrated, p_calibrated):
    """Calcula EL individual y de cartera a un mes, en NT$, con LGD y EAD hipotéticas."""
    if any(isinstance(p, pd.Series) and not exposures.index.equals(p.index)
           for p in (p_uncalibrated, p_calibrated)):
        raise ValueError("Las exposiciones y PD deben conservar el mismo índice.")
    raw, calibrated = map(_probability_vector, (p_uncalibrated, p_calibrated))
    if len(raw) != len(exposures) or len(calibrated) != len(exposures):
        raise ValueError("Se requieren ambas PD para cada exposición.")
    limit, bill = exposures[["LIMIT_BAL", "BILL_AMT1"]].to_numpy(dtype=float).T
    if not (np.isfinite(limit).all() and np.isfinite(bill).all() and (limit >= 0).all()):
        raise ValueError("Los saldos y límites deben ser finitos y los límites no negativos.")
    drawn = np.maximum(bill, 0)
    unused = np.maximum(limit - drawn, 0)
    individual = exposures[["LIMIT_BAL", "BILL_AMT1"]].assign(
        pd_cruda=raw, pd_calibrada=calibrated,
    )
    rows = []
    for scenario, lgd, ccf in [("Base", 0.55, 0.50), ("Adverso", 0.70, 0.75),
                               ("Severo", 0.85, 1.00)]:
        ead = limit if scenario == "Severo" else drawn + ccf * unused
        el_raw, el_calibrated = raw * lgd * ead, calibrated * lgd * ead
        delta = el_calibrated - el_raw
        for name, values in [("ead", ead), ("el_cruda", el_raw),
                             ("el_calibrada", el_calibrated), ("delta", delta)]:
            individual[f"{name}_{scenario.lower()}_ntd"] = values
        total_ead, total_raw, total_calibrated = ead.sum(), el_raw.sum(), el_calibrated.sum()
        rows.append({
            "escenario": scenario, "LGD": lgd, "CCF": ccf, "n": len(exposures),
            "ead_total_ntd": total_ead, "el_cruda_ntd": total_raw,
            "el_calibrada_ntd": total_calibrated, "delta_cal_menos_cruda_ntd": delta.sum(),
            "delta_relativo_a_calibrada": delta.sum() / total_calibrated if total_calibrated else np.nan,
            "pd_cruda_ponderada_ead": total_raw / (lgd * total_ead) if total_ead else np.nan,
            "pd_calibrada_ponderada_ead": total_calibrated / (lgd * total_ead) if total_ead else np.nan,
        })
    return individual, pd.DataFrame(rows).set_index("escenario")


def youden_threshold(y, p) -> float:
    """Maximiza sensibilidad + especificidad − 1. En empate, usa el mayor corte."""
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    j = (tpr - fpr)[1:]
    return float(thresholds[1:][np.flatnonzero(np.isclose(j, j.max(), rtol=0, atol=1e-14))[0]])


def threshold_metrics(y, p, threshold: float) -> dict[str, float | int]:
    tn, fp, fn, tp = map(int, confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel())
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "umbral": threshold, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Accuracy": (tn + tp) / (tn + fp + fn + tp),
        "Precisión": tp / (tp + fp) if tp + fp else 0.0,
        "Sensibilidad": sensitivity, "Especificidad": specificity,
        "VPN": tn / (tn + fn) if tn + fn else 0.0,
        "F1": 2 * tp / (2 * tp + fp + fn),
        "Accuracy balanceada": (sensitivity + specificity) / 2,
        "J": sensitivity + specificity - 1,
    }


def summarize_search(search: RandomizedSearchCV) -> dict:
    """Resume el candidato elegido y conserva las métricas por pliegue."""
    row = search.best_index_
    results = search.cv_results_
    summary = {}
    for score, label, sign in [
        ("roc_auc", "ROC-AUC", 1),
        ("pr_auc", "PR-AUC (AP)", 1),
        ("neg_log_loss", "Log Loss", -1),
    ]:
        summary[label] = {}
        for split in ("train", "test"):
            values = [
                float(sign * results[f"split{i}_{split}_{score}"][row])
                for i in range(search.n_splits_)
            ]
            summary[label]["cv_validación" if split == "test" else "cv_train"] = {
                "media": float(np.mean(values)), "de": float(np.std(values, ddof=0)),
                "pliegues": values,
            }
    return {
        "candidatos": len(results["params"]),
        "parámetros": {key.removeprefix("model__"): value for key, value in search.best_params_.items()},
        "métricas": summary,
    }


def compare_feature_importance(
    pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, *, n_repeats: int = 12,
) -> pd.DataFrame:
    """Combina importancia interna agrupada y caída de ROC-AUC en validación."""
    design = pipeline["design"]
    model = pipeline["model"]
    names = design.get_feature_names_out()
    if isinstance(model, HistGradientBoostingClassifier):
        weights = np.zeros(len(names))
        for stage in model._predictors:
            for tree in stage:
                nodes = tree.nodes[tree.nodes["is_leaf"] == 0]
                np.add.at(weights, nodes["feature_idx"].astype(np.intp), nodes["gain"])
        if weights.sum() > 0:
            weights /= weights.sum()
    else:
        weights = model.feature_importances_
    groups = {column: column for column in X_val.columns}
    categories = design.named_transformers_["categorical"]
    for column, encoded in zip(CATEGORICAL, categories.categories_):
        groups.update({f"{column}_{level}": column for level in encoded})
    internal = pd.Series(weights, index=[groups[name] for name in names]).groupby(level=0).sum()
    with threadpool_limits(limits=1):
        perm = permutation_importance(
            pipeline, X_val, y_val, scoring="roc_auc", random_state=42,
            n_repeats=n_repeats, n_jobs=1,
        )
    return pd.DataFrame({
        "interna": internal.reindex(X_val.columns, fill_value=0.0),
        "permutación_media": perm.importances_mean,
        "permutación_de": perm.importances_std,
    }).rename_axis("variable")
