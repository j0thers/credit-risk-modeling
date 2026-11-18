"""Calibración probabilística y simulación de pérdida esperada. Ejecutar con python notebooks/phase7_calibration.py."""

import hashlib
import json
from io import BytesIO
from pathlib import Path
import sys
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import (
    audit_subgroups, calibration_bins, calibration_metrics, fit_probability_calibrator,
    probability_metrics, simulate_expected_losses, youden_threshold,
)
from src.preprocessing import load_and_split


LABELS = {
    "prob_logreg": "Regresión logística", "prob_dt": "Árbol de decisión",
    "prob_rf": "Random Forest", "prob_xgb": "XGBoost",
}
METHODS = {"cruda": "Sin calibrar", "sigmoid": "Platt", "isotonic": "Isotónica"}
COLORS = ["#2166ac", "#a63d40", "#1b7964", "#83509b"]
FIGURES = (
    "11_calibration_curves_comparison.png", "12_demographic_fairness_audit.png",
    "13_expected_loss_scenarios.png",
)
DEMOGRAPHIC_ORDER = {
    "SEX": {1: "Hombres", 2: "Mujeres"},
    "EDUCATION": {1: "Posgrado", 2: "Universidad", 3: "Secundaria", 4: "Otros"},
    "MARRIAGE": {1: "Casado", 2: "Soltero", 3: "Otros"},
    "AGE": {"<30": "<30", "30–45": "30–45", ">45": ">45"},
}


def read_frozen_predictions(path, y):
    # La alineación con los índices archivados evita asociar PD y respuestas de clientes distintos.
    frame = pd.read_csv(path, float_precision="round_trip")
    np.testing.assert_array_equal(frame["y_true"], y.to_numpy())
    return frame.loc[:, list(LABELS)].set_axis(y.index)


def calibration_candidates(val, y_val):
    """Elige candidatos por ROC-AUC y evalúa calibración fuera de pliegue en val."""
    ranking = pd.DataFrame({
        model: probability_metrics(y_val, val[model]) for model in LABELS
    }).T.rename_axis("modelo").sort_values(
        ["ROC-AUC", "PR-AUC (AP)", "Log Loss"], ascending=[False, False, True],
    )
    candidates = list(dict.fromkeys([*ranking.index[:2], "prob_logreg"]))
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(val, y_val))
    rows = []
    for model in candidates:
        raw = val[model].to_numpy()
        for method in METHODS:
            if method == "cruda":
                scores = raw
            else:
                scores = np.empty(len(val), dtype=float)
                for fit, holdout in folds:
                    calibrate = fit_probability_calibrator(y_val.iloc[fit], raw[fit], method=method)
                    scores[holdout] = calibrate(raw[holdout])
            rows.append({"modelo": model, "método": method, **probability_metrics(y_val, scores)})
    comparison = pd.DataFrame(rows).set_index(["modelo", "método"])
    calibrated = comparison.query("método != 'cruda'").sort_values(["Brier Score", "Log Loss"])
    winner = calibrated.index[0]
    calibrators = {
        (model, method): fit_probability_calibrator(y_val, val[model], method=method)
        for model in candidates for method in ("sigmoid", "isotonic")
    }
    return ranking, comparison, winner, calibrators


def plot_calibration(y, test, scores, model):
    fig = plt.figure(figsize=(12, 6.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[3.4, 1])
    ax_raw = fig.add_subplot(grid[0, 0])
    ax_cal = fig.add_subplot(grid[0, 1], sharex=ax_raw, sharey=ax_raw)
    ax_hist = fig.add_subplot(grid[1, 1], sharex=ax_cal)
    fig.suptitle("Calibración de la probabilidad de incumplimiento a un mes", fontweight="bold")

    for axis, title, curves in [
        (ax_raw, "A. Modelos sin calibrar", [(LABELS[key], test[key]) for key in LABELS]),
        (ax_cal, f"B. {LABELS[model]} antes y después",
         [(label, scores[model, method]) for method, label in METHODS.items()]),
    ]:
        axis.plot([0, 1], [0, 1], "--", color="#68717a", linewidth=1, label="Calibración perfecta")
        for (label, p), color, marker in zip(curves, COLORS, ["o", "s", "^", "D"]):
            bins = calibration_bins(y, p)
            axis.plot(bins["pd_media"], bins["tasa_observada"], marker=marker, markersize=4,
                      color=color, linewidth=1.6, label=label)
        axis.set(title=title, xlim=(0, 1), ylim=(0, 1.03))
        axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.grid(alpha=0.19)
        axis.legend(loc="upper left", frameon=False, fontsize=9)
    ax_raw.set(xlabel="PD media predicha", ylabel="Fracción observada de incumplimientos")
    ax_raw.text(0, -0.28, f"Taiwán, 2005 · prueba (n = {len(y):,})\n"
                         "10 intervalos uniformes. Se muestran los intervalos con observaciones.",
                transform=ax_raw.transAxes, color="#475569", fontsize=9, va="top")
    ax_cal.tick_params(labelbottom=False)
    for method, color in zip(METHODS, COLORS):
        ax_hist.hist(scores[model, method], bins=np.linspace(0, 1, 31), histtype="step",
                     color=color, linewidth=1.5)
    ax_hist.set(xlabel="PD predicha", ylabel="Clientes")
    ax_hist.grid(axis="y", alpha=0.19)
    return fig


def plot_demographics(audit, title):
    fig, axes = plt.subplots(
        1, 4, sharey=True, figsize=(13, 6.5), layout="constrained",
        gridspec_kw={"width_ratios": [1.25, 1.3, 0.85, 0.85]},
    )
    fig.suptitle(f"Auditoría demográfica en prueba (n = 4.500) · {title}", fontweight="bold")
    positions = np.arange(len(audit), dtype=float)
    groups = audit.index.get_level_values("variable")
    changes = np.flatnonzero(groups[1:] != groups[:-1]) + 1
    for split in changes:
        positions[split:] += 0.55
    labels = [f"{group}  ·  n = {n:,}" for (_, group), n in zip(audit.index, audit["n"])]
    axes[0].barh(positions, 100 * audit["sesgo_pd"], height=0.58,
                 color=np.where(audit["sesgo_pd"] < 0, COLORS[0], COLORS[1]))
    axes[0].axvline(0, color="#68717a", linewidth=1)
    axes[0].set(title="Sesgo de PD", xlabel="PD − observado (pp)", yticks=positions, yticklabels=labels)
    axes[0].invert_yaxis()
    for metric, marker, color in [("FPR", "o", COLORS[0]), ("FNR", "D", COLORS[1])]:
        axes[1].scatter(audit[metric], positions, label=metric, marker=marker, color=color, s=28)
    axes[1].set(title="Errores de decisión", xlabel="Corte de Youden (validación)", xlim=(0, 1))
    axes[1].xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].legend(frameon=False, loc="lower right", ncols=2, fontsize=8)
    for ax, metric, bounds in zip(axes[2:], ["ROC-AUC", "Brier Score"], [(0.5, 1), (0, 0.30)]):
        ax.scatter(audit[metric], positions, color=COLORS[2], s=25)
        ax.set(title=metric, xlim=bounds)
    for ax in axes:
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        for split in changes:
            ax.axhline((positions[split - 1] + positions[split]) / 2, color="#e2e8f0", linewidth=1)
    axes[0].set_ylim(positions[-1] + 1, -0.65)
    return fig


def plot_losses(losses):
    fig, ax = plt.subplots(figsize=(9.5, 5.2), layout="constrained")
    width = 0.33
    x = np.arange(len(losses))
    for shift, column, label, color in [
        (-width / 2, "el_cruda_ntd", "PD cruda", "#b4c7db"),
        (width / 2, "el_calibrada_ntd", "PD calibrada", COLORS[0]),
    ]:
        bars = ax.bar(x + shift, losses[column] / 1e6, width, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    for i, (cruda, cal, delta) in enumerate(losses[[
        "el_cruda_ntd", "el_calibrada_ntd", "delta_cal_menos_cruda_ntd",
    ]].itertuples(index=False, name=None)):
        ax.text(i, max(cruda, cal) / 1e6 + 5.8, f"Δ {delta / 1e6:+.2f}",
                ha="center", color="#1e293b", fontsize=10)
    ticks = [f"{name}\nLGD {row.LGD:.0%} · CCF {row.CCF:.0%}"
             for name, row in losses.iterrows()]
    ax.set(title="Pérdida crediticia esperada a un mes · cartera de prueba",
           ylabel="EL total (millones de NT$)", xticks=x, xticklabels=ticks,
           ylim=(0, losses[["el_cruda_ntd", "el_calibrada_ntd"]].max().max() / 1e6 + 10))
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    return fig


def run_phase7(root=ROOT):
    root = Path(root)
    reports, figures = root / "reports", root / "figures"
    figures.mkdir(exist_ok=True)
    archive_path = root / "sources" / "phase1_deliverables.zip"
    val_path = reports / "06_probabilities_val.csv"
    test_path = reports / "06_probabilities_test.csv"
    protocol_path = reports / "06_frozen_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    splits = load_and_split(archive_path)
    with ZipFile(archive_path) as archive:
        archived = archive.read("data/phase1/split_indices.npz")
    with np.load(BytesIO(archived)) as indices:
        for split, key in [("train", "train"), ("val", "validation"), ("test", "test")]:
            np.testing.assert_array_equal(splits[split][0].index, indices[key])
    _, y_val = splits["val"]
    val = read_frozen_predictions(val_path, y_val)
    ranking, selection, winner, calibrators = calibration_candidates(val, y_val)
    model, method = winner
    cutoff = protocol["classification_cutoffs"]["youden_val"][model]
    np.testing.assert_allclose(cutoff, youden_threshold(y_val, val[model]), rtol=0, atol=1e-15)
    mapped_cutoff = float(calibrators[winner]([cutoff])[0])

    X_test, y_test = splits["test"]
    test = read_frozen_predictions(test_path, y_test)
    scores = {(name, "cruda"): test[name] for name in LABELS}
    scores.update({
        key: pd.Series(calibrate(test[key[0]]), index=test.index)
        for key, calibrate in calibrators.items()
    })
    metrics = pd.DataFrame.from_dict({
        key: calibration_metrics(y_test, p) for key, p in scores.items()
    }, orient="index").rename_axis(["modelo", "método"])
    metrics["Delta ROC-AUC vs cruda"] = metrics["ROC-AUC"] - metrics.index.get_level_values(
        "modelo").map(metrics.xs("cruda", level="método")["ROC-AUC"])

    groups = X_test[["SEX", "EDUCATION", "MARRIAGE"]].copy()
    groups["AGE"] = pd.cut(X_test["AGE"], [-np.inf, 29, 45, np.inf],
                           labels=list(DEMOGRAPHIC_ORDER["AGE"]))
    order = []
    for variable, labels in DEMOGRAPHIC_ORDER.items():
        groups[variable] = groups[variable].map(labels)
        order.extend((variable, group) for group in labels.values())
    raw, final = test[model], scores[winner]
    frozen_decisions = raw >= cutoff
    audit = audit_subgroups(groups, y_test, final, threshold=cutoff).reindex(order)
    individual, losses = simulate_expected_losses(X_test, raw, final)
    individual.index.name = "fila_original_0based"

    metadata = {
        "modelo_final": model, "calibrador_final": method, "seed": 42, "pliegues_val": 5,
        "n_train_val_test": [len(splits[k][1]) for k in ("train", "val", "test")],
        "incumplimientos_test": int(y_test.sum()),
        "youden_cruda_fase6": cutoff, "pd_calibrada_en_el_corte": mapped_cutoff,
        "decisiones_distintas_corte_transformado": int(((final >= mapped_cutoff) != frozen_decisions).sum()),
        "decisiones_distintas_corte_literal": int(((final >= cutoff) != frozen_decisions).sum()),
        "pd_media_cruda": float(raw.mean()), "pd_media_calibrada": float(final.mean()),
        "pd_extremas_incorrectas_en_test": int(
            (((final == 0) & (y_test == 1)) | ((final == 1) & (y_test == 0))).sum()
        ),
        "bills_negativos": int((X_test["BILL_AMT1"] < 0).sum()),
        "bills_superiores_al_limite": int((X_test["BILL_AMT1"] > X_test["LIMIT_BAL"]).sum()),
        "entradas_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (val_path, test_path, protocol_path, archive_path)
        },
    }
    tables = {
        "07_validation_ranking": ranking, "07_validation_selection": selection,
        "07_calibration_metrics": metrics, "07_demographic_audit": audit,
        "07_expected_loss_scenarios": losses, "07_expected_loss_individual": individual,
    }
    for name, table in tables.items():
        table.to_csv(reports / f"{name}.csv")
    (reports / "07_protocol.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    with plt.rc_context({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "semibold"}):
        for name, fig in zip(FIGURES, [
            plot_calibration(y_test, test, scores, model),
            plot_demographics(audit, f"{LABELS[model]} + {METHODS[method]}"),
            plot_losses(losses),
        ]):
            fig.savefig(figures / name, dpi=300, facecolor="white")
            plt.close(fig)
    return tables, metadata


if __name__ == "__main__":
    results, protocol = run_phase7()
    print(json.dumps(protocol, indent=2, ensure_ascii=False))
    for name in ("07_calibration_metrics", "07_demographic_audit", "07_expected_loss_scenarios"):
        print(results[name].to_string())
