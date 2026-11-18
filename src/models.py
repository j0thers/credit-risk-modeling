"""Modelos de incumplimiento, búsqueda estratificada y persistencia."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import loguniform
from scipy.special import xlogy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from threadpoolctl import threadpool_limits

from .preprocessing import make_design_pipeline, with_bill_change

LIMIT_CURVES = ("LIMIT_BAL_xlogx", "LIMIT_BAL_log", "LIMIT_BAL_sqrt")
AGE_CURVE = "AGE_xlogx"
INTERACTIONS = {
    "PAY_0_x_LIMIT_BAL": "LIMIT_BAL",
    "PAY_0_x_PAY_2": "PAY_2",
    "PAY_0_x_BILL_AMT1": "BILL_AMT1",
}
# Resultado del criterio de selección de la Fase 4 sobre el train archivado.
FINAL_CURVES = ("LIMIT_BAL_log",)
FINAL_INTERACTIONS = ("PAY_0_x_LIMIT_BAL", "PAY_0_x_BILL_AMT1")


def _derive_curves(X: pd.DataFrame, curves: tuple[str, ...]) -> pd.DataFrame:
    limit = X["LIMIT_BAL"] / 100_000
    functions = {
        "LIMIT_BAL_xlogx": lambda: xlogy(limit, limit),
        "LIMIT_BAL_log": lambda: np.log(limit),
        "LIMIT_BAL_sqrt": lambda: np.sqrt(limit),
        "AGE_xlogx": lambda: xlogy(X["AGE"] / 10, X["AGE"] / 10),
    }
    return with_bill_change(X).assign(**{name: functions[name]() for name in curves})


def add_logistic_terms(design: pd.DataFrame, interactions: tuple[str, ...]) -> pd.DataFrame:
    products = {name: design["PAY_0"] * design[INTERACTIONS[name]] for name in interactions}
    return sm.add_constant(design.assign(**products), has_constant="add")


def make_logistic_design(curves: tuple[str, ...] = ()) -> Pipeline:
    """Construye la matriz reducida con transformaciones aún sin ajustar."""
    design = make_design_pipeline(reduced_bills=True, extra_continuous=curves)
    return Pipeline([
        ("features", FunctionTransformer(_derive_curves, kw_args={"curves": curves})),
        ("design", design.named_steps["design"]),
    ])


def fit_binomial(design: pd.DataFrame, y: pd.Series) -> GLMResultsWrapper:
    result = sm.GLM(y, design, family=sm.families.Binomial(), missing="raise").fit()
    if not result.converged or not np.isfinite(result.llf):
        raise RuntimeError("El GLM binomial no convergió a una verosimilitud finita.")
    return result


@dataclass
class Phase4Logistic:
    preprocessing: Pipeline
    result: GLMResultsWrapper
    interactions: tuple[str, ...]

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Devuelve P(default=1) con el índice de entrada."""
        design = add_logistic_terms(self.preprocessing.transform(X), self.interactions)
        return self.result.predict(design.loc[:, self.result.params.index]).rename("prob_default")


def fit_phase4_logistic(X_train: pd.DataFrame, y_train: pd.Series) -> Phase4Logistic:
    """Ajusta el GLM final y aprende la codificación y escalas solo de X_train.

    Recibe los predictores limpios de load_and_split y el objetivo binario alineado.
    `result` expone la inferencia de statsmodels y `predict_proba` estima la PD.
    """
    preprocessing = make_logistic_design(FINAL_CURVES)
    design = add_logistic_terms(preprocessing.fit_transform(X_train), FINAL_INTERACTIONS)
    return Phase4Logistic(preprocessing, fit_binomial(design, y_train), FINAL_INTERACTIONS)


ModelName = Literal["logistic", "tree", "forest", "xgboost", "hist_gradient_boosting"]
SEED = 42
CV_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "neg_log_loss": "neg_log_loss",
}


class Phase4BinomialClassifier(ClassifierMixin, BaseEstimator):
    """Expone la especificación fija de Fase 4 a los scorers de scikit-learn."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Phase4BinomialClassifier":
        design = add_logistic_terms(X, FINAL_INTERACTIONS)
        self.result_ = fit_binomial(design, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        p = self.result_.predict(add_logistic_terms(X, FINAL_INTERACTIONS)).to_numpy()
        return np.column_stack((1 - p, p))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_model_pipeline(name: ModelName) -> Pipeline:
    """Construye un pipeline sin ajustar para una entrada limpia de load_and_split."""
    if name == "logistic":
        return Pipeline([
            *make_logistic_design(FINAL_CURVES).steps,
            ("model", Phase4BinomialClassifier()),
        ])
    if name == "tree":
        model = DecisionTreeClassifier(random_state=SEED)
    elif name == "forest":
        model = RandomForestClassifier(random_state=SEED, n_jobs=1)
    elif name == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            importance_type="total_gain", random_state=SEED, n_jobs=1,
        )
    elif name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            loss="log_loss", early_stopping=False, random_state=SEED,
        )
    else:
        raise ValueError(f"Modelo desconocido: {name}")
    return Pipeline([*make_design_pipeline().steps, ("model", model)])


def _search_space(name: ModelName) -> tuple[int, dict]:
    if name == "logistic":
        return 1, {}
    if name == "tree":
        return 16, {
            "max_depth": [3, 4, 5, 6, 8, 10, 14, None],
            "min_samples_split": [10, 25, 50, 100, 200],
            "min_samples_leaf": [10, 25, 50, 100, 200],
            "ccp_alpha": [0, 0.00001, 0.00005, 0.0001, 0.00025, 0.0005],
        }
    if name == "forest":
        return 12, {
            "n_estimators": [120, 180, 250],
            "max_depth": [5, 8, 12, 16, None],
            "min_samples_leaf": [5, 12, 25, 50],
            "max_features": ["sqrt", 0.4, 0.65, 1.0],
            "max_samples": [0.55, 0.70, 0.85, 1.0],
        }
    if name == "xgboost":
        return 16, {
            "n_estimators": [100, 160, 240, 360],
            "learning_rate": loguniform(0.015, 0.15),
            "max_depth": [2, 3, 4, 5, 6],
            "subsample": [0.65, 0.8, 1.0],
            "colsample_bytree": [0.55, 0.75, 1.0],
            "reg_alpha": loguniform(0.001, 10),
            "reg_lambda": loguniform(0.1, 30),
        }
    if name == "hist_gradient_boosting":
        return 16, {
            "max_iter": [100, 160, 240, 360],
            "learning_rate": loguniform(0.015, 0.15),
            "max_depth": [2, 3, 4, 5, 6],
            "max_leaf_nodes": [7, 15, 31],
            "min_samples_leaf": [20, 50, 100],
            "l2_regularization": loguniform(0.01, 30),
        }
    raise ValueError(f"Modelo desconocido: {name}")


def train_model(
    name: ModelName, X_train: pd.DataFrame, y_train: pd.Series, **model_params: object,
) -> Pipeline:
    """Ajusta el pipeline completo con los parámetros indicados para el estimador."""
    pipeline = make_model_pipeline(name)
    pipeline.set_params(**{f"model__{key}": value for key, value in model_params.items()})
    with threadpool_limits(limits=1):
        return pipeline.fit(X_train, y_train)


def optimize_model(
    name: ModelName, X_train: pd.DataFrame, y_train: pd.Series, *, n_jobs: int = 4,
) -> RandomizedSearchCV:
    """Busca en train y reajusta por ROC-AUC. La logística evalúa una especificación fija."""
    pipeline = make_model_pipeline(name)
    n_iter, space = _search_space(name)
    search = RandomizedSearchCV(
        pipeline, {f"model__{key}": values for key, values in space.items()},
        n_iter=n_iter, scoring=CV_SCORING, refit="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        random_state=SEED, n_jobs=n_jobs, return_train_score=True, error_score="raise",
    )
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1), threadpool_limits(1):
        return search.fit(X_train, y_train)


def save_model(model: Pipeline | Phase4Logistic, path: str | Path) -> Path:
    """Serializa el estimador ajustado junto con sus transformaciones."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path


def load_model(path: str | Path) -> Pipeline | Phase4Logistic:
    """Recupera un artefacto joblib de origen confiable."""
    return joblib.load(path)


def fit_ablation_model(frozen: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                       columns: list[str]) -> Pipeline | Phase4Logistic:
    """Reajusta con las mismas decisiones previas y solo las columnas permitidas.

    Para el GLM se conservan las transformaciones e interacciones cuyos términos
    están disponibles. La variante completa puede reutilizar el modelo congelado.
    """
    from sklearn.base import clone
    from .preprocessing import BILL_CHANGE, BILLS

    fitted = clone(frozen)
    design = fitted["design"]
    allowed = set(columns)
    is_logistic = isinstance(fitted["model"], Phase4BinomialClassifier)
    has_financial = "LIMIT_BAL" in allowed
    if is_logistic and has_financial:
        allowed.update((BILL_CHANGE, *FINAL_CURVES))
    design.transformers = [
        (name, transformer, retained)
        for name, transformer, original in design.transformers
        if (retained := [column for column in original if column in allowed])
    ]
    X_selected = X_train.loc[:, columns]
    with threadpool_limits(limits=1):
        if is_logistic:
            steps = fitted.steps[:-1] if has_financial else [("design", design)]
            preprocessing = Pipeline(steps)
            interactions = FINAL_INTERACTIONS if {"PAY_0", "LIMIT_BAL", BILLS[0]} <= allowed else ()
            matrix = add_logistic_terms(preprocessing.fit_transform(X_selected), interactions)
            return Phase4Logistic(preprocessing, fit_binomial(matrix, y_train), interactions)
        return fitted.fit(X_selected, y_train)
