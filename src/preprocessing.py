"""Carga y preparación de predictores para Default of Credit Card Clients."""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "default.payment.next.month"
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
PAYMENT_STATUS = ["PAY_0", *[f"PAY_{month}" for month in range(2, 7)]]
BILLS = [f"BILL_AMT{month}" for month in range(1, 7)]
PAYMENTS = [f"PAY_AMT{month}" for month in range(1, 7)]
CONTINUOUS = ["LIMIT_BAL", "AGE", *BILLS, *PAYMENTS]
FEATURES = [*CATEGORICAL, *CONTINUOUS, *PAYMENT_STATUS]
BILL_CHANGE = "BILL_AMT1_MINUS_PREV5_MEAN"
PHASE1_DATA = "data/phase1/credit_clients_clean.csv"


def load_and_split(path: str | Path) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Carga el ZIP de Fase 1, su CSV extraído o el XLS de UCI (requiere xlrd).

    Conserva el orden original y reproduce la partición de la Fase 1 en dos
    pasos estratificados. Las recodificaciones son reglas fijas por fila.
    """
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive, archive.open(PHASE1_DATA) as stream:
            frame = pd.read_csv(stream)
    elif path.suffix.lower() == ".xls":
        frame = pd.read_excel(path, header=1, engine="xlrd")
    else:
        frame = pd.read_csv(path)

    frame = frame.drop(columns="ID", errors="ignore").rename(columns={
        "DEFAULT_PAYMENT_NEXT_MONTH": TARGET,
        "default payment next month": TARGET,
    })
    y = frame[TARGET]
    if not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError("La variable objetivo debe contener solo las clases 0 y 1.")
    X = frame.loc[:, FEATURES].copy()
    X["EDUCATION"] = X["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    X["MARRIAGE"] = X["MARRIAGE"].replace({0: 3})
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=42, stratify=y_rest,
    )
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def with_bill_change(X: pd.DataFrame) -> pd.DataFrame:
    """Añade la diferencia en NT$ entre el último saldo y el promedio anterior."""
    return X.assign(**{BILL_CHANGE: X[BILLS[0]] - X[BILLS[1:]].mean(axis=1)})


def make_design_pipeline(
    *, reduced_bills: bool = False, extra_continuous: tuple[str, ...] = (),
) -> Pipeline:
    """Devuelve una pipeline sin ajustar para la especificación indicada."""
    continuous = CONTINUOUS
    if reduced_bills:
        continuous = ["LIMIT_BAL", "AGE", BILLS[0], BILL_CHANGE, *PAYMENTS]
    design = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL),
            ("continuous", StandardScaler(), [*continuous, *extra_continuous]),
            ("payment_status", "passthrough", PAYMENT_STATUS),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("design", design)]).set_output(transform="pandas")
