"""Local sklearn baseline. Run before BQML to sanity-check features."""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


META_COLS = {"path", "run", "test_id", "board", "ts", "label", "replicate", "split",
             "wave_type_1", "start_freq_1", "end_freq_1", "sweep_time_1", "amp_1",
             "wave_type_2", "start_freq_2", "end_freq_2"}


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in META_COLS]


def get_xy(df: pd.DataFrame, split_value: str):
    sub = df[df["split"] == split_value]
    cols = feature_columns(df)
    X = sub[cols].fillna(0.0).to_numpy()
    y = sub["label"].to_numpy()
    return X, y, cols


def train_and_evaluate(features_df: pd.DataFrame, model_name: str = "gbm"):
    X_train, y_train, cols = get_xy(features_df, "train")
    X_test, y_test, _ = get_xy(features_df, "test")

    if model_name == "lr":
        clf = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    elif model_name == "rf":
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    elif model_name == "gbm":
        clf = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(model_name)

    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"\n[{model_name}] train acc: {train_acc:.3f} | test acc: {test_acc:.3f}")
    print(classification_report(y_test, clf.predict(X_test)))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, clf.predict(X_test)))
    return clf, {"train_acc": train_acc, "test_acc": test_acc}


def evaluate_all_splits(features_df_with_split, model_name="gbm"):
    """Run model under multiple split strategies — main reporting table."""
    from src.splitter import split as splitter
    rows = []
    for strategy in ["random", "by_replicate", "by_board", "by_excitation", "combined"]:
        df_s = splitter(features_df_with_split.drop(columns=["split"], errors="ignore"), strategy=strategy)
        _, metrics = train_and_evaluate(df_s, model_name=model_name)
        rows.append({"strategy": strategy, **metrics})
    return pd.DataFrame(rows)
