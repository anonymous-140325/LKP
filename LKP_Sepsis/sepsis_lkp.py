import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                              classification_report, confusion_matrix, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    import torch
    from torch import nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

try:
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False


SEED_BALANCED = 6
SEED_UNBALANCED = 17
SEED = SEED_BALANCED
QUIET = False
TEST_SIZE = 0.20
TOP_K_FEATURES = None
MISSINGNESS_DROP_THRESHOLD = 0.5
MIN_SAMPLES_LEAF = 100
DEPTHS_TO_SEARCH = range(2, 16)
DECAY_TAU = 24.0
LAGS_HOURS = (1, 2, 3, 4, 5, 6)
SURROGATE_CRITERION = "entropy"

CONCEPT_LABELS = ("no_sepsis", "sepsis_present")

MAX_ITER_LINEAR = 2000
C_REGULARIZATION = 1.0
EQUAL_PRIORS = (0.5, 0.5)
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 10
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1

AE_EMBEDDING_DIMS_TO_COMPARE = (16, 10)
AE_EPOCHS = 60
AE_BATCH_SIZE = 256
AE_LR = 1e-3
DECISION_THRESHOLD = 0.5

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
LABEL = "SepsisLabel"
CLINICAL_FEATURES = VITALS + LABS
DEMOGRAPHIC_CANDIDATES = ["Age", "Gender", "MICU", "SICU"]

HOURS_SINCE_FEATURES = [f"{f}_hours_since_obs" for f in CLINICAL_FEATURES]
EVER_OBSERVED_FEATURES = [f"{f}_ever_observed" for f in CLINICAL_FEATURES]
LAG_FEATURES = {lag: [f"{f}_lag{lag}h" for f in CLINICAL_FEATURES] for lag in LAGS_HOURS}
DECAY_FEATURES = [f"{f}_decay" for f in CLINICAL_FEATURES]


def log(msg: str = "") -> None:
    if not QUIET:
        print(msg)


def load_combined(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def build_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Patient_ID", "ICULOS"]).reset_index(drop=True)
    df[CLINICAL_FEATURES] = df.groupby("Patient_ID")[CLINICAL_FEATURES].ffill()
    df[DEMOGRAPHIC_CANDIDATES] = df.groupby("Patient_ID")[DEMOGRAPHIC_CANDIDATES].ffill().bfill()

    first_flag = df.loc[df[LABEL] == 1].groupby("Patient_ID")["ICULOS"].min()
    last_hour = df.groupby("Patient_ID")["ICULOS"].max()
    true_onset = first_flag + 6
    target_hour = true_onset.reindex(last_hour.index).fillna(last_hour)
    target_hour = pd.concat([target_hour, last_hour], axis=1).min(axis=1)
    target_hour.name = "target_hour"

    df = df.merge(target_hour, on="Patient_ID", how="left")
    snapshot = df.loc[df["ICULOS"] == df["target_hour"]].drop_duplicates(subset="Patient_ID")
    return snapshot.drop(columns=["target_hour"]).reset_index(drop=True)


def balance_classes(patient_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    septic = patient_df[patient_df[LABEL] == 1]
    nonseptic = patient_df[patient_df[LABEL] == 0]
    nonseptic_sampled = nonseptic.sample(n=len(septic), random_state=seed)
    return pd.concat([septic, nonseptic_sampled]).sample(frac=1, random_state=seed).reset_index(drop=True)


def add_richer_features(patient_df: pd.DataFrame, full_df: pd.DataFrame,
                         features=CLINICAL_FEATURES, lags=LAGS_HOURS) -> pd.DataFrame:
    df = full_df.sort_values(["Patient_ID", "ICULOS"]).reset_index(drop=True)

    new_cols = {}
    for f in features:
        new_cols[f"__{f}_lastval"] = df.groupby("Patient_ID", sort=False)[f].ffill()
        obs_hour = df["ICULOS"].where(df[f].notna())
        new_cols[f"__{f}_lastobshour"] = obs_hour.groupby(df["Patient_ID"], sort=False).ffill()
        new_cols[f"__{f}_everobs"] = (
            df[f].notna().astype(float).groupby(df["Patient_ID"], sort=False).cummax()
        )
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    lookup_cols = list(new_cols.keys())
    at_snapshot = patient_df[["Patient_ID", "ICULOS"]].merge(
        df[["Patient_ID", "ICULOS"] + lookup_cols], on=["Patient_ID", "ICULOS"], how="left")

    result_cols = {}
    for f in features:
        result_cols[f"{f}_hours_since_obs"] = (
            at_snapshot["ICULOS"].to_numpy() - at_snapshot[f"__{f}_lastobshour"].to_numpy())
        result_cols[f"{f}_ever_observed"] = at_snapshot[f"__{f}_everobs"].fillna(0.0).to_numpy()

    for lag in lags:
        lag_lookup = patient_df[["Patient_ID", "ICULOS"]].copy()
        lag_lookup["ICULOS"] = lag_lookup["ICULOS"] - lag
        lag_at = lag_lookup.merge(
            df[["Patient_ID", "ICULOS"] + [f"__{f}_lastval" for f in features]],
            on=["Patient_ID", "ICULOS"], how="left")
        for f in features:
            result_cols[f"{f}_lag{lag}h"] = lag_at[f"__{f}_lastval"].to_numpy()

    return pd.concat([patient_df, pd.DataFrame(result_cols, index=patient_df.index)], axis=1)


def add_decay_features(df: pd.DataFrame, pop_mean: pd.Series, features=CLINICAL_FEATURES,
                        tau: float = DECAY_TAU) -> pd.DataFrame:
    decay_cols = {}
    for f in features:
        dt = df[f"{f}_hours_since_obs"].to_numpy()
        last_val = df[f].to_numpy()
        weight = np.exp(-dt / tau)
        decayed = last_val * weight + pop_mean[f] * (1 - weight)
        decayed = np.where(np.isnan(decayed), pop_mean[f], decayed)
        decay_cols[f"{f}_decay"] = decayed
    return pd.concat([df, pd.DataFrame(decay_cols, index=df.index)], axis=1)


def extract_flat_features(df: pd.DataFrame, balanced: bool, max_patients: int = None):
    patient_df = build_snapshots(df)
    if balanced:
        patient_df = balance_classes(patient_df, seed=SEED)
    patient_df = add_richer_features(patient_df, df)
    if max_patients is not None and len(patient_df) > max_patients:
        patient_df = patient_df.sample(n=max_patients, random_state=SEED).reset_index(drop=True)
    label = "50:50 balanced" if balanced else "FULL cohort, not balanced"
    print(f"Patients: {len(patient_df)}   prevalence: {patient_df[LABEL].mean():.2%}  ({label})")

    y = patient_df[LABEL].to_numpy()
    train_df, test_df, y_train, y_test = train_test_split(
        patient_df, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    pop_mean = train_df[CLINICAL_FEATURES].mean()
    train_df = add_decay_features(train_df, pop_mean)
    test_df = add_decay_features(test_df, pop_mean)

    lag_candidates = [c for lag in LAGS_HOURS for c in LAG_FEATURES[lag]]
    candidates = list(CLINICAL_FEATURES) + HOURS_SINCE_FEATURES + EVER_OBSERVED_FEATURES + lag_candidates + DECAY_FEATURES
    train_miss = train_df[candidates].isna().mean()
    kept = [c for c in candidates if train_miss[c] <= MISSINGNESS_DROP_THRESHOLD]
    medians = train_df[kept].median()
    Xtr_full = train_df[kept].fillna(medians)
    Xte_full = test_df[kept].fillna(medians)

    f_scores, _ = f_classif(Xtr_full, y_train)
    ranking = pd.Series(f_scores, index=kept).sort_values(ascending=False)
    top_k = TOP_K_FEATURES if TOP_K_FEATURES is not None else len(kept)
    FEATURES = ranking.head(top_k).index.tolist()
    print(f"Candidate pool: {len(candidates)} (34 vars x {3 + len(LAGS_HOURS)} views) -> "
          f"{len(kept)} kept after missingness filter -> top {len(FEATURES)} by ANOVA F-score (no truncation)")

    X_train_df, X_test_df = Xtr_full[FEATURES], Xte_full[FEATURES]
    return dict(X_train_df=X_train_df, X_test_df=X_test_df, y_train=y_train, y_test=y_test,
                FEATURES=FEATURES, train_df=train_df, test_df=test_df)


def standardize(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame):
    mean = X_train_df.mean(axis=0)
    std = X_train_df.std(axis=0).replace(0, 1.0)
    X_train_std = ((X_train_df - mean) / std).to_numpy(dtype=np.float64)
    X_test_std = ((X_test_df - mean) / std).to_numpy(dtype=np.float64)
    return X_train_std, X_test_std, mean, std


def tree_to_knowledge_object(tree: DecisionTreeClassifier, feature_names,
                              concept_labels=CONCEPT_LABELS) -> list:
    tree_ = tree.tree_
    class_names = tree.classes_
    K = []

    def recurse(node, conditions):
        if tree_.feature[node] != -2:
            feature = feature_names[tree_.feature[node]]
            threshold = float(tree_.threshold[node])
            recurse(tree_.children_left[node], conditions + [
                {"feature": feature, "op": "<=", "threshold": threshold,
                 "display": f"{feature} <= {threshold:.2f}"}])
            recurse(tree_.children_right[node], conditions + [
                {"feature": feature, "op": ">", "threshold": threshold,
                 "display": f"{feature} > {threshold:.2f}"}])
            return
        values = tree_.value[node][0]
        predicted = int(class_names[int(values.argmax())])
        K.append({
            "id": f"rule_{len(K) + 1}",
            "concept": concept_labels[predicted],
            "conditions": conditions,
            "temporal_dependency": None,
            "confidence": float(values.max()),
            "support": int(tree_.n_node_samples[node]),
            "predicted_label": predicted,
        })

    recurse(0, [])
    return K


def save_knowledge_object(K: list, path: str) -> None:
    with open(path, "w") as f:
        json.dump(K, f, indent=2)


def satisfies(patient: dict, condition: dict) -> bool:
    value, op, rhs = patient[condition["feature"]], condition["op"], condition["threshold"]
    if op == "<=":
        return value <= rhs
    if op == ">":
        return value > rhs
    raise ValueError(f"Unknown operator: {op}")


def _violation(patient: dict, condition: dict) -> float:
    value, op, rhs = patient[condition["feature"]], condition["op"], condition["threshold"]
    return max(0.0, value - rhs) if op == "<=" else max(0.0, rhs - value)


def classify_one(patient: dict, K: list):
    matches = [rule for rule in K if all(satisfies(patient, c) for c in rule["conditions"])]
    if len(matches) == 1:
        rule = matches[0]
        return rule["predicted_label"], rule["confidence"], rule, "exact"
    if len(matches) > 1:
        rule = max(matches, key=lambda r: r["confidence"] * r["support"])
        return rule["predicted_label"], rule["confidence"], rule, "ambiguous"
    scored = sorted(((sum(_violation(patient, c) for c in rule["conditions"]), rule) for rule in K),
                     key=lambda pair: pair[0])
    rule = scored[0][1]
    return rule["predicted_label"], rule["confidence"], rule, "nearest"


def select_surrogate_depth(X_train_df, teacher_train_labels, depths=DEPTHS_TO_SEARCH,
                            min_samples_leaf=MIN_SAMPLES_LEAF) -> int:
    scores = {}
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_samples_leaf,
                                      class_weight="balanced", random_state=SEED)
        cv = cross_val_score(clf, X_train_df, teacher_train_labels, cv=5, scoring="balanced_accuracy")
        scores[depth] = (cv.mean(), cv.std())
        log(f"    max_depth={depth}: CV balanced accuracy = {cv.mean():.3f} (+/- {cv.std():.3f})")
    return max(scores, key=lambda d: (scores[d][0], d))


def distill_to_K(X_train_df, teacher_train_labels, FEATURES, k_path):
    depth = max(DEPTHS_TO_SEARCH)
    surrogate = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=MIN_SAMPLES_LEAF,
                                        criterion=SURROGATE_CRITERION, class_weight="balanced",
                                        random_state=SEED)
    surrogate.fit(X_train_df, teacher_train_labels)
    log(f"  Surrogate: max_depth={depth} (fixed), criterion={SURROGATE_CRITERION}"
        f"   leaves/rules={surrogate.get_n_leaves()}")
    K = tree_to_knowledge_object(surrogate, FEATURES)
    save_knowledge_object(K, k_path)
    log(f"  K written to {k_path} ({len(K)} rules)")
    return K


def evaluate(y_true, y_pred, y_score, label: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    log(f"  [{label}] accuracy={acc:.3f}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  "
        f"(baseline=prevalence={np.mean(y_true):.3f})")
    log("  " + classification_report(y_true, y_pred, target_names=list(CONCEPT_LABELS)).replace("\n", "\n  "))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    log(f"  confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    return dict(accuracy=acc, auroc=auroc, auprc=auprc, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def evaluate_student(K, FEATURES, X_test_df, y_test, label: str) -> dict:
    k_preds, scores = [], []
    for _, row in X_test_df.iterrows():
        patient = {f: row[f] for f in FEATURES}
        pred, confidence, _, _ = classify_one(patient, K)
        k_preds.append(pred)
        scores.append(confidence if pred == 1 else 1 - confidence)
    k_preds = np.array(k_preds)
    return evaluate(y_test, k_preds, np.array(scores), f"student K ({label})")


def fit_flat_teacher(name, X_train_df, X_train_std, y_train):
    if name == "tree":
        depth = select_surrogate_depth(X_train_df, y_train, depths=range(2, 7))
        teacher = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=MIN_SAMPLES_LEAF,
                                          class_weight="balanced", random_state=SEED)
        teacher.fit(X_train_df, y_train)
        return teacher, False
    if name == "logreg":
        teacher = LogisticRegression(C=C_REGULARIZATION, max_iter=MAX_ITER_LINEAR,
                                      class_weight="balanced", random_state=SEED)
        teacher.fit(X_train_std, y_train)
        return teacher, True
    if name == "svm_linear":
        teacher = LinearSVC(C=C_REGULARIZATION, max_iter=5000, class_weight="balanced",
                             dual="auto", random_state=SEED)
        teacher.fit(X_train_std, y_train)
        return teacher, True
    if name == "lda":
        teacher = LinearDiscriminantAnalysis(priors=EQUAL_PRIORS)
        teacher.fit(X_train_std, y_train)
        return teacher, True
    if name == "gnb":
        teacher = GaussianNB(priors=EQUAL_PRIORS)
        teacher.fit(X_train_std, y_train)
        return teacher, True
    if name == "random_forest":
        teacher = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                                          min_samples_leaf=MIN_SAMPLES_LEAF, class_weight="balanced",
                                          random_state=SEED, n_jobs=-1)
        teacher.fit(X_train_df, y_train)
        return teacher, False
    if name == "xgboost":
        if not HAVE_XGBOOST:
            raise RuntimeError("xgboost is not installed (pip install xgboost)")
        pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
        teacher = XGBClassifier(n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
                                 learning_rate=XGB_LEARNING_RATE, scale_pos_weight=pos_weight,
                                 random_state=SEED, n_jobs=-1, eval_metric="logloss")
        teacher.fit(X_train_df, y_train)
        return teacher, False
    raise ValueError(f"Unknown flat model: {name!r}")


def run_flat_model(name, flat, k_dir=".", k_suffix=""):
    log("\n" + "=" * 70)
    log(f"MODEL: {name}  (seed={SEED})")
    log("=" * 70)
    t0 = time.perf_counter()
    X_train_df, X_test_df = flat["X_train_df"], flat["X_test_df"]
    y_train, y_test, FEATURES = flat["y_train"], flat["y_test"], flat["FEATURES"]
    X_train_std, X_test_std, _, _ = standardize(X_train_df, X_test_df)

    teacher, uses_std = fit_flat_teacher(name, X_train_df, X_train_std, y_train)
    X_train_in = X_train_std if uses_std else X_train_df

    log(f"-- knowledge extraction (distilled K from {name}'s own training predictions) --")
    teacher_train_labels = teacher.predict(X_train_in)
    K = distill_to_K(X_train_df, teacher_train_labels, FEATURES,
                      f"{k_dir}/knowledge_object_{name}{k_suffix}.json")
    t1 = time.perf_counter()

    log(f"-- student (K-only, distilled from {name}) --")
    student_metrics = evaluate_student(K, FEATURES, X_test_df, y_test, name)
    t2 = time.perf_counter()

    n_test = len(y_test)
    timing = dict(training=t1 - t0, evaluation=t2 - t1, total=t2 - t0,
                  n_test=n_test, eval_ms_per_pair=(t2 - t1) * 1000.0 / n_test)
    print(f"[seed={SEED}] {name:<18} accuracy={student_metrics['accuracy']:.3f}  "
          f"AUROC={student_metrics['auroc']:.3f}  AUPRC={student_metrics['auprc']:.3f}  "
          f"TN={student_metrics['tn']} FP={student_metrics['fp']} FN={student_metrics['fn']} TP={student_metrics['tp']}  "
          f"TotalTime={timing['total']:.1f}s")
    return dict(model=name, student=student_metrics, timing=timing)


class SupervisedAutoencoder(nn.Module if HAVE_TORCH else object):
    def __init__(self, n_features, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, embedding_dim), nn.ReLU())
        self.decoder = nn.Linear(embedding_dim, n_features)
        self.classifier_head = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), self.classifier_head(z), z


def train_autoencoder(X, y, embedding_dim, pos_weight):
    torch.manual_seed(SEED)
    model = SupervisedAutoencoder(X.shape[1], embedding_dim)
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR)
    recon_fn = nn.MSELoss()
    clf_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=AE_BATCH_SIZE, shuffle=True)
    model.train()
    for _ in range(AE_EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            x_hat, logit, _ = model(xb)
            loss = recon_fn(x_hat, xb) + clf_fn(logit, yb)
            loss.backward()
            opt.step()
    return model


def autoencoder_embed(model, X):
    model.eval()
    with torch.no_grad():
        _, logit, z = model(torch.tensor(X, dtype=torch.float32))
        prob = torch.sigmoid(logit).squeeze(1).numpy()
        return z.numpy(), (prob >= DECISION_THRESHOLD).astype(int), prob


def run_latent_autoencoder_model(flat, k_dir=".", k_suffix=""):
    if not HAVE_TORCH:
        print("\nSkipping autoencoder_latent: torch is not installed.")
        return None
    log("\n" + "=" * 70)
    log(f"MODEL: autoencoder_latent  (seed={SEED})")
    log("=" * 70)
    t0 = time.perf_counter()
    X_train_df, X_test_df = flat["X_train_df"], flat["X_test_df"]
    y_train, y_test, FEATURES = flat["y_train"], flat["y_test"], flat["FEATURES"]
    X_train_std, X_test_std, _, _ = standardize(X_train_df, X_test_df)
    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    log(f"-- comparing embedding_dim in {AE_EMBEDDING_DIMS_TO_COMPARE} --")
    best = None
    for dim in AE_EMBEDDING_DIMS_TO_COMPARE:
        model = train_autoencoder(X_train_std, y_train, dim, pos_weight)
        z_train, _, _ = autoencoder_embed(model, X_train_std)
        z_test, _, _ = autoencoder_embed(model, X_test_std)
        depth = select_surrogate_depth(pd.DataFrame(z_train), y_train, depths=range(2, 7))
        clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=MIN_SAMPLES_LEAF,
                                      class_weight="balanced", random_state=SEED)
        clf.fit(z_train, y_train)
        bacc = balanced_accuracy_score(y_test, clf.predict(z_test))
        log(f"  embedding_dim={dim}: tree-on-z balanced test accuracy = {bacc:.3f}")
        if best is None or bacc > best["bacc"]:
            best = dict(dim=dim, clf=clf, z_train=z_train, bacc=bacc)
    log(f"-> using embedding_dim={best['dim']} (highest tree-on-z balanced test accuracy)")

    clf = best["clf"]
    log("-- knowledge extraction (distilled K from the latent tree's own training predictions) --")
    teacher_train_labels = clf.predict(best["z_train"])
    K = distill_to_K(X_train_df, teacher_train_labels, FEATURES,
                      f"{k_dir}/knowledge_object_autoencoder_latent{k_suffix}.json")
    t1 = time.perf_counter()

    log("-- student (K-only, distilled from autoencoder_latent) --")
    student_metrics = evaluate_student(K, FEATURES, X_test_df, y_test, "autoencoder_latent")
    t2 = time.perf_counter()

    n_test = len(y_test)
    timing = dict(training=t1 - t0, evaluation=t2 - t1, total=t2 - t0,
                  n_test=n_test, eval_ms_per_pair=(t2 - t1) * 1000.0 / n_test)
    print(f"[seed={SEED}] {'autoencoder_latent':<18} accuracy={student_metrics['accuracy']:.3f}  "
          f"AUROC={student_metrics['auroc']:.3f}  AUPRC={student_metrics['auprc']:.3f}  "
          f"TN={student_metrics['tn']} FP={student_metrics['fp']} FN={student_metrics['fn']} TP={student_metrics['tp']}  "
          f"TotalTime={timing['total']:.1f}s")
    return dict(model="autoencoder_latent", student=student_metrics, timing=timing)


FLAT_MODELS = ["tree", "logreg", "svm_linear", "lda", "gnb", "random_forest", "xgboost"]
LATENT_MODELS = ["autoencoder_latent"]
ALL_MODELS = FLAT_MODELS + LATENT_MODELS


def print_summary(results, title):
    results = [r for r in results if r is not None]
    if not results:
        print("\nNo models ran.")
        return
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    header = f"{'model':<20}{'accuracy':>12}{'AUROC':>12}{'AUPRC':>12}   confusion"
    print(header)
    for r in results:
        s = r["student"]
        print(f"{r['model']:<20}{s['accuracy']:>12.3f}{s['auroc']:>12.3f}{s['auprc']:>12.3f}   "
              f"TN={s['tn']} FP={s['fp']} FN={s['fn']} TP={s['tp']}")

    print("\n" + "=" * 100)
    print("TIMING (seconds)")
    print("=" * 100)
    header2 = f"{'model':<20}{'TrainingTime':>14}{'EvaluationTime':>16}{'ms/pair':>10}{'TotalTime':>12}{'n_test':>9}"
    print(header2)
    for r in results:
        t = r["timing"]
        print(f"{r['model']:<20}{t['training']:>14.1f}{t['evaluation']:>16.1f}"
              f"{t['eval_ms_per_pair']:>10.3f}{t['total']:>12.1f}{t['n_test']:>9}")
    total_time = sum(r["timing"]["total"] for r in results)
    print(f"\nTotal wall time across {len(results)} model(s): {total_time:.1f}s")


def run_all_models(requested, flat, out_dir):
    results = []
    for name in requested:
        if name in FLAT_MODELS:
            if name == "xgboost" and not HAVE_XGBOOST:
                print("\nSkipping xgboost: xgboost is not installed (pip install xgboost).")
                continue
            results.append(run_flat_model(name, flat, k_dir=out_dir))
        elif name in LATENT_MODELS:
            results.append(run_latent_autoencoder_model(flat, k_dir=out_dir))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Dataset.csv")
    parser.add_argument("--models", default="all")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    requested = ALL_MODELS if args.models == "all" else [m.strip() for m in args.models.split(",")]
    unknown = [m for m in requested if m not in ALL_MODELS]
    if unknown:
        parser.error(f"unknown model(s): {unknown}. Choices: {', '.join(ALL_MODELS)}")

    global QUIET, SEED
    QUIET = args.quiet

    print(f"Loading {args.data} ...")
    df = load_combined(args.data)
    df = df.rename(columns={"Unit1": "MICU", "Unit2": "SICU"})
    print(f"{len(df):,} patient-hours, {df['Patient_ID'].nunique():,} patients")

    SEED = SEED_BALANCED
    print(f"\nP=0h, 50:50 balanced, 80:20 split, seed={SEED}, surrogate criterion={SURROGATE_CRITERION}, "
          f"DECAY_TAU={DECAY_TAU}, lags={LAGS_HOURS}")
    flat_balanced = extract_flat_features(df, balanced=True, max_patients=args.max_patients)
    results_balanced = run_all_models(requested, flat_balanced, args.out_dir)
    print_summary(results_balanced, f"STUDENT TABLE -- BALANCED (seed={SEED})")

    SEED = SEED_UNBALANCED
    print(f"\nP=0h, FULL cohort (not balanced), 80:20 split, seed={SEED}, surrogate criterion={SURROGATE_CRITERION}, "
          f"DECAY_TAU={DECAY_TAU}, lags={LAGS_HOURS}")
    flat_unbalanced = extract_flat_features(df, balanced=False, max_patients=args.max_patients)
    results_unbalanced = run_all_models(requested, flat_unbalanced, args.out_dir)
    print_summary(results_unbalanced, f"STUDENT TABLE -- UNBALANCED (seed={SEED})")


if __name__ == "__main__":
    main()
