from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------

DATA_FILE = "data/Soybean.csv"

MARKETS = [
    "Amravati", "Baran", "Bhiwani", "Karanja", "Khamgaon",
    "Kota", "Latur", "Dewas", "Ujjain"
]

MARKET_ALIASES = {
    "Amravati": ["Amravati", "Amrawati"],
    "Baran": ["Baran"],
    "Bhiwani": ["Bhiwani", "Bhawani"],
    "Karanja": ["Karanja"],
    "Khamgaon": ["Khamgaon"],
    "Kota": ["Kota"],
    "Latur": ["Latur"],
    "Dewas": ["Dewas"],
    "Ujjain": ["Ujjain"]
}

N_LAGS = 12
N_QUBITS = 4
SHOTS = 1024
RANDOM_SEED = 42
CLASSES = [0, 1, 2]

TRAIN_START, TRAIN_END = "2011-01", "2020-07"
VALID_START, VALID_END = "2020-08", "2021-12"
TEST_START, TEST_END = "2022-01", "2024-09"

OUTPUT_DIR = Path("soybean_results")
for folder in ["data_prepared", "quantum_features", "predictions",
               "confusion_matrices", "metrics", "models"]:
    (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# 2. Loading Data
# ---------------------------------------------------------------------

data = pd.read_csv(DATA_FILE, parse_dates=["Date"], dayfirst=True)
data = data.set_index("Date").sort_index()

# Match the market names used in the analysis with the names in the CSV.
column_names = {str(col).strip().lower(): col for col in data.columns}
selected_columns = {}

for market in MARKETS:
    for alias in MARKET_ALIASES[market]:
        if alias.lower() in column_names:
            selected_columns[market] = column_names[alias.lower()]
            break

price_data = data[[selected_columns[market] for market in MARKETS]].copy()
price_data.columns = MARKETS
price_data = price_data.apply(pd.to_numeric, errors="coerce")

# Relative change from the average price of the preceding 12 months.
changes = pd.DataFrame(index=price_data.index)

for market in MARKETS:
    previous_12_month_mean = price_data[market].rolling(12).mean().shift(1)
    changes[market] = (
        price_data[market] - previous_12_month_mean
    ) / previous_12_month_mean

# Class cut-offs are calculated from the training period only.
training_changes = changes[
    (changes.index.to_period("M") >= pd.Period(TRAIN_START)) &
    (changes.index.to_period("M") <= pd.Period(TRAIN_END))
].stack().dropna()

q40 = training_changes.quantile(0.40)
q60 = training_changes.quantile(0.60)

market_data = {}

for market in MARKETS:
    temp = pd.DataFrame(index=price_data.index)
    temp["price"] = price_data[market]

    for lag in range(1, N_LAGS + 1):
        temp[f"x{lag}"] = price_data[market].shift(lag)

    temp["y"] = np.select(
        [changes[market] <= q40, changes[market] <= q60],
        [0, 1],
        default=2
    )
    temp.loc[changes[market].isna(), "y"] = np.nan

    # Keep January 2011 to September 2024 as target observations.
    month = temp.index.to_period("M")
    temp = temp[(month >= pd.Period("2011-01")) &
                (month <= pd.Period("2024-09"))].copy()

    lag_columns = [f"x{i}" for i in range(1, N_LAGS + 1)]
    temp = temp.dropna(subset=lag_columns + ["y"])
    temp["y"] = temp["y"].astype(int)

    # Relative lag values, as used in the Qiskit feature-extraction script.
    lag_mean = temp[lag_columns].mean(axis=1)
    temp[lag_columns] = temp[lag_columns].div(lag_mean, axis=0) - 1

    market_data[market] = temp


# ---------------------------------------------------------------------
# 3. Quantum Feature Extraction
# ---------------------------------------------------------------------

def make_circuit(features):
    circuit = QuantumCircuit(N_QUBITS, N_QUBITS)
    circuit.h(range(N_QUBITS))

    # Three Ry rotations on each of the four qubits (12 input features).
    for qubit in range(N_QUBITS):
        for position in range(3):
            circuit.ry(float(features[3 * qubit + position]), qubit)

    # Fully connected CNOT layer.
    for control in range(N_QUBITS):
        for target in range(control + 1, N_QUBITS):
            circuit.cx(control, target)

    circuit.h(range(N_QUBITS))
    circuit.measure(range(N_QUBITS), range(N_QUBITS))
    return circuit


def quantum_features(X, backend):
    circuits = [make_circuit(row) for row in X]
    compiled = transpile(circuits, backend, optimization_level=0,
                         seed_transpiler=RANDOM_SEED)
    result = backend.run(compiled, shots=SHOTS,
                         seed_simulator=RANDOM_SEED).result()

    probabilities = np.zeros((len(circuits), 2 ** N_QUBITS))

    for i in range(len(circuits)):
        counts = result.get_counts(i)
        total = sum(counts.values())
        for bitstring, count in counts.items():
            state = int(bitstring.replace(" ", ""), 2)
            probabilities[i, state] = count / total

    return probabilities


# ---------------------------------------------------------------------
# 4. Preparing Models
# ---------------------------------------------------------------------

def get_models():
    return {
        "Logistic_Regression": LogisticRegression(
            random_state=RANDOM_SEED, max_iter=10000
        ),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Random_Forest": RandomForestClassifier(
            n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "SVM": SVC(random_state=RANDOM_SEED, probability=True),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(45, 20), random_state=RANDOM_SEED,
            max_iter=1000
        )
    }


def get_metrics(actual, predicted):
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision_weighted": precision_score(
            actual, predicted, labels=CLASSES,
            average="weighted", zero_division=0
        ),
        "Recall_weighted": recall_score(
            actual, predicted, labels=CLASSES,
            average="weighted", zero_division=0
        ),
        "F1_weighted": f1_score(
            actual, predicted, labels=CLASSES,
            average="weighted", zero_division=0
        )
    }


def save_results(market, split_name, actual, predictions):
    prediction_table = pd.DataFrame({"Actual": actual})
    metric_rows = []

    for model_name, predicted in predictions.items():
        prediction_table[model_name] = predicted

        cm = confusion_matrix(actual, predicted, labels=CLASSES)
        cm_table = pd.DataFrame(
            cm,
            index=[f"Actual_{c}" for c in CLASSES],
            columns=[f"Predicted_{c}" for c in CLASSES]
        )
        cm_table.to_csv(
            OUTPUT_DIR / "confusion_matrices" /
            f"{market}_{split_name}_{model_name}.csv"
        )

        metric_rows.append({
            "Market": market,
            "Split": split_name,
            "Model": model_name,
            **get_metrics(actual, predicted)
        })

    prediction_table.to_csv(
        OUTPUT_DIR / "predictions" /
        f"{market}_{split_name}_actual_predicted.csv",
        index_label="Date"
    )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        OUTPUT_DIR / "metrics" / f"{market}_{split_name}_metrics.csv",
        index=False
    )
    return metrics


# ---------------------------------------------------------------------
# 5. Training the Models
# ---------------------------------------------------------------------

backend = AerSimulator()
all_metrics = []

for market in MARKETS:
    print(f"\nProcessing {market}...")

    temp = market_data[market]
    temp.to_csv(
        OUTPUT_DIR / "data_prepared" / f"{market}_lagged_data.csv",
        index_label="Date"
    )

    # Split by calendar month, so first-of-month and month-end dates both work.
    month = temp.index.to_period("M")
    train = temp[(month >= pd.Period(TRAIN_START)) &
                 (month <= pd.Period(TRAIN_END))]
    validation = temp[(month >= pd.Period(VALID_START)) &
                      (month <= pd.Period(VALID_END))]
    test = temp[(month >= pd.Period(TEST_START)) &
                (month <= pd.Period(TEST_END))]

    print(f"Train: {len(train)}, Validation: {len(validation)}, Test: {len(test)}")

    lag_columns = [f"x{i}" for i in range(1, N_LAGS + 1)]

    X_train = train[lag_columns]
    y_train = train["y"]
    X_validation = validation[lag_columns]
    y_validation = validation["y"]
    X_test = test[lag_columns]
    y_test = test["y"]

    # Standardise using training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(
        scaler, OUTPUT_DIR / "models" / f"{market}_feature_scaler.joblib"
    )

    # Generate the 16 quantum measurement probabilities.
    q_train = quantum_features(X_train_scaled, backend)
    q_validation = quantum_features(X_validation_scaled, backend)
    q_test = quantum_features(X_test_scaled, backend)

    # Classical and quantum-enhanced feature matrices.
    Xc_train = X_train_scaled
    Xc_validation = X_validation_scaled
    Xc_test = X_test_scaled

    Xq_train = np.hstack([X_train_scaled, q_train])
    Xq_validation = np.hstack([X_validation_scaled, q_validation])
    Xq_test = np.hstack([X_test_scaled, q_test])

    # Save feature matrices.
    quantum_columns = [f"qprob_{i:04b}" for i in range(16)]
    classical_columns = lag_columns
    enhanced_columns = classical_columns + quantum_columns

    for split_name, features, columns, index in [
        ("train_classical", Xc_train, classical_columns, X_train.index),
        ("validation_classical", Xc_validation, classical_columns, X_validation.index),
        ("test_classical", Xc_test, classical_columns, X_test.index),
        ("train_quantum", Xq_train, enhanced_columns, X_train.index),
        ("validation_quantum", Xq_validation, enhanced_columns, X_validation.index),
        ("test_quantum", Xq_test, enhanced_columns, X_test.index)
    ]:
        pd.DataFrame(features, index=index, columns=columns).to_csv(
            OUTPUT_DIR / "quantum_features" / f"{market}_{split_name}.csv",
            index_label="Date"
        )

    prediction_sets = {}
    fitted_models = {}

    # Fit the five classifiers on classical and quantum-enhanced data.
    for feature_name, train_features, validation_features, test_features in [
        ("Classical", Xc_train, Xc_validation, Xc_test),
        ("Quantum", Xq_train, Xq_validation, Xq_test)
    ]:
        for model_name, model in get_models().items():
            print(f"  {feature_name}: {model_name}")
            model.fit(train_features, y_train)

            key = f"{market}_{feature_name}_{model_name}"
            fitted_models[key] = model

            prediction_sets[f"{feature_name}_{model_name}"] = {
                "train": model.predict(train_features),
                "validation": model.predict(validation_features),
                "test": model.predict(test_features)
            }

    # Weighted ensemble of SVM and Random Forest using quantum features.
    svm_validation = SVC(
        random_state=RANDOM_SEED, probability=True
    ).fit(Xq_train, y_train)
    rf_validation = RandomForestClassifier(
        n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1
    ).fit(Xq_train, y_train)

    svm_val_prob = svm_validation.predict_proba(Xq_validation)
    rf_val_prob = rf_validation.predict_proba(Xq_validation)

    # Find the SVM/RF weights that give the highest validation accuracy.
    best_accuracy = -1
    best_weights = (0.5, 0.5)

    for svm_weight in np.arange(0, 1.01, 0.01):
        rf_weight = 1 - svm_weight
        combined_prob = svm_weight * svm_val_prob + rf_weight * rf_val_prob
        predicted = np.array(CLASSES)[np.argmax(combined_prob, axis=1)]
        accuracy = accuracy_score(y_validation, predicted)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = (svm_weight, rf_weight)

    svm_weight, rf_weight = best_weights
    validation_ensemble = np.array(CLASSES)[
        np.argmax(
            svm_weight * svm_val_prob + rf_weight * rf_val_prob, axis=1
        )
    ]

    # Refit both ensemble models on training + validation data.
    X_train_validation = np.vstack([Xq_train, Xq_validation])
    y_train_validation = pd.concat([y_train, y_validation])

    final_svm = SVC(
        random_state=RANDOM_SEED, probability=True
    ).fit(X_train_validation, y_train_validation)
    final_rf = RandomForestClassifier(
        n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1
    ).fit(X_train_validation, y_train_validation)

    fitted_models[f"{market}_Quantum_Ensemble_SVM"] = final_svm
    fitted_models[f"{market}_Quantum_Ensemble_RF"] = final_rf

    svm_test_prob = final_svm.predict_proba(Xq_test)
    rf_test_prob = final_rf.predict_proba(Xq_test)
    test_ensemble = np.array(CLASSES)[
        np.argmax(svm_weight * svm_test_prob + rf_weight * rf_test_prob, axis=1)
    ]

    # Training predictions are in-sample predictions from the final refitted models.
    svm_train_prob = final_svm.predict_proba(Xq_train)
    rf_train_prob = final_rf.predict_proba(Xq_train)
    train_ensemble = np.array(CLASSES)[
        np.argmax(svm_weight * svm_train_prob + rf_weight * rf_train_prob, axis=1)
    ]

    prediction_sets["Quantum_Weighted_Ensemble"] = {
        "train": train_ensemble,
        "validation": validation_ensemble,
        "test": test_ensemble
    }

    # Save fitted models.
    for model_name, model in fitted_models.items():
        joblib.dump(model, OUTPUT_DIR / "models" / f"{model_name}.joblib")

    # Save actual values, predictions, confusion matrices and metrics.
    train_predictions = {
        name: values["train"] for name, values in prediction_sets.items()
    }
    validation_predictions = {
        name: values["validation"] for name, values in prediction_sets.items()
    }
    test_predictions = {
        name: values["test"] for name, values in prediction_sets.items()
    }

    all_metrics.append(save_results(
        market, "train", y_train, train_predictions
    ))
    all_metrics.append(save_results(
        market, "validation", y_validation, validation_predictions
    ))
    all_metrics.append(save_results(
        market, "test", y_test, test_predictions
    ))

    print(
        f"  Ensemble weights: SVM={svm_weight:.2f}, "
        f"Random Forest={rf_weight:.2f}"
    )

    with open(OUTPUT_DIR / f"{market}_ensemble_weights.json", "w") as file:
        json.dump({
            "SVM_weight": float(svm_weight),
            "Random_Forest_weight": float(rf_weight),
            "validation_accuracy": float(best_accuracy)
        }, file, indent=4)


# ---------------------------------------------------------------------
# 6. Saving the Metrics
# ---------------------------------------------------------------------

combined_metrics = pd.concat(all_metrics, ignore_index=True)
combined_metrics.to_csv(
    OUTPUT_DIR / "all_markets_all_metrics.csv", index=False
)

combined_metrics[combined_metrics["Split"] == "test"].to_csv(
    OUTPUT_DIR / "test_metrics_all_markets.csv", index=False
)

print("\nAnalysis completed.")
print(f"Results saved in: {OUTPUT_DIR.resolve()}")
