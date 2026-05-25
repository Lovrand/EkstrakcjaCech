import pandas as pd
import numpy as np
import time
import glob
import os
import warnings
import xgboost as xgb

# https://www.kaggle.com/datasets/surajsooraj26/iot-23
# Disabling logs and warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
pd.set_option('future.no_silent_downcasting', True)

# Disabling logs of TensorFlow (oneDNN)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from xgboost import XGBClassifier

# import TensorFlow (Autoenkoder)
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    import tensorflow as tf

    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("\n[Attention] No TensorFlow library.")


def prepare_data(filepaths, samples_per_class=10000):
    print("--- Step1: Loading and merging files ---")
    dataframes = []
    for file in filepaths:
        try:
            df_temp = pd.read_csv(file, low_memory=False)
            dataframes.append(df_temp)
            print(f" -> Loaded successfully {file}")
        except Exception as e:
            print(f" -> Loading error {file}: {e}")
            pass

    if not dataframes:
        raise ValueError("No data loaded")

    df = pd.concat(dataframes, ignore_index=True)

    print("\n--- Step2: Repairing the structure and splitting last column(in 3) ---")
    last_col = df.columns[-1]
    if 'label' in last_col.lower() or len(df.columns) == 21:
        df[['tunnel_parents', 'label', 'detailed-label']] = df[last_col].str.split(expand=True).iloc[:, :3]
        df = df.drop(columns=[last_col])

    print("\n--- Step3: Label classification and data set balancing ---")
    df['label'] = df['label'].astype(str).str.lower()
    df['target'] = df['label'].apply(lambda x: 0 if 'benign' in x else 1)

    # Creating 2 tables - safe and dangerous
    df_benign = df[df['target'] == 0]
    df_malicious = df[df['target'] == 1]

    # FULL SET
    # samples_per_class = min(len(df_benign), len(df_malicious))
    n_benign = min(samples_per_class, len(df_benign))
    n_malicious = min(samples_per_class, len(df_malicious))

    df_balanced = pd.concat([
        df_benign.sample(n=n_benign, random_state=44),
        df_malicious.sample(n=n_malicious, random_state=44)
    ], ignore_index=True)

    print(f"Size of set: {len(df_balanced)} (Benign: {n_benign}, Malicious: {n_malicious})")

    print("\n--- Step4: Data cleaning ---")
    cols_to_drop = ['ts', 'uid', 'id.orig_h', 'id.resp_h', 'detailed-label', 'tunnel_parents', 'label', 'target']
    X = df_balanced.drop(columns=[c for c in cols_to_drop if c in df_balanced.columns])
    y = df_balanced['target']

    # Exchanging '-' signs on NaN with type actualization
    X = X.replace('-', np.nan).infer_objects()

    for col in X.columns:
        numeric_attempt = pd.to_numeric(X[col], errors='coerce')
        if numeric_attempt.notna().sum() > 0:
            X[col] = numeric_attempt.fillna(0)
        else:
            # empty cells in word columns named unknown
            X[col] = X[col].fillna('unknown')
            # every word in column is given its own number
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # normalization
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    # deleting columns with same values in all cells
    X_scaled = X_scaled.loc[:, X_scaled.nunique() > 1]

    return X_scaled, y


# ==========================================
# Functions for feature extraction
# ==========================================

def get_rfecv_features(X_train, y_train, xgb_estimator, tolerance=0.02):
    print(f"\n[Method 1] Running RFECV...")
    cv = StratifiedKFold(3)
    # MODYFIKACJA [GPU]: Usunięto n_jobs=-1, ponieważ wielowątkowość CPU koliduje z pracą rdzeni CUDA
    rfecv = RFECV(estimator=xgb_estimator, step=1, cv=cv, scoring='accuracy')
    rfecv.fit(X_train, y_train)

    scores = rfecv.cv_results_['mean_test_score']
    max_score = np.max(scores)
    threshold = max_score - tolerance

    optimal_n_features = np.where(scores >= threshold)[0][0] + 1
    selected_indices = np.argsort(rfecv.ranking_)[:optimal_n_features]
    selected_features = X_train.columns[selected_indices].tolist()

    return selected_features, int(optimal_n_features)


def apply_select_k_best(X_train, y_train, X_test, k):
    print(f"\n[Method 2] Running SelectKBest (K={k})...")
    selector = SelectKBest(score_func=f_classif, k=k)  # ANOVA F-value
    X_train_red = selector.fit_transform(X_train, y_train)
    X_test_red = selector.transform(X_test)

    selected_indices = selector.get_support(indices=True)
    selected_features = X_train.columns[selected_indices].tolist()

    return X_train_red, X_test_red, selected_features


def apply_pca(X_train, X_test, n_components):
    print(f"\n[Method 3] Running PCA (Components={n_components})...")
    pca = PCA(n_components=n_components, random_state=44)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca


def apply_autoencoder(X_train, X_test, encoding_dim):
    print(f"\n[Method 4] Running Autoencoder (Latent space={encoding_dim})...")
    input_dim = X_train.shape[1]

    encoding_dim = int(encoding_dim)

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation='relu')(input_layer)
    bottleneck = Dense(encoding_dim, activation='linear')(encoded)
    decoded = Dense(16, activation='relu')(bottleneck)
    output_layer = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder = Model(inputs=input_layer, outputs=bottleneck)

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, epochs=15, batch_size=256, shuffle=True, validation_split=0.1, verbose=0)

    X_train_ae = encoder.predict(X_train, verbose=0)
    X_test_ae = encoder.predict(X_test, verbose=0)

    return X_train_ae, X_test_ae


# ==========================================
# Managing and testing
# ==========================================

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, base_time=None):
    # Trenowanie modelu (dane automatycznie trafią na GPU dzięki konfiguracji xgb_base)
    model.fit(X_tr, y_tr)

    # Classification time measurement
    start_time = time.perf_counter()

    # POPRAWKA [GPU]: Używamy model.predict() bezpośrednio na danych z CPU.
    # XGBoost w nowej wersji sam zoptymalizuje transfer do pamięci VRAM.
    preds = model.predict(X_te)

    inf_time = time.perf_counter() - start_time

    # Calculating metrics
    acc = accuracy_score(y_te, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_te, preds, average='macro')
    support = len(y_te)

    num_features = X_tr.shape[1]
    speedup = base_time / inf_time if base_time else 1.0

    print(f" -> Ended. Accuracy: {acc:.4f} | Classification time: {inf_time:.5f}s")

    return {
        'Method': name,
        'Dimensions': num_features,
        'Accuracy': f"{acc:.4f}",
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'F1-Score': f"{f1:.4f}",
        'Support': support,
        'Time [s]': f"{inf_time:.5f}",
        'Speed up': f"{speedup:.2f}x"
    }


def run_experiments(X, y):
    print("\n" + "=" * 80)
    print(" COMPARISON OF FEATURE EXTRACTION AND SELECTION METHODS (XGBOOST)")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44, stratify=y)

    # MODYFIKACJA [GPU]: Zmieniono n_jobs=-1 na obsługę akceleracji CUDA za pomocą 'tree_method' i 'device'
    xgb_base = XGBClassifier(
        random_state=44,
        eval_metric='logloss',
        tree_method='hist',
        device='cuda'
    )

    results = []

    # 0. BASE (all features)
    print("\n[Method 0] Base classifier (All features)")
    base_res = evaluate_model("Base (XGBoost)", xgb_base, X_train, y_train, X_test, y_test)
    base_time = float(base_res['Time [s]'])
    results.append(base_res)
    print(f" -> Used features: All {X_train.shape[1]} original features.")

    # 1. RFECV
    selected_features, target_k = get_rfecv_features(X_train, y_train, xgb_base)
    X_train_rfecv = X_train[selected_features]
    X_test_rfecv = X_test[selected_features]
    res_rfecv = evaluate_model("RFECV", xgb_base, X_train_rfecv, y_train, X_test_rfecv, y_test, base_time)
    results.append(res_rfecv)
    print(f" -> Selected Features ({target_k}): {selected_features}")

    # 2. SelectKBest
    X_train_kb, X_test_kb, kb_features = apply_select_k_best(X_train, y_train, X_test, k=target_k)
    res_kb = evaluate_model("SelectKBest", xgb_base, X_train_kb, y_train, X_test_kb, y_test, base_time)
    results.append(res_kb)
    print(f" -> Selected Features ({target_k}): {kb_features}")

    # 3. PCA
    X_train_pca, X_test_pca = apply_pca(X_train, X_test, n_components=target_k)
    res_pca = evaluate_model("PCA", xgb_base, X_train_pca, y_train, X_test_pca, y_test, base_time)
    results.append(res_pca)
    print(f" -> Used features: {target_k} new Principal Components (mathematical combinations of all features).")

    # 4. AUTOENCODER
    if TF_AVAILABLE:
        X_train_ae, X_test_ae = apply_autoencoder(X_train, X_test, encoding_dim=target_k)
        res_ae = evaluate_model("Autoencoder (DL)", xgb_base, X_train_ae, y_train, X_test_ae, y_test, base_time)
        results.append(res_ae)
        print(f" -> Used features: {target_k} new Latent Features (compressed by neural network).")

    # ==========================================
    # DISPLAYING THE RESULTS TABLE
    # ==========================================
    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY: METRICS AND PERFORMANCE")
    print("=" * 100)

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    folder = 'iot23_csv'
    paths = glob.glob(os.path.join(folder, '*.csv'))

    if not paths:
        print(f"No files in folder {folder}")
    else:
        try:
            X, y = prepare_data(paths, samples_per_class=500000)
            run_experiments(X, y)
        except Exception as e:
            print(f"Error: {e}")